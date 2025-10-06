# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Optional
import time
import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict


from amzn_sagemaker_checkpointing.config.sagemaker_checkpoint_config import SageMakerCheckpointConfig
from amzn_sagemaker_checkpointing.checkpointing.filesystem.filesystem import (
    SageMakerTieredStorageWriter,
    SageMakerTieredStorageReader
)

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.checkpoint import load
import torch.distributed.checkpoint as dist_cp

from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def upload_folder_to_s3(local_folder, bucket_name, s3_folder_prefix=""):
    s3_client = boto3.client('s3')
    # Walk the local folder recursively
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create relative path and then S3 object key
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_key = os.path.join(s3_folder_prefix, relative_path).replace("\\", "/")
            try:
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
            except FileNotFoundError:
                print(f"File {local_file_path} was not found.")
            except NoCredentialsError:
                print("AWS credentials not available.")
                return
            except ClientError as e:
                print(f"Failed to upload {local_file_path}: {e}")

def split_s3_uri(s3_uri):
    # Remove the s3:// prefix
    if s3_uri.startswith('s3://'):
        path = s3_uri[5:]
    else:
        path = s3_uri
    # Split on the first slash to get bucket and prefix
    parts = path.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    return bucket, prefix


# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class CheckpointState(Stateful):
    def __init__(self, model, optimizer=None, lr_scheduler=None, rng_state_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.rng_state_fn = rng_state_fn  # function that returns RNG state dict

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler else None
        extra_state = {
            "lr_scheduler": lr_scheduler_state_dict,
            "rng": self.rng_state_fn() if self.rng_state_fn else None,
        }
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "extra": extra_state
        }

    def load_state_dict(self, state_dict):
        # Implement loading logic, if needed
        self.model.load_state_dict(state_dict["model"])

        # Load optimizer state dict if optimizer exists and state is provided
        if self.optimizer and state_dict.get("optimizer") is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        # Load lr_scheduler state dict if lr_scheduler exists and state is provided
        if self.lr_scheduler and state_dict.get("extra", {}).get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(state_dict["extra"]["lr_scheduler"])

            # Restore RNG state if rng_state_fn and corresponding state is provided
        if self.rng_state_fn and state_dict.get("extra", {}).get("rng") is not None:
            rng_state = state_dict["extra"]["rng"]
            # Assume rng_state_fn is a setter function or callable to restore RNG state
            # If rng_state_fn is a getter, then you need to define a setter separately
            # This example assumes rng_state_fn sets RNG state when called with arg
            self.rng_state_fn(rng_state)

@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, s3_base_path: str= None, ckpt_namespace:str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # Ensure required components exist
        assert self.model is not None, "Model must be provided for checkpoint loading"

        # Wrap stateful holder (will call load_state_dict automatically)
        state = CheckpointState(
            self.model,
            self.optimizer if self.should_load_optimizer else None,
            self.lr_scheduler if self.should_load_extra else None,
            rng_state_fn=(lambda rng_state: torch.set_rng_state(rng_state["cpu_rng_state"]))
                         if self.should_load_extra else None,
        )

        # Setup SageMaker Tiered Storage Reader
        smcheckpointconfig = SageMakerCheckpointConfig(
            namespace=ckpt_namespace,
            world_size=torch.distributed.get_world_size(),
            s3_tier_base_path=s3_base_path,
            logger=logger,
            save_to_s3=True
        )
        checkpoint_reader = SageMakerTieredStorageReader(
            checkpoint_config=smcheckpointconfig
        )

        # Typically we load the latest step, so try to infer a checkpoint_id
        # For example: look for "step_x" directories inside local_path
        checkpoint_id = None
        if os.path.exists(local_path):
            candidates = [d for d in os.listdir(local_path) if d.startswith("step_")]
            if candidates:
                checkpoint_id = sorted(candidates)[-1]  # latest checkpoint
        if checkpoint_id is None:
            logger.warning(f"No checkpoint found under {local_path}")
            return

        # Now perform distributed checkpoint load
        load_future = dcp.async_load(
            state_dict={"app": state},
            storage_reader=checkpoint_reader,
            checkpoint_id=checkpoint_id,
        )
        load_future.result()

        torch.distributed.barrier()



    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0,s3_base_path: str = None, ckpt_namespace: str = None , max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path, only rank 0 should do this
        if local_path is None:
            return

        # Rotate old checkpoints manually as before (rank 0 only)
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # Ensure required components exist
        assert self.model is not None, "Model must be provided for checkpointing"
        if self.should_save_optimizer:
            assert self.optimizer is not None, "Optimizer must be provided for checkpointing"

        # Define a RNG state retrieval function if needed
        def get_rng_state():
            # Could return CPU and GPU RNG states as dict
            return {
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        # Wrap into DCP Stateful object
        state = CheckpointState(
            self.model,
            self.optimizer if self.should_save_optimizer else None,
            self.lr_scheduler if self.should_save_extra else None,
            rng_state_fn=get_rng_state if self.should_save_extra else None,
        )

        # Create StorageWriter (cache pinned memory for speed on repeat saves)
        #if not hasattr(self, "checkpoint_writer"):
        torch.cuda.empty_cache()
        #namespace=os.environ.get('TRAINING_JOB_NAME', f'job-{int(time.time())}')
        # s3_ckpt_freq = 100
        # save_to_s3 = global_step % s3_ckpt_freq == 0
        smcheckpointconfig = SageMakerCheckpointConfig(
                namespace=ckpt_namespace,
                world_size=torch.distributed.get_world_size(),
                s3_tier_base_path=s3_base_path,
                logger=logger,
                save_to_s3=False
            )
        self.checkpoint_writer = SageMakerTieredStorageWriter(
            checkpoint_config=smcheckpointconfig,
            step=global_step
            )
        
        # Await previous async save result to avoid multiple concurrent saves

        if self.checkpoint_future is not None:
            exc = self.checkpoint_future.exception()
            if exc:
                print(f"Failure in saving previous checkpoint:{str(exc)}")
                #Handle failures as required
            else:
                result = self.checkpoint_future.result()
        # if hasattr(self, "checkpoint_future") and self.checkpoint_future is not None:
        #     self.checkpoint_future.result()

        checkpoint_id = f"step_{global_step}"
        self.checkpoint_future = dcp.async_save(
            state_dict={"app": state},
            storage_writer=self.checkpoint_writer,
            checkpoint_id=checkpoint_id,
        )
        self.checkpoint_future.result()

        torch.distributed.barrier()

        # Rank 0: Save HF tokenizer and configs in HF standardized directory
        if self.rank == -1:
            hf_dir = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_dir)
            # Unwrap model for Hugging Face config
            unwrap_model = getattr(self.model, "_fsdp_wrapped_module", self.model)

            # Save Model config
            model_config = unwrap_model.config
            model_config.save_pretrained(hf_dir)

            # Save tokenizer or processing class if defined
            if hasattr(self, "processing_class"):
                self.processing_class.save_pretrained(hf_dir)

            # Save generation config if available
            generation_config = None
            if hasattr(unwrap_model, "can_generate") and unwrap_model.can_generate():
                try:
                    from transformers import GenerationConfig
                    if hasattr(model_config, "name_or_path") and model_config.name_or_path:
                        generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                        generation_config.save_pretrained(hf_dir)
                except Exception:
                    pass
            # Save fsdp config JSON
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = {
                "FSDP_version": fsdp_version(self.model),
                "world_size": self.world_size,
            }
            with open(fsdp_config_path, "w") as f:
                json.dump(fsdp_config, f, indent=4)
            # If needed, save custom model code or auto_map logic here as well
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_dir, config=model_config)
            bucket, prefix = split_s3_uri(s3_base_path)
            upload_folder_to_s3(local_path,bucket, prefix + "/" + ckpt_namespace)


        torch.distributed.barrier()

        # Optionally save full non-sharded HF model (rank 0 only)
        if self.should_save_hf_model and self.rank == 0:
            # Get full FSDP state dict offloaded to CPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            arch = unwrap_model.config.architectures[0] if hasattr(unwrap_model.config, "architectures") else ""
            if "ForTokenClassification" in arch:
                from transformers import AutoModelForTokenClassification
                auto_model_cls = AutoModelForTokenClassification
            elif "ForCausalLM" in arch:
                from transformers import AutoModelForCausalLM
                auto_model_cls = AutoModelForCausalLM
            elif "ForConditionalGeneration" in arch:
                # Handle transformers versions
                import transformers
                from packaging import version
                if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                    from transformers import AutoModelForImageTextToText
                    auto_model_cls = AutoModelForImageTextToText
                else:
                    from transformers import AutoModelForVision2Seq
                    auto_model_cls = AutoModelForVision2Seq
            else:
                raise NotImplementedError(f"Unknown architecture {arch}")

            from accelerate import init_empty_weights
            with init_empty_weights():
                save_model = auto_model_cls.from_config(unwrap_model.config, torch_dtype=torch.bfloat16)
            save_model.to_empty(device="cpu")

            if save_model.can_generate():
                if generation_config is not None:
                    save_model.generation_config = generation_config
                else:
                    print("Warning: Generation config file not found, using model config fallback when saving hf_model.")

            save_model.save_pretrained(hf_local_path, state_dict=state_dict)
            bucket, prefix = split_s3_uri(s3_base_path)

            upload_folder_to_s3(hf_local_path,bucket,prefix + "/" + ckpt_namespace + "/final_model")


            del state_dict
            del save_model

        torch.distributed.barrier()

        # Record this local path for checkpoint rotation
        self.previous_saved_paths.append(local_path)

   