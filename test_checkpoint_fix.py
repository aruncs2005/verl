#!/usr/bin/env python3
"""
Test script to verify the checkpoint manager fix
"""
import torch
import tempfile
import os
from verl.utils.checkpoint.fsdp_checkpoint_manager import CheckpointState

def test_rng_state_consistency():
    """Test that RNG state functions work consistently between save and load"""
    
    print("=== Testing RNG State Function Consistency ===")
    
    # Create a simple model for testing
    model = torch.nn.Linear(10, 1)
    
    # Define RNG state getter function (for saving)
    def get_rng_state():
        return {
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    
    # Test save scenario
    print("1. Testing save scenario (rng_state_fn as getter)...")
    save_state = CheckpointState(
        model=model,
        rng_state_fn=get_rng_state,  # Getter function
        global_step=100
    )
    
    try:
        save_state_dict = save_state.state_dict()
        print("✓ Save state_dict() succeeded")
        print(f"  RNG state keys: {save_state_dict['extra']['rng'].keys() if save_state_dict['extra']['rng'] else 'None'}")
    except Exception as e:
        print(f"✗ Save state_dict() failed: {e}")
        return False
    
    # Test load scenario
    print("2. Testing load scenario (no rng_state_fn)...")
    load_state = CheckpointState(
        model=model,
        rng_state_fn=None,  # No function for loading
        global_step=0
    )
    
    try:
        load_state.load_state_dict(save_state_dict)
        print("✓ Load state_dict() succeeded")
        print(f"  Loaded global_step: {load_state.global_step}")
    except Exception as e:
        print(f"✗ Load state_dict() failed: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

def test_function_signature_mismatch():
    """Test the original problematic scenario"""
    
    print("\n=== Testing Original Problem Scenario ===")
    
    model = torch.nn.Linear(10, 1)
    
    # This was the problematic lambda from the original code
    problematic_lambda = lambda rng_state: torch.set_rng_state(rng_state["cpu_rng_state"])
    
    print("1. Testing problematic lambda as rng_state_fn...")
    problem_state = CheckpointState(
        model=model,
        rng_state_fn=problematic_lambda,  # This expects 1 argument
        global_step=100
    )
    
    try:
        # This should fail because state_dict() calls rng_state_fn() with no arguments
        # but the lambda expects 1 argument
        problem_state_dict = problem_state.state_dict()
        print("✗ This should have failed but didn't!")
        return False
    except TypeError as e:
        print(f"✓ Expected failure occurred: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Checkpoint Manager Fix Verification")
    print("=" * 50)
    
    # Test the fix
    success1 = test_rng_state_consistency()
    
    # Test the original problem
    success2 = test_function_signature_mismatch()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")