#!/usr/bin/env python3
"""
Script to run test_net.py on all eccv checkpoints with automatic config updates
based on checkpoint modalities.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

# Configuration
CHECKPOINTS_DIR = Path("/vol/bitbucket/sna21/CUENet/best_checkpoints/eccv")
CONFIG_TEMPLATE = Path("/vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/exp/RWF_exp/config.yaml")
TOOLS_DIR = Path("/vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/tools")
RESULTS_DIR = Path("/vol/bitbucket/sna21/CUENet/test_results")


class CheckpointConfig:
    """Represents modality configuration extracted from checkpoint name"""
    
    def __init__(self, checkpoint_name: str):
        self.name = checkpoint_name
        name_lower = checkpoint_name.lower()
        self.use_rgb = "rgb" in name_lower
        self.use_pose = "pose" in name_lower
        self.use_text = "text" in name_lower
        # Only enable LoRA if "lora" is present AND "no_lora" is not present
        self.enable_lora = "lora" in name_lower and "no_lora" not in name_lower
        self.soft_prompt = "soft" in name_lower
        self.use_velocity = "velo" in name_lower and "no_velo" not in name_lower
    
    def __str__(self):
        return (f"RGB={self.use_rgb}, Pose={self.use_pose}, Text={self.use_text}, "
                f"LoRA={self.enable_lora}, Soft={self.soft_prompt}, Velocity={self.use_velocity}")
    
    def to_dict(self):
        """Convert to dictionary for YAML update"""
        return {
            'use_rgb': self.use_rgb,
            'use_pose': self.use_pose,
            'use_text': self.use_text,
            'enable_lora': self.enable_lora,
            'soft_prompt': self.soft_prompt,
            'use_velocity': self.use_velocity,
        }


def load_config(config_path: Path) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path) -> None:
    """Save YAML configuration"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config_for_checkpoint(
    config: dict, 
    checkpoint_path: Path, 
    modality_config: CheckpointConfig,
    checkpoint_name: str
) -> None:
    """Update config with checkpoint path and modality settings"""
    
    # Ensure TEST section exists
    if 'TEST' not in config:
        config['TEST'] = {}
    
    # Update checkpoint path
    config['TEST']['CHECKPOINT_FILE_PATH'] = str(checkpoint_path)
    
    # Generate prediction CSV path based on checkpoint name
    predictions_dir = Path("/vol/bitbucket/sna21/dataset/predictions/eccv")
    csv_filename = f"{checkpoint_name}.csv"
    config['TEST']['PREDICTION_CSV_PATH'] = str(predictions_dir / csv_filename)
    
    # Ensure MODEL section exists
    if 'MODEL' not in config:
        config['MODEL'] = {}
    
    # Update modality settings
    config['MODEL']['USE_RGB'] = modality_config.use_rgb
    config['MODEL']['USE_POSE'] = modality_config.use_pose
    config['MODEL']['USE_TEXT'] = modality_config.use_text
    config['MODEL']['ENABLE_LORA'] = modality_config.enable_lora
    config['MODEL']['SOFT_PROMPT'] = modality_config.soft_prompt
    config['MODEL']['POSE_USE_VELOCITY'] = modality_config.use_velocity


def run_test(config_path: Path, checkpoint_name: str, results_dir: Path) -> bool:
    """Run test_net.py with given config"""
    
    log_file = results_dir / f"{checkpoint_name}_test.log"
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                [sys.executable, 'test_net.py', '--cfg', str(config_path)],
                cwd=str(TOOLS_DIR),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600  # 1 hour timeout
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timed out for {checkpoint_name}")
        return False
    except Exception as e:
        print(f"  ✗ Error running test for {checkpoint_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests on all eccv checkpoints")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without running tests")
    parser.add_argument('--checkpoint', type=str, help="Run test only for specific checkpoint")
    args = parser.parse_args()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config = load_config(CONFIG_TEMPLATE)
    
    # Collect all checkpoints
    checkpoints = []
    for checkpoint_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if checkpoint_dir.is_dir():
            checkpoint_file = checkpoint_dir / "best.pyth"
            if checkpoint_file.exists():
                checkpoints.append((checkpoint_dir.name, checkpoint_file))
            else:
                print(f"Warning: Checkpoint file not found in {checkpoint_dir}")
    
    if not checkpoints:
        print("Error: No checkpoints found!")
        return 1
    
    # Filter by specific checkpoint if requested
    if args.checkpoint:
        checkpoints = [(name, path) for name, path in checkpoints if args.checkpoint in name]
        if not checkpoints:
            print(f"Error: No checkpoints matching '{args.checkpoint}'")
            return 1
    
    print(f"Found {len(checkpoints)} checkpoint(s) to process")
    print("=" * 60)
    
    results = {}
    
    for checkpoint_name, checkpoint_file in checkpoints:
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        print(f"  Checkpoint path: {checkpoint_file}")
        
        # Determine modalities
        modality_config = CheckpointConfig(checkpoint_name)
        print(f"  Configuration: {modality_config}")
        
        if args.dry_run:
            print("  [DRY RUN] Would run test with this configuration")
            results[checkpoint_name] = "dry-run"
            continue
        
        # Update config
        config = load_config(CONFIG_TEMPLATE)
        update_config_for_checkpoint(config, checkpoint_file, modality_config, checkpoint_name)
        
        # Save temporary config
        temp_config_path = RESULTS_DIR / f"{checkpoint_name}_config.yaml"
        save_config(config, temp_config_path)
        
        # Run test
        print("  Running test...")
        success = run_test(temp_config_path, checkpoint_name, RESULTS_DIR)
        
        if success:
            print(f"  ✓ Test completed successfully")
            results[checkpoint_name] = "success"
        else:
            print(f"  ✗ Test failed (check {RESULTS_DIR}/{checkpoint_name}_test.log for details)")
            results[checkpoint_name] = "failed"
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for checkpoint_name, status in results.items():
        symbol = "✓" if status == "success" else "✗" if status == "failed" else "→"
        print(f"{symbol} {checkpoint_name}: {status}")
    
    if not args.dry_run:
        success_count = sum(1 for s in results.values() if s == "success")
        failed_count = sum(1 for s in results.values() if s == "failed")
        print(f"\nTotal: {success_count} succeeded, {failed_count} failed")
        print(f"Results saved in: {RESULTS_DIR}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
