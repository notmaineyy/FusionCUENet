# Running Tests on All ECCV Checkpoints

This directory contains scripts to automatically run `test_net.py` on all checkpoints in the `best_checkpoints/eccv/` directory, with automatic config updates based on checkpoint modalities.

## Overview

Different checkpoints use different modality combinations (RGB, Pose, Text, LoRA, Soft Prompts, etc.). The scripts automatically:
1. Parse the checkpoint directory name to infer modalities
2. Update `config.yaml` with the appropriate settings
3. Run `test_net.py` with the updated configuration
4. Save test logs and results

## Scripts

### 1. Python Script (Recommended) - `run_all_tests.py`

**Advantages:**
- Better error handling and logging
- More flexible and maintainable
- Cross-platform compatible
- Supports dry-run mode
- Can filter specific checkpoints

**Usage:**

```bash
# Run all checkpoints
python3 run_all_tests.py

# Dry-run (show what would be done without running tests)
python3 run_all_tests.py --dry-run

# Run only specific checkpoints matching a pattern
python3 run_all_tests.py --checkpoint "rgb_pose"
```

### 2. Bash Script - `run_all_tests.sh`

**Usage:**

```bash
# Run all checkpoints
bash run_all_tests.sh

# Or make it executable first
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## Configuration Inference

The scripts automatically determine modalities from checkpoint directory names:

| Pattern | Setting | Default |
|---------|---------|---------|
| `rgb` | USE_RGB | False if absent |
| `pose` | USE_POSE | False if absent |
| `text` | USE_TEXT | False if absent |
| `lora` | ENABLE_LORA | False if absent |
| `soft` | SOFT_PROMPT | False if absent |
| `velo` | POSE_USE_VELOCITY | False if absent |

**Examples:**
- `8_rgb_pose_velo_text/` → RGB=True, Pose=True, Text=True, Velocity=True
- `10_text_only/` → Text=True, others=False
- `13_rgb_lora/` → RGB=True, LoRA=True, others=False

## Output

Test results are saved in `/vol/bitbucket/sna21/CUENet/test_results/`:

- `{checkpoint_name}_test.log` - Full test output and any errors
- `{checkpoint_name}_config.yaml` - The config used for that specific test

## Example Checkpoints Being Tested

The following checkpoints will be processed:

1. `10_text_only/` - Text only
2. `11_rgb_pose_velo_text_no_lora_with_soft/` - RGB + Pose + Text (soft prompt)
3. `12_rgb_pose_no_velo_lora/` - RGB + Pose with LoRA
4. `13_rgb_lora/` - RGB only with LoRA
5. `14_rgb_pose_velo_lora/` - RGB + Pose + Velocity with LoRA
6. `15_rgb_text_soft_lora/` - RGB + Text with soft prompt and LoRA
7. `16_pose_no_velo_text_soft_lora/` - Pose + Text with soft prompt and LoRA
8. `17_pose_velo_text_soft_lora/` - Pose + Velocity + Text with soft prompt and LoRA
9. `18_rgb_pose_velo_text_lora/` - All modalities with LoRA
10. `19_rgb_text_lora/` - RGB + Text with LoRA
11. `6_rgb_pose_no_velo_text_soft_lora/` - RGB + Pose + Text (soft prompt, LoRA)
12. `8_rgb_pose_velo_text/` - All modalities without LoRA
13. `9_text_with_soft/` - Text only with soft prompt

## Manually Testing Specific Checkpoint

If you want to test a single checkpoint manually:

```bash
cd /vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/tools

# Update the config.yaml with your desired settings
# Then run:
python3 test_net.py --cfg ../exp/RWF_exp/config.yaml
```

## Notes

- Each test run creates a config backup with the checkpoint name prefix
- Logs are preserved for debugging if a test fails
- The base config template is not modified; only temporary copies are created
- Default timeout for each test is 1 hour (3600 seconds)
