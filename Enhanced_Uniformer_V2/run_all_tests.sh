#!/bin/bash

# Script to run tests on all eccv checkpoints with appropriate modality configurations

CHECKPOINTS_DIR="/vol/bitbucket/sna21/CUENet/best_checkpoints/eccv"
CONFIG_TEMPLATE="/vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/exp/RWF_exp/config.yaml"
TOOLS_DIR="/vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/tools"
RESULTS_DIR="/vol/bitbucket/sna21/CUENet/test_results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to determine modalities from checkpoint name
get_modalities() {
    local checkpoint_name=$1
    local use_rgb=False
    local use_pose=False
    local use_text=False
    local enable_lora=False
    local soft_prompt=False
    local use_velocity=False
    
    # Parse checkpoint name to determine modalities
    if [[ $checkpoint_name == *"rgb"* ]]; then
        use_rgb=True
    fi
    if [[ $checkpoint_name == *"pose"* ]]; then
        use_pose=True
    fi
    if [[ $checkpoint_name == *"text"* ]]; then
        use_text=True
    fi
    # Only enable LoRA if "lora" is present AND "no_lora" is not present
    if [[ $checkpoint_name == *"lora"* ]] && [[ $checkpoint_name != *"no_lora"* ]]; then
        enable_lora=True
    fi
    if [[ $checkpoint_name == *"soft"* ]]; then
        soft_prompt=True
    fi
    if [[ $checkpoint_name == *"velo"* ]] && [[ $checkpoint_name != *"no_velo"* ]]; then
        use_velocity=True
    fi
    
    echo "$use_rgb,$use_pose,$use_text,$enable_lora,$soft_prompt,$use_velocity"
}

# Function to update config.yaml
update_config() {
    local config_file=$1
    local checkpoint_path=$2
    local use_rgb=$3
    local use_pose=$4
    local use_text=$5
    local enable_lora=$6
    local soft_prompt=$7
    local use_velocity=$8
    
    # Use Python to update YAML config
    python3 << EOF
import yaml

config_file = "$config_file"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Update checkpoint path
config['TEST']['CHECKPOINT_FILE_PATH'] = "$checkpoint_path"

# Update modality settings
config['MODEL']['USE_RGB'] = $use_rgb
config['MODEL']['USE_POSE'] = $use_pose
config['MODEL']['USE_TEXT'] = $use_text
config['MODEL']['ENABLE_LORA'] = $enable_lora
config['MODEL']['SOFT_PROMPT'] = $soft_prompt
config['MODEL']['POSE_USE_VELOCITY'] = $use_velocity

# Write back
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Updated config: RGB={$use_rgb}, Pose={$use_pose}, Text={$use_text}, LoRA={$enable_lora}, Soft={$soft_prompt}, Velocity={$use_velocity}")
EOF
}

# Main loop through all checkpoints
echo "Starting tests for all eccv checkpoints..."
echo "=========================================="

for checkpoint_dir in $(ls -d "$CHECKPOINTS_DIR"/*/ | sort); do
    checkpoint_name=$(basename "$checkpoint_dir")
    checkpoint_file="$checkpoint_dir/best.pyth"
    
    if [ ! -f "$checkpoint_file" ]; then
        echo "Warning: Checkpoint file not found: $checkpoint_file"
        continue
    fi
    
    echo ""
    echo "Processing checkpoint: $checkpoint_name"
    echo "Checkpoint path: $checkpoint_file"
    
    # Get modalities
    modalities=$(get_modalities "$checkpoint_name")
    IFS=',' read -r use_rgb use_pose use_text enable_lora soft_prompt use_velocity <<< "$modalities"
    
    echo "Configuration: RGB=$use_rgb, Pose=$use_pose, Text=$use_text, LoRA=$enable_lora, Soft=$soft_prompt, Velocity=$use_velocity"
    
    # Update config
    update_config "$CONFIG_TEMPLATE" "$checkpoint_file" "$use_rgb" "$use_pose" "$use_text" "$enable_lora" "$soft_prompt" "$use_velocity"
    
    # Run test
    echo "Running test_net.py..."
    cd "$TOOLS_DIR"
    python3 test_net.py --cfg "$CONFIG_TEMPLATE" 2>&1 | tee "$RESULTS_DIR/${checkpoint_name}_test.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Test completed successfully for $checkpoint_name"
    else
        echo "✗ Test failed for $checkpoint_name"
    fi
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved in: $RESULTS_DIR"
