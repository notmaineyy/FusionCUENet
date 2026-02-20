# Dynamic Prediction CSV Path Configuration

## Changes Made

The scripts have been updated to dynamically generate prediction CSV file paths based on checkpoint names, instead of using a hardcoded path.

### Files Modified

1. **test_net.py**
   - Replaced hardcoded `prediction_csv_path` with a dynamic initialization function
   - Added `initialize_prediction_csv(csv_path)` function that:
     - Creates the output directory if it doesn't exist
     - Initializes the CSV file with headers
   - Updated the `test()` function to read `PREDICTION_CSV_PATH` from config and initialize it

2. **run_all_tests.py**
   - Updated `update_config_for_checkpoint()` function to accept `checkpoint_name` parameter
   - Generates prediction CSV path based on checkpoint name: `/vol/bitbucket/sna21/dataset/predictions/eccv/{checkpoint_name}.csv`
   - Stores the path in `config['TEST']['PREDICTION_CSV_PATH']`
   - Updated the function call to pass `checkpoint_name`

### CSV Path Generation

For each checkpoint, the CSV file will be saved as:
```
/vol/bitbucket/sna21/dataset/predictions/eccv/{checkpoint_name}.csv
```

**Examples:**
- `10_text_only/` → `/vol/bitbucket/sna21/dataset/predictions/eccv/10_text_only.csv`
- `8_rgb_pose_velo_text/` → `/vol/bitbucket/sna21/dataset/predictions/eccv/8_rgb_pose_velo_text.csv`
- `15_rgb_text_soft_lora/` → `/vol/bitbucket/sna21/dataset/predictions/eccv/15_rgb_text_soft_lora.csv`

### How It Works

1. `run_all_tests.py` generates a unique CSV path for each checkpoint based on its name
2. The path is added to the temporary config file as `TEST.PREDICTION_CSV_PATH`
3. `test_net.py` reads this path from the config during initialization
4. The `initialize_prediction_csv()` function creates the directory and CSV file with headers
5. Predictions are saved to the checkpoint-specific CSV file

### Benefits

✓ Each checkpoint gets its own CSV output file
✓ Easy to track which predictions came from which checkpoint
✓ No risk of overwriting previous results
✓ Directory is created automatically if it doesn't exist
✓ Config is the single source of truth for the CSV path
