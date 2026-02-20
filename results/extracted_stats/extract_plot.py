import json
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

# === INPUT CONFIG ===
# /vol/bitbucket/sna21/logs/eccv/run_net_217110.out #lora rgb pose no velo
#'/vol/bitbucket/sna21/logs/eccv/run_net_216401.out' lora all (done)
#/vol/bitbucket/sna21/logs/eccv/run_net_217095.out lora rgb only (done)
#/vol/bitbucket/sna21/logs/eccv/run_net_217122.out lora rgb pose
#/vol/bitbucket/sna21/logs/eccv/run_net_217211.out lora_rgb_text 
#/vol/bitbucket/sna21/logs/eccv/run_net_217398.out lora_pose_no_velo_text 
#/vol/bitbucket/sna21/logs/eccv/run_net_217399.out lora_pose_with_velo_text 
#/vol/bitbucket/sna21/logs/eccv/run_net_217400.out lora_rgb_pose_no_velo_text 


log_file_path = '/vol/bitbucket/sna21/logs/eccv/run_net_217110.out' 

#'/vol/bitbucket/sna21/logs/eccv/run_net_216401.out' lora all
#'/vol/bitbucket/sna21/logs/eccv/run_net_211903.out'
#'/vol/bitbucket/sna21/logs/eccv/run_net_211903.out' # velo & tet
# '/vol/bitbucket/sna21/logs/eccv/run_net_211866.out' #eccv all
#'/vol/bitbucket/sna21/logs/eccv/run_net_211523.out' #kd try 1 continue
#'/vol/bitbucket/sna21/logs/eccv/run_net_211487.out' #kd try 1
#"/vol/bitbucket/sna21/logs/cvpr/run_net_207165.out" #cvpr rlv
#"/vol/bitbucket/sna21/logs/cvpr/run_net_206802.out" #cvpr rf only
# "/vol/bitbucket/sna21/logs/run_net_200832.out" #ubi from scratch
#"/vol/bitbucket/sna21/logs/run_net_200795.out" #ubi finetune
#"/vol/bitbucket/sna21/logs/run_net_200280.out" #rwf only rgb pose
#"/vol/bitbucket/sna21/logs/run_net_199997.out" # rgb pose no velo
# "/vol/bitbucket/sna21/logs/run_net_197495.out" text only
# "/vol/bitbucket/sna21/logs/run_net_196892.out" pose text
# "/vol/bitbucket/sna21/logs/run_net_196517.out" rgb text
# "/vol/bitbucket/sna21/logs/run_net_195168.out"  rgb pose
#"/vol/bitbucket/sna21/logs/run_net_194471.out" all
#  #"/vol/bitbucket/sna21/logs/run_net_194610.out" poe only
output_dir = "/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/lora_rgb_pose_no_velo/"
# lora_rgb_pose_no_velo  lora_rgb lora_rgb_text


#"/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/gemini_pose_velo_text/" 
# #"/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/gemini_all/" #eccv all
train_output_path = os.path.join(output_dir, "train_epoch_stats.csv")
val_output_path = os.path.join(output_dir, "val_epoch_stats.csv")
plot_output_path = os.path.join(output_dir, "training_validation_curves.png")
start_line = 1  # Start processing from this line
plot_output_path_acc = os.path.join(output_dir, "training_validation_accuracy.png")
plot_output_path_loss = os.path.join(output_dir, "training_validation_loss.png")
start_line = 1
# === REGEX & FIELDS ===
pattern_json = re.compile(r'json_stats:\s*({.*})')
train_fields = ["epoch", "accuracy", "loss", "lr", "RAM", "gpu_mem"]
val_fields = ["epoch", "accuracy", "loss", "RAM", "gpu_mem"]
train_epoch_data = []
val_epoch_data = []

# === EXTRACTION ===
with open(log_file_path, "r") as log_file:
    for _ in range(start_line - 1):
        log_file.readline()
    
    while True:
        line = log_file.readline()
        if not line:
            break
        
        match_json = pattern_json.search(line)
        if match_json:
            try:
                json_str = match_json.group(1)
                data = json.loads(json_str)
                _type = data.get('_type')

                # Only save train_epoch and val_epoch lines
                if _type == 'train_epoch':
                    record = {field: data.get(field) for field in train_fields}
                    train_epoch_data.append(record)
                    print(f"Train record added: Epoch {record['epoch']}")

                elif _type == 'val_epoch':
                    # Create record from first val_epoch line
                    record = {field: data.get(field) for field in ['epoch', 'accuracy', 'loss', 'RAM', 'gpu_mem']}

                    # If accuracy contains %, clean it
                    if record['accuracy'] and isinstance(record['accuracy'], str):
                        record['accuracy'] = record['accuracy'].replace('%', '')

                    # Peek ahead at next line to merge RAM/gpu info if available
                    pos = log_file.tell()  # remember file pointer
                    next_line = log_file.readline()

                    if next_line:
                        match_next = pattern_json.search(next_line)
                        if match_next:
                            try:
                                next_data = json.loads(match_next.group(1))
                                # If next line is also val_epoch, merge fields
                                if next_data.get('_type') == 'val_epoch':
                                    # Merge missing fields
                                    for field in ['RAM', 'gpu_mem']:
                                        if not record.get(field) and next_data.get(field):
                                            record[field] = next_data.get(field)
                            except json.JSONDecodeError:
                                pass
                    else:
                        break

                    # Ensure all fields exist
                    for field in ['loss', 'RAM', 'gpu_mem']:
                        if field not in record:
                            record[field] = ""

                    # Save single merged record
                    val_epoch_data.append(record)
                    print(f" Val record added: Epoch {record['epoch']} | Acc {record['accuracy']} | Loss {record['loss']}")



            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

# === SAVE CSVs ===
os.makedirs(output_dir, exist_ok=True)

if train_epoch_data:
    with open(train_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_fields)
        writer.writeheader()
        writer.writerows(train_epoch_data)
    print(f"Training CSV written to {train_output_path}")

if val_epoch_data:
    with open(val_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=val_fields)
        writer.writeheader()
        writer.writerows(val_epoch_data)
    print(f"Validation CSV written to {val_output_path}")

# === PLOTTING ===
def extract_epoch(epoch_str):
    match = re.match(r'(\d+)', str(epoch_str))
    return int(match.group()) if match else None

train_df = pd.read_csv(train_output_path)
val_df = pd.read_csv(val_output_path)
train_df['epoch'] = train_df['epoch'].apply(extract_epoch)
val_df['epoch'] = val_df['epoch'].apply(extract_epoch)

df = pd.merge(train_df[['epoch', 'accuracy', 'loss']],
              val_df[['epoch', 'accuracy', 'loss']],
              on='epoch',
              suffixes=('_train', '_val'))

# Combined plot: Accuracy (left) and Loss (right)
plt.figure(figsize=(14, 6))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(df['epoch']-15, df['accuracy_train'], label='Train Accuracy', marker='o')
plt.plot(df['epoch']-15, df['accuracy_val'], label='Val Accuracy', marker='o')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(df['epoch']-15, df['loss_train'], label='Train Loss', marker='o')
plt.plot(df['epoch']-15, df['loss_val'], label='Val Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_output_path)
plt.show()
print(f"Combined Accuracy & Loss plot saved to {plot_output_path}")
