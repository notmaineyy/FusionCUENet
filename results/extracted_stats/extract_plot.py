import json
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

# === INPUT CONFIG ===
log_file_path = "/vol/bitbucket/sna21/logs/run_net_174538.out"
output_dir = "/vol/bitbucket/sna21/CUENet/results/extracted_stats/fl_tvr"
train_output_path = os.path.join(output_dir, "train_epoch_stats.csv")
val_output_path = os.path.join(output_dir, "val_epoch_stats.csv")
plot_output_path = os.path.join(output_dir, "training_validation_curves.png")
start_line = 1  # Start processing from this line

# === REGEX & FIELDS ===
pattern_json = re.compile(r'json_stats:\s*({.*})')
train_fields = ["epoch", "accuracy", "loss", "lr", "RAM", "gpu_mem"]
val_fields = ["epoch", "accuracy", "loss", "RAM", "gpu_mem"]
train_epoch_data = []
val_epoch_data = []

def parse_val_results_line(line):
    if "Val results:" not in line:
        return None
    parts = line.split("Val results:")[1].strip().split(", ")
    for part in parts:
        if part.startswith("Loss="):
            return part.split('=')[1]
    return None

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
                
                if _type == 'train_epoch':
                    record = {field: data.get(field) for field in train_fields}
                    train_epoch_data.append(record)
                
                elif _type == 'val_epoch':
                    record = {field: data.get(field) for field in ['epoch', 'accuracy', 'RAM', 'gpu_mem']}
                    next_line = log_file.readline()
                    if not next_line:
                        break
                    loss_val = parse_val_results_line(next_line)
                    if loss_val is not None:
                        record['loss'] = loss_val
                        val_epoch_data.append(record)
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
    return int(match.group()) - 15 if match else None

train_df = pd.read_csv(train_output_path)
val_df = pd.read_csv(val_output_path)
train_df['epoch'] = train_df['epoch'].apply(extract_epoch)
val_df['epoch'] = val_df['epoch'].apply(extract_epoch)

df = pd.merge(train_df[['epoch', 'accuracy', 'loss']],
              val_df[['epoch', 'accuracy', 'loss']],
              on='epoch',
              suffixes=('_train', '_val'))

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['accuracy_train'], label='Train Accuracy', marker='o', color='blue')
plt.plot(df['epoch'], df['accuracy_val'], label='Val Accuracy', marker='o', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['loss_train'], label='Train Loss', marker='o', color='red')
plt.plot(df['epoch'], df['loss_val'], label='Val Loss', marker='o', color='green')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_output_path)
plt.show()
print(f"Plot saved to {plot_output_path}")
