import json
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# === INPUT CONFIG ===
# Folder containing log files to process
log_folder_path = '/vol/bitbucket/sna21/logs/eccv/'  # Process all .out files in this folder
output_base_dir = "/vol/bitbucket/sna21/CUENet/results/extracted_stats/batch_extract/"
summary_output_file = os.path.join(output_base_dir, "best_epochs_summary.txt")

start_line = 1  # Start processing from this line
# === REGEX & FIELDS ===
pattern_json = re.compile(r'json_stats:\s*({.*})')
pattern_modality = re.compile(r'Learned modality weights:\s*\[(.*?)\]')
train_fields = ["epoch", "accuracy", "loss", "lr", "RAM", "gpu_mem", "modality_weights"]
val_fields = ["epoch", "accuracy", "loss", "RAM", "gpu_mem", "modality_weights"]

# === FUNCTION TO PROCESS A SINGLE LOG FILE ===
def process_log_file(log_file_path):
    """Process a single log file and return train/val data and best epoch info"""
    train_epoch_data = []
    val_epoch_data = []
    last_modality_weights = None  # Track the last logged modality weights
    
    try:
        with open(log_file_path, "r") as log_file:
            for _ in range(start_line - 1):
                log_file.readline()
            
            while True:
                line = log_file.readline()
                if not line:
                    break
                
                # Check for modality weights in this line
                modality_match = pattern_modality.search(line)
                if modality_match:
                    weights_str = modality_match.group(1).strip()
                    last_modality_weights = weights_str
                
                match_json = pattern_json.search(line)
                if match_json:
                    try:
                        json_str = match_json.group(1)
                        data = json.loads(json_str)
                        _type = data.get('_type')

                        # Only save train_epoch and val_epoch lines
                        if _type == 'train_epoch':
                            record = {field: data.get(field) for field in ['epoch', 'accuracy', 'loss', 'lr', 'RAM', 'gpu_mem']}
                            record['modality_weights'] = last_modality_weights
                            train_epoch_data.append(record)
                            print(f"Train record added: Epoch {record['epoch']} | Modality weights: {record['modality_weights']}")

                        elif _type == 'val_epoch':
                            # Create record from first val_epoch line
                            record = {field: data.get(field) for field in ['epoch', 'accuracy', 'loss', 'RAM', 'gpu_mem']}
                            record['modality_weights'] = last_modality_weights

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
                            print(f" Val record added: Epoch {record['epoch']} | Acc {record['accuracy']} | Loss {record['loss']} | Modality weights: {record['modality_weights']}")

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue
        
        return train_epoch_data, val_epoch_data
    
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return [], []

# === FUNCTION TO FIND BEST EPOCH ===
def find_best_epoch(val_epoch_data):
    """Find epoch(s) with best validation accuracy and loss, along with modality weights"""
    if not val_epoch_data:
        return None
    
    # Convert accuracy and loss to float for comparison
    best_info = {
        'epochs': [],
        'accuracy': -float('inf'),
        'loss': float('inf'),
        'modality_weights': None
    }
    
    for record in val_epoch_data:
        try:
            acc = float(record['accuracy']) if record['accuracy'] else -float('inf')
            loss = float(record['loss']) if record['loss'] else float('inf')
            
            # Update best accuracy
            if acc > best_info['accuracy']:
                best_info['accuracy'] = acc
            
            # Update best loss
            if loss < best_info['loss']:
                best_info['loss'] = loss
        except (ValueError, TypeError):
            continue
    
    # Find epochs with best accuracy and loss
    best_epochs = set()
    for record in val_epoch_data:
        try:
            acc = float(record['accuracy']) if record['accuracy'] else -float('inf')
            loss = float(record['loss']) if record['loss'] else float('inf')
            
            if acc == best_info['accuracy'] and loss == best_info['loss']:
                best_epochs.add(record['epoch'])
                best_info['modality_weights'] = record.get('modality_weights')
        except (ValueError, TypeError):
            continue
    
    best_info['epochs'] = sorted(list(best_epochs))
    return best_info

# === MAIN PROCESSING LOOP ===
os.makedirs(output_base_dir, exist_ok=True)
best_epochs_list = []

# Get all .out files in the log folder
log_files = sorted(Path(log_folder_path).glob('*.out'))

if not log_files:
    print(f"No .out files found in {log_folder_path}")
else:
    print(f"Found {len(log_files)} log files to process\n")
    
    for log_file_path in log_files:
        model_name = log_file_path.stem  # Get filename without extension
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        # Create output directory for this model
        output_dir = os.path.join(output_base_dir, model_name)
        train_output_path = os.path.join(output_dir, "train_epoch_stats.csv")
        val_output_path = os.path.join(output_dir, "val_epoch_stats.csv")
        plot_output_path = os.path.join(output_dir, "training_validation_curves.png")
        
        # Process the log file
        train_epoch_data, val_epoch_data = process_log_file(str(log_file_path))
        
        # Save CSVs
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
            
            # Find and store best epoch info
            best_epoch_info = find_best_epoch(val_epoch_data)
            if best_epoch_info:
                best_epochs_list.append({
                    'model': model_name,
                    'epochs': best_epoch_info['epochs'],
                    'accuracy': best_epoch_info['accuracy'],
                    'loss': best_epoch_info['loss'],
                    'modality_weights': best_epoch_info['modality_weights']
                })
                print(f"Best Epoch(s): {best_epoch_info['epochs']} | Acc: {best_epoch_info['accuracy']:.6f} | Loss: {best_epoch_info['loss']:.6f} | Modality weights: {best_epoch_info['modality_weights']}")
        
        # === PLOTTING ===
        if train_epoch_data and val_epoch_data:
            try:
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
                plt.plot(df['epoch'], df['accuracy_train'], label='Train Accuracy', marker='o')
                plt.plot(df['epoch'], df['accuracy_val'], label='Val Accuracy', marker='o')
                plt.title('Accuracy Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                # Loss subplot
                plt.subplot(1, 2, 2)
                plt.plot(df['epoch'], df['loss_train'], label='Train Loss', marker='o')
                plt.plot(df['epoch'], df['loss_val'], label='Val Loss', marker='o')
                plt.title('Loss Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(plot_output_path)
                plt.close()
                print(f"Combined Accuracy & Loss plot saved to {plot_output_path}")
            except Exception as e:
                print(f"Error creating plots for {model_name}: {e}")

# === WRITE SUMMARY FILE ===
print(f"\n{'='*60}")
print("Writing summary file...")
print(f"{'='*60}\n")

with open(summary_output_file, 'w') as summary_file:
    summary_file.write("=" * 100 + "\n")
    summary_file.write("BEST EPOCHS SUMMARY\n")
    summary_file.write("=" * 100 + "\n\n")
    
    for entry in best_epochs_list:
        epochs_str = ", ".join(map(str, entry['epochs']))
        summary_file.write(f"Model: {entry['model']}\n")
        summary_file.write(f"Best Epoch(s): {epochs_str}\n")
        summary_file.write(f"Validation Accuracy: {entry['accuracy']:.6f}\n")
        summary_file.write(f"Validation Loss: {entry['loss']:.6f}\n")
        summary_file.write(f"Modality Weights: {entry['modality_weights']}\n")
        summary_file.write("-" * 100 + "\n\n")

print(f"Summary file written to {summary_output_file}")
print(f"\nProcessing complete! All results saved to {output_base_dir}")
