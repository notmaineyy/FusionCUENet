import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
LOG_FILE = '/vol/bitbucket/sna21/logs/eccv/run_net_216401.out'
#"/vol/bitbucket/sna21/logs/eccv/run_net_211903.out" #'/vol/bitbucket/sna21/logs/eccv/run_net_211866.out' #eccv all
OUTPUT_DIR = "/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/lora/plots"
# "/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/gemini_pose_velo_text/plots" 
# "/vol/bitbucket/sna21/CUENet/results/extracted_stats/eccv/gemini_all/plots"

def parse_logs_by_time(filepath):
    """
    Parses logs using timestamps to align disjoint data streams.
    """
    data_points = []
    
    current_year = datetime.now().year
    start_time = None

    with open(filepath, 'r') as f:
        for line in f:
            # 1. Extract Timestamp [12/07 10:44:34]
            # Assumes format: [MM/DD HH:MM:SS]
            ts_match = re.search(r'^\[(\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]', line)
            if not ts_match:
                continue
                
            ts_str = f"{current_year}/{ts_match.group(1)}"
            try:
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue

            if start_time is None:
                start_time = dt
            
            # Calculate elapsed minutes
            elapsed_min = (dt - start_time).total_seconds() / 60.0

            # 2. Extract Modality Weights
            if "Learned modality weights:" in line:
                # Clean up the array string to handle multiple spaces/newlines
                raw_nums = line.split("weights: [")[1].split("]")[0]
                # Replace newlines and multiple spaces with single space
                clean_nums = re.sub(r'\s+', ' ', raw_nums).strip()
                try:
                    w = [float(x) for x in clean_nums.split(' ')]
                    if len(w) == 3:
                        data_points.append({
                            "time": elapsed_min,
                            "type": "weight",
                            "rgb": w[0],
                            "pose": w[1],
                            "text": w[2]
                        })
                    elif len(w)==2:
                        data_points.append({
                            "time": elapsed_min,
                            "type": "weight",
                            "pose": w[0],
                            "text": w[1]
                        })
                except:
                    pass

            # 3. Extract Training Stats
            elif "json_stats" in line and "_type" in line:
                try:
                    json_str = line.split("json_stats: ")[1].strip()
                    stats = json.loads(json_str)
                    
                    if stats["_type"] == "train_iter":
                        data_points.append({
                            "time": elapsed_min,
                            "type": "train",
                            "loss": float(stats.get("loss", 0)),
                            "accuracy": float(stats.get("accuracy", 0)),
                            "epoch": stats.get("epoch", "")
                        })
                    elif stats["_type"] == "val_epoch":
                         # Handle "71.55%" string format if present
                        acc = stats.get("accuracy", 0)
                        if isinstance(acc, str): acc = float(acc.replace("%", ""))
                        
                        data_points.append({
                            "time": elapsed_min,
                            "type": "val",
                            "accuracy": acc,
                            "loss": float(stats.get("loss", 0)),
                            "epoch": stats.get("epoch", "")
                        })
                except:
                    pass

    return pd.DataFrame(data_points)

def generate_plots(df):
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ---------------------------------------------------------
    # PLOT 1: Modality Competition (The "Contribution" Plot)
    # ---------------------------------------------------------
    w_df = df[df["type"] == "weight"].copy()
    if not w_df.empty:
        # Normalize to show relative change from start
        # Or just plot raw values since they are softmaxed
        plt.figure(figsize=(10, 6))
        
        #plt.plot(w_df["time"], w_df["rgb"], label="RGB (Visual)", color="#e74c3c", linewidth=2)
        plt.plot(w_df["time"], w_df["pose"], label="Pose (Kinematic)", color="#3498db", linewidth=2)
        plt.plot(w_df["time"], w_df["text"], label="Text (Semantic)", color="#2ecc71", linewidth=2)
        
        plt.title("Dynamic Evolution of Modality Importance", fontsize=14)
        plt.xlabel("Training Time (Minutes)", fontsize=12)
        plt.ylabel("Attention Weight", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(f"{OUTPUT_DIR}/modality_dynamics.png", dpi=300)
        plt.close()
        print("Generated modality_dynamics.png")

    # ---------------------------------------------------------
    # PLOT 2: Training Stability (Loss vs Acc)
    # ---------------------------------------------------------
    t_df = df[df["type"] == "train"].copy()
    v_df = df[df["type"] == "val"].copy()
    
    if not t_df.empty:
        # Smooth the noisy training data
        t_df["loss_smooth"] = t_df["loss"].rolling(10).mean()
        t_df["acc_smooth"] = t_df["accuracy"].rolling(10).mean()

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Loss (Left Axis)
        ax1.set_xlabel('Training Time (Minutes)')
        ax1.set_ylabel('Training Loss', color='tab:red')
        ax1.plot(t_df["time"], t_df["loss_smooth"], color='tab:red', alpha=0.5, label="Train Loss (Smoothed)")
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Accuracy (Right Axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')
        ax2.plot(t_df["time"], t_df["acc_smooth"], color='tab:blue', label="Train Acc (Smoothed)")
        
        # Validation Markers
        if not v_df.empty:
            ax2.scatter(v_df["time"], v_df["accuracy"], color='navy', s=100, marker='*', label="Validation Acc", zorder=10)
            for _, row in v_df.iterrows():
                ax2.annotate(f"{row['accuracy']:.1f}%", (row['time'], row['accuracy']), 
                             xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

        plt.title("Training Dynamics: FusionCUENet", fontsize=14)
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_dynamics.png", dpi=300)
        plt.close()
        print("Generated training_dynamics.png")

if __name__ == "__main__":
    print(f"Reading logs from {LOG_FILE}...")
    df = parse_logs_by_time(LOG_FILE)
    
    if df.empty:
        print("Error: No data parsed. Check log format.")
    else:
        print(f"Parsed {len(df)} data points.")
        generate_plots(df)
        print(f"Plots saved to {OUTPUT_DIR}/ folder.")