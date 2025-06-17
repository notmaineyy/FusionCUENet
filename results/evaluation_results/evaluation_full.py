import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# === Paths ===
input_dir = "/vol/bitbucket/sna21/dataset/predictions/rlvs"
base_output_dir = "/vol/bitbucket/sna21/CUENet/results/evaluation_results/rlvs"
os.makedirs(base_output_dir, exist_ok=True)

# === Initialize list to collect all metrics ===
all_metrics = []

# === Process each CSV file in input directory ===
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        csv_path = os.path.join(input_dir, file)
        model_name = os.path.splitext(file)[0]
        output_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        # === Load CSV ===
        df = pd.read_csv(csv_path)
        y_true = df["true_class"]
        y_pred = df["predicted_class"]
        confidences = df["confidence"]

        metrics_dict = {"model": model_name}

        # === Accuracy ===
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)

        # === Macro scores ===
        metrics_dict["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics_dict["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics_dict["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # === Per-class scores ===
        labels = sorted(y_true.unique())
        precs = precision_score(y_true, y_pred, average=None, zero_division=0)
        recs = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, label in enumerate(labels):
            metrics_dict[f"precision_class_{label}"] = precs[i]
            metrics_dict[f"recall_class_{label}"] = recs[i]
            metrics_dict[f"f1_class_{label}"] = f1s[i]

        # === Confusion Matrix Plot ===
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()

        # === Confidence Histogram ===
        plt.figure(figsize=(6, 4))
        plt.hist(confidences, bins=20, color='green', alpha=0.7)
        plt.title("Prediction Confidence Histogram")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confidence_histogram.png"))
        plt.close()

        # === ROC Curve & AUC (Binary only) ===
        if len(labels) == 2:
            y_score = df["confidence"]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            metrics_dict["auc"] = auc

            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--', label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
            plt.close()

        # === Save individual metrics CSV ===
        pd.DataFrame([metrics_dict]).to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)

        # === Append to overall list ===
        all_metrics.append(metrics_dict)

# === Save all metrics in one CSV ===
all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv(os.path.join(base_output_dir, "all_models_metrics.csv"), index=False)
