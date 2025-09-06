import cv2
import os
import csv

def get_video_duration(video_path):
    """Return duration of video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return None

def find_short_videos(folder_path, output_csv="/vol/bitbucket/sna21/dataset/UBI_FIGHTS/short_videos.csv", max_duration=120):
    """Write names of videos shorter than max_duration (seconds) into a CSV file."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(folder_path, filename)
                duration = get_video_duration(video_path)
                if duration is not None and duration <= max_duration:
                    writer.writerow([filename, "0"])
                    print(f"Added: {filename} ({duration:.2f} sec)")

import csv

def filter_csv(file1, file2, output_file):
    """Remove rows from file1 if filename exists in file2, save result to output_file."""
    
    # Read filenames from file2
    with open(file2, "r") as f2:
        reader2 = csv.reader(f2)
        file2_names = {row[0] for row in reader2 if row}  # collect all filenames
    
    # Process file1 and write only rows not in file2
    with open(file1, "r") as f1, open(output_file, "w", newline="") as out:
        reader1 = csv.reader(f1)
        writer = csv.writer(out)
        
        for row in reader1:
            if row and row[0] not in file2_names:
                writer.writerow(row)

    print(f"Filtered file saved to {output_file}")

import csv
from collections import Counter
from sklearn.model_selection import train_test_split

def split_csv_stratified(input_file, 
                         train_file="/vol/bitbucket/sna21/dataset/UBI_FIGHTS/train.csv", 
                         val_file="/vol/bitbucket/sna21/dataset/UBI_FIGHTS/val.csv", 
                         train_ratio=0.8, seed=42):
    """Split a CSV file into train and val sets, stratified by label, and print label counts."""
    
    # Read all rows
    with open(input_file, "r") as f:
        reader = list(csv.reader(f))
    
    if not reader:
        print("Input CSV is empty.")
        return
    
    # Separate filenames and labels
    filenames = [row[0] for row in reader]
    labels = [row[1] for row in reader]
    
    # Stratified split
    train_idx, val_idx = train_test_split(
        range(len(filenames)), 
        train_size=train_ratio, 
        random_state=seed, 
        stratify=labels
    )
    
    # Prepare rows
    train_rows = [reader[i] for i in train_idx]
    val_rows = [reader[i] for i in val_idx]
    
    # Count labels
    train_labels_count = Counter([row[1] for row in train_rows])
    val_labels_count = Counter([row[1] for row in val_rows])
    
    # Print label distributions
    print("Train label distribution:")
    for label, count in train_labels_count.items():
        print(f"  Label {label}: {count}")
    
    print("Validation label distribution:")
    for label, count in val_labels_count.items():
        print(f"  Label {label}: {count}")
    
    # Write train.csv
    with open(train_file, "w", newline="") as f_train:
        writer = csv.writer(f_train)
        writer.writerows(train_rows)
    
    # Write val.csv
    with open(val_file, "w", newline="") as f_val:
        writer = csv.writer(f_val)
        writer.writerows(val_rows)
    
    print(f"✅ Stratified split complete: {len(train_rows)} train, {len(val_rows)} val")

if __name__ == "__main__":
    input_csv = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/train_val.csv"
    split_csv_stratified(input_csv)



""" if __name__ == "__main__":
    csv_file1 = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/short_videos.csv"   # <- path to first csv
    csv_file2 = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/test.csv"   # <- path to second csv
    output_csv = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/train.csv"
    filter_csv(csv_file1, csv_file2, output_csv) """
""" 

if __name__ == "__main__":
    folder = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/fight"  # <- change this to your folder path
    find_short_videos(folder)
 """