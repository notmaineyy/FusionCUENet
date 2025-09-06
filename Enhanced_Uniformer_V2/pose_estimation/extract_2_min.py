import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    else:
        return 0
        

def process_videos_in_folder(folder_path, output_folder, max_persons=5):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in video_extensions:
            print(f"Skipping {filename} (unsupported extension)")
            continue

        video_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, f"{name}_pose.npy")

        # ✅ Skip if pose.npy already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename} (pose file already exists at {output_path})")
            continue

        # ✅ Skip if video longer than 2 minutes
        duration = get_video_duration(video_path)
        if duration > 120:
            print(f"Skipping {filename} (too long: {duration:.1f} seconds)")
            continue

        print(f"Processing {filename} ({duration:.1f} seconds)...")
        pose_data = extract_pose_data(video_path, max_persons=max_persons)

        if pose_data:  
            try:
                final_pose_array = np.array(pose_data, dtype=np.float32)
                np.save(output_path, final_pose_array)
                print(f"Saved multi-person pose data to {output_path} with shape {final_pose_array.shape}\n")
            except ValueError as e:
                print(f"Warning: Could not save {output_path} as a contiguous array. Saving as object array. Error: {e}")
                np.save(output_path, np.array(pose_data, dtype=object))
        else:
            print(f"No pose data extracted for {filename}. Skipping save.\n")

def extract_pose_data(video_path, max_persons=5):
    yolo_model = YOLO('yolov8n.pt')
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    pose_data = []  # Will hold list of lists of landmarks per frame

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # YOLO detect persons in frame
            # We explicitly ask for person class (class_id=0)
            results = yolo_model(image_rgb, classes=0, verbose=False)[0]

            # Extract person bounding boxes (xyxy format)
            # Filter by confidence if needed (e.g., box.conf > 0.5)
            #print(f"Frame {frame_idx}: Detected {len(results.boxes)} boxes")
            person_boxes = results.boxes.xyxy.cpu().numpy()
            #print(f"Frame {frame_idx}: Detected {len(person_boxes)} persons")
            person_confidences = results.boxes.conf.cpu().numpy()
            #print(f"Frame {frame_idx}: Person confidences: {person_confidences}")

            # Sort persons by confidence and take top max_persons
            # Combine boxes and confidences, then sort
            sorted_persons = sorted(zip(person_boxes, person_confidences), key=lambda x: x[1], reverse=True)
            person_boxes_filtered = [box for box, conf in sorted_persons[:max_persons]]

            frame_poses = [] # List to store poses for the current frame

            if not person_boxes_filtered:
                # No person detected, append max_persons number of zero arrays
                for _ in range(max_persons):
                    frame_poses.append(np.zeros((33, 4)))
            else:
                for box in person_boxes_filtered:
                    x1, y1, x2, y2 = map(int, box)

                    # Clamp bbox within image dimensions to prevent out-of-bounds errors
                    h_img, w_img, _ = image.shape
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w_img - 1), min(y2, h_img - 1)

                    person_crop = image_rgb[y1:y2, x1:x2]

                    if person_crop.size == 0: # Check for empty crop (e.g., if x1=x2 or y1=y2)
                        landmarks = np.zeros((33, 4))
                    else:
                        # Run MediaPipe Pose on the cropped image
                        results_pose = pose.process(person_crop)

                        if results_pose.pose_landmarks:
                            landmarks = []
                            for lmk in results_pose.pose_landmarks.landmark:
                                # Normalize landmarks relative to the cropped region and then to full frame
                                # (lmk.x, lmk.y) are already normalized to the crop (0-1)
                                # We need to re-normalize them to the original frame coordinates
                                global_x = x1 + lmk.x * (x2 - x1)
                                global_y = y1 + lmk.y * (y2 - y1)
                                landmarks.append([global_x / w_img, global_y / h_img, lmk.z, lmk.visibility])
                            landmarks = np.array(landmarks)
                        else:
                            landmarks = np.zeros((33, 4))
                    frame_poses.append(landmarks)
                
                # Pad with zeros if fewer than max_persons were found
                while len(frame_poses) < max_persons:
                    frame_poses.append(np.zeros((33, 4)))

            pose_data.append(frame_poses)
            frame_idx += 1

    cap.release()
    return pose_data # This is a list of frames, each frame is list of person poses

if __name__ == "__main__":
    print("Starting pose extraction...")
    input_folder = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/normal"      
    output_folder = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/multi/pose_outputs/normal"  
    
    MAX_PERSONS_TO_TRACK = 10
    print(f"Processing videos in {input_folder} and saving pose data to {output_folder} with max persons: {MAX_PERSONS_TO_TRACK}")
    process_videos_in_folder(input_folder, output_folder, max_persons=MAX_PERSONS_TO_TRACK)
