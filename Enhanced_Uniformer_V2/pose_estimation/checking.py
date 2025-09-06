import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# MediaPipe-style skeleton connections
POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # Arms (Shoulder to Elbow, Elbow to Wrist)
    (11, 12),  # Shoulders
    (23, 24),  # Hips
    (11, 23), (12, 24),  # Torso (Shoulder to Hip)
    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs (Hip to Knee, Knee to Ankle)
    (15, 17), (15, 19), (15, 21), # Wrists and Fingers
    (16, 18), (16, 20), (16, 22),
    (27, 29), (27, 31), # Feet
    (28, 30), (28, 32)
]


def get_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def plot_frame_and_pose(frame_img, poses, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    h, w, _ = frame_img.shape

    # === Plot original frame with overlayed poses ===
    axs[0].imshow(frame_img)
    axs[0].set_title("Original Frame with Poses")
    axs[0].axis('off')
    axs[0].set_aspect('equal')  # Keep aspect ratio of the image

    axs[1].set_title("Pose Landmarks (Normalized)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].invert_yaxis() # MediaPipe Y-axis increases downwards, typical plot Y-axis increases upwards
    axs[1].set_aspect('equal')  # Equal aspect for normalized plot
    axs[1].set_xlim([-0.2, 1.2]) # Extend limits slightly for better visualization
    axs[1].set_ylim([1.2, -0.2]) # Invert and extend Y-axis for consistency with image


    colors = plt.cm.get_cmap('tab10', len(poses)) # Get distinct colors for each person

    for i, landmarks in enumerate(poses):
        # Check if the landmarks are all zeros (indicating no person or no pose detected for this slot)
        if np.all(landmarks == 0):
            continue

        color = colors(i) # Assign a unique color to each person

        # Draw keypoints on image (scaled to image dimensions)
        for kp_idx, kp in enumerate(landmarks):
            x, y = kp[:2] # Extract x, y (normalized)
            # Only plot visible landmarks if desired (kp[3] is visibility)
            if kp[3] > 0.5: # Example threshold for visibility
                axs[0].scatter(x * w, y * h, c=[color], s=15, zorder=5) # zorder to ensure points are on top

        # Draw connections on image (scaled to image dimensions)
        for start_idx, end_idx in POSE_CONNECTIONS:
            # Check if indices are within bounds and landmarks are visible
            if start_idx < len(landmarks) and end_idx < len(landmarks) and \
               landmarks[start_idx, 3] > 0.5 and landmarks[end_idx, 3] > 0.5:
                x1, y1 = landmarks[start_idx][:2]
                x2, y2 = landmarks[end_idx][:2]
                axs[0].plot([x1 * w, x2 * w], [y1 * h, y2 * h], c=color, linewidth=2, zorder=4)

        # Draw normalized pose
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        axs[1].scatter(xs, ys, c=[color], s=15)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks) and \
               landmarks[start_idx, 3] > 0.5 and landmarks[end_idx, 3] > 0.5:
                x1, y1 = landmarks[start_idx][:2]
                x2, y2 = landmarks[end_idx][:2]
                axs[1].plot([x1, x2], [y1, y2], c=color, linewidth=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300) # Higher DPI for better quality
    plt.show()

# Example: plot a specific frame
if __name__ == "__main__":
    # Ensure these paths are correct after running the data extraction
    pose_path = "/vol/bitbucket/sna21/dataset/VioGuard/multi/pose_outputs/rwf2000_VbwTMJroTbI_0_0_0_pose.npy"
    video_path = "/vol/bitbucket/sna21/dataset/VioGuard/videos/rwf2000_VbwTMJroTbI_0_0_0.avi"

    pose = np.load(pose_path)  # shape should be (T, P, J, D)

    print("Shape of loaded pose file:", pose.shape)

    # Make sure the .npy file exists before trying to load
    if not os.path.exists(pose_path):
        print(f"Error: Pose data file not found at {pose_path}. Please run data extraction first.")
    elif not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}. Please check path.")
    else:
        pose_data = np.load(pose_path, allow_pickle=True)

        # Example: plot a random frame
        if len(pose_data) > 0:
            frame_index = random.randint(0, len(pose_data) - 1)
            print(f"Plotting a random frame: {frame_index}")
        else:
            frame_index = 0
            print("No pose data available to plot.")
            exit() # Exit if no data

        try:
            frame_img = get_video_frame(video_path, frame_index)

            # poses in that frame (list of np arrays, each shape (33,4))
            # If `pose_data` is a 3D array (num_frames, max_persons, 33, 4), then:
            if pose_data.ndim == 4: # Check if it's a consistently shaped array
                frame_poses = pose_data[frame_index]
            else: # It's likely an object array of lists
                frame_poses = pose_data[frame_index]
            
            output_vis_folder = "/vol/bitbucket/sna21/dataset/VioGuard/frame_with_pose"
            os.makedirs(output_vis_folder, exist_ok=True)

            plot_frame_and_pose(
                frame_img, frame_poses,
                save_path=f"{output_vis_folder}/frame_{frame_index}_with_pose.png"
            )
        except Exception as e:
            print(f"An error occurred during plotting: {e}")