import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import CUENet.Enhanced_Uniformer_V2.pose_estimation.mp_thing as mp
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slowfast.models.build import build_model
from slowfast.utils.parser import load_config
from slowfast.config.defaults import get_cfg

# ----- MediaPipe Pose Landmarker Setup -----
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/vol/bitbucket/sna21/CUENet/pose_estimation/pose_landmarker_full.task'),
    running_mode=VisionRunningMode.IMAGE
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# ----- Dataset Definition -----
class VideoPoseDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        video_np = np.stack(frames, axis=0)
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()

        pose_features = []
        for frame in frames:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            res = pose_landmarker.detect(mp_img)
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                coords = []
                for kp in lm:
                    coords.extend([kp.x, kp.y, kp.z])
            else:
                coords = [0.0] * (33 * 3)
            pose_features.append(coords)

        pose_tensor = torch.tensor(pose_features).float().mean(dim=0)
        label = torch.tensor(self.labels[idx]).long()
        return video_tensor, pose_tensor, label

# ----- Pose Fusion Model -----
class PoseFusionModel(nn.Module):
    def __init__(self, visual_model, visual_dim=1024, pose_dim=99, hidden_dim=256):
        super().__init__()
        self.visual_model = visual_model
        for param in self.visual_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, video, pose):
        visual_feat = self.visual_model.extract_features(video)
        fused = torch.cat([visual_feat, pose], dim=1)
        return self.classifier(fused)

# ----- Extract feature method monkey patch -----
def extract_features(model, x):
    x = model.blocks(x)
    return x.mean(dim=1)

# ----- Training Script -----
if __name__ == '__main__':
    # Load config and model
    cfg = get_cfg()
    cfg.merge_from_file("/vol/bitbucket/sna21/CUENet/Enhanced_Uniformer_V2/exp/RWF_exp/config.yaml")
    cfg.TEST.ENABLE = False
    model = build_model(cfg)

    checkpoint = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.extract_features = lambda x: extract_features(model, x)

    # Load dataset
    df = pd.read_csv("/vol/bitbucket/sna21/dataset/VioGuard/train_full.csv")
    video_paths = df['video_path'].tolist()
    labels = df['label'].tolist()
    dataset = VideoPoseDataset(video_paths, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Create fusion model
    fusion_model = PoseFusionModel(model, visual_dim=1024, pose_dim=99)
    fusion_model = fusion_model.cuda()

    optimizer = optim.Adam(fusion_model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        fusion_model.train()
        total_loss = 0.0
        for video, pose, label in loader:
            video = video.cuda()
            pose = pose.cuda()
            label = label.cuda()

            logits = fusion_model(video, pose)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * video.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(fusion_model.state_dict(), '/vol/bitbucket/sna21/CUENet/pose_estimation/pose_fused_model.pth')
