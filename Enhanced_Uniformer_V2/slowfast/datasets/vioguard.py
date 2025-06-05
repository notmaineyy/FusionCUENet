from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets.ava_helper import load_image_lists  # Or your own helper
from slowfast.datasets.utils import tensor_normalize, spatial_sampling
from slowfast.datasets.video_dataset import VideoDataset
import torch
import os

@DATASET_REGISTRY.register()
class VioGuard(VideoDataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split  # 'train', 'val', or 'test'
        self._construct_loader()

    def _construct_loader(self):
        # Example: parse CSV to get (video_path, label)
        self._path_to_videos = []
        self._labels = []

        csv_path = self.cfg.DATA.PATH_TO_TEST_FILE if self.split == "test" else self.cfg.DATA.PATH_TO_TRAIN_FILE
        with open(csv_path, "r") as f:
            for line in f:
                path, label = line.strip().split(",")
                self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, path))
                self._labels.append(int(label))

    def __getitem__(self, index):
        # Load video clip and apply transforms (you may use the parent class utilities)
        video = self._load_video(self._path_to_videos[index])  # Implement this or use decord/av loader
        label = self._labels[index]
        return video, label, index, {}

    def __len__(self):
        return len(self._path_to_videos)
