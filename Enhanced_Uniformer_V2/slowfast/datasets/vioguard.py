import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class VioGuard(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing VioGuard {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

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
