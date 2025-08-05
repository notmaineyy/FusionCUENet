# dataloader/fusion_blip.py
import os
import random
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from .build import DATASET_REGISTRY
import numpy as np
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from . import video_container as container

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Fusion_blip_dataset(Dataset):
    def __init__(self, cfg, mode, num_retries=10, transform=None):
        assert mode in ["train", "val", "test"], f"Split '{mode}' not supported"
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.caption_path_prefix = cfg.DATA.CAPTION_PATH_PREFIX

        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode == "test":
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing RGB+BLIP Dataset {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct video + caption loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._path_to_captions = []

        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2
                path, label = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)

                for idx in range(self._num_clips):
                    video_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    caption_name = os.path.splitext(os.path.basename(path))[0] + "_caption.txt"
                    caption_path = os.path.join(self.caption_path_prefix, caption_name)

                    self._path_to_videos.append(video_path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                    self._path_to_captions.append(caption_path)

        assert len(self._path_to_videos) > 0, f"Failed to load videos from {path_to_file}"
        logger.info(f"Constructed RGB+BLIP dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        # Sampling params
        if self.mode == "train":
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        else:  # val / test
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)

        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        for i_try in range(self._num_retries):
            # Load video
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(f"Failed to load video {self._path_to_videos[index]}: {e}")

            if video_container is None:
                logger.warning(f"Retry {i_try}: failed to load video idx {index}")
                if self.mode != "test" and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            )

            if frames is None:
                logger.warning(f"Retry {i_try}: failed to decode video idx {index}")
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Normalize + spatial sampling
            frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
            frames = frames.permute(3, 0, 1, 2)
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)

            # Load caption & tokenize
            try:
                with open(self._path_to_captions[index], 'r') as f:
                    caption = f.read().strip()
                inputs = self.tokenizer(
                    caption,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = inputs['input_ids'].squeeze(0)
                attention_mask = inputs['attention_mask'].squeeze(0)
            except Exception as e:
                logger.warning(f"Failed to load caption: {e}")
                input_ids = torch.zeros(128, dtype=torch.long)
                attention_mask = torch.zeros(128, dtype=torch.long)

            label = self._labels[index]
            return (frames, input_ids, attention_mask), label, index, {}

        raise RuntimeError(f"Failed to fetch video after {self._num_retries} retries.")

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
