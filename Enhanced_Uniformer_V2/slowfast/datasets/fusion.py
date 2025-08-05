# dataloader/fusion_dataset.py
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
import torchvision.transforms as T
import cv2
from . import video_container as container

logger = logging.get_logger(__name__)

MAX_NUM_PEOPLE = 24

@DATASET_REGISTRY.register()
class Fusion_dataset(Dataset):
    def __init__(self, cfg, mode, num_retries=10, transform=None):
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Fusion Dataset {}...".format(mode))
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
        """
        Construct the video + pose loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_poses = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._path_to_flows = []


        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2
                path, label = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)

                for idx in range(self._num_clips):
                    video_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    pose_name = os.path.splitext(os.path.basename(path))[0] + "_pose.npy"
                    pose_path = os.path.join(self.cfg.DATA.POSE_PATH_PREFIX, pose_name)

                    #flow_name = os.path.splitext(os.path.basename(path))[0] + "_flow.npy"
                    #flow_path = os.path.join(self.cfg.DATA.FLOW_PATH_PREFIX, flow_name)

                    #self._path_to_flows.append(flow_path)
                    self._path_to_videos.append(video_path)
                    self._path_to_poses.append(pose_path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert len(self._path_to_videos) > 0, f"Failed to load videos from {path_to_file}"
        logger.info(f"Constructed fusion dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

    def load_video_tensor(self, path, num_frames=16):
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total // num_frames, 1)

        for i in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(T.ToPILImage()(frame))
            frames.append(frame)
            if len(frames) == num_frames:
                break
        cap.release()
        return torch.stack(frames)  # (T, C, H, W)

    def _load_flow_tensor(self, path, num_frames):
        try:
            flow_array = np.load(path)  # shape: (T, H, W, 2)
            total_frames = flow_array.shape[0]
            idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)
            flow_array = flow_array[idxs]  # sample to match RGB/Pose

            # Convert to tensor: (T, H, W, 2) → (T, 2, H, W)
            flow_tensor = torch.from_numpy(flow_array).permute(0, 3, 1, 2).float()
            return flow_tensor
        except Exception as e:
            logger.warning(f"Failed to load flow at {path}: {e}")
            return torch.zeros((num_frames, 2, self.cfg.DATA.RESIZE_HEIGHT, self.cfg.DATA.RESIZE_WIDTH))  # fallback

    def __getitem__(self, index):
        """
        Returns:
            frames (tensor): Video frames [C x T x H x W].
            pose_tensor (tensor): Pose data [D x T].
            label (int): Class label.
            index (int): Index of the sample.
        """
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
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
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported")

        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(f"Failed to load video from {self._path_to_videos[index]}: {e}")

            if video_container is None:
                logger.warning(f"Retry {i_try}: failed to load video idx {index}")
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
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

            # Load corresponding pose data
            pose_tensor = None
            try:
                # In Fusion_dataset.__getitem__

                pose_array = np.load(self._path_to_poses[index])  # shape: (T, P, J, 4)
                total_frames = pose_array.shape[0]
                idxs = np.linspace(0, total_frames - 1, self.cfg.DATA.NUM_FRAMES).astype(int)
                pose_array = pose_array[idxs]  # shape: (T, P, J, 4)

                P = pose_array.shape[1]
                if P < MAX_NUM_PEOPLE:
                    # Pad with zeros
                    pad = np.zeros((pose_array.shape[0], MAX_NUM_PEOPLE - P, pose_array.shape[2], pose_array.shape[3]))
                    pose_array = np.concatenate([pose_array, pad], axis=1)
                elif P > MAX_NUM_PEOPLE:
                    # Truncate
                    pose_array = pose_array[:, :MAX_NUM_PEOPLE, :, :]
                # Optional: keep only x, y if you want
                pose_array = pose_array[:, :, :, :4]  # (T, P, J, 2)

                # To tensor
                pose_tensor = torch.from_numpy(pose_array).float()  # shape: (T, P, J, D)
                #flow_tensor = self._load_flow_tensor(self._path_to_flows[index], self.cfg.DATA.NUM_FRAMES)



            except Exception as e:
                logger.warning(f"Failed to load pose for idx {index}: {e}")
                pose_tensor = torch.zeros((
                    self.cfg.DATA.NUM_FRAMES, 
                    self.cfg.DATA.NUM_JOINTS,  # Add this to your config (e.g., 33 for NTURGB+D)
                    3
                ))


            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:
                    frame_list = []
                    label_list = []
                    index_list = []
                    pose_list = []

                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        pose_list.append(pose_tensor)
                        label_list.append(label)
                        index_list.append(index)

                    return (frame_list, pose_list), label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )
            else:
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

            return (frames, pose_tensor), label, index, {}

        raise RuntimeError(f"Failed to fetch video after {self._num_retries} retries.")

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

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

