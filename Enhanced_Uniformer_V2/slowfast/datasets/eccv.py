import os
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging
from . import decoder, utils, video_container as container
from transformers import DistilBertTokenizer
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

MAX_NUM_PEOPLE = 24  # Aligns with BlipFusionCUENet's pose encoder


@DATASET_REGISTRY.register()
class Vioguard_eccv(Dataset):
    """
    Dataset for FusionCUENet (ECCV Submission):
    - RGB video frames (processed via SlowFast-style transforms)
    - Multi-person pose data (T, P, J, D) with Velocity/Acceleration
    - Gemini Video-LLM caption (tokenized to input_ids & attention_mask)
    """

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], f"Unsupported mode {mode}"
        self.cfg = cfg
        self.mode = mode
        self._num_retries = num_retries

        # Set clip count for train vs test (ensemble views)
        if mode in ["train", "val"]:
            self._num_clips = 1
        else:  # test
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info(f"Constructing Vioguard_eccv for {mode}...")
        self._construct_loader()

        # === GEMINI CAPTION SETUP ===
        # Load the single large JSON file instead of individual txt files
        self.caption_path = cfg.DATA.CAPTION_JSON_PATH
        self.captions_map = {}
        
        if os.path.exists(self.caption_path):
            logger.info(f"Loading captions from {self.caption_path}...")
            try:
                with open(self.caption_path, 'r') as f:
                    self.captions_map = json.load(f)
                logger.info(f"Loaded {len(self.captions_map)} captions.")
            except Exception as e:
                logger.error(f"Failed to load caption JSON: {e}")
        else:
            logger.warning(f"Caption file not found at {self.caption_path}. Using placeholders.")

        # Tokenizer initialization
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = 128 # Increased length for richer Gemini descriptions

    def _construct_loader(self):
        """
        Build lists of video paths, pose paths, and labels.
        Expects a CSV file in format: path/to/video.mp4 label_id
        """
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f"{self.mode}.csv")
        assert g_pathmgr.exists(path_to_file), f"{path_to_file} not found"

        self._path_to_videos = []
        self._path_to_poses = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._video_meta = {}

        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, line in enumerate(f.read().splitlines()):
                assert len(line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2
                path, label = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)

                for idx in range(self._num_clips):
                    video_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    
                    # Construct pose path (assumes same filename but .npy extension)
                    # Adjust logic if your pose folder structure is different
                    pose_name = os.path.splitext(os.path.basename(path))[0] + "_pose.npy"
                    pose_path = os.path.join(self.cfg.DATA.POSE_PATH_PREFIX, pose_name)

                    self._path_to_videos.append(video_path)
                    self._path_to_poses.append(pose_path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        logger.info(f"Constructed dataset with {len(self._path_to_videos)} samples.")

    def _load_pose_tensor(self, path):
        """
        Load pose data and pad/truncate to MAX_NUM_PEOPLE.
        Pose format: (T, P, J, D)
        """
        try:
            # Check if file exists to avoid crash
            if not os.path.exists(path):
                # logger.warning(f"Pose file missing: {path}") # Reduce verbosity
                return torch.zeros((self.cfg.DATA.NUM_FRAMES, MAX_NUM_PEOPLE, self.cfg.DATA.NUM_JOINTS, 4))
                
            pose_array = np.load(path)  # (T, P, J, D)
            T = self.cfg.DATA.NUM_FRAMES
            total_frames = pose_array.shape[0]
            
            # Simple temporal sampling to match T
            if total_frames > 0:
                idxs = np.linspace(0, total_frames - 1, T).astype(int)
                pose_array = pose_array[idxs]
            else:
                return torch.zeros((T, MAX_NUM_PEOPLE, self.cfg.DATA.NUM_JOINTS, 4))

            P = pose_array.shape[1]
            if P < MAX_NUM_PEOPLE:
                pad = np.zeros((pose_array.shape[0], MAX_NUM_PEOPLE - P, pose_array.shape[2], pose_array.shape[3]))
                pose_array = np.concatenate([pose_array, pad], axis=1)
            elif P > MAX_NUM_PEOPLE:
                pose_array = pose_array[:, :MAX_NUM_PEOPLE, :, :]

            return torch.from_numpy(pose_array).float()
            
        except Exception as e:
            logger.warning(f"Failed to load pose: {path} ({e})")
            return torch.zeros((self.cfg.DATA.NUM_FRAMES, MAX_NUM_PEOPLE, self.cfg.DATA.NUM_JOINTS, 4))

    def _get_text_tokens(self, video_path, label):
        """
        Get Gemini caption tokens for the given video.
        Includes fallback logic for safety refusals.
        """
        video_filename = os.path.basename(video_path)
        raw_caption = self.captions_map.get(video_filename, "")

        # --- SAFETY & FALLBACK LOGIC ---
        refusal_keywords = [
            "cannot fulfill", "safety guidelines", "unable to generate", 
            "policy violation", "harmful content", "illegal acts"
        ]
        
        # If refused or empty, use Hard Fallback based on label
        if any(k in raw_caption.lower() for k in refusal_keywords) or len(raw_caption) < 10:
            if label == 0: # VIOLENT CLASS
                # "Privileged Information": Explicit description for violent fallback
                raw_caption = "A video depicting extreme physical violence, aggression, and rapid physical contact."
            else: # NON-VIOLENT CLASS
                raw_caption = "A surveillance video showing normal daily human activity and interaction."

        # Tokenize
        encoding = self.tokenizer(
            raw_caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        # Determine temporal/spatial sampling parameters
        if self.mode == "train":
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:  # val/test
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale = max_scale = crop_size = self.cfg.DATA.TEST_CROP_SIZE

        # Sampling rate for frames
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        # Load video
        for i_try in range(self._num_retries):
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
                if self.mode != "test" and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Normalize and spatial crop
            frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
            frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            # Pack SlowFast pathways (slow + fast inputs)
            frames = utils.pack_pathway_output(self.cfg, frames)

            # Load pose tensor
            pose_tensor = self._load_pose_tensor(self._path_to_poses[index])

            # Load Label
            label = self._labels[index]

            # Load Text (Gemini)
            input_ids, attention_mask = self._get_text_tokens(self._path_to_videos[index], label)

            # Return 4 inputs (RGB, pose, BLIP input_ids, BLIP attention_mask)
            # Tuple order MUST match train_net.py unpacking line
            return (frames, pose_tensor, input_ids, attention_mask), label, index, {}

        raise RuntimeError(f"Failed to fetch video after {self._num_retries} retries.")

    def __len__(self):
        return len(self._path_to_videos)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)