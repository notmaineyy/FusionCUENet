#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.POSE_PATH_PREFIX = "/vol/bitbucket/sna21/dataset/VioGuard/multi/pose_outputs"
    _C.DATA.POSE_DIM = 34
    _C.DATA.NUM_JOINTS = 33
    _C.MODEL.FREEZE_TEXT_ENCODER = False  # Whether to freeze the text encoder during training
    _C.DATA.CAPTION_PATH_PREFIX = "/vol/bitbucket/sna21/dataset/VioGuard/blip"
    _C.MODEL.USE_RGB = True
    _C.MODEL.USE_POSE = False
    _C.MODEL.USE_TEXT = False
    _C.MODEL.POSE_USE_VELOCITY = False
    _C.MODEL.ENABLE_LORA = False
    _C.MODEL.SOFT_PROMPT = False
    _C.MODEL.SOFT_PROMPT_N_CTX = 16
    _C.TEST.PREDICTION_CSV_PATH = ""

    pass
