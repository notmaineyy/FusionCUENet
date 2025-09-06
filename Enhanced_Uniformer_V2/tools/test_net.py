#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model with ablation support."""

import numpy as np
import os
import pickle
import torch
import sys
from iopath.common.file_io import g_pathmgr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter

import csv
import os

# CSV file path
prediction_csv_path = "/vol/bitbucket/sna21/dataset/predictions/ubi_fights/original_cuenet.csv"

# Initialize CSV with headers (do this once before calling perform_test)
with open(prediction_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_index", "predicted_class", "true_class", "confidence"])

logger = logging.get_logger(__name__)


def extract_modalities_from_inputs(inputs, cfg):
    """
    Extract different modalities from inputs based on what's available and what's enabled in config.
    Returns: (video_tensor, pose_tensor, input_ids, attention_mask)
    """
    video_tensor = None
    pose_tensor = None
    input_ids = None
    attention_mask = None

    print(len(inputs))
    
    if isinstance(inputs, (list, tuple)):
        # Different input formats based on dataset configuration
        if len(inputs) == 2:
            # Could be (video, pose) or (video, text_data) or (pose, text_data)
            if hasattr(cfg.MODEL, 'USE_RGB') and cfg.MODEL.USE_RGB and hasattr(cfg.MODEL, 'USE_POSE') and cfg.MODEL.USE_POSE:
                video_tensor, pose_tensor = inputs
            elif hasattr(cfg.MODEL, 'USE_RGB') and cfg.MODEL.USE_RGB and hasattr(cfg.MODEL, 'USE_TEXT') and cfg.MODEL.USE_TEXT:
                video_tensor, text_data = inputs
                if isinstance(text_data, dict):
                    input_ids = text_data.get('input_ids')
                    attention_mask = text_data.get('attention_mask')
                elif isinstance(text_data, (list, tuple)) and len(text_data) == 2:
                    input_ids, attention_mask = text_data
            elif hasattr(cfg.MODEL, 'USE_POSE') and cfg.MODEL.USE_POSE and hasattr(cfg.MODEL, 'USE_TEXT') and cfg.MODEL.USE_TEXT:
                pose_tensor, text_data = inputs
                if isinstance(text_data, dict):
                    input_ids = text_data.get('input_ids')
                    attention_mask = text_data.get('attention_mask')
                elif isinstance(text_data, (list, tuple)) and len(text_data) == 2:
                    input_ids, attention_mask = text_data
                    
        elif len(inputs) == 3:
            # (video, input_ids, attention_mask) or (video, pose, text_data) or (pose, input_ids, attention_mask)
            if hasattr(cfg.MODEL, 'USE_RGB') and cfg.MODEL.USE_RGB and hasattr(cfg.MODEL, 'USE_TEXT') and cfg.MODEL.USE_TEXT and not (hasattr(cfg.MODEL, 'USE_POSE') and cfg.MODEL.USE_POSE):
                video_tensor, input_ids, attention_mask = inputs
            elif hasattr(cfg.MODEL, 'USE_POSE') and cfg.MODEL.USE_POSE and hasattr(cfg.MODEL, 'USE_TEXT') and cfg.MODEL.USE_TEXT and not (hasattr(cfg.MODEL, 'USE_RGB') and cfg.MODEL.USE_RGB):
                pose_tensor, input_ids, attention_mask = inputs
            elif hasattr(cfg.MODEL, 'USE_RGB') and cfg.MODEL.USE_RGB and hasattr(cfg.MODEL, 'USE_POSE') and cfg.MODEL.USE_POSE:
                video_tensor, pose_tensor, text_data = inputs
                if isinstance(text_data, dict):
                    input_ids = text_data.get('input_ids')
                    attention_mask = text_data.get('attention_mask')
                elif isinstance(text_data, (list, tuple)) and len(text_data) == 2:
                    input_ids, attention_mask = text_data
                    
        elif len(inputs) == 4:
            # All modalities: (video, pose, input_ids, attention_mask)
            video_tensor, pose_tensor, input_ids, attention_mask = inputs
            
        # Handle case where video is a list that needs stacking
        if video_tensor is not None and isinstance(video_tensor, list):
            video_tensor = torch.stack(video_tensor, dim=1)
            
    else:
        # Single input - assume it's video
        video_tensor = inputs
    
    return video_tensor, pose_tensor, input_ids, attention_mask


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    Perform multi-view testing with ablation support for different modality combinations.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    
    # Log ablation configuration
    use_rgb = getattr(cfg.MODEL, 'USE_RGB', True)
    use_pose = getattr(cfg.MODEL, 'USE_POSE', True)
    use_text = getattr(cfg.MODEL, 'USE_TEXT', True)
    
    logger.info("Testing with ablation configuration:")
    logger.info(f"  RGB: {'ENABLED' if use_rgb else 'DISABLED'}")
    logger.info(f"  Pose: {'ENABLED' if use_pose else 'DISABLED'}")  
    logger.info(f"  Text: {'ENABLED' if use_text else 'DISABLED'}")

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Extract modalities from inputs
            video_tensor, pose_tensor, input_ids, attention_mask = extract_modalities_from_inputs(inputs, cfg)
            
            # Move tensors to GPU based on what's available and enabled
            if video_tensor is not None and use_rgb:
                video_tensor = video_tensor.cuda(non_blocking=True)
            else:
                video_tensor = None
                
            if pose_tensor is not None and use_pose:
                pose_tensor = pose_tensor.cuda(non_blocking=True)
            else:
                pose_tensor = None
                
            if input_ids is not None and attention_mask is not None and use_text:
                input_ids = input_ids.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
            else:
                input_ids = None
                attention_mask = None

            # Transfer labels and metadata to GPU
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
                    
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions for detection
            preds = model(video_tensor, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform forward pass for classification with ablation support
            try:
                # Call model with only the enabled modalities
                model_kwargs = {}
                if video_tensor is not None:
                    model_kwargs['video_tensor'] = video_tensor
                if pose_tensor is not None:
                    model_kwargs['pose_tensor'] = pose_tensor
                if input_ids is not None:
                    model_kwargs['input_ids'] = input_ids
                if attention_mask is not None:
                    model_kwargs['attention_mask'] = attention_mask
                
                if cfg.TEST.ADD_SOFTMAX:
                    preds = model(**model_kwargs).softmax(-1)
                else:
                    preds = model(**model_kwargs)
                    
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                logger.error(f"Model inputs: {list(model_kwargs.keys())}")
                logger.error(f"Shapes: {[(k, v.shape if v is not None else None) for k, v in model_kwargs.items()]}")
                raise e

            # Gather all predictions across devices
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
                
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()

            # Save predictions to CSV
            with open(prediction_csv_path, mode="a", newline="") as f:
                csv_writer = csv.writer(f)
                for i in range(preds.shape[0]):
                    pred_class = preds[i].argmax().item()
                    true_class = labels[i].item()
                    confidence = preds[i].max().item()
                    video_id = video_idx[i].item()
                    csv_writer.writerow([video_id, pred_class, true_class, confidence])
                    
            # Update test meter
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print final testing results
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics(ks=(1, 2))
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model with ablation support.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment
    du.init_distributed_training(cfg)
    # Set random seed from configs
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics
    model = build_model(cfg)
    
    # Load test checkpoint
    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    logger.info("Test completed successfully")
    
    if writer is not None:
        writer.close()

    # Save results
    file_name = f'{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}.pkl'
    with g_pathmgr.open(os.path.join(cfg.OUTPUT_DIR, file_name), 'wb') as f:
        result = {
            'video_preds': test_meter.video_preds.cpu().numpy(),
            'video_labels': test_meter.video_labels.cpu().numpy()
        }
        logger.info(f"Saving results with shape: preds={result['video_preds'].shape}, labels={result['video_labels'].shape}")
        pickle.dump(result, f)