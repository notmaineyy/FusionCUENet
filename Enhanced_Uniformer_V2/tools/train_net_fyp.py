#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train BlipFusionCUENet: RGB + Pose + BLIP fusion model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint_amp as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)

# ------------------------------------------------------------
# Helper: move nested data structures to device
# ------------------------------------------------------------
def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, list):
        return [move_to_device(i, device) for i in x]
    elif isinstance(x, tuple):
        return tuple(move_to_device(i, device) for i in x)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


# ------------------------------------------------------------
# Train one epoch
# ------------------------------------------------------------
def train_epoch(train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer=None):
    """
    Training loop for BlipFusionCUENet (RGB + Pose + BLIP fusion).
    """
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # Mixup for RGB only
    mixup_fn = None
    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # ---------------------------------------------------------
        # Inputs format: (frames, pose, input_ids, attention_mask)
        # ---------------------------------------------------------
        frames, pose, input_ids, attention_mask = inputs

        # Stack if frames are list
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)

        # Move to GPU
        frames = frames.cuda(non_blocking=True)
        pose = pose.cuda(non_blocking=True)
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Meta (e.g., boxes for detection)
        meta = move_to_device(meta, torch.device("cuda"))

        # Learning rate schedule
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Mixup (only RGB frames)
        if mixup_fn:
            frames, labels = mixup_fn(frames, labels)

        # Forward pass with autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if cfg.DETECTION.ENABLE:
                preds = model((frames, pose, input_ids, attention_mask), meta["boxes"])
            else:
                print(f"Pose tensor shape going into model: {pose.shape}")
                preds = model(frames, pose, input_ids, attention_mask)

            # Compute loss
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            loss = loss_fun(preds, labels)

        misc.check_nan_losses(loss)

        # Backpropagation
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(
            loss, optimizer,
            clip_grad=cfg.SOLVER.CLIP_GRADIENT,
            parameters=model.parameters(),
            create_graph=is_second_order
        )

        # Compute accuracy (single-label only)
        accuracy = None
        if not cfg.DATA.MULTI_LABEL:
            preds_class = preds.argmax(dim=1)
            correct = (preds_class == labels).sum().item()
            accuracy = 100.0 * correct / labels.size(0)

        # Reduce loss across GPUs
        if cfg.NUM_GPUS > 1:
            [loss] = du.all_reduce([loss])
        loss = loss.item()

        # Update meter
        train_meter.update_stats(accuracy, loss, lr, frames[0].size(0) * max(cfg.NUM_GPUS, 1))

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalars(
                {"Train/loss": loss, "Train/lr": lr, "Train/accuracy": accuracy},
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # End of epoch
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


# ------------------------------------------------------------
# Evaluate one epoch
# ------------------------------------------------------------
@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer=None):
    """
    Evaluation loop for BlipFusionCUENet (RGB + Pose + BLIP fusion).
    """
    model.eval()
    val_meter.iter_tic()

    total_loss = 0.0
    total_samples = 0
    total_accuracy = 0.0

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        frames, pose, input_ids, attention_mask = inputs

        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)

        # Move tensors to GPU
        frames = frames.cuda(non_blocking=True)
        pose = pose.cuda(non_blocking=True)
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        meta = move_to_device(meta, torch.device("cuda"))

        val_meter.data_toc()

        # Forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if cfg.DETECTION.ENABLE:
                preds = model((frames, pose, input_ids, attention_mask), meta["boxes"])
            else:
                preds = model(frames, pose, input_ids, attention_mask)

            # Loss
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            loss = loss_fun(preds, labels)
            total_loss += loss.item() * frames.size(0)
            total_samples += frames.size(0)

            # Accuracy (single-label)
            if not cfg.DATA.MULTI_LABEL:
                preds_class = preds.argmax(dim=1)
                correct = (preds_class == labels).sum().item()
                total_accuracy += 100.0 * correct / labels.size(0) * frames.size(0)

                val_meter.update_stats(100.0 * correct / labels.size(0), frames.size(0) * max(cfg.NUM_GPUS, 1))

                if writer is not None:
                    writer.add_scalars(
                        {"Val/Accuracy": 100.0 * correct / labels.size(0)},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

        if cfg.NUM_GPUS:
            torch.cuda.empty_cache()

    # Epoch summary
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc = total_accuracy / total_samples if total_samples > 0 else 0.0

    logger.info(f"Val results: Loss={avg_loss:.4f}, Accuracy={acc:.4f}%, Samples={total_samples}")
    logging.log_json_stats({
        "_type": "val_epoch",
        "epoch": f"{cur_epoch + 1}/{cfg.SOLVER.MAX_EPOCH}",
        "loss": f"{avg_loss:.4f}",
        "accuracy": f"{acc:.4f}%",
        "gpu_mem": f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}G",
        "samples": total_samples,
    })

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


# ------------------------------------------------------------
# Precise BN computation
# ------------------------------------------------------------
@torch.no_grad()
def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update batch norm statistics precisely using a subset of the training set.
    """
    bn_layers = get_bn_modules(model)
    if len(bn_layers) == 0:
        return

    logger.info("Computing precise BN statistics...")
    update_bn_stats(model, loader, num_iters)


# ------------------------------------------------------------
# Build and run the training pipeline
# ------------------------------------------------------------
def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Loss scaler
    loss_scaler = NativeScaler()

    # Build the video model and print model statistics.
    model = build_model(cfg)
    #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #misc.log_model_info(model, cfg, use_train_input=True)

    print("Total parameters of the model == ", misc.params_count(model))

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loss_scaler,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch or cfg.TRAIN.SAVE_LATEST:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            flag = eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer)
            if flag:
                cu.save_best_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
        if cfg.NUM_GPUS:
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    """
    Entry point for training BlipFusionCUENet.
    """
    # Load config
    from slowfast.config.defaults import get_cfg
    from slowfast.utils.parser import load_config

    cfg = get_cfg()
    args = load_config(cfg)

    # Initialize distributed training
    du.init_distributed_training(cfg)

    # Set random seed
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Train
    train(cfg)


if __name__ == "__main__":
    main()
