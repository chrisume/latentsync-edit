import os
import torch
import argparse
import yaml
import json
import time
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from latentsync.models.unet import HighResUNet
from latentsync.models.phoneme_mapper import EnhancedPhonemeMapper
from latentsync.training.dataset import LatentSyncDataset
from latentsync.training.loss import LatentSyncLoss
from latentsync.utils.batch_utils import auto_configure_batch_size, get_vram_info

def parse_args():
    parser = argparse.ArgumentParser(description="Train LatentSync high-resolution model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/high_res_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/latentsync-highres",
        help="Path to output directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA for model weights",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution",
    )
    
    return parser.parse_args()

def setup_distributed():
    """Setup distributed training"""
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed training if needed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        setup_distributed()
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    config = OmegaConf.create(config)
    
    # Override configuration with command line arguments
    config.model.resolution = args.resolution
    
    # Create output directory
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)
    
    # Get VRAM info
    vram_info = get_vram_info()
    
    # Auto-configure batch size if needed
    if config.training.auto_batch_size:
        batch_size = auto_configure_batch_size(config)
        if global_rank == 0:
            print(f"Auto-configured batch size: {batch_size}")
    else:
        batch_size = config.training.batch_size
    
    # Create dataset and dataloader
    train_dataset = LatentSyncDataset(
        data_dir=args.data_dir,
        split="train",
        resolution=config.model.resolution
    )
    
    val_dataset = LatentSyncDataset(
        data_dir=args.data_dir,
        split="val",
        resolution=config.model.resolution
    )
    
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create models
    unet = HighResUNet(config).to(device)
    phoneme_mapper = EnhancedPhonemeMapper(config).to(device)
    
    # Enable gradient checkpointing if configured
    if config.model.use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        phoneme_mapper.apply(lambda m: setattr(m, "use_checkpoint", True) if hasattr(m, "use_checkpoint") else None)
    
    # Enable xformers memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        if hasattr(config.model, "attention_type"):
            config.model.attention_type = "xformers"
            if global_rank == 0:
                print("Using xformers memory efficient attention")
    
    # Create EMA models if requested
    if args.use_ema:
        from torch_ema import ExponentialMovingAverage
        ema_unet = ExponentialMovingAverage(unet.parameters(), decay=0.9999)
        ema_phoneme_mapper = ExponentialMovingAverage(phoneme_mapper.parameters(), decay=0.9999)
    
    # Wrap models with DDP if distributed
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
        phoneme_mapper = DDP(phoneme_mapper, device_ids=[local_rank], output_device=local_rank)
    
    # Create optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                list(unet.parameters()) + list(phoneme_mapper.parameters()),
                lr=args.learning_rate,
                betas=(config.training.adam_beta1, config.training.adam_beta2),
                weight_decay=config.training.adam_weight_decay,
                eps=config.training.adam_epsilon
            )
            if global_rank == 0:
                print("Using 8-bit Adam optimizer")
        except ImportError:
            print("bitsandbytes not found, using regular AdamW")
            optimizer = AdamW(
                list(unet.parameters()) + list(phoneme_mapper.parameters()),
                lr=args.learning_rate,
                betas=(config.training.adam_beta1, config.training.adam_beta2),
                weight_decay=config.training.adam_weight_decay,
                eps=config.training.adam_epsilon
            )
    else:
        optimizer = AdamW(
            list(unet.parameters()) + list(phoneme_mapper.parameters()),
            lr=args.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
            eps=config.training.adam_epsilon
        )
    
    # Create learning rate scheduler
    if config.training.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.max_train_steps,
            eta_min=1e-6
        )
    elif config.training.lr_scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=args.max_train_steps
        )
    else:
        lr_scheduler = None
    
    # Create loss function
    loss_fn = LatentSyncLoss(config)
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision != "no" else None
    
    # Create tensorboard writer
    if global_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)
        unet.load_state_dict(checkpoint["unet"])
        phoneme_mapper.load_state_dict(checkpoint["phoneme_mapper"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if args.use_ema:
            ema_unet.load_state_dict(checkpoint["ema_unet"])
            ema_phoneme_mapper.load_state_dict(checkpoint["ema_phoneme_mapper"])
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1
        if global_rank == 0:
            print(f"Resumed from checkpoint: {args.resume_from} at step {start_step}")
    
    # Training loop
    global_step = start_step
    progress_bar = range(global_step, args.max_train_steps)
    if global_rank == 0:
        from tqdm import tqdm
        progress_bar = tqdm(progress_bar, desc="Training", dynamic_ncols=True)
    
    # Set models to training mode
    unet.train()
    phoneme_mapper.train()
    
    # Training loop
    for step in progress_bar:
        # Reset gradients
        optimizer.zero_grad()
        
        # Accumulate gradients over multiple steps
        for _ in range(args.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except (StopIteration, NameError):
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            if args.mixed_precision != "no":
                with autocast(dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16):
                    # Process phoneme features
                    phoneme_output = phoneme_mapper(
                        phoneme_features=batch["phoneme_features"],
                        audio_features=batch.get("audio_features")
                    )
                    
                    # Get visual features
                    visual_features = phoneme_output["visual_features"]
                    
                    # Process with UNet
                    unet_output = unet(
                        x=batch["reference"],
                        timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                        audio_features=visual_features
                    )
                    
                    # Calculate loss
                    loss_dict = loss_fn(
                        pred=unet_output,
                        target=batch["target"],
                        phoneme_output=phoneme_output
                    )
                    
                    # Get total loss
                    loss = loss_dict["total_loss"] / args.gradient_accumulation_steps
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
            else:
                # Process phoneme features
                phoneme_output = phoneme_mapper(
                    phoneme_features=batch["phoneme_features"],
                    audio_features=batch.get("audio_features")
                )
                
                # Get visual features
                visual_features = phoneme_output["visual_features"]
                
                # Process with UNet
                unet_output = unet(
                    x=batch["reference"],
                    timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                    audio_features=visual_features
                )
                
                # Calculate loss
                loss_dict = loss_fn(
                    pred=unet_output,
                    target=batch["target"],
                    phoneme_output=phoneme_output
                )
                
                # Get total loss
                loss = loss_dict["total_loss"] / args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
        
        # Update weights
        if args.mixed_precision != "no":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update EMA models if enabled
        if args.use_ema:
            ema_unet.update()
            ema_phoneme_mapper.update()
        
        # Log progress
        if global_rank == 0:
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss_dict["total_loss"].item(),
                "lr": optimizer.param_groups[0]["lr"]
            })
            
            # Log to tensorboard
            writer.add_scalar("train/total_loss", loss_dict["total_loss"].item(), global_step)
            writer.add_scalar("train/reconstruction_loss", loss_dict["reconstruction_loss"].item(), global_step)
            writer.add_scalar("train/perceptual_loss", loss_dict.get("perceptual_loss", 0.0), global_step)
            writer.add_scalar("train/viseme_loss", loss_dict.get("viseme_loss", 0.0), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        
        # Run validation
        if (step + 1) % args.validation_steps == 0:
            run_validation(
                unet=unet,
                phoneme_mapper=phoneme_mapper,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
                global_step=global_step,
                writer=writer if global_rank == 0 else None,
                args=args,
                config=config,
                ema_unet=ema_unet if args.use_ema else None,
                ema_phoneme_mapper=ema_phoneme_mapper if args.use_ema else None
            )
        
        # Save checkpoint
        if (step + 1) % args.checkpointing_steps == 0:
            if global_rank == 0:
                save_checkpoint(
                    unet=unet,
                    phoneme_mapper=phoneme_mapper,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    step=step,
                    output_dir=args.output_dir,
                    ema_unet=ema_unet if args.use_ema else None,
                    ema_phoneme_mapper=ema_phoneme_mapper if args.use_ema else None
                )
        
        # Update global step
        global_step += 1
        
        # Check if we reached max steps
        if global_step >= args.max_train_steps:
            break
    
    # Save final checkpoint
    if global_rank == 0:
        save_checkpoint(
            unet=unet,
            phoneme_mapper=phoneme_mapper,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            step=global_step - 1,
            output_dir=args.output_dir,
            ema_unet=ema_unet if args.use_ema else None,
            ema_phoneme_mapper=ema_phoneme_mapper if args.use_ema else None,
            is_final=True
        )
    
    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()

def run_validation(
    unet,
    phoneme_mapper,
    val_dataloader,
    loss_fn,
    device,
    global_step,
    writer=None,
    args=None,
    config=None,
    ema_unet=None,
    ema_phoneme_mapper=None
):
    """Run validation"""
    # Set models to eval mode
    unet.eval()
    phoneme_mapper.eval()
    
    # Initialize validation metrics
    val_loss = 0.0
    val_reconstruction_loss = 0.0
    val_perceptual_loss = 0.0
    val_viseme_loss = 0.0
    num_batches = 0
    
    # Run validation with no gradients
    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Use EMA models if available
            if ema_unet is not None and ema_phoneme_mapper is not None:
                with ema_unet.average_parameters(), ema_phoneme_mapper.average_parameters():
                    # Process phoneme features
                    phoneme_output = phoneme_mapper(
                        phoneme_features=batch["phoneme_features"],
                        audio_features=batch.get("audio_features")
                    )
                    
                    # Get visual features
                    visual_features = phoneme_output["visual_features"]
                    
                    # Process with UNet
                    unet_output = unet(
                        x=batch["reference"],
                        timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                        audio_features=visual_features
                    )
            else:
                # Process phoneme features
                phoneme_output = phoneme_mapper(
                    phoneme_features=batch["phoneme_features"],
                    audio_features=batch.get("audio_features")
                )
                
                # Get visual features
                visual_features = phoneme_output["visual_features"]
                
                # Process with UNet
                unet_output = unet(
                    x=batch["reference"],
                    timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                    audio_features=visual_features
                )
            
            # Calculate loss
            loss_dict = loss_fn(
                pred=unet_output,
                target=batch["target"],
                phoneme_output=phoneme_output
            )
            
            # Update validation metrics
            val_loss += loss_dict["total_loss"].item()
            val_reconstruction_loss += loss_dict["reconstruction_loss"].item()
            val_perceptual_loss += loss_dict.get("perceptual_loss", 0.0)
            val_viseme_loss += loss_dict.get("viseme_loss", 0.0)
            num_batches += 1
    
    # Calculate average metrics
    val_loss /= num_batches
    val_reconstruction_loss /= num_batches
    val_perceptual_loss /= num_batches
    val_viseme_loss /= num_batches
    
    # Log validation metrics
    if writer is not None:
        writer.add_scalar("val/total_loss", val_loss, global_step)
        writer.add_scalar("val/reconstruction_loss", val_reconstruction_loss, global_step)
        writer.add_scalar("val/perceptual_loss", val_perceptual_loss, global_step)
        writer.add_scalar("val/viseme_loss", val_viseme_loss, global_step)
        
        # Log sample images
        if num_batches > 0:
            # Get first batch
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Use EMA models if available
                if ema_unet is not None and ema_phoneme_mapper is not None:
                    with ema_unet.average_parameters(), ema_phoneme_mapper.average_parameters():
                        # Process phoneme features
                        phoneme_output = phoneme_mapper(
                            phoneme_features=batch["phoneme_features"],
                            audio_features=batch.get("audio_features")
                        )
                        
                        # Get visual features
                        visual_features = phoneme_output["visual_features"]
                        
                        # Process with UNet
                        unet_output = unet(
                            x=batch["reference"],
                            timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                            audio_features=visual_features
                        )
                else:
                    # Process phoneme features
                    phoneme_output = phoneme_mapper(
                        phoneme_features=batch["phoneme_features"],
                        audio_features=batch.get("audio_features")
                    )
                    
                    # Get visual features
                    visual_features = phoneme_output["visual_features"]
                    
                    # Process with UNet
                    unet_output = unet(
                        x=batch["reference"],
                        timesteps=torch.zeros(batch["reference"].shape[0], device=device),
                        audio_features=visual_features
                    )
                
                # Log images
                for i in range(min(4, batch["reference"].shape[0])):
                    # Create grid of reference, target, and prediction
                    from torchvision.utils import make_grid
                    grid = make_grid([
                        batch["reference"][i].cpu(),
                        batch["target"][i].cpu(),
                        unet_output[i].cpu()
                    ], nrow=3)
                    
                    # Add to tensorboard
                    writer.add_image(f"val/sample_{i}", grid, global_step)
                
                break
    
    # Print validation metrics
    print(f"Validation: loss={val_loss:.4f}, reconstruction={val_reconstruction_loss:.4f}, perceptual={val_perceptual_loss:.4f}, viseme={val_viseme_loss:.4f}")
    
    # Set models back to train mode
    unet.train()
    phoneme_mapper.train()

def save_checkpoint(
    unet,
    phoneme_mapper,
    optimizer,
    lr_scheduler,
    scaler,
    step,
    output_dir,
    ema_unet=None,
    ema_phoneme_mapper=None,
    is_final=False
):
    """Save checkpoint"""
    # Create checkpoint directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        "unet": unet.module.state_dict() if hasattr(unet, "module") else unet.state_dict(),
        "phoneme_mapper": phoneme_mapper.module.state_dict() if hasattr(phoneme_mapper, "module") else phoneme_mapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }
    
    # Add learning rate scheduler if available
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    
    # Add gradient scaler if available
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    # Add EMA models if available
    if ema_unet is not None:
        checkpoint["ema_unet"] = ema_unet.state_dict()
    
    if ema_phoneme_mapper is not None:
        checkpoint["ema_phoneme_mapper"] = ema_phoneme_mapper.state_dict()
    
    # Save checkpoint
    if is_final:
        checkpoint_path = os.path.join(output_dir, "checkpoint-final.pt")
    else:
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{step:06d}.pt")
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
