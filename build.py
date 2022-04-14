import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from dataset.dataset import FSC_Dataset
from model.Extractor import (Resnet18FPN, Resnet50FPN)
from model.Regressor import CountRegressor
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from loss import BasicLoss



def build_optimizer(params, opt: dict):
    # opt = opt["optimizer"]
    optimizer_name = opt["name"].lower()
    if optimizer_name == "adam":
        optimizer = Adam(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    elif optimizer_name == "adamw":
        optimizer = AdamW(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    else:
        ValueError(f"Not availble optimzer f{optimizer_name}")

    return optimizer


def build_model(opt: dict, extractor: bool = False) -> nn.Module:
    # opt = opt["model"]

    if extractor:
        model_name = opt["extractor"]["name"].lower()
        if model_name == "resnet18":
            return Resnet18FPN()
        elif model_name == "resnet50":
            return Resnet50FPN()
        else:
            ValueError("Not available extractor")
    else:
        model_name = opt["regressor"]["name"].lower()
        if model_name == "countregressor":
            return CountRegressor(6, pool="mean")
        else:
            ValueError("Not availble regressor")

    bn_momentum = opt["bn_momentum"]
    if bn_momentum is not None:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.momentum = bn_momentum
    return model


def build_criterion(opt: dict, name):
    # opt = opt["loss"]
    criterion_name = name.lower()
    if criterion_name == "baseline" or criterion_name == "advanced":
        loss = BasicLoss(mse_weight=opt["mse_weight"])
    else:
        ValueError("No Available Type of Loss")

    return loss


def build_scheduler(opt: dict, optimizer, loader, start_epoch):
    # opt = opt BE CAREFUL!
    scheduler_type = opt["scheduler"]['name'].lower()

    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # noqa
            optimizer,
            max_lr=opt['optimizer']['lr'],
            epochs=opt['train']['epoch'] + 1,
            steps_per_epoch=len(loader) // opt["train"]["num_accum"],
            cycle_momentum=opt["scheduler"].get("cycle_momentum", True),
            base_momentum=0.85,
            max_momentum=0.95,
            pct_start=opt["scheduler"]["pct_start"],
            last_epoch=start_epoch - 1,
            div_factor=opt["scheduler"]['div_factor'],
            final_div_factor=opt["scheduler"]['final_div_factor']
        )
    elif scheduler_type == "cos_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=opt["scheduler"]["t_max"], eta_min=opt["scheduler"]["eta_min"])
    else:
        raise ValueError(f"Unsupported scheduler type {scheduler_type}.")

    return scheduler

def build_dataset(opt: dict, mode: str = "train") -> FSC_Dataset:
    # opt = opt['dataset']
    return FSC_Dataset(
        data_path=opt["data_path"],
        data_type=opt["data_type"],
        mode=mode
    )


def build_dataloader(dataset: FSC_Dataset, opt: dict, shuffle: bool = True) -> DataLoader:
    # opt = opt["dataloader"]
    if not dist.is_initialized():
        return DataLoader(
            dataset,
            batch_size=opt["batch_size"],
            shuffle=shuffle,
            num_workers=opt.get("num_workers", 4),
            pin_memory=True,
            drop_last=shuffle,
        )
    else:
        assert dist.is_available() and dist.is_initialized()
        ddp_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=shuffle,
        )
        world_size = dist.get_world_size()
        return DataLoader(
            dataset,
            batch_size=opt["batch_size"] // world_size,
            num_workers=(opt.get("num_workers", 4) + world_size - 1) // world_size,
            pin_memory=True,
            sampler=ddp_sampler,
        )


def split_params_for_optimizer(model, opt):
    # opt = opt["optimizer"]
    params_small_lr = []
    params_small_lr_no_wd = []
    params_base_lr = []
    params_base_lr_no_wd = []
    for param_name, param_value in model.named_parameters():
        param_value: torch.Tensor
        if "encoder" in param_name:
            if param_value.ndim > 1:
                params_small_lr.append(param_value)
            else:
                params_small_lr_no_wd.append(param_value)
        else:  # decoder
            if param_value.ndim > 1:
                params_base_lr.append(param_value)
            else:
                params_base_lr_no_wd.append(param_value)
    params_for_optimizer = [
        {"params": params_base_lr},
        {"params": params_base_lr_no_wd, "weight_decay": 0.0},
        {"params": params_small_lr, "lr": opt["lr"] * 0.1},
        {"params": params_small_lr_no_wd, "lr": opt["lr"] * 0.1, "weight_decay": 0.0},
    ]
    return params_for_optimizer
