import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from CLIP.clip import create_model
from CLIP.adapter import AVADino
from tools.loss import FocalLoss, BinaryDiceLoss
from tools.utils import ava_dino_forward
from Datasets import DATASET_REGISTRY


use_cuda = torch.cuda.is_available()
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


def prepare_train_loader(dataset_name, args):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname='ALL',
        resize=512,
        imagesize=512,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )


def train_one_epoch(model, dino_model, clip_model, loader, optimizer, scheduler,
                    loss_focal, loss_dice, epoch, args, device):
    model.train()
    losses = {'total': [], 'awareness': [], 'seg': [], 'global': [], 'routing': []}
    routing_log = {'w_normal': [], 'w_anomaly': []}
    start = time.time()

    for idx, image_info in enumerate(loader):
        ava_map, mask, anomaly_map_cm, global_score, routing_info = \
            ava_dino_forward(clip_model, image_info, device, model, dino_model)

        loss_aware   = loss_focal(ava_map, mask) + loss_dice(ava_map[:, 1], mask)
        loss_seg     = loss_focal(anomaly_map_cm, mask)    + loss_dice(anomaly_map_cm[:, 1], mask)
        loss_global  = F.cross_entropy(global_score, image_info["is_anomaly"].to(device).long())
        loss_routing = routing_info['routing_loss']

        loss = 0.25 * loss_aware + 0.5 * loss_seg + 0.25 * loss_global + 0.1 * loss_routing

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses['total'].append(loss.item())
        losses['awareness'].append(loss_aware.item())
        losses['seg'].append(loss_seg.item())
        losses['global'].append(loss_global.item())
        losses['routing'].append(loss_routing.item())
        routing_log['w_normal'].append(routing_info['w_normal'].mean().item())
        routing_log['w_anomaly'].append(routing_info['w_anomaly'].mean().item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | Batch {idx+1}/{len(loader)} "
            f"| loss: {loss.item():.4f} | aware: {loss_aware.item():.4f} "
            f"| seg: {loss_seg.item():.4f} | global: {loss_global.item():.4f} "
            f"| routing: {loss_routing.item():.4f} "
            f"| w(N/A): {routing_info['w_normal'].mean():.3f}/{routing_info['w_anomaly'].mean():.3f} "
            f"| {time.time()-start:.1f}s",
            end='\r', flush=True
        )

    print()
    return {k: np.mean(v) for k, v in losses.items()}, \
           {k: np.mean(v) for k, v in routing_log.items()}


def main():
    parser = argparse.ArgumentParser(description="AVA-DINO Training")
    parser.add_argument("--dataset",     type=str,   required=True,        help="Training dataset (mvtec | visa)")
    parser.add_argument("--result_path", type=str,   default="./results",  help="Output directory")
    parser.add_argument("--device",      type=str,   default="cuda:0")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(args.result_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Backbone models (frozen) ──────────────────────────────────────────────
    print("Loading DINOv3...")
    dino_model = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local',
                                weights='./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    dino_model.to(device).eval()

    print("Loading CLIP...")
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512,
                              device=device, pretrained='openai', require_pretrained=True)
    clip_model.to(device).eval()

    # ── AVA-DINO adapter (trainable) ─────────────────────────────────────────
    model = AVADino(c_in=1024, device=device).to(device)

    trainable_keys = [
        'patch_token_adapter_normal', 'patch_token_adapter_anomaly',
        'cls_token_adapter_normal',   'cls_token_adapter_anomaly',
        'prompt_adapter',             'text_projection',
    ]
    params = [p for n, p in model.named_parameters()
              if any(k in n for k in trainable_keys)]
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    train_loader = prepare_train_loader(args.dataset, args)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.03 * total_steps)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s) / max(1, warmup_steps) if s < warmup_steps else 1.0
    )

    loss_focal = FocalLoss()
    loss_dice  = BinaryDiceLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.result_path, "loss.txt")
    for epoch in range(args.epochs):
        losses, routing = train_one_epoch(
            model, dino_model, clip_model, train_loader,
            optimizer, scheduler, loss_focal, loss_dice, epoch, args, device
        )

        print(
            f"Epoch {epoch}: total={losses['total']:.6f}  aware={losses['awareness']:.6f}  "
            f"seg={losses['seg']:.6f}  global={losses['global']:.6f}  "
            f"routing={losses['routing']:.6f}  "
            f"w(N/A)={routing['w_normal']:.4f}/{routing['w_anomaly']:.4f}"
        )

        torch.save({
            'epoch': epoch,
            'cls_token_adapter_normal':    model.cls_token_adapter_normal.state_dict(),
            'cls_token_adapter_anomaly':   model.cls_token_adapter_anomaly.state_dict(),
            'patch_token_adapter_normal':  model.patch_token_adapter_normal.state_dict(),
            'patch_token_adapter_anomaly': model.patch_token_adapter_anomaly.state_dict(),
            'prompt_adapter':              model.prompt_adapter.state_dict(),
            'text_projection':             model.text_projection.state_dict(),
        }, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        with open(log_path, 'a') as f:
            f.write(
                f"epoch_{epoch}: "
                f"total={losses['total']:.6f}\t"
                f"aware={losses['awareness']:.6f}\t"
                f"seg={losses['seg']:.6f}\t"
                f"global={losses['global']:.6f}\t"
                f"routing={losses['routing']:.6f}\t"
                f"w(N/A)={routing['w_normal']:.4f}/{routing['w_anomaly']:.4f}\n"
            )


if __name__ == "__main__":
    main()
