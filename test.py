import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve

from CLIP.clip import create_model
from CLIP.adapter import AVADino
from tools.utils import ava_dino_forward
from tools.visualization import visualization
from Datasets import DATASET_REGISTRY, DATASET_CLASSES


use_cuda = torch.cuda.is_available()
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


def prepare_test_loader(dataset_name, category, args):
    dataset_name = dataset_name.lower()
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        source=root_path, split=split_cls.TEST,
        classname=category, resize=512, imagesize=512,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.cls_token_adapter_normal.load_state_dict(ckpt['cls_token_adapter_normal'])
    model.cls_token_adapter_anomaly.load_state_dict(ckpt['cls_token_adapter_anomaly'])
    model.patch_token_adapter_normal.load_state_dict(ckpt['patch_token_adapter_normal'])
    model.patch_token_adapter_anomaly.load_state_dict(ckpt['patch_token_adapter_anomaly'])
    model.prompt_adapter.load_state_dict(ckpt['prompt_adapter'])
    model.text_projection.load_state_dict(ckpt['text_projection'])
    return model


def evaluate(clip_model, dino_model, model, args, device, epoch):
    """Run evaluation on all categories of a dataset. Returns mean P-AUROC, I-AUROC, F1."""
    pixel_auroc_list, image_auroc_list, f1_list = [], [], []

    sep = '=' * 100
    print(f"\n{sep}\nDataset: {args.dataset}  |  Epoch: {epoch}\n{sep}")

    with torch.no_grad():
        for category in sorted(DATASET_CLASSES[args.dataset]):
            out_dir = os.path.join(args.result_path, category)
            os.makedirs(out_dir, exist_ok=True)

            pixel_pred, pixel_gt, image_pred, image_gt, img_paths = [], [], [], [], []
            loader = prepare_test_loader(args.dataset, category, args)

            for image_info in tqdm(loader, desc=f"{category:<20s}"):
                _, mask, cm_map, _, _ = ava_dino_forward(
                    clip_model, image_info, device, model, dino_model
                )
                scores = cm_map[:, 1]

                pixel_pred.extend(scores.cpu().numpy())
                pixel_gt.extend(mask.squeeze(1).cpu().numpy())
                image_pred.extend(scores.flatten(1).max(1)[0].cpu().numpy())
                image_gt.extend(image_info["is_anomaly"].numpy())
                img_paths.extend(image_info["image_path"])
                torch.cuda.empty_cache()

            pixel_pred = np.array(pixel_pred)
            pixel_gt   = np.array(pixel_gt)
            image_pred = np.array(image_pred)
            image_gt   = np.array(image_gt)

            # Normalize to [0, 1]
            pixel_pred = (pixel_pred - pixel_pred.min()) / (pixel_pred.max() - pixel_pred.min() + 1e-8)
            image_pred = (image_pred - image_pred.min()) / (image_pred.max() - image_pred.min() + 1e-8)

            visualization(img_paths, pixel_pred, pixel_gt, category, args.result_path)

            p_auc = roc_auc_score(pixel_gt.flatten(), pixel_pred.flatten()) \
                    if pixel_gt.sum() > 0 else np.nan
            i_auc = roc_auc_score(image_gt, image_pred) \
                    if len(np.unique(image_gt)) > 1 else np.nan

            if pixel_gt.sum() > 0:
                prec, rec, _ = precision_recall_curve(pixel_gt.flatten(), pixel_pred.flatten())
                f1s = 2 * prec * rec / (prec + rec + 1e-8)
                f1  = np.max(f1s[np.isfinite(f1s)])
            else:
                f1 = np.nan

            pixel_auroc_list.append(p_auc)
            image_auroc_list.append(i_auc)
            f1_list.append(f1)

            print(f"  {category:<25s} P-AUROC: {p_auc:.5f}  I-AUROC: {i_auc:.5f}  F1: {f1:.5f}")
            torch.cuda.empty_cache()

    mean_p = np.nanmean(pixel_auroc_list)
    mean_i = np.nanmean(image_auroc_list)
    mean_f = np.nanmean(f1_list)
    print(f"\n  {'MEAN':<25s} P-AUROC: {mean_p:.5f}  I-AUROC: {mean_i:.5f}  F1: {mean_f:.5f}")
    print(sep)

    # Save per-epoch metrics
    metric_path = os.path.join(args.result_path, "metrics.txt")
    with open(metric_path, 'a') as f:
        f.write(f"\nDataset: {args.dataset} | Epoch: {epoch}\n")
        f.write(f"{'Category':<25s} {'P-AUROC':<12s} {'I-AUROC':<12s} {'F1'}\n")
        f.write('-' * 60 + '\n')
        for i, cat in enumerate(sorted(DATASET_CLASSES[args.dataset])):
            f.write(f"{cat:<25s} {pixel_auroc_list[i]:<12.5f} {image_auroc_list[i]:<12.5f} {f1_list[i]:.5f}\n")
        f.write(f"{'MEAN':<25s} {mean_p:<12.5f} {mean_i:<12.5f} {mean_f:.5f}\n")

    return mean_p, mean_i, mean_f


def main():
    parser = argparse.ArgumentParser(description="AVA-DINO Evaluation")
    parser.add_argument("--dataset",        type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing epoch_N.pth files")
    parser.add_argument("--result_path",    type=str, default="./results_test")
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--start_epoch",    type=int, default=0)
    parser.add_argument("--num_epochs",     type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_path, exist_ok=True)

    print("Loading DINOv3...")
    dino_model = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local',
                                weights='./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    dino_model.to(device).eval()

    print("Loading CLIP...")
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512,
                              device=device, pretrained='openai', require_pretrained=True)
    clip_model.to(device).eval()

    model = AVADino(c_in=1024, device=device).to(device).eval()

    best = {'epoch': -1, 'p_auc': 0.0, 'i_auc': 0.0, 'f1': 0.0}
    results = []

    for epoch in range(args.start_epoch, args.num_epochs):
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping.")
            continue

        print(f"\nLoading checkpoint: epoch_{epoch}.pth")
        model = load_checkpoint(model, ckpt_path, device)
        torch.cuda.empty_cache()

        p_auc, i_auc, f1 = evaluate(clip_model, dino_model, model, args, device, epoch)
        results.append({'epoch': epoch, 'p_auc': p_auc, 'i_auc': i_auc, 'f1': f1})

        if p_auc > best['p_auc']:
            best = {'epoch': epoch, 'p_auc': p_auc, 'i_auc': i_auc, 'f1': f1}

        torch.cuda.empty_cache()

    print(f"\nBest model — Epoch: {best['epoch']}  "
          f"P-AUROC: {best['p_auc']:.5f}  I-AUROC: {best['i_auc']:.5f}  F1: {best['f1']:.5f}")

    summary_path = os.path.join(args.result_path, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint dir: {args.checkpoint_dir}\n\n")
        f.write(f"{'Epoch':<8s} {'P-AUROC':<12s} {'I-AUROC':<12s} {'F1'}\n")
        f.write('-' * 45 + '\n')
        for r in results:
            marker = '  <-- best' if r['epoch'] == best['epoch'] else ''
            f.write(f"{r['epoch']:<8d} {r['p_auc']:<12.5f} {r['i_auc']:<12.5f} {r['f1']:.5f}{marker}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
