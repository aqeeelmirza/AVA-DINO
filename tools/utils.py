import torch
import torch.nn.functional as F
from utils import encode_text_with_prompt_ensemble


def extract_dinov3_features(image_path, batch_img, device, dino_model):
    """
    Extract multi-scale patch and CLS tokens from DINOv3 via forward hooks.

    Hooks are registered on layers {5, 11, 17, 23} of the ViT-L/16 backbone.
    Patch tokens are layer-normalized and mean-centered per sample.

    Args:
        image_path: list of str, length B
        batch_img:  [B, 3, H, W]
        device:     torch.device
        dino_model: frozen DINOv3 ViT-L/16

    Returns:
        cls_tokens:    list of 4 tensors, each [B, 1, 1024]
        patch_tokens:  list of 4 tensors, each [B, N, 1024]
    """
    layers = [5, 11, 17, 23]
    patch_tokens_dict = {i: [] for i in layers}
    cls_tokens_dict   = {i: [] for i in layers}

    anchor = getattr(dino_model, "norm", None) or getattr(dino_model, "fc_norm", None)
    assert anchor is not None, "Cannot find norm layer in DINOv3 — print(dino_model) to verify."

    with torch.no_grad():
        for j in range(len(image_path)):
            tokens_dict = {}
            handles = []
            image = batch_img[j].unsqueeze(0).to(device)

            for i in layers:
                def _mk_hook(idx):
                    def _hook(module, inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()
                    return _hook
                handles.append(dino_model.blocks[i].register_forward_hook(_mk_hook(i)))

            with torch.inference_mode():
                dino_model(image)

            for h in handles:
                h.remove()

            for i, toks in tokens_dict.items():
                patches = toks[:, 5:, :]
                patches = (patches - patches.mean(dim=1, keepdim=True)) / (
                    patches.std(dim=1, keepdim=True) + 1e-6
                )
                patch_tokens_dict[i].append(patches)
                cls_tokens_dict[i].append(toks[:, 0, :].unsqueeze(1))

    patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0).to(device) for i in layers]
    cls_tokens   = [torch.cat(cls_tokens_dict[i],   dim=0).to(device) for i in layers]
    return cls_tokens, patch_tokens


def ava_dino_forward(clip_model, image_info, device, model, dino_model):
    """
    Full forward pass: extract features, apply dual adapters, compute anomaly maps.

    Args:
        clip_model:  frozen CLIP ViT-L/14-336
        image_info:  dict with keys: image, mask, is_anomaly, image_path
        device:      torch.device
        model:       AVADino (AVA-DINO adapter)
        dino_model:  frozen DINOv3 ViT-L/16

    Returns:
        ava_map:      [B, 2, H, W]  AACM map (softmax, class-dim=1)
        mask:                   [B, 1, H, W]  binarized GT mask
        anomaly_map_cross_modal:[B, 2, H, W]  cross-modal map
        global_anomaly_score:   [B, 2]        image-level logits
        routing_info:           dict with w_normal, w_anomaly, routing_loss
    """
    image      = image_info["image"].to(device)
    image_path = image_info["image_path"]
    mask       = image_info["mask"].to(device)
    mask[mask > 0.5]  = 1
    mask[mask <= 0.5] = 0
    y = image_info["is_anomaly"]

    B = len(image_path)

    # ── Text embeddings ──────────────────────────────────────────────────────
    text_feature          = torch.zeros(B, 768, 2).to(device)
    text_feature_normal   = torch.zeros(B, 768).to(device)
    text_feature_anomaly  = torch.zeros(B, 768).to(device)

    with torch.no_grad():
        for i in range(B):
            obj = image_path[i].split('/')[-4]
            tf  = encode_text_with_prompt_ensemble(clip_model, obj, device, '', y)
            text_feature[i]         = tf
            text_feature_normal[i]  = tf[:, 0]
            text_feature_anomaly[i] = tf[:, 1]

    # Prompt adapter on text features
    adj0 = torch.stack([model.prompt_adapter[0](text_feature[i, :, 0])[0] for i in range(B)])
    adj1 = torch.stack([model.prompt_adapter[1](text_feature[i, :, 1])[0] for i in range(B)])
    adjusted_text = torch.cat([adj0, adj1], dim=1).view(B, 768, 2)

    # Project to DINOv3 space for cross-modal matching
    adjusted_text_proj = model.text_projection(
        adjusted_text.permute(0, 2, 1)
    ).permute(0, 2, 1)  # [B, 1024, 2]

    # ── Visual features ──────────────────────────────────────────────────────
    cls_tokens, patch_tokens = extract_dinov3_features(image_path, image, device, dino_model)

    # ── Dual-adapter forward + anomaly maps ──────────────────────────────────
    ava_maps, cross_modal_maps, global_scores, routing_weights = [], [], [], []

    for i in range(4):
        cls_i   = cls_tokens[i].squeeze(1)    # [B, 1024]
        patch_i = patch_tokens[i]             # [B, N, 1024]

        w_n, w_a = model.compute_routing_weights(cls_i, text_feature_normal, text_feature_anomaly)
        routing_weights.append((w_n, w_a))

        # CLS token adaptation
        _, cls_norm = model.cls_token_adapter_normal[i](cls_i)
        _, cls_anom = model.cls_token_adapter_anomaly[i](cls_i)
        cls_feat = w_n.view(-1, 1) * cls_norm + w_a.view(-1, 1) * cls_anom + cls_i
        cls_feat = F.normalize(cls_feat, dim=-1)

        # Patch token adaptation
        _, p_norm = model.patch_token_adapter_normal[i](patch_i)
        _, p_anom = model.patch_token_adapter_anomaly[i](patch_i)
        patch_feat = w_n.view(-1, 1, 1) * p_norm + w_a.view(-1, 1, 1) * p_anom + patch_i
        patch_feat = F.normalize(patch_feat, dim=-1)

        # Cross-modal map [B, 2, H, W]
        cm = 100 * patch_feat @ adjusted_text_proj
        cm = F.interpolate(cm.permute(0, 2, 1).view(-1, 2, 32, 32), size=512,
                           mode='bilinear', align_corners=True)
        cross_modal_maps.append(torch.softmax(cm, dim=1))

        # Anomaly-Aware Calibration Map (AACM) [B, 2, H, W]
        aacm = 10 * patch_feat @ cls_feat.unsqueeze(-1)
        aacm = F.interpolate(aacm.permute(0, 2, 1).view(-1, 1, 32, 32), size=512,
                              mode='bilinear', align_corners=True)
        aacm = torch.sigmoid(aacm)
        ava_maps.append(torch.cat([1 - aacm, aacm], dim=1))

        # Image-level score [B, 2]
        global_scores.append((100 * (cls_feat.unsqueeze(1) @ adjusted_text_proj)).squeeze(1))

    anomaly_map_cross_modal = torch.mean(torch.stack(cross_modal_maps), dim=0)
    ava_map       = torch.mean(torch.stack(ava_maps),     dim=0)
    global_anomaly_score    = torch.mean(torch.stack(global_scores),    dim=0)

    avg_w_n = torch.mean(torch.stack([w[0] for w in routing_weights]), dim=0)
    avg_w_a = torch.mean(torch.stack([w[1] for w in routing_weights]), dim=0)
    routing_loss = model.compute_routing_loss(avg_w_n, avg_w_a, y.to(device))

    routing_info = {
        'w_normal':     avg_w_n.detach().cpu(),
        'w_anomaly':    avg_w_a.detach().cpu(),
        'routing_loss': routing_loss,
    }

    return ava_map, mask, anomaly_map_cross_modal, global_anomaly_score, routing_info
