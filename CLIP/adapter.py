import torch
from torch import nn
from torch.nn import functional as F


class BottleneckAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_in, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc2 = nn.Sequential(nn.Linear(bottleneck, c_in, bias=False), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        h = self.fc1(x)
        return h, self.fc2(h)


class AVADino(nn.Module):
    """
    AVA-DINO: Dual-branch adapter with CLIP text-guided dynamic routing.

    Architecture:
      - Normal branch adapters: adapt DINOv3 features toward normal appearance.
      - Anomaly branch adapters: adapt DINOv3 features toward anomalous appearance.
      - Text-guided routing: cosine similarity between CLS token and CLIP text
        embeddings determines per-sample branch mixing weights.
      - Routing regularization: MSE loss encourages branch specialization.
    """

    def __init__(self, c_in: int, device):
        super().__init__()
        self.device = device

        self.cls_token_adapter_normal   = nn.ModuleList([BottleneckAdapter(c_in=1024) for _ in range(4)])
        self.patch_token_adapter_normal = nn.ModuleList([BottleneckAdapter(c_in=1024) for _ in range(4)])
        self.cls_token_adapter_anomaly   = nn.ModuleList([BottleneckAdapter(c_in=1024) for _ in range(4)])
        self.patch_token_adapter_anomaly = nn.ModuleList([BottleneckAdapter(c_in=1024) for _ in range(4)])
        self.prompt_adapter = nn.ModuleList([BottleneckAdapter(c_in=768) for _ in range(2)])

        # Projects CLIP text space (768-d) into DINOv3 feature space (1024-d) for routing.
        self.text_projection = nn.Linear(768, 1024, bias=False)

        self.routing_temperature = 0.1

    def compute_routing_weights(self, features, text_normal, text_anomaly):
        """
        Compute per-sample routing weights via temperature-scaled cosine similarity.

        Args:
            features:      [B, 1024] DINOv3 CLS token
            text_normal:   [B, 768]  CLIP normal prompt embedding
            text_anomaly:  [B, 768]  CLIP anomaly prompt embedding

        Returns:
            w_normal, w_anomaly: [B] each, summing to 1.
        """
        t_n = F.normalize(self.text_projection(text_normal), dim=-1)
        t_a = F.normalize(self.text_projection(text_anomaly), dim=-1)
        f   = F.normalize(features, dim=-1)

        logits = torch.stack([(f * t_n).sum(-1), (f * t_a).sum(-1)], dim=1)
        w = F.softmax(logits / self.routing_temperature, dim=1)
        return w[:, 0], w[:, 1]

    def compute_routing_loss(self, w_normal, w_anomaly, is_anomaly):
        """
        Routing regularization: encourage branch specialization via MSE.

        Normal samples  → w_normal = 1, w_anomaly = 0
        Anomaly samples → w_normal = 0, w_anomaly = 1
        """
        is_anomaly = is_anomaly.float().to(w_normal.device)
        return (
            F.mse_loss(w_normal,  1.0 - is_anomaly) +
            F.mse_loss(w_anomaly, is_anomaly)
        )

    def _apply_dual_adapters(self, features_list, adapters_normal, adapters_anomaly,
                              w_normal, w_anomaly):
        """Apply parallel adapters and fuse with routing weights + residual."""
        out = []
        for i, feat in enumerate(features_list):
            _, f_n = adapters_normal[i](feat)
            _, f_a = adapters_anomaly[i](feat)
            if feat.dim() == 3:
                w_n, w_a = w_normal.view(-1, 1, 1), w_anomaly.view(-1, 1, 1)
            else:
                w_n, w_a = w_normal.view(-1, 1), w_anomaly.view(-1, 1)
            out.append(w_n * f_n + w_a * f_a + feat)
        return out

    def forward(self, patch_tokens_list, cls_token, text_normal, text_anomaly):
        """
        Args:
            patch_tokens_list: list of [B, N, 1024] from DINOv3 layers {5,11,17,23}
            cls_token:         [B, 1024]
            text_normal:       [B, 768]
            text_anomaly:      [B, 768]

        Returns:
            adapted_patches:   list of [B, N, 1024]
            adapted_cls:       [B, 1024]
            w_normal:          [B]
            w_anomaly:         [B]
        """
        w_normal, w_anomaly = self.compute_routing_weights(cls_token, text_normal, text_anomaly)

        adapted_patches = self._apply_dual_adapters(
            patch_tokens_list,
            self.patch_token_adapter_normal, self.patch_token_adapter_anomaly,
            w_normal, w_anomaly
        )

        _, cls_n = self.cls_token_adapter_normal[0](cls_token)
        _, cls_a = self.cls_token_adapter_anomaly[0](cls_token)
        adapted_cls = w_normal.view(-1, 1) * cls_n + w_anomaly.view(-1, 1) * cls_a + cls_token

        return adapted_patches, adapted_cls, w_normal, w_anomaly
