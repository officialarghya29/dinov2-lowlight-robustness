"""LoRA for DINOv2 attention (and optionally MLP) projections.

Implementation notes that matter for correctness:
- B is initialized to ZERO (Hu et al. 2021), so the adapted model is exactly
  the pretrained model at step 0. The original repo script used randn for both
  factors, which injects a random feature perturbation before training starts.
- Optimizer groups are constructed disjointly (the head must not appear in both
  groups — the original script's duplication made AdamW raise).
- Seeds are set inside train_lora so different (rank, target) configs are
  comparable run-to-run.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:  # allows analysis-only environments to import the module
    TORCH_AVAILABLE = False

from src.corruptions import low_light

if not TORCH_AVAILABLE:
    raise ImportError(
        "src.lora requires torch; install requirements.txt to run LoRA experiments")


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model: nn.Module, rank: int, alpha: int, target: str = "attn"):
    """target: 'attn' -> qkv+proj; 'attn_mlp' -> qkv+proj+fc1+fc2."""
    for block in model.blocks:
        block.attn.qkv = LoRALayer(block.attn.qkv, r=rank, alpha=alpha)
        block.attn.proj = LoRALayer(block.attn.proj, r=rank, alpha=alpha)
        if target == "attn_mlp":
            block.mlp.fc1 = LoRALayer(block.mlp.fc1, r=rank, alpha=alpha)
            block.mlp.fc2 = LoRALayer(block.mlp.fc2, r=rank, alpha=alpha)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    return model


class LowLightAugCIFAR(Dataset):
    """70% of samples randomly degraded to severity 2-4, 50% horizontal flip."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].copy()
        if np.random.random() < 0.7:
            img = low_light(img, np.random.randint(2, 5))
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        return self.transform(img), self.labels[i]


class BackboneClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        if out.dim() == 3:
            out = out[:, 0, :]
        return self.head(out)


def train_lora(pretrained_model, train_images, train_labels, preprocess, cfg: dict,
               rank: int, target: str, device: str, epochs: int = None, verbose: bool = True):
    """Train a fresh LoRA adapter set + head. Returns the eval-mode classifier."""
    import time

    rank = int(rank)
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    model = pretrained_model  # reused; apply_lora mutates in place
    apply_lora(model, rank, alpha=cfg.get("alpha_multiplier", 2) * rank, target=target)
    clf = BackboneClassifier(model).to(device)

    head_params = list(clf.head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for n, p in clf.named_parameters()
                   if p.requires_grad and id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.get("lr_lora", 5e-5), "weight_decay": cfg.get("weight_decay", 0.01)},
        {"params": head_params, "lr": cfg.get("lr_head", 1e-3), "weight_decay": cfg.get("weight_decay", 0.01)},
    ])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(LowLightAugCIFAR(train_images, train_labels, preprocess),
                        batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=2)

    epochs = epochs or cfg.get("epochs", 10)
    clf.train()
    t0 = time.time()
    for epoch in range(epochs):
        running, correct, total = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = clf(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
        if verbose:
            print(f"  [r={rank} {target}] epoch {epoch+1}/{epochs}: "
                  f"loss={running/total:.4f} acc={correct/total:.4f} ({time.time()-t0:.0f}s)")
    clf.eval()
    return clf


@torch.no_grad()
def extract_backbone(clf, image_list, preprocess, device, batch_size: int = 64):
    import torch as _torch

    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = clf.backbone(tensors)
        if out.dim() == 3:
            out = out[:, 0, :]
        feats.append(out.float().cpu())
    return _torch.cat(feats, dim=0).numpy()
