"""
Notebook 3 runner for Colab — Mechanistic Ablation + Restoration Analysis.
Same experiments as run_notebook3.py but packaged for a headless Colab GPU session:
  - /content working dir, headless matplotlib
  - artifacts written to /content/output and mirrored back to colab_results/notebook3/
Upload this file to Colab and run: %run run_notebook3_colab.py
"""

import sys, os, subprocess
os.makedirs("/content/output", exist_ok=True)
os.chdir("/content")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("Notebook 3: Mechanistic Ablation + Restoration Analysis")
print("=" * 60)

# Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn", "scikit-image", "matplotlib", "scipy"], check=False)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import csv, json

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ========== Cell 1: Data (same seed/subsets as notebooks 1 and 2) ==========
N_IMAGES = 1000
N_TRAIN = 5000

raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
rng = np.random.default_rng(42)
idx = rng.choice(len(raw_ds), size=N_IMAGES, replace=False)
images, labels = [], []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))
    labels.append(label)
labels = np.array(labels)

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
train_idx = rng.choice(len(train_ds), size=N_TRAIN, replace=False)
train_images, train_labels = [], []
for i in train_idx:
    img, label = train_ds[i]
    train_images.append(np.array(img))
    train_labels.append(label)
train_labels = np.array(train_labels)
print(f"Test subset: {len(images)}  Train subset: {len(train_images)}")

# ========== Cell 2: Corruption functions ==========
def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    img = image.astype(np.float32) * brightness_factors[severity]
    if noise_std[severity] > 0:
        img = img + np.random.normal(0, noise_std[severity], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_low_light_to_stage1(image, severity):
    """Stage 1 only: photometric loss WITHOUT the uint8 clip (float32, [0,255])."""
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    img = image.astype(np.float32) * brightness_factors[severity]
    if noise_std[severity] > 0:
        img = img + np.random.normal(0, noise_std[severity], img.shape)
    return np.clip(img, 0, 255)

def two_stage_low_light(image, severity):
    """Stage 1 + stage 2 (uint8 quantization) — the standard pipeline."""
    return apply_low_light_to_stage1(image, severity).astype(np.uint8)

# ========== Cell 3: Classical mitigation baselines ==========
import skimage.color
import skimage.exposure as skexposure

def simple_gain(image, severity):
    factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    img = np.asarray(image).astype(np.float32) / factors[severity]
    return np.clip(img, 0, 255).astype(np.uint8)

def gamma_correct(image, severity, base=0.35):
    exponent = 1.0 + base * max(severity, 1)
    x = np.asarray(image).astype(np.float32) / 255.0
    out = np.power(x, 1.0 / exponent) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_enhance(image, severity=None):
    img = np.asarray(image).astype(np.float32) / 255.0
    lab = skimage.color.rgb2lab(img)
    lab[:, :, 0] = skexposure.equalize_adapthist(lab[:, :, 0] / 100.0, clip_limit=0.03) * 100.0
    return (np.clip(skimage.color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)

# ========== Cell 4: Model + extraction ==========
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_embeddings(image_list, batch_size=64, float_input=False):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        if float_input:
            tensors = torch.stack([
                T.functional.normalize(
                    T.functional.resize(
                        torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0,
                        (224, 224), antialias=True),
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for img in batch
            ]).to(device)
        else:
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = dinov2(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

print("DINOv2 ViT-S/14 loaded.")

# ========== Cell 5: Q1 — mechanism (fixed vs retrained probe) ==========
def fit_probe(X, y):
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X, y)
    return probe

embeddings_by_severity = {}
for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    embeddings_by_severity[severity] = get_embeddings(degraded)
    print(f"severity {severity}: extracted")

X_clean = embeddings_by_severity[0]
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

fixed_probe = fit_probe(X_train, y_train)

def logit_margins(probe, X, y):
    scores = probe.decision_function(X)
    srt = np.sort(scores, axis=1)
    margins = srt[:, -1] - srt[:, -2]
    preds = probe.predict(X)
    correct = (preds == y).astype(float)
    return correct.mean(), margins.mean(), margins[preds == y].mean()

mechanism_rows = []
for severity in range(6):
    acc_fixed, margin_all, margin_correct = logit_margins(
        fixed_probe, embeddings_by_severity[severity][idx_test], y_test)
    probe_s = fit_probe(embeddings_by_severity[severity][idx_train], y_train)
    acc_adapted = accuracy_score(y_test, probe_s.predict(embeddings_by_severity[severity][idx_test]))
    mechanism_rows.append({
        "severity": severity,
        "acc_fixed_probe": acc_fixed,
        "acc_probe_retrained_on_severity": acc_adapted,
        "mean_logit_margin": margin_all,
        "mean_logit_margin_correct": margin_correct,
    })
    print(mechanism_rows[-1])

# ========== Cell 6: Q2 — participation ratio (geometry) ==========
def participation_ratio(X):
    S = np.cov(X.T)
    eig = np.linalg.eigvalsh(S)[::-1]
    return float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))

pr_clean = participation_ratio(embeddings_by_severity[0])
pr_by_severity = []
for severity in range(6):
    pr_dark = participation_ratio(embeddings_by_severity[severity])
    pr_by_severity.append({"severity": severity, "pr_clean": pr_clean,
                           "pr_dark": pr_dark, "ratio": pr_dark / pr_clean})
    print(pr_by_severity[-1])

# ========== Cell 7: Q3 — quantization vs contrast ==========
quant_rows = []
for severity in range(6):
    float_imgs = [apply_low_light_to_stage1(img, severity) for img in images]
    uint8_imgs = [two_stage_low_light(img, severity) for img in images]
    E_float = get_embeddings(float_imgs, float_input=True)
    E_uint8 = get_embeddings(uint8_imgs)
    acc_float = accuracy_score(y_test, fixed_probe.predict(E_float[idx_test]))
    acc_uint8 = accuracy_score(y_test, fixed_probe.predict(E_uint8[idx_test]))
    quant_rows.append({"severity": severity, "acc_float_stage1": acc_float, "acc_uint8_full": acc_uint8})
    print(quant_rows[-1])

# ========== Cell 8: Q4 — probe vs nearest-prototype (zero-training head) ==========
protos = np.stack([X_train[y_train == c].mean(0) for c in range(10)])
proto_norms = np.linalg.norm(protos, axis=1)

def proto_accuracy(E):
    sims = (E @ protos.T) / (np.linalg.norm(E, axis=1, keepdims=True) * proto_norms + 1e-8)
    return accuracy_score(y_test, sims.argmax(1))

zero_shot_rows = []
for severity in range(6):
    E = embeddings_by_severity[severity]
    zero_shot_rows.append({
        "severity": severity,
        "acc_probe": accuracy_score(y_test, fixed_probe.predict(E[idx_test])),
        "acc_nearest_prototype": proto_accuracy(E[idx_test]),
    })
    print(zero_shot_rows[-1])

# ========== Cell 9: Q5 — LoRA ablation (rank x target) ==========
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
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

def build_model(rank, alpha, target):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().to(device)
    for block in model.blocks:
        if target in ("attn", "attn_mlp"):
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

class AugmentedCIFAR10(Dataset):
    def __init__(self, imgs, lbls, transform):
        self.images, self.labels, self.transform = imgs, lbls, transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        img = self.images[i].copy()
        if np.random.random() < 0.7:
            img = low_light(img, np.random.randint(2, 5))
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        return self.transform(img), self.labels[i]

class Classifier(nn.Module):
    def __init__(self, backbone, n_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)
    def forward(self, x):
        out = self.backbone(x)
        return self.head(out[:, 0, :] if out.dim() == 3 else out)

def train_lora(rank, target, epochs=6):
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_model(rank, alpha=2 * rank, target=target)
    clf = Classifier(model).to(device)
    lora_params = [p for n, p in clf.named_parameters()
                   if p.requires_grad and not n.startswith("head.")]
    head_params = list(clf.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 5e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
    ])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(AugmentedCIFAR10(train_images, train_labels, preprocess),
                        batch_size=64, shuffle=True, num_workers=2)
    clf.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for bi_img, bi_lbl in loader:
            bi_img, bi_lbl = bi_img.to(device), bi_lbl.to(device)
            optimizer.zero_grad()
            out = clf(bi_img)
            loss = criterion(out, bi_lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bi_img.size(0)
            correct += out.argmax(1).eq(bi_lbl).sum().item()
            total += bi_img.size(0)
        print(f"  [r={rank} {target}] epoch {epoch+1}/{epochs}: "
              f"loss={running_loss/total:.4f} acc={correct/total:.4f}")
    clf.eval()
    return clf

@torch.no_grad()
def extract_backbone(clf, image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = clf.backbone(tensors)
        feats.append((out[:, 0, :] if out.dim() == 3 else out).cpu())
    return torch.cat(feats, dim=0).numpy()

ABLATIONS = [
    (4, "attn"),
    (8, "attn"),
    (16, "attn"),
    (8, "attn_mlp"),
    (16, "attn_mlp"),
]

ablation_rows = []
for rank, target in ABLATIONS:
    print(f"\n=== LoRA r={rank} target={target} ===")
    clf = train_lora(rank, target, epochs=6)
    pooled = {s: extract_backbone(clf, [low_light(im, s) for im in images]) for s in range(6)}
    Xc = pooled[0]
    Xtr, Xte, ytr, yte, _, ite = train_test_split(Xc, labels, np.arange(len(labels)),
                                                  test_size=0.3, random_state=42, stratify=labels)
    probe = fit_probe(Xtr, ytr)
    accs = [accuracy_score(yte, probe.predict(pooled[s][ite])) for s in range(6)]
    n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    ablation_rows.append({
        "rank": rank, "target": target, "trainable_params": int(n_trainable),
        "acc_sev0": accs[0], "acc_sev3": accs[3], "acc_sev5": accs[5],
        "mean_acc": float(np.mean(accs)),
    })
    print(ablation_rows[-1])
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ========== Cell 10: Q6 — classical mitigation vs no mitigation ==========
baseline_accs = [accuracy_score(y_test, fixed_probe.predict(embeddings_by_severity[s][idx_test]))
                 for s in range(6)]

mitigation_rows = []
for severity in range(6):
    dark = [low_light(img, severity) for img in images]
    row = {"severity": severity, "no_mitigation": baseline_accs[severity]}
    for name, fn in [("gain", simple_gain), ("gamma", gamma_correct), ("clahe", clahe_enhance)]:
        if name == "gain":
            restored = [simple_gain(img, severity) for img in dark]
        elif name == "gamma":
            restored = [gamma_correct(img, severity) for img in dark]
        else:
            restored = [clahe_enhance(img) for img in dark]
        E = get_embeddings(restored)
        row[f"acc_{name}"] = accuracy_score(y_test, fixed_probe.predict(E[idx_test]))
    mitigation_rows.append(row)
    print(row)

# ========== Cell 11: Save + plots ==========
with open("output/notebook3_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "acc_fixed_probe", "acc_probe_retrained_on_severity",
                                      "mean_logit_margin", "mean_logit_margin_correct"])
    w.writeheader()
    for r in mechanism_rows:
        w.writerow(r)

with open("output/ablation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rank", "target", "trainable_params",
                                      "acc_sev0", "acc_sev3", "acc_sev5", "mean_acc"])
    w.writeheader()
    for r in ablation_rows:
        w.writerow(r)

with open("output/mitigation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "no_mitigation", "acc_gain", "acc_gamma", "acc_clahe"])
    w.writeheader()
    for r in mitigation_rows:
        w.writerow(r)

with open("output/quantization_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "acc_float_stage1", "acc_uint8_full"])
    w.writeheader()
    for r in quant_rows:
        w.writerow(r)

with open("output/geometry_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "pr_clean", "pr_dark", "ratio"])
    w.writeheader()
    for r in pr_by_severity:
        w.writerow(r)

with open("output/zero_shot_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "acc_probe", "acc_nearest_prototype"])
    w.writeheader()
    for r in zero_shot_rows:
        w.writerow(r)

# Plot: mechanism
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r["severity"] for r in mechanism_rows],
        [r["acc_fixed_probe"] for r in mechanism_rows], marker="o",
        label="Fixed probe (trained on clean)")
ax.plot([r["severity"] for r in mechanism_rows],
        [r["acc_probe_retrained_on_severity"] for r in mechanism_rows], marker="s",
        label="Probe retrained on same severity")
ax.set_xlabel("Low-light severity"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("Q1: is the failure in the representation or the probe?")
ax.legend(); plt.tight_layout(); plt.savefig("output/mechanism.png", dpi=150); plt.close()

# Plot: geometry
plt.figure(figsize=(8, 5))
plt.plot(range(6), [r["ratio"] for r in pr_by_severity], marker="o", color="tab:purple")
plt.xlabel("Low-light severity"); plt.ylabel("Participation ratio (dark / clean)")
plt.title("Q2: embedding effective dimensionality under low light")
plt.tight_layout(); plt.savefig("output/geometry.png", dpi=150); plt.close()

# Plot: quantization gap
plt.figure(figsize=(8, 5))
plt.plot([r["severity"] for r in quant_rows], [r["acc_float_stage1"] for r in quant_rows],
         marker="o", label="Stage-1 only (float, recoverable)")
plt.plot([r["severity"] for r in quant_rows], [r["acc_uint8_full"] for r in quant_rows],
         marker="s", label="Full two-stage (uint8 clip, irreversible)")
plt.xlabel("Low-light severity"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
plt.title("Q3: how much damage does uint8 quantization cause?")
plt.legend(); plt.tight_layout(); plt.savefig("output/quantization.png", dpi=150); plt.close()

# Plot: ablation Pareto
plt.figure(figsize=(8, 5))
for (rank, target) in ABLATIONS:
    row = next(r for r in ablation_rows if r["rank"] == rank and r["target"] == target)
    plt.scatter(row["trainable_params"], row["mean_acc"], s=90)
    plt.annotate(f"r={rank}\n{target}", (row["trainable_params"], row["mean_acc"]),
                 fontsize=8, textcoords="offset points", xytext=(6, 4))
plt.xscale("log"); plt.xlabel("Trainable parameters (log)")
plt.ylabel("Mean accuracy over severities 0-5")
plt.title("Q5: LoRA rank x target Pareto front")
plt.tight_layout(); plt.savefig("output/ablation_pareto.png", dpi=150); plt.close()

# Plot: mitigation
plt.figure(figsize=(8, 5))
plt.plot(range(6), [r["no_mitigation"] for r in mitigation_rows], marker="o", label="No mitigation")
plt.plot(range(6), [r["acc_gain"] for r in mitigation_rows], marker="s", label="Gain x(1/a)")
plt.plot(range(6), [r["acc_gamma"] for r in mitigation_rows], marker="^", label="Gamma correction")
plt.plot(range(6), [r["acc_clahe"] for r in mitigation_rows], marker="d", label="CLAHE")
plt.xlabel("Low-light severity"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
plt.title("Q6: zero-training classical mitigation")
plt.legend(); plt.tight_layout(); plt.savefig("output/mitigation.png", dpi=150); plt.close()

print("\n" + "=" * 60)
print("NOTEBOOK 3 COMPLETE - SUMMARY")
print("=" * 60)

print("\n--- Q1 Mechanism (fixed vs retrained probe) ---")
for r in mechanism_rows:
    print(f"  sev {r['severity']}: fixed={r['acc_fixed_probe']:.4f}  "
          f"retrained={r['acc_probe_retrained_on_severity']:.4f}  "
          f"margin={r['mean_logit_margin']:.3f}")

print("\n--- Q2 Geometry (participation ratio, dark/clean) ---")
for r in pr_by_severity:
    print(f"  sev {r['severity']}: PR={r['pr_dark']:.2f}  ratio={r['ratio']:.3f}")

print("\n--- Q3 Quantization gap (float stage-1 vs uint8 full) ---")
for r in quant_rows:
    print(f"  sev {r['severity']}: float={r['acc_float_stage1']:.4f}  "
          f"uint8={r['acc_uint8_full']:.4f}  gap={r['acc_float_stage1']-r['acc_uint8_full']:+.4f}")

print("\n--- Q4 Probe vs nearest-prototype ---")
for r in zero_shot_rows:
    print(f"  sev {r['severity']}: probe={r['acc_probe']:.4f}  prototype={r['acc_nearest_prototype']:.4f}")

print("\n--- Q5 LoRA ablation ---")
for r in ablation_rows:
    print(f"  r={r['rank']:<3} {r['target']:<9} params={r['trainable_params']:>9,}  "
          f"sev0={r['acc_sev0']:.4f} sev3={r['acc_sev3']:.4f} sev5={r['acc_sev5']:.4f}  "
          f"mean={r['mean_acc']:.4f}")

print("\n--- Q6 Classical mitigation ---")
print(f"  {'sev':<4}{'none':>8}{'gain':>8}{'gamma':>8}{'clahe':>8}")
for r in mitigation_rows:
    print(f"  {r['severity']:<4}{r['no_mitigation']:>8.4f}{r['acc_gain']:>8.4f}"
          f"{r['acc_gamma']:>8.4f}{r['acc_clahe']:>8.4f}")

print("\nArtifacts written to /content/output/:")
for name in ["notebook3_results.csv", "ablation_results.csv", "mitigation_results.csv",
             "quantization_results.csv", "geometry_results.csv", "zero_shot_results.csv",
             "mechanism.png", "geometry.png", "quantization.png",
             "ablation_pareto.png", "mitigation.png"]:
    print(" -", name)
print("=" * 60)
