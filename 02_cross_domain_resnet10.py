# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Domain Few-Shot with ResNet-10 (Matching Guo et al. 2020)
#
# ## What this notebook does
#
# This is a **strict reproduction** of
# [Guo et al. (2020, ECCV)](https://doi.org/10.1007/978-3-030-58583-9_8)
# Table 1 — Prototypical Networks on EuroSAT — using the same **ResNet-10
# backbone** they used.
#
# Our [companion notebook](01_cross_domain_eurosat.py) used a simpler
# 4-block CNN backbone (~110K parameters) and got 67.4% vs Guo's 73.3%
# on 5-way 5-shot. The gap was likely due to the backbone difference.
# This notebook tests that hypothesis by switching to ResNet-10 (~5M
# parameters), matching their architecture exactly.
#
# If our numbers align with theirs, the reproduction is validated.
# If they still differ, there must be other factors (training recipe,
# data preprocessing, random seed).
#
# ## Published baseline (Guo et al. 2020, Table 1)
#
# | Method | 5-way 5-shot | 5-way 20-shot | 5-way 50-shot |
# |--------|-------------|---------------|---------------|
# | ProtoNet | 73.29% ± 0.71 | 82.27% ± 0.57 | 80.48% ± 0.57 |
# | MAML | 71.70% ± 0.72 | 81.95% ± 0.55 | — |
# | MatchingNet | 64.45% ± 0.63 | 77.10% ± 0.57 | 84.44% ± 0.47 |
#
# Our goal: reproduce the ProtoNet row.
#
# ## The data
#
# **Training domain** (mini-ImageNet): 60,000 photographs of 100 everyday
# object categories (animals, vehicles, household items), 84×84 pixels.
# The model learns visual features from these — but never sees any
# satellite imagery during training.
#
# **Test domain** (EuroSAT): 27,000 real Sentinel-2 satellite image patches
# (64×64 pixels, 10 m ground resolution), 10 land cover classes. The model
# must classify these using only a few labeled examples per class.

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from PIL import Image

# %% [markdown]
# ## Configuration

# %%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Few-shot settings (matching Guo et al.)
N_WAY = 5
K_SHOT = 5
N_QUERY = 16  # Guo: "n_query = max(1, int(16 * test_n_way / train_n_way))" = 16

# Training settings (matching Guo et al. exactly)
# From their code: stop_epoch=400, each epoch = 100 episodes = 40,000 total
# Image size = 224 (not 84!)
# Episodic training for ProtoNet (not supervised classification)
IMAGE_SIZE = 224
CI_MODE = os.environ.get("CI", "").lower() in ("true", "1")
N_TRAIN_EPISODES = 1000 if CI_MODE else 40000  # 400 epochs × 100 episodes
N_EPISODES = 100 if CI_MODE else 600           # evaluation episodes

if CI_MODE:
    print("CI mode: reduced episodes for faster execution")

# %% [markdown]
# ## 1. Load mini-ImageNet (training domain)
#
# mini-ImageNet contains 100 classes split into 64 train / 16 validation /
# 20 test. We train our embedding network on the 64 training classes.
# The model never sees EuroSAT during training.

# %%
# Transforms matching Guo et al. code exactly:
# Train (aug=True): RandomSizedCrop(224), ImageJitter(B=0.4,C=0.4,Color=0.4), RandomHorizontalFlip
# Eval (aug=False): Scale(1.15*224=257), CenterCrop(224)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_eval = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading mini-ImageNet from HuggingFace...")
mini_train_hf = load_dataset("GATE-engine/mini_imagenet", split="train")
print(f"  Raw train: {len(mini_train_hf)} images")

# Preload PIL images into memory (transforms applied on-the-fly)
print("  Preloading PIL images into memory...", flush=True)
mini_pil_images = []
mini_labels = []
for item in mini_train_hf:
    img = item["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    mini_pil_images.append(img)
    mini_labels.append(item["label"])

mini_labels = torch.tensor(mini_labels)
train_label_set = sorted(set(mini_labels.tolist()))
print(f"  Loaded: {len(mini_pil_images)} images, {len(train_label_set)} classes")

# Build class-to-indices mapping
mini_class_indices = defaultdict(list)
for idx, label in enumerate(mini_labels.tolist()):
    mini_class_indices[label].append(idx)

del mini_train_hf

# %% [markdown]
# ## 2. Load EuroSAT (test domain)
#
# The model will be evaluated on EuroSAT — satellite imagery it has
# never seen during training. This tests cross-domain transfer.

# %%
transform_eurosat = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eurosat = EuroSAT(root=str(DATA_DIR), download=True, transform=transform_eurosat)
print(f"EuroSAT: {len(eurosat)} images, {len(eurosat.classes)} classes")
print(f"Classes: {eurosat.classes}")

# Build class-to-indices mapping for EuroSAT
eurosat_class_indices = defaultdict(list)
for idx in range(len(eurosat)):
    _, label = eurosat[idx]
    eurosat_class_indices[label].append(idx)

# %% [markdown]
# ## 3. Prepare mini-ImageNet for episodic training

# %%
# %% [markdown]
# ## 4. Embedding network — ResNet-10
#
# Guo et al. (2020) used ResNet-10 as the backbone for all their
# experiments. ResNet-10 is a variant of ResNet with [1, 1, 1, 1]
# residual blocks — smaller than the standard ResNet-18 [2, 2, 2, 2]
# but much deeper than the 4-block CNN used in the original ProtoNet
# paper.
#
# We build ResNet-10 from torchvision's ResNet building blocks,
# remove the final classification layer, and use the 512-dimensional
# feature vector as the embedding for Prototypical Networks.

# %%
import math


def init_layer(L):
    """Weight initialization matching Guo et al. backbone.py."""
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class SimpleBlock(nn.Module):
    """Residual block matching Guo et al. backbone.py exactly."""

    def __init__(self, indim, outdim, half_res):
        super().__init__()
        self.C1 = nn.Conv2d(indim, outdim, 3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, 3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut_type = "identity"
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.shortcut_type = "projection"

        for L in [self.C1, self.C2, self.BN1, self.BN2]:
            init_layer(L)

    def forward(self, x):
        short_out = x if self.shortcut_type == "identity" else self.BNshortcut(self.shortcut(x))
        out = self.relu(self.BN1(self.C1(x)))
        out = self.BN2(self.C2(out))
        out = out + short_out
        return self.relu(out)


class ResNet10(nn.Module):
    """ResNet-10 matching Guo et al. backbone.py exactly.

    - SimpleBlock with [1,1,1,1] layers
    - Channel dims [64, 128, 256, 512]
    - AvgPool2d(7) for 224×224 input
    - Custom weight initialization
    """

    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1)]

        indim = 64
        for i, outdim in enumerate([64, 128, 256, 512]):
            half_res = (i >= 1)
            trunk.append(SimpleBlock(indim, outdim, half_res))
            indim = outdim

        trunk.append(nn.AvgPool2d(7))
        trunk.append(nn.Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 512

    def forward(self, x):
        return self.trunk(x)


model = ResNet10().to(DEVICE)
embed_dim = model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)).shape[1]
n_params = sum(p.numel() for p in model.parameters())
print(f"Backbone: ResNet-10 (matching Guo et al. exactly)")
print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"Embedding dimension: {embed_dim}")
print(f"Parameters: {n_params:,}")

# %% [markdown]
# ## 5. Episode sampling

# %%
def sample_episode_from_pil(pil_images, class_indices, classes, n_way, k_shot, n_query, transform):
    """Sample a few-shot episode from PIL images with on-the-fly transform."""
    selected = random.sample(classes, n_way)
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for i, cls in enumerate(selected):
        indices = random.sample(class_indices[cls], k_shot + n_query)
        for idx in indices[:k_shot]:
            support_images.append(transform(pil_images[idx]))
            support_labels.append(i)
        for idx in indices[k_shot:]:
            query_images.append(transform(pil_images[idx]))
            query_labels.append(i)

    return (torch.stack(support_images), torch.tensor(support_labels),
            torch.stack(query_images), torch.tensor(query_labels))


def sample_episode_from_eurosat(class_indices, classes, n_way, k_shot, n_query, dataset):
    """Sample a few-shot episode from EuroSAT."""
    selected = random.sample(classes, n_way)
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for i, cls in enumerate(selected):
        indices = random.sample(class_indices[cls], k_shot + n_query)
        for idx in indices[:k_shot]:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(i)
        for idx in indices[k_shot:]:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(i)

    return (torch.stack(support_images), torch.tensor(support_labels),
            torch.stack(query_images), torch.tensor(query_labels))

# %% [markdown]
# ## 6. Train on mini-ImageNet (episodic, matching Guo et al.)
#
# From Guo et al. code: for the ProtoNet row in Table 1, they used
# **episodic meta-training** (not supervised classification).
# Their `stop_epoch=400` with 100 episodes per epoch = **40,000 episodes**.
#
# Each episode: sample 5 classes, 5 support + 16 query images per class,
# compute prototypes, classify queries, update the ResNet-10 backbone.
#
# Data augmentation: `RandomSizedCrop(224)`, `ImageJitter`, `RandomHorizontalFlip`.

# %%
optimizer = torch.optim.Adam(model.parameters())

print(f"Episodic training on mini-ImageNet ({len(train_label_set)} classes, "
      f"{N_TRAIN_EPISODES} episodes, {IMAGE_SIZE}×{IMAGE_SIZE} images)...")
print(f"Data augmentation: RandomResizedCrop, ColorJitter, RandomHorizontalFlip", flush=True)
model.train()
losses = []

for ep in range(N_TRAIN_EPISODES):
    s_img, s_lbl, q_img, q_lbl = sample_episode_from_pil(
        mini_pil_images, mini_class_indices, train_label_set,
        N_WAY, K_SHOT, N_QUERY, transform_train
    )
    s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
    q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)

    s_emb = model(s_img)
    q_emb = model(q_img)

    prototypes = torch.stack([s_emb[s_lbl == c].mean(0) for c in range(N_WAY)])
    dists = torch.pow(q_emb.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
    scores = -dists
    loss = F.cross_entropy(scores, q_lbl)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (ep + 1) % 5000 == 0 or ep == 0:
        avg = np.mean(losses[-5000:]) if ep > 0 else losses[0]
        acc = (scores.argmax(1) == q_lbl).float().mean().item()
        print(f"  Episode {ep+1}/{N_TRAIN_EPISODES}: loss={avg:.3f}, acc={acc:.1%}", flush=True)

print("Training complete (on everyday photos — no satellite data seen).", flush=True)

# %% [markdown]
# ## 7. Evaluate on EuroSAT (cross-domain transfer)
#
# Now the critical test: the model trained **only on everyday photographs**
# must classify **Sentinel-2 satellite land cover types** it has never seen.
# We give it just K labeled satellite images per class and ask it to
# classify new ones.
#
# This directly replicates Guo et al. (2020) Table 1 — ProtoNet row.

# %%
model.eval()
ALL_EUROSAT = list(range(10))

shot_results = {}
for k in [5, 20, 50]:
    accs = []
    with torch.no_grad():
        for ep in range(N_EPISODES):
            s_img, s_lbl, q_img, q_lbl = sample_episode_from_eurosat(
                eurosat_class_indices, ALL_EUROSAT, N_WAY, k, N_QUERY, eurosat
            )
            s_img, q_img = s_img.to(DEVICE), q_img.to(DEVICE)
            q_lbl = q_lbl.to(DEVICE)

            s_emb = model(s_img)
            q_emb = model(q_img)

            prototypes = torch.stack([s_emb[s_lbl.to(DEVICE) == c].mean(0)
                                      for c in range(N_WAY)])
            dists = torch.cdist(q_emb, prototypes)
            preds = (-dists).argmax(1)
            acc = (preds == q_lbl).float().mean().item()
            accs.append(acc)

    mean = np.mean(accs)
    ci = 1.96 * np.std(accs) / np.sqrt(N_EPISODES)
    shot_results[k] = (mean, ci)
    print(f"  {N_WAY}-way {k}-shot: {mean:.1%} +/- {ci:.1%}")

# %% [markdown]
# ## 8. Compare with published baselines
#
# Guo et al. (2020) reported these results for ProtoNet on EuroSAT
# (trained on mini-ImageNet, ResNet-10 backbone, 600 episodes):
#
# | Setting | Guo et al. (2020) | Our reproduction |
# |---------|-------------------|------------------|

# %%
guo_baselines = {
    5:  (0.7329, 0.0071),
    20: (0.8227, 0.0057),
    50: (0.8048, 0.0057),
}

print("Comparison with Guo et al. (2020) Table 1 — ProtoNet on EuroSAT:")
print(f"{'Setting':<15} {'Guo et al.':<20} {'Ours':<20} {'Match?'}")
print("-" * 70)
for k in [5, 20, 50]:
    guo_mean, guo_ci = guo_baselines[k]
    our_mean, our_ci = shot_results[k]
    # Check if within overlapping confidence intervals
    overlap = abs(our_mean - guo_mean) < (our_ci + guo_ci)
    status = "WITHIN CI" if overlap else "DIFFERS"
    print(f"{N_WAY}-way {k}-shot   "
          f"{guo_mean:.1%} +/- {guo_ci:.1%}    "
          f"{our_mean:.1%} +/- {our_ci:.1%}    "
          f"{status}")

# %% [markdown]
# ## 9. Results and interpretation

# %%
import json

results = {
    "method": "Prototypical Networks (Snell et al. 2017)",
    "training_domain": "mini-ImageNet (64 classes, everyday photos)",
    "test_domain": "EuroSAT (10 classes, Sentinel-2 satellite imagery)",
    "replicates": "Guo et al. 2020, ECCV, Table 1, ProtoNet row",
    "replicates_doi": "10.1007/978-3-030-58583-9_8",
    "device": DEVICE,
    "n_train_episodes": N_TRAIN_EPISODES,
    "n_eval_episodes": N_EPISODES,
    "our_results": {
        f"{N_WAY}way_{k}shot": {"accuracy": f"{m:.4f}", "ci95": f"{c:.4f}"}
        for k, (m, c) in shot_results.items()
    },
    "guo_baselines": {
        f"{N_WAY}way_{k}shot": {"accuracy": f"{m:.4f}", "ci95": f"{c:.4f}"}
        for k, (m, c) in guo_baselines.items()
    },
}

with open(RESULTS_DIR / "cross_domain_resnet10_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'cross_domain_resnet10_results.json'}")

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: our results vs Guo baselines
    ax = axes[0]
    shots = sorted(shot_results.keys())
    our_means = [shot_results[k][0] for k in shots]
    our_cis = [shot_results[k][1] for k in shots]
    guo_means = [guo_baselines[k][0] for k in shots]
    guo_cis = [guo_baselines[k][1] for k in shots]

    x = np.arange(len(shots))
    w = 0.35
    ax.bar(x - w/2, guo_means, w, yerr=guo_cis, label="Guo et al. 2020",
           color="lightcoral", capsize=5)
    ax.bar(x + w/2, our_means, w, yerr=our_cis, label="Our reproduction",
           color="steelblue", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-shot" for k in shots])
    ax.set_ylabel("Accuracy")
    ax.set_title("ProtoNet on EuroSAT (trained on mini-ImageNet)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: training loss
    ax = axes[1]
    window = 100
    smoothed = [np.mean(losses[max(0, i-window):i+1]) for i in range(len(losses))]
    ax.plot(smoothed, linewidth=1, color="steelblue")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training on mini-ImageNet (everyday photos)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Domain Few-Shot: Everyday Photos → Sentinel-2 Satellite",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cross_domain_resnet10.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {RESULTS_DIR / 'cross_domain_resnet10.png'}")
except ImportError:
    print("matplotlib not available")

torch.save(model.state_dict(), RESULTS_DIR / "resnet10_mini_to_eurosat.pth")
print(f"Model saved: {RESULTS_DIR / 'resnet10_mini_to_eurosat.pth'}")

# %% [markdown]
# ## 10. What does this mean?
#
# **The big picture**: a model that has only ever seen photos of dogs,
# cars, and household objects can classify satellite land cover types
# with reasonable accuracy (~73%) from just 5 labeled satellite images.
# This is remarkable — and practically useful, because it means
# organizations with limited satellite training data can bootstrap
# classifiers from abundant non-satellite image datasets.
#
# **The limitation**: performance is lower than training on satellite
# data directly (82% in our within-domain experiment vs. ~73% here).
# The domain gap — perspective, spectral content, semantics — is real.
# For operational habitat monitoring, some domain-specific training
# data is still needed.
#
# ## Replication context
#
# This is a strict reproduction of [Guo et al. (2020, ECCV)](https://doi.org/10.1007/978-3-030-58583-9_8)
# Table 1 — ProtoNet row on EuroSAT. Part of the
# [Science Live](https://platform.sciencelive4all.org) FORRT replication
# initiative.
#
# - **Zenodo DOI**: (to be created on release)
# - **Within-domain companion**: [few-shot-eurosat-within-domain](https://github.com/annefou/few-shot-eurosat-within-domain)
