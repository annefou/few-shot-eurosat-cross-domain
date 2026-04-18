---
title: Few-Shot EuroSAT Cross-Domain
subtitle: Can everyday photos teach AI to classify satellite imagery?
---

## The question

You have Sentinel-2 satellite imagery of a region and need to classify land cover types — but you only have a handful of labeled examples. Deep learning models typically need thousands of labeled images to train. What if you could use the millions of labeled everyday photographs (dogs, cars, furniture) that already exist?

This project tests exactly that: train an AI model on everyday photographs, then ask it to classify satellite imagery it has never seen, using only 5 labeled satellite images per class.

## What we tested

We ran three experiments, each using [Prototypical Networks](https://arxiv.org/abs/1703.05175) — an AI method that classifies new categories by comparing images to a few labeled examples:

| Experiment | Training data | Backbone | 5-shot accuracy |
|------------|--------------|----------|----------------|
| Simple backbone | 5,000 episodes on everyday photos (84×84 px) | 4-block CNN | 67.4% |
| Matching published setup | 40,000 episodes on everyday photos (224×224 px) | ResNet-10 | 75.0% |
| Supervised pretraining | 10 epochs classification on everyday photos (224×224 px) | ResNet-10 | 76.2% |

For comparison, random guessing on a 5-way task gives 20%.

## What we learned

### 1. Cross-domain transfer works — but with limits

A model trained only on photographs of dogs, cars, and household objects can classify satellite land cover types at 67–76% accuracy from just 5 labeled satellite images. This is well above random (20%) and shows that low-level visual features (edges, textures, colour patterns) transfer across domains.

However, 76% accuracy on broad land cover categories (forest vs. highway vs. sea) is not sufficient for most operational EO applications. Real habitat classification — distinguishing two types of grassland, or identifying specific Natura 2000 habitat types — requires finer discrimination that RGB features from everyday photos cannot provide.

### 2. Architecture and image resolution matter more than training method

The biggest accuracy improvement came from switching to a deeper backbone (ResNet-10) and higher resolution images (224×224 instead of 84×84) — not from the training algorithm. This means EO researchers don't need to learn complex meta-learning techniques. A standard pretrained model (available in one line of PyTorch code) is a good starting point.

### 3. Supervised pretraining is as effective as meta-learning

Standard supervised training (classifying everyday objects for 10 epochs, ~15 minutes) achieves the same cross-domain accuracy as episodic meta-learning (40,000 episodes, ~3 hours). This confirms the original paper's finding and has a practical implication: **you don't need specialised meta-learning tools — any off-the-shelf ImageNet-pretrained model works**.

### 4. The domain gap is real

Compared to our [within-domain experiment](https://github.com/annefou/few-shot-eurosat-within-domain) where we trained on common satellite classes and tested on rare ones (82% accuracy), cross-domain transfer loses about 6–15 percentage points. The model has never seen overhead perspective, land cover semantics, or the spectral characteristics of satellite imagery during training — and that gap shows.

## What we recommend for EO researchers

Based on these experiments, here is our practical advice:

**If you have some labeled satellite data** (even for other classes in your region): use the [within-domain approach](https://github.com/annefou/few-shot-eurosat-within-domain). Train on your common classes, then classify rare classes with few examples. This gives you 82% accuracy and the model already understands satellite imagery.

**If you have no labeled satellite data at all**: use an off-the-shelf ImageNet-pretrained model (e.g. `torchvision.models.resnet18(pretrained=True)`) as a feature extractor. Feed your few labeled satellite patches through it, compute class prototypes, classify by nearest prototype. This gives you ~76% as a starting point — enough for initial screening, not for final results. Then invest in labeling a small satellite training set to move to the within-domain approach.

**What won't work well**: trying to distinguish spectrally similar classes (two vegetation types, or crop stages) using only RGB bands and ImageNet features. For this you need Sentinel-2's near-infrared and shortwave infrared bands, which require domain-specific models.

## Results compared

| Approach | Training data | 5-shot | 20-shot | 50-shot |
|----------|--------------|--------|---------|---------|
| **Cross-domain** (this project) | Everyday photos | 76.2% | 79.5% | 81.5% |
| **Within-domain** ([companion](https://github.com/annefou/few-shot-eurosat-within-domain)) | Satellite (common classes) | 82.1% | — | — |
| **Random baseline** | — | 20.0% | 20.0% | 20.0% |

The within-domain approach is consistently better. If you can collect even a small amount of labeled satellite data for your region, it's worth it.

## FORRT nanopublications

This work is published as three chains of [nanopublications](https://nanopub.net/) on [Science Live](https://platform.sciencelive4all.org), following the [FORRT](https://forrt.org/) replication framework:

### Chain 1: Simple backbone baseline
| Step | Nanopublication |
|------|----------------|
| Quotation | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA-bVr4LQBaoZPSWEyLFLWdoC7BTcObJ6ykgtBSml5cp4) |
| AIDA sentence | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAO6jpjWhEJQL4bKkTti0H9eqzYUm5LrLr2Vj3cZRJIb4) |
| FORRT Claim | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/np/RAgZGw58tmKN2aaPtkWv4uBJ_tc3U1RtDzh5mjPy8BNKw) |
| FORRT Study | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAytukwQbTLwKSG4r7oEP7jRtgoBKTQVvO0EAzj6xEUiY) |
| FORRT Outcome | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA7OZmOmun07jDm8q6lq7Ris1W0MU3rptcp3bphOWUJj8) |

### Chain 2: Faithful reproduction with matching architecture
| Step | Nanopublication |
|------|----------------|
| Quotation | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA-bVr4LQBaoZPSWEyLFLWdoC7BTcObJ6ykgtBSml5cp4) |
| AIDA sentence | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAxT8-efe0SW_OT_pGFjKHHlBjrfHdgF8Lw8nQpcPp-jI) |
| FORRT Claim | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/np/RALaapCREr1eEa-abjVo3M7Ago23rN2ISt3gMjYXq2cKo) |
| FORRT Study | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAW_JRkV23_hp26WaP0POGns07s3eJXzItiJEINUIcdes) |
| FORRT Outcome | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAqs99x7CDi1tutYKJ6J1zes8PEbUhfjbpf3dkdqSoffQ) |

### Chain 3: Supervised pretraining vs meta-learning
| Step | Nanopublication |
|------|----------------|
| Quotation | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA-bVr4LQBaoZPSWEyLFLWdoC7BTcObJ6ykgtBSml5cp4) |
| AIDA sentence | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA4cXCCMH0rK_0mZn0ShUOa3HjSP6tqxaLQFyR3FVmZdg) |
| FORRT Claim | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/np/RAC32x9lWdLTKRpiz7utsMmLf0_JzadtA4kCvvbrJ_KK8) |
| FORRT Study | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RAW3U5bhevQdUMc51N_QsJSuCrV8j8QauGh9cU6H0o-2Q) |
| FORRT Outcome | [view](https://platform.sciencelive4all.org/np/?uri=https://w3id.org/sciencelive/np/RA9PP1TVbvRwEv9UNHYfGWvfCXtxCe3f6_xJnf-YbAgSc) |

## Reproducibility

- **Code**: [GitHub](https://github.com/annefou/few-shot-eurosat-cross-domain)
- **Archive**: [Zenodo DOI 10.5281/zenodo.19643138](https://doi.org/10.5281/zenodo.19643138)
- **Data**: mini-ImageNet (HuggingFace) + EuroSAT (torchvision) — both downloaded automatically
- **CI**: GitHub Actions runs the experiment on every push
- **Runtime**: Conv4 ~5 min, supervised ~15 min, episodic ~3 hours (Apple M1 Pro)

## Quick start

```bash
git clone https://github.com/annefou/few-shot-eurosat-cross-domain
cd few-shot-eurosat-cross-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat-cross
python 01_cross_domain_eurosat.py
```
