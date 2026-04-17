---
title: Few-Shot EuroSAT Cross-Domain
subtitle: Can everyday photos teach AI to classify satellite imagery?
---

## The cross-domain challenge

Earth observation relies on satellite sensors that produce imagery fundamentally different from everyday photographs. Sentinel-2 captures multispectral data at 10 m resolution, looking straight down at landscapes. Meanwhile, the largest labeled datasets in computer vision contain photos of dogs, cars, and furniture taken at human eye level. Can visual features learned from these everyday photos transfer to satellite land cover classification?

This is the **cross-domain few-shot learning** problem studied by [Guo et al. (2020, ECCV)](https://doi.org/10.1007/978-3-030-58583-9_8). It is harder than within-domain transfer because the training and test images come from completely different imaging modalities.

## What we did

1. **Trained a Prototypical Network on mini-ImageNet** -- 60,000 natural photographs of everyday objects (dogs, buses, furniture, etc.) across 100 classes
2. **Froze the feature extractor** -- no satellite data is used during training at all
3. **Evaluated on EuroSAT** -- 27,000 real Sentinel-2 satellite patches covering 10 European land cover types
4. **Tested with 1, 5, and 20 labeled satellite examples per class** -- replicating the exact protocol from Guo et al. (2020) Table 1

## Why this matters for Earth observation

For EO practitioners, this experiment answers a practical question: when you have a new satellite classification task with very few labeled examples, how much can you rely on models pre-trained on non-satellite data?

- **If cross-domain transfer works well**: you can bootstrap satellite classifiers from general-purpose models, reducing the need for expensive labeled satellite training data
- **If it fails**: domain-specific training on satellite imagery is essential, even for few-shot learning -- motivating investments in labeled EO datasets

## Companion project

The [within-domain companion](https://github.com/annefou/few-shot-eurosat-within-domain) trains and tests entirely on EuroSAT, simulating a Natura 2000 habitat monitoring scenario. Comparing the two reveals the cost of the domain gap between natural photos and satellite imagery.

## Reproducibility

This experiment is fully reproducible:

- **Code**: [GitHub repository](https://github.com/annefou/few-shot-eurosat-cross-domain) with Jupytext notebook, Snakemake pipeline, Dockerfile
- **Data**: [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet) via HuggingFace `datasets` + [EuroSAT](https://zenodo.org/records/7711810) -- both downloaded automatically
- **CI**: GitHub Actions runs the full experiment on every push

## Quick start

```bash
# Clone and run
git clone https://github.com/annefou/few-shot-eurosat-cross-domain
cd few-shot-eurosat-cross-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat-cross
python 01_cross_domain_eurosat.py
```

Or with Docker:
```bash
docker pull ghcr.io/annefou/few-shot-eurosat-cross-domain
docker run ghcr.io/annefou/few-shot-eurosat-cross-domain
```
