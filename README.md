# Few-Shot EuroSAT Cross-Domain: From Everyday Photos to Satellite Imagery

[![Run Few-Shot Experiment](https://github.com/annefou/few-shot-eurosat-cross-domain/actions/workflows/run-experiment.yml/badge.svg)](https://github.com/annefou/few-shot-eurosat-cross-domain/actions/workflows/run-experiment.yml)
[![Jupyter Book](https://img.shields.io/badge/Jupyter%20Book-live-orange)](https://annefou.github.io/few-shot-eurosat-cross-domain/)

Can a model trained on everyday photographs (mini-ImageNet) classify satellite land cover types (EuroSAT) with only a handful of labeled examples?

This project is a strict replication of [Guo et al. (2020, ECCV)](https://doi.org/10.1007/978-3-030-58583-9_8) Table 1, applying [Prototypical Networks](https://arxiv.org/abs/1703.05175) (Snell et al., NeurIPS 2017) in a **cross-domain** setting: the model is trained on [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet) (natural photos of everyday objects) and evaluated on [EuroSAT](https://github.com/phelber/EuroSAT) (real [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) satellite imagery). This tests whether visual features learned from ordinary photographs transfer to a radically different imaging domain.

## Quick start

```bash
git clone https://github.com/annefou/few-shot-eurosat-cross-domain
cd few-shot-eurosat-cross-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat-cross
python 01_cross_domain_eurosat.py
```

Or with Docker:

```bash
docker build -t few-shot-eurosat-cross .
docker run few-shot-eurosat-cross
```

## Datasets

- **Training**: [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet) — 60,000 natural photographs (84x84 px) across 100 everyday object classes, downloaded via HuggingFace `datasets`
- **Testing**: [EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 Sentinel-2 satellite image patches (64x64 px, 10 m ground resolution), 10 land use/land cover classes across Europe

## Method

Prototypical Networks learn to map images into a feature space where similar classes cluster together. In this cross-domain setting, the feature extractor is trained entirely on mini-ImageNet (photos of dogs, buses, furniture, etc.) and then applied zero-shot to satellite imagery. At test time, the model computes a "prototype" from the few labeled satellite examples of each class and classifies new satellite images by nearest-prototype distance.

The key question: do visual features learned from everyday photos generalize to a completely different imaging modality (multispectral satellite data rendered as RGB)?

## Related

- [few-shot-eurosat-within-domain](https://github.com/annefou/few-shot-eurosat-within-domain) — Within-domain companion: trains and tests on EuroSAT, simulating Natura 2000 monitoring

## Citation

```bibtex
@software{fouilloux2026fewshot_cross,
  author = {Fouilloux, Anne},
  title = {Few-Shot EuroSAT Cross-Domain: From Everyday Photos to Satellite Imagery},
  year = {2026},
  url = {https://github.com/annefou/few-shot-eurosat-cross-domain}
}
```

## License

MIT -- see [LICENSE](LICENSE).
