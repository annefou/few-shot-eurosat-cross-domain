# Few-Shot EuroSAT Cross-Domain: From Everyday Photos to Satellite Imagery

[![Run Few-Shot Experiment](https://github.com/annefou/few-shot-eurosat-cross-domain/actions/workflows/run-experiment.yml/badge.svg)](https://github.com/annefou/few-shot-eurosat-cross-domain/actions/workflows/run-experiment.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19643138.svg)](https://doi.org/10.5281/zenodo.19643138)
[![Jupyter Book](https://img.shields.io/badge/Jupyter%20Book-live-orange)](https://annefou.github.io/few-shot-eurosat-cross-domain/)

Can a model trained on everyday photographs classify satellite land cover types with only a handful of labeled examples?

This project applies [Prototypical Networks](https://arxiv.org/abs/1703.05175) in a **cross-domain** setting: train on [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet) (everyday photos), test on [EuroSAT](https://github.com/phelber/EuroSAT) (real [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) satellite imagery). Three approaches are compared, replicating [Guo et al. (2020, ECCV)](https://doi.org/10.1007/978-3-030-58583-9_8).

## Results

| Approach | 5-shot | 20-shot | 50-shot | Training time |
|----------|--------|---------|---------|---------------|
| Conv4 episodic (84×84) | 67.4% | 73.1% | 74.7% | ~5 min |
| ResNet-10 episodic (224×224, 40k ep) | **75.0%** | **81.9%** | **82.9%** | ~3 hours |
| ResNet-10 supervised (224×224, 10 ep) | **76.2%** | 79.5% | 81.5% | ~15 min |
| Guo et al. (2020) baseline | 73.3% | 82.3% | 80.5% | — |

## FORRT Nanopublications

Published as three chains on [Science Live](https://platform.sciencelive4all.org):

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

## Quick start

```bash
git clone https://github.com/annefou/few-shot-eurosat-cross-domain
cd few-shot-eurosat-cross-domain
mamba env create -f environment.yml
mamba activate few-shot-eurosat-cross
python 01_cross_domain_eurosat.py
```

## Datasets

- **Training**: [mini-ImageNet](https://paperswithcode.com/dataset/mini-imagenet) — 60,000 photographs of everyday objects, downloaded via HuggingFace `datasets`
- **Testing**: [EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 real Sentinel-2 satellite image patches, 10 land cover classes

## Citation

```bibtex
@software{fouilloux2026fewshot_cross,
  author = {Fouilloux, Anne},
  title = {Few-Shot EuroSAT Cross-Domain: From Everyday Photos to Satellite Imagery},
  year = {2026},
  doi = {10.5281/zenodo.19643138},
  url = {https://github.com/annefou/few-shot-eurosat-cross-domain}
}
```

## Related

- [few-shot-eurosat-within-domain](https://github.com/annefou/few-shot-eurosat-within-domain) — Within-domain companion: trains and tests on EuroSAT

## License

MIT — see [LICENSE](LICENSE).
