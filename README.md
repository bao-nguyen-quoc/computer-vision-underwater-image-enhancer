# computer-vision-underwater-image-enhancer

Underwater image enhancer project to support learning purpose on university (Computer vision).

The goal is a selector that picks, per image, the best version (original / SCNet / Semi-UIR) as input for a YOLO11 detector, evaluated on RUOD.

## Kaggle dataset

https://www.kaggle.com/datasets/phanlvnminh/ruod640?select=ruod640

## Kaggle notebook

The notebook is private and only shared in private group

## Enhancers setup

The project's enhancers are split across 3 repos:

| Repo | Role |
|---|---|
| [computer-vision-SCNet](https://github.com/bao-nguyen-quoc/computer-vision-SCNet) | Fork of [zhenqifu/SCNet](https://github.com/zhenqifu/SCNet), with small compatibility fixes |
| [computer-vision-Semi-UIR](https://github.com/bao-nguyen-quoc/computer-vision-Semi-UIR) | Fork of [huang-shirui/semi-uir](https://github.com/huang-shirui/semi-uir), with small compatibility fixes |
| [computer-vision-underwater-image-enhancer](https://github.com/bao-nguyen-quoc/computer-vision-underwater-image-enhancer) | This repo: project scripts (setup, enhance, checks) |

The forks only contain fixes needed to run the original code on a current environment. All project-specific scripts live in this repo and call the forks through `--repo`.

```
enhancers/
  setup.sh # clone forks, install deps, download SCNet weights (idempotent)
  enhance_semiuir.py # enhance one image or a folder with Semi-UIR
  enhance_scnet.py # enhance one image or a folder with SCNet
  check_outputs.py # verify enhanced images (count, size, NaN/black)
samples/
  test_img.png # sample image for smoke tests
```