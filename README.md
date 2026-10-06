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
  run_ruod.sh # enhance the whole ruod640 dataset with both models (resumable)
  enhance_semiuir.py # enhance one image or a folder with Semi-UIR
  enhance_scnet.py # enhance one image or a folder with SCNet
  check_outputs.py # verify enhanced images (count, size, NaN/black)
samples/
  test_img.png # sample image for smoke tests
```

## Enhance the whole RUOD dataset (Kaggle)

1. Add the `ruod640` dataset as notebook input and enable **GPU T4 x2**.
2. Run:

```bash
git clone -q https://github.com/bao-nguyen-quoc/computer-vision-underwater-image-enhancer /kaggle/working/computer-vision-underwater-image-enhancer
bash /kaggle/working/computer-vision-underwater-image-enhancer/enhancers/setup.sh /kaggle/working
bash /kaggle/working/computer-vision-underwater-image-enhancer/enhancers/run_ruod.sh /kaggle/working
```

3. Run it with **Save Version → Save & Run All** so `/kaggle/working/enhanced` is kept.

Output (same file names as `ruod640/orig`, JPEG quality 95, labels copied):

```
enhanced/
  scnet/images/{train,test}/    scnet/labels/{train,test}/
  semiuir/images/{train,test}/  semiuir/labels/{train,test}/
```

SCNet runs on GPU 0 and Semi-UIR on GPU 1 in parallel; logs go to `scnet.log` / `semiuir.log`. Re-running skips images that are already done.
