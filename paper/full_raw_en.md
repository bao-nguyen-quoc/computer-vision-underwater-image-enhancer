# Selective Underwater Image Enhancement for Object Detection with YOLO11

**Nguyen Quoc Bao**$^1$, **Phan Le Van Minh**$^1$  
$^1$*The University of Danang, University of Science and Technology, Da Nang, Vietnam*  
Email: `{102250412, 102250417}@hv.dut.udn.vn`

---

## Abstract

Underwater image enhancement (UIE) is often applied before object detection. Recent studies indicate that using one enhancement method for every image does not raise detection accuracy. We investigate a selective alternative on the RUOD benchmark with a YOLO11n detector. A compact ResNet18 selector picks, for each image, among the raw input, SCNet, and Semi-UIR. Its training targets stem from an out-of-fold protocol, preventing any detector from scoring images seen during training. The selector matches the mAP of raw images by leaving most images unchanged. We also introduce a control in which an oracle chooses between a raw image and its horizontal reflection; this geometric control yields an mAP gain comparable to oracles formed over enhanced candidates, indicating that detector sensitivity to spatial shifts contributes to the oracle margin. Fusing predictions of four views with weighted boxes fusion gives a modest gain over raw images. Finally, training detectors on both raw and enhanced versions matches compute-matched raw baselines on clean images while providing higher mAP across 12 unseen degradation conditions. In our experiments, enhancement proved more effective as multi-domain training data for robustness than as a per-image choice at test time.

**Keywords:** Underwater image enhancement, Underwater object detection, Selective enhancement, YOLO11, RUOD, UIEB.

---

## 1. Introduction

Cameras on autonomous underwater vehicles (AUVs) and remotely operated vehicles (ROVs) record images affected by wavelength-dependent light absorption and scattering. Red light attenuates within shallow depths. Consequently, captured scenes frequently exhibit low contrast, green or blue color casts, soft boundaries, and forward-scattering haze.

Deep learning architectures for underwater image enhancement (UIE), such as SCNet and Semi-UIR, deliver high PSNR and SSIM against reference images. Conventional practice inserts such models as an upstream preprocessing step, assuming that cleaner visual quality translates into higher detection accuracy.

Awad et al. demonstrated that this assumption does not hold universally. Across several detectors, applying a single enhancer to all images reduced average detection accuracy relative to raw images, despite individual frames showing improvement. On the RUOD benchmark, they computed an idealized mixed-set upper bound by selecting per image the best among ten candidates (the raw image and nine enhanced variants) using ground-truth annotations. That oracle lifted the mean per-image mAP from 0.68 to 0.77 (+0.09).

This theoretical upper bound motivates two questions. First, can an automated selector choose the winning version directly from raw pixels without ground-truth annotations? Second, how much of the oracle margin stems from genuine optical restoration rather than detector variance across different representations of the same scene?

We examine these questions with YOLO11n on the 14,000-image RUOD benchmark. Candidate versions for each image include the raw input, SCNet, and Semi-UIR. Our findings include:
- We formulate an out-of-fold labeling protocol with a margin $\delta = 0.01$. A ResNet18 selector operating on $224 \times 224$ raw inputs requires 7.7 ms per frame.
- On 4,200 test images, the thresholded selector yields 0.6159 mAP@0.5:0.95 (+3.2% of the oracle gap closed) against 0.6155 for raw images. It keeps the raw input for 85.5% of queries. Uniform SCNet gives 0.6129, Semi-UIR gives 0.6084, and random selection gives 0.6120.
- In a control experiment without enhancement, an oracle choosing between a raw image and its horizontal reflection gains +0.0207 per-image mAP. This matches 144% of the +0.0144 gain achieved by the oracle over raw and SCNet candidates.
- Fusing detections across four views via weighted boxes fusion (WBF) gives +0.0023 mAP@0.5:0.95 on the complete test set, with a 95% confidence interval of $[+0.0014, +0.0051]$ strictly above zero.
- Multi-version training across raw and enhanced variants increases raw-image mAP@0.5:0.95 from 0.5990 to 0.6155 over a 50-epoch baseline, matching an epoch-matched 150-epoch raw detector (0.6150). It further provides higher detection mAP across 12 unseen degradation conditions, although it receives three times more image presentations during training.

---

## 2. Related Work

### 2.1. Underwater Image Enhancement

SCNet targets variable water conditions. At several scales of a U-Net, it decorrelates spatial activations through instance whitening and standardizes channel statistics, ensuring that intermediate features depend less on specific water types.

Semi-UIR mitigates paired training data scarcity via a mean-teacher framework with contrastive regularization. A reliable memory bank stores pseudo-labels filtered by a no-reference quality metric. At inference time, an attention-guided illumination modulation network (AIMnet) processes multi-scale Gaussian luminance representations.

Both models were benchmarked on UIEB using PSNR and SSIM. These metrics assess fidelity against reference photographs rather than downstream detection utility.

### 2.2. Enhancement for Underwater Object Detection

Fu et al. introduced RUOD for underwater detection, observing variable interactions between common enhancers and detector architectures. Luo et al. proposed MAS-YOLOv11 with multi-scale attention heads for cluttered underwater environments. Huang et al. developed AMIEOD to route low-illumination inputs dynamically across specialized enhancement modules.

Awad et al. compared nine enhancers across modern detectors. Uniform application of any enhancer failed to outperform raw inputs over the full test distribution. They introduced a per-image mixed-set oracle, though it relies on test-time annotations and separately trained domain detectors.

### 2.3. Box Fusion and Test-Time Augmentation

Test-time augmentation combines detections from transformed input views. Standard non-maximum suppression (NMS) retains one bounding box per cluster and discards overlapping candidates. Weighted boxes fusion (WBF) instead clusters overlapping boxes and computes confidence-weighted coordinate averages.

---

## 3. Methodology

### 3.1. System Overview

Let $x \in \mathbb{R}^{H \times W \times 3}$ denote an underwater RGB frame. We employ two public pre-trained enhancers with author-released weights: SCNet ($E_{\mathrm{SC}}$) and Semi-UIR ($E_{\mathrm{SU}}$). Candidate representations comprise:
$$\mathcal{V}(x) = \{\, v_0, v_1, v_2 \,\} = \{\, x,\; E_{\mathrm{SC}}(x),\; E_{\mathrm{SU}}(x) \,\}.$$
A selector $f_\theta$ processes raw frame $x$ and outputs index $\hat{k} \in \{0, 1, 2\}$. Chosen candidate $v_{\hat{k}}$ is then supplied to fine-tuned YOLO11n detector $D$. Fig. 1 shows a sample underwater image before and after enhancement with Semi-UIR and SCNet. The overall workflow, encompassing out-of-fold label generation and selective inference, is illustrated in Fig. 2.

![Sample image before and after enhancement. From left to right: raw input, Semi-UIR and SCNet.](figures/sample_enhancement.png)
*Figure 1: Sample image before and after enhancement. From left to right: raw input, Semi-UIR and SCNet.*

![Pipeline. Each training image is enhanced by SCNet and Semi-UIR; out-of-fold YOLO11n detectors score the three versions per image; the best version (with margin $\delta$) becomes the label of a ResNet18 selector that sees only the raw image.](figures/pipeline_en.png)
*Figure 2: Pipeline. Each training image is enhanced by SCNet and Semi-UIR; out-of-fold YOLO11n detectors score the three versions per image; the best version (with margin $\delta$) becomes the label of a ResNet18 selector that sees only the raw image.*

### 3.2. Enhancers and Processing Pipeline

Both enhancers run at native resolution (640 px long edge) with released model parameters.
- **SCNet:** We isolate the enhancement decoder branch, excluding the auxiliary VGG16 perceptual network. Inputs are reflect-padded to multiples of 8, and outputs are cropped and quantized to $[0, 255]$.
- **Semi-UIR:** Luminance map $L(x)$ is computed by convolving the luminance channel with Gaussian kernels at scales $\sigma \in \{15, 60, 90\}$, averaging logarithmic responses, and normalizing to $[0, 255]$. Inputs and luminance maps are reflect-padded to multiples of 16. Processing averages 2 s per frame.

Outputs are stored as JPEG files ($\text{quality} = 95$) sharing original annotations.

### 3.3. Out-of-Fold (OOF) Selector Labeling

Evaluating a detector on its training data inflates scores, biasing target labels. We employ a three-fold out-of-fold protocol:
1. The 9,800 RUOD training frames are partitioned into three folds $\mathcal{F}_0, \mathcal{F}_1, \mathcal{F}_2$ with seed 0 (MD5 hash: `ea44b385539c`).
2. For fold $k \in \{0, 1, 2\}$, a YOLO11n detector is trained for 30 epochs on all three candidate versions of the remaining folds ($\approx 19,600$ images).
3. Checkpoint `last.pt` evaluates candidates of held-out fold $\mathcal{F}_k$ ($\text{conf} = 0.001$, $\text{IoU} = 0.7$).
4. Using `pycocotools`, per-image mAP@0.5:0.95 values $m_{i,0}, m_{i,1}, m_{i,2}$ are recorded. Unannotated frames receive 0.

With margin $\delta = 0.01$, target label $y_i$ is assigned as:
$$y_i = \begin{cases} 0 & \text{if } \max_{k \in \{1, 2\}} m_{i,k} - m_{i,0} < \delta, \\ \arg\max_{k} m_{i,k} & \text{otherwise}. \end{cases}$$

### 3.4. Selector Architecture and Decision Rules

The selector is a ResNet18 backbone pretrained on ImageNet, with a three-way linear head. It processes raw inputs resized to $224 \times 224 \times 3$ (7.7 ms on an NVIDIA Tesla T4 GPU). We train it via two objectives:
- **Class-weighted cross-entropy (`ce`):**
  $$\mathcal{L}_{\mathrm{ce}} = -\sum_{c=0}^2 w_c \cdot \mathbb{I}(y = c) \log p_c,$$
  where $w_c$ denotes inverse class frequency.
- **mAP regression (`reg`):** the network predicts mAP values $(\hat{m}_0, \hat{m}_1, \hat{m}_2)$ supervised by Smooth L1 loss.

#### Decision Threshold ($\tau$)
To prevent low-confidence enhancement calls, we select $k^* = \arg\max_{k \in \{1, 2\}} p_k$ only if $p_{k^*} - p_0 > \tau$; otherwise, the raw image is retained. Under regression, the condition is $\hat{m}_{k^*} - \hat{m}_0 > \tau$.

#### Validation Selection Criterion
Always predicting raw achieves 58.2% accuracy on training labels and 62.4% on test labels. Simple classification accuracy is therefore an uninformative objective. Model selection and threshold tuning rely on chosen-mAP across a held-out 20% validation subset (1,960 images):
$$\text{chosen-mAP} = \frac{1}{N_{\mathrm{val}}} \sum_{i=1}^{N_{\mathrm{val}}} m_{i, \hat{y}_i}.$$
Tuned thresholds are $\tau = 0.625$ for `ce` and $\tau = 0.040$ for `reg`.

### 3.5. Evaluation by Composing Saved Predictions

Evaluation on 4,200 RUOD test images proceeds in four stages:
1. A single detector (`full.pt`) trains for 50 epochs on all three versions of 9,800 training images (29,400 frames total, 10,732 s training time).
2. Predictions across all versions are precomputed and cached. Each policy is scored by selecting corresponding predictions and invoking `pycocotools`.
3. Paired bootstrap resampling (1,000 iterations) computes 95% confidence intervals and empirical p-values $p_{\text{boot}}(\Delta \le 0)$ for per-image mAP differences against raw baseline.
4. Closed gap percentage is computed as:
   $$G = \frac{\mathrm{mAP}_{\mathrm{policy}} - \mathrm{mAP}_{\mathrm{orig}}}{\mathrm{mAP}_{\mathrm{oracle}} - \mathrm{mAP}_{\mathrm{orig}}} \times 100\%.$$

### 3.6. Oracle Control with Horizontal Flip

To examine whether oracle gains reflect enhancer processing or detector variance, we introduce a geometric control via horizontal reflection $T_{\mathrm{flip}}$. For input $x$, bounding boxes predicted on $T_{\mathrm{flip}}(x)$ are mapped to original coordinates via inverse transform $T_{\mathrm{flip}}^{-1}$. Flipped frames preserve exact color and contrast distributions. Differences between $m_{i, \mathrm{orig}}$ and $m_{i, \mathrm{flip}}$ thus reflect detector spatial sensitivity and NMS clustering. We define:
$$\text{Oracle}_{\{\mathrm{orig}, \mathrm{flip}\}}(i) = \max \left( m_{i, \mathrm{orig}},\; m_{i, \mathrm{flip}} \right).$$
If this control yields gains comparable to enhancement oracles, a portion of the apparent upper bound can be attributed to detector sensitivity rather than optical improvement.

### 3.7. Multi-View Weighted Boxes Fusion

We fuse candidate detections across raw, flipped, SCNet, and Semi-UIR inputs using WBF with fixed parameters: $\text{IoU}_{\mathrm{thr}} = 0.6$, $\text{skip\_box\_thr} = 0.001$, and maximum 150 boxes per view.

---

## 4. Experimental Setup

### 4.1. Datasets

**UIEB** comprises 890 paired underwater images. We employ its 90-image benchmark split at native resolution to verify published restoration metrics.

**RUOD** contains 14,000 underwater images (9,800 train, 4,200 test) across 10 categories: *holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish*. Long edges are scaled to 640 px.

### 4.2. Implementation Details

Experiments ran on Kaggle with dual NVIDIA Tesla T4 GPUs, PyTorch 2.11 (CUDA 12.8), and Ultralytics 8.4.174. YOLO11n used $640 \times 640$ input resolution, batch size 32, and seed 0. ResNet18 trained for 15 epochs with AdamW (learning rate $3 \times 10^{-4}$, weight decay $10^{-4}$) on an 80/20 train/validation split (7,840 train, 1,960 val).

### 4.3. Metrics

PSNR and SSIM are evaluated via `scikit-image`. Detection accuracy is measured by COCO mAP@0.5 and mAP@0.5:0.95 through `pycocotools`. Reported confidence intervals represent paired bootstrap 95% intervals over 1,000 iterations.

---

## 5. Results and Discussion

### 5.1. Enhancer Check on UIEB

Table 1 compares our measurements on the 90-image UIEB benchmark with published values. Our measured PSNR matches published reports within 0.35 dB for SCNet and 0.38 dB for Semi-UIR. SSIM differs by under 0.02. Both methods boost PSNR by over 4.5 dB over raw inputs.

#### Table 1: PSNR and SSIM on UIEB test set (90 images, mean $\pm$ std)

| Method | Ours PSNR (dB) | Ours SSIM | Published PSNR | Published SSIM |
|:---|:---:|:---:|:---:|:---:|
| Raw input | $17.90 \pm 5.35$ | $0.776 \pm 0.137$ | -- | -- |
| SCNet | $22.43 \pm 4.43$ | $0.879 \pm 0.082$ | 22.08 | 0.8625 |
| Semi-UIR | **24.21** $\pm 4.03$ | **0.886** $\pm 0.069$ | 24.59 | 0.9010 |
| Per-image oracle$^\dagger$ | 24.95 | 0.9006 | -- | -- |

$^\dagger$*Best candidate selected per image independently for each metric.*

#### Table 2: Number of UIEB test images (out of 90) winning per metric

| Metric | Raw input | SCNet | Semi-UIR |
|:---|:---:|:---:|:---:|
| PSNR | 6 | 24 | 60 |
| SSIM | 10 | 29 | 51 |

No individual model wins uniformly across test images (Table 2). Under PSNR, Semi-UIR tops 60 images, SCNet leads 24, and raw input wins 6. The per-image oracle achieves 24.95 dB PSNR and 0.9006 SSIM. For image 703, raw PSNR is 40.63 dB, while SCNet and Semi-UIR drop to 27.54 dB and 26.38 dB. Conversely, on image 713, raw PSNR is 9.75 dB, whereas SCNet reaches 19.64 dB and Semi-UIR reaches 28.54 dB.

### 5.2. Out-of-Fold Labels on the RUOD Training Set

The three-fold out-of-fold procedure over 9,800 training images yielded 5,704 raw labels (58.2%), 2,208 SCNet labels (22.5%), and 1,888 Semi-UIR labels (19.3%).

On out-of-fold detections, raw images reach 0.5808 mAP@0.5:0.95 (0.6873 per image). Uniform SCNet achieves 0.5785 (0.6859 per image) and uniform Semi-UIR gives 0.5729 (0.6817 per image). The out-of-fold oracle achieves 0.7164 per-image mAP (+0.0291 above raw).

### 5.3. Detection Results on the RUOD Test Set

Table 3 summarizes policy evaluations on 4,200 test images using a common detector.

#### Table 3: Object detection performance on RUOD test set (4,200 images) with YOLO11n

| Policy | mAP@0.5 | mAP@0.5:0.95 | Per-image mAP | Gap Closed (All / Per) | $\Delta$ vs. Raw [95% CI] | $p_{\text{boot}}(\Delta \le 0)$ | Selection (Orig / SC / Semi) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Always Raw (`orig`) | 0.8468 | 0.6155 | 0.7186 | 0.0% / 0.0% | -- (Baseline) | -- | 4200 / 0 / 0 |
| Always SCNet | 0.8449 | 0.6129 | 0.7154 | -18.1% / -13.6% | -0.0031 [-0.0049; -0.0013] | 1.000 | 0 / 4200 / 0 |
| Always Semi-UIR | 0.8422 | 0.6084 | 0.7130 | -50.0% / -24.2% | -0.0056 [-0.0077; -0.0034] | 1.000 | 0 / 0 / 4200 |
| Random Selection | 0.8444 | 0.6120 | 0.7148 | -24.7% / -16.3% | -0.0038 [-0.0054; -0.0021] | 1.000 | 1387 / 1401 / 1412 |
| Selector `ce` | 0.8472 | 0.6146 | 0.7173 | -6.4% / -5.4% | -0.0012 [-0.0025; 0.0000] | 0.970 | 2210 / 1101 / 889 |
| **Selector `ce+$\tau$`** | **0.8474** | **0.6159** | 0.7182 | **+3.2%** / -1.5% | **-0.0004 [-0.0010; +0.0003]** | 0.851 | **3593 / 319 / 288** |
| Selector `reg` | 0.8457 | 0.6142 | 0.7182 | -8.8% / -1.7% | -0.0004 [-0.0018; +0.0009] | 0.734 | 2133 / 1373 / 694 |
| Selector `reg+$\tau$` | 0.8462 | 0.6146 | 0.7183 | -5.9% / -1.4% | -0.0003 [-0.0014; +0.0008] | 0.745 | 3019 / 828 / 353 |
| Oracle$^\dagger$ | **0.8524** | **0.6296** | **0.7417** | 100.0% / 100.0% | +0.0231 [+0.0216; +0.0247] | 0.000 | 2620 / 846 / 734 |

$^\dagger$*Oracle evaluated using the training label margin $\delta = 0.01$ (defaulting to raw unless an enhancer gains $\ge \delta$).*

#### Table 4: Validation performance on 1,960 held-out images and test confusion matrix of Selector `ce+$\tau$` on 4,200 test images

**A. Validation Split Performance (1,960 images)**

| Variant | $\tau$ | Chosen-mAP | Gap Closed |
|:---|:---:|:---:|:---:|
| Always Raw | -- | 0.6924 | 0.0% |
| `ce` | 0 | 0.6923 | -0.4% |
| **`ce+$\tau$`** | **0.625** | **0.6931** | **+2.5%** |
| `reg` | 0 | 0.6918 | -2.4% |
| `reg+$\tau$` | 0.040 | 0.6931 | +2.2% |
| Oracle | -- | 0.7202 | 100.0% |

**B. Test Confusion Matrix (Rows: True Oracle, Columns: Pred.)**

| Oracle Label ($\delta = 0.01$) | Pred. Raw | Pred. SCNet | Pred. Semi-UIR | Total |
|:---|:---:|:---:|:---:|:---:|
| Raw (`orig`) | **2,259** (86.2%) | 176 (6.7%) | 185 (7.1%) | 2,620 |
| SCNet | 708 (83.7%) | **83** (9.8%) | 55 (6.5%) | 846 |
| Semi-UIR | 626 (85.3%) | 60 (8.2%) | **48** (6.5%) | 734 |
| Total Predicted | 3,593 (85.5%) | 319 (7.6%) | 288 (6.9%) | 4,200 |

#### Table 5: Per-class AP@0.5:0.95 across input policies on RUOD test set

| Class | Raw | SCNet | Semi-UIR | Selector `ce+$\tau$` | Oracle |
|:---|:---:|:---:|:---:|:---:|:---:|
| holothurian | 0.4966 | 0.4910 | 0.4877 | 0.4966 | 0.5102 |
| echinus | 0.5305 | 0.5250 | 0.5187 | 0.5298 | 0.5446 |
| scallop | 0.4957 | 0.4941 | 0.4791 | 0.4958 | 0.5058 |
| starfish | 0.5508 | 0.5409 | 0.5431 | 0.5506 | 0.5652 |
| fish | 0.5198 | 0.5196 | 0.5181 | 0.5191 | 0.5337 |
| corals | 0.5386 | 0.5367 | 0.5338 | 0.5377 | 0.5519 |
| diver | 0.7456 | 0.7467 | 0.7397 | 0.7462 | 0.7638 |
| cuttlefish | 0.8360 | 0.8320 | 0.8315 | 0.8361 | 0.8487 |
| turtle | 0.8383 | 0.8390 | 0.8330 | 0.8384 | 0.8518 |
| jellyfish | 0.6026 | 0.6041 | 0.5991 | **0.6089** | 0.6208 |
| **Mean** | **0.6155** | **0.6129** | **0.6084** | **0.6159** | **0.6296** |

#### Uniform Enhancement
Applying SCNet uniformly lowers mAP@0.5:0.95 by 0.0026 ($-0.0031$ per image, CI $[-0.0049, -0.0013]$). Semi-UIR lowers mAP by 0.0071 ($-0.0056$ per image, CI $[-0.0077, -0.0034]$). Both confidence intervals lie below zero ($p_{\text{boot}} = 1.000$). Random selection drops accuracy by 0.0038 per image.

#### Selector Accuracy
Selectors lacking a threshold (`ce` and `reg`) lose detection accuracy, reflecting unnecessary enhancement calls. Adding decision threshold $\tau = 0.625$ allows Selector `ce+$\tau$` to reach 0.6159 mAP@0.5:0.95 (+3.2% gap closed), comparable to the 0.6155 baseline. The 95% CI of its per-image difference, $[-0.0010, +0.0003]$, spans zero ($p_{\text{boot}} = 0.851$).

#### Confusion Matrix
Table 4 presents validation metrics and the test confusion matrix. Overall test accuracy is 56.9%, trailing the 62.4% obtainable by defaulting to raw images. Under the labeling margin $\delta = 0.01$, SCNet was the winning candidate on 846 test images (20.1% of inputs); the selector selected SCNet on 83 of these cases (9.8% recall). Semi-UIR won on 734 test images (17.5%); the selector picked it on 48 instances (6.5% recall). When unconstrained by $\delta$, pairwise oracles show that SCNet exceeds raw inputs on 31.6% of images (1,327 frames), and Semi-UIR on 31.7% (1,331 frames).

The selector matches the raw baseline primarily because threshold $\tau = 0.625$ leaves inputs unenhanced for 85.5% of test images (3,593 cases). Overhead is 7.7 ms per frame.

#### Per-Class Results
Table 5 lists per-class AP across all 10 categories. For *holothurian*, *echinus*, and *scallop*, the selector remains close to raw baselines. The largest variation occurs on *jellyfish* (0.6089 versus 0.6026), an isolated class effect not subjected to separate confidence bounds.

### 5.4. Oracle Gain With and Without Enhancement

Table 6 assesses oracle bounds across candidate configurations on 4,200 test images.

#### Table 6: Oracle comparison between optical enhancers and horizontal reflection on RUOD test set

| Oracle Candidate Set | Size | mAP@0.5:0.95 | $\Delta$ Per-Image [95% CI] | Non-Raw % |
|:---|:---:|:---:|:---:|:---:|
| $\{\text{orig}, \text{orig rerun}\}$ | 2 | 0.6159 | +0.0005 [+0.0003; +0.0006] | 2.3% |
| $\{\text{orig}, \text{scnet}\}$ | 2 | 0.6250 | +0.0144 [+0.0133; +0.0154] | 31.6% |
| $\{\text{orig}, \text{semiuir}\}$ | 2 | 0.6250 | +0.0164 [+0.0150; +0.0179] | 31.7% |
| $\{\text{orig}, \text{flip}\}$ (control) | 2 | 0.6300 | +0.0207 [+0.0193; +0.0221] | 38.5% |
| UIE oracle (orig, SCNet, Semi-UIR)$^\dagger$ | 3 | 0.6300 | +0.0234 [+0.0219; +0.0250] | 44.2% |
| $\{\text{orig}, \text{flip}, \text{SC}, \text{Semi}\}$ | 4 | 0.6386 | +0.0353 [+0.0336; +0.0371] | 58.6% |

$^\dagger$*Unconstrained oracle ($\delta = 0$). In Table 3, the oracle applies $\delta = 0.01$, yielding 0.6296 mAP and 37.6% non-raw selections.*

The horizontal reflection oracle reaches 0.6300 mAP@0.5:0.95, matching the three-candidate unconstrained optical oracle ($\{\text{orig}, \text{SC}, \text{Semi}\}$). On a per-image basis, the flip oracle gains +0.0207, which is 144% of the gain provided by SCNet (+0.0144). The three-version oracle yields +0.0234 per image. Because reflected images undergo no enhancement, this margin reflects purely how the detector responds to mirrored coordinates.

With four candidates (raw, flip, SCNet, Semi-UIR), the oracle attains 0.6386. The expected maximum over multiple variables grows monotonically with candidate set size. Adding candidates expands the oracle envelope regardless of whether added variants contain complementary visual information.

In Table 3, the oracle uses the labeling margin $\delta = 0.01$, selecting an enhanced version on 37.6% of images (1,580 frames) and achieving 0.6296 mAP. In Table 6, the unconstrained oracle ($\delta = 0$) takes the raw maximum, selecting non-raw candidates on 44.2% of images and reaching 0.6300 mAP.

Selector target labels stem from out-of-fold evaluations. The correlation between flip gain and SCNet gain is 0.306. When target labels incorporate detector response variability, a classifier trained solely on raw image pixels faces inherent ambiguity, explaining the selector behavior observed.

### 5.5. Fusion and Confidence-Based Selection

Table 7 evaluates policies operating without test-time annotations.

#### Table 7: Multi-view fusion and confidence selection on RUOD test set

| Method | mAP@0.5:0.95 | $\Delta$ Full | $\Delta$ Per-Image [95% CI] |
|:---|:---:|:---:|:---:|
| Raw Baseline | 0.6155 | -- | -- |
| WBF(orig, flip) | 0.6145 | -0.0009 | -0.0002 [-0.0019; +0.0014] |
| WBF(orig, SCNet) | 0.6085 | -0.0070 | -0.0050 [-0.0065; -0.0035] |
| WBF(orig, SCNet, Semi-UIR) | 0.6111 | -0.0044 | -0.0025 [-0.0044; -0.0007] |
| **WBF(orig, flip, SC, Semi)** | **0.6177** | **+0.0023** | **+0.0032 [+0.0014; +0.0051]** |
| Confidence `conf_top5` | 0.6129 | -0.0025 | -0.0021 [-0.0035; -0.0006] |
| Confidence `conf_sum` | 0.6136 | -0.0018 | -0.0018 [-0.0034; -0.0003] |

#### Selection by Confidence
Routing images by detector confidence (`conf_top5` or `conf_sum`) reduced accuracy by 0.0025 and 0.0018 mAP. Enhancement operations can elevate background false positive scores, misguiding selection.

#### Four-View WBF
WBF over two views, or over three optical variants, fails to surpass raw inputs. In contrast, four-view WBF (raw, flip, SCNet, Semi-UIR) achieves 0.6177 mAP@0.5:0.95 (+0.0023 full set, +0.0032 per-image). The 95% CI of per-image difference, $[+0.0014, +0.0051]$, lies above zero. This constitutes the sole test-time approach demonstrating a strictly positive interval, though requiring four detector passes and two enhancement evaluations per image.

### 5.6. Training Detectors on Multiple Domains

We investigated whether enhanced imagery offers value during detector training. Table 8 evaluates single-domain baseline A (50 epochs on raw images), single-domain detector B (50 epochs on Semi-UIR images), compute-matched baseline A150 (150 epochs on raw images), and multi-version detector C (50 epochs on raw, SCNet, and Semi-UIR), all evaluated on raw test images using `pycocotools`.

#### Table 8: Comparison of YOLO11n detectors evaluated on raw RUOD test images (`pycocotools`)

| Detector | Training Data | Epochs | Image Passes | mAP@0.5 | mAP@0.5:0.95 |
|:---|:---|:---:|:---:|:---:|:---:|
| A (Baseline) | 9,800 raw images | 50 | $1\times$ | 0.8367 | 0.5990 |
| B (Semi-UIR) | 9,800 Semi-UIR images | 50 | $1\times$ | 0.7749 | 0.5475 |
| A150 (Matched) | 9,800 raw images | 150 | $3\times$ | **0.8470** | 0.6150 |
| **C (Multi-version)** | **29,400 images (3 versions)** | **50** | **$3\times$** | 0.8468 | **0.6155** |

Training across raw, SCNet, and Semi-UIR frames (Detector C) lifts clean raw mAP@0.5:0.95 from 0.5990 to 0.6155 (+0.0165) over the 50-epoch baseline A. Single-domain detector B, trained exclusively on Semi-UIR images, reaches only 0.5475 on raw test images ($-0.0515$ against baseline A), indicating that single-domain enhancement training harms accuracy on raw inputs. Meanwhile, Detector A150, trained on raw frames for 150 epochs to match total image passes ($3\times$), achieves 0.6150 mAP. The difference between C and A150 on clean raw images (+0.0005, 95% CI $[-0.0003, +0.0056]$) spans zero. On clean raw images alone, performance gains over baseline A stem from prolonged training rather than domain diversity.

Per-class AP@0.5:0.95 scores on clean test images show close correspondence between A150 and C: A150 leads slightly on *holothurian* (0.5007 vs 0.4966), *echinus* (0.5379 vs 0.5305), and *scallop* (0.5042 vs 0.4957), whereas C leads on *fish* (0.5198 vs 0.5149), *corals* (0.5386 vs 0.5379), *diver* (0.7456 vs 0.7315), *cuttlefish* (0.8360 vs 0.8296), *turtle* (0.8383 vs 0.8347), and *jellyfish* (0.6026 vs 0.5991).

### 5.7. Robustness to Unseen Degradations

While Detectors C and A150 exhibit matching performance on clean inputs, real underwater missions encounter fluctuating water conditions. To test cross-domain generalization, we evaluated the models across 12 synthetic degradation conditions spanning four distortion types not seen during training: low illumination (`dark1`–`3`), turbidity (`turbid1`–`3`), depth-dependent color cast (`cast1`–`3`), and blur with compression (`blur1`–`3`). Fig. 3 illustrates these synthetic degradations applied to a representative RUOD test frame across all four categories and three severity levels.

![Synthetic degradations applied to one RUOD test image. Rows: dark, turbid, color cast, blur + noise; columns: severity levels 1–3.](figures/degradation_grid.jpg)
*Figure 3: Synthetic degradations applied to one RUOD test image. Rows: dark, turbid, color cast, blur + noise; columns: severity levels 1–3.*

#### Table 9: mAP@0.5:0.95 under 12 unseen degradation conditions on RUOD test set (`pycocotools`)

| Condition | Baseline A | Semi-UIR B | Matched A150 | Multi-Version C | C $-$ A150 [95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|
| Clean | 0.5991 | 0.5476 | 0.6151 | **0.6155** | +0.0027 [$-0.0003$, $+0.0056$] |
| `dark1` | 0.5386 | 0.4919 | 0.5431 | **0.5606** | +0.0181 [$+0.0146$, $+0.0218$] |
| `dark2` | 0.3041 | 0.3110 | 0.2632 | **0.3245** | +0.0664 [$+0.0611$, $+0.0717$] |
| `dark3` | 0.0555 | **0.0634** | 0.0402 | 0.0554 | +0.0295 [$+0.0247$, $+0.0346$] |
| `turbid1` | 0.5824 | 0.4668 | 0.5861 | **0.5912** | +0.0067 [$+0.0034$, $+0.0100$] |
| `turbid2` | 0.5060 | 0.3326 | 0.4867 | **0.5086** | +0.0193 [$+0.0150$, $+0.0234$] |
| `turbid3` | **0.3099** | 0.1476 | 0.2738 | 0.3020 | +0.0274 [$+0.0220$, $+0.0323$] |
| `cast1` | 0.5824 | 0.5028 | 0.5939 | **0.6018** | +0.0072 [$+0.0039$, $+0.0102$] |
| `cast2` | 0.5587 | 0.4633 | 0.5662 | **0.5838** | +0.0165 [$+0.0130$, $+0.0198$] |
| `cast3` | 0.5303 | 0.4179 | 0.5349 | **0.5584** | +0.0248 [$+0.0210$, $+0.0287$] |
| `blur1` | 0.5028 | 0.4596 | 0.5142 | **0.5280** | +0.0175 [$+0.0141$, $+0.0215$] |
| `blur2` | 0.2709 | 0.2655 | 0.2768 | **0.3050** | +0.0251 [$+0.0199$, $+0.0304$] |
| `blur3` | 0.1064 | 0.1145 | 0.1212 | **0.1383** | +0.0093 [$+0.0036$, $+0.0148$] |
| **Mean retention** | 66.8% | 61.4% | 65.0% | **68.5%** | -- |

![mAP@0.5:0.95 of the four detectors as severity increases for each degradation type (clean = undegraded test images).](figures/robustness_en.png)
*Figure 4: mAP@0.5:0.95 of the four detectors as severity increases for each degradation type (clean = undegraded test images).*

As shown in Table 9 and plotted across degradation severities in Fig. 4, Detector C outperforms Detector A150 across all 12 degradation settings, with paired bootstrap 95% confidence intervals strictly above zero ($p_{\text{boot}} < 0.05$). Fig. 4 traces how each detector degrades as severity escalates: single-domain models deteriorate rapidly, whereas multi-version training maintains higher retention across increasing distortion levels. Across all conditions, C preserves an average of 68.5% of its clean mAP, compared to 65.0% for A150. Detector B, trained exclusively on Semi-UIR, yields the lowest mean retention (61.4%); it achieves the highest mAP under extreme darkness (`dark3`, 0.0634) because Semi-UIR boosts illumination, yet it drops to 0.1476 in `turbid3` and trails on color casts.

The largest margins of C over A150 emerge under moderate disruptions: in `dark2`, C achieves 0.3245 against 0.2632 for A150 (+0.0664, a 23% relative margin); in `turbid3`, C yields 0.3020 against 0.2738 (+0.0274).

Notably, extending single-domain training on raw images reduced resilience. While A150 gained +0.0160 on clean images over baseline A, it lagged behind A on `dark2`, `dark3`, `turbid2`, and `turbid3`. In contrast, multi-domain training allowed Detector C to secure the clean accuracy of A150 while building protection against environmental shifts without test-time inference penalty.

---

## 6. Conclusion

We explored selective underwater image enhancement for object detection using YOLO11n on the RUOD benchmark.

Applying SCNet or Semi-UIR across all inputs reduced mAP@0.5:0.95 by 0.0026 and 0.0071 relative to raw inputs. A ResNet18 selector (7.7 ms latency) with threshold $\tau = 0.625$ matched raw baseline mAP (0.6159 versus 0.6155), retaining raw inputs for 85.5% of images.

An oracle choosing between raw frames and their horizontal reflections gained +0.0207 per-image mAP, reaching 144% of the SCNet optical oracle gain (+0.0144). This indicates that detector sensitivity to spatial alterations accounts for an appreciable fraction of the apparent oracle gap. Four-view WBF yielded +0.0023 mAP over the full test set (95% CI $[+0.0014, +0.0051]$).

Multi-version training elevated raw-frame mAP@0.5:0.95 from 0.5990 to 0.6155 over 50-epoch raw baselines and matched compute-matched 150-epoch raw detectors (0.6150). It further outperformed raw detectors across 12 unseen degradation conditions, retaining 68.5% of clean mAP versus 65.0%. In our experiments, enhanced imagery yielded its primary benefit as multi-domain training data for environmental robustness rather than as a per-image test-time choice.

---

## References

1. Ali Awad, Ashraf Saleem, Sidike Paheding, Evan Lucas, Serein Al-Ratrout, and Timothy C. Havens. Revisiting underwater image enhancement for object detection: A unified quality–detection evaluation framework. *Journal of Imaging*, 12(1):18, 2026. doi:10.3390/jimaging12010018.
2. Shirui Huang, Keyan Wang, Huan Liu, Jun Chen, and Yunsong Li. Contrastive semi-supervised learning for underwater image restoration via reliable bank. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 18145–18155, 2023. doi:10.1109/CVPR52729.2023.01739.
3. Chenping Fu, Risheng Liu, Xin Fan, Puyang Chen, Hao Fu, Wanqi Yuan, Ming Zhu, and Zhongxuan Luo. Rethinking general underwater object detection: Datasets, challenges, and solutions. *Neurocomputing*, 517:243–256, 2023. doi:10.1016/j.neucom.2022.10.039.
4. Zhenqi Fu, Xiaopeng Lin, Wu Wang, Yue Huang, and Xinghao Ding. Underwater image enhancement via learning water type desensitized representations. *arXiv preprint arXiv:2102.00676*, 2022. doi:10.48550/arXiv.2102.00676.
5. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770–778, 2016. doi:10.1109/CVPR.2016.90.
6. Xiaochen Huang, Honggang Chen, Weicheng Zhang, Xiaobo Dai, Yongyi Li, Linbo Qing, and Xiaohai He. AMIEOD: Adaptive multi-experts image enhancement for object detection in low-illumination scenes. *arXiv preprint arXiv:2605.06084*, 2026.
7. Glenn Jocher and Jing Qiu. Ultralytics YOLO11, version 8.3.0, 2024. URL: https://github.com/ultralytics/ultralytics.
8. Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, and Dacheng Tao. An underwater image enhancement benchmark dataset and beyond. *IEEE Transactions on Image Processing*, 29:4376–4389, 2020. doi:10.1109/TIP.2019.2955241.
9. Yang Luo, Aiping Wu, and Qingqing Fu. MAS-YOLOv11: An improved underwater object detection algorithm based on YOLOv11. *Sensors*, 25(11):3433, 2025. doi:10.3390/s25113433.
10. Roman Solovyev, Weimin Wang, and Tatiana Gabruseva. Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107:104117, 2021. doi:10.1016/j.imavis.2021.104117.
