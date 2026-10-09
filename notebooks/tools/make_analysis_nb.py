"""Sinh notebook analysis_noise_fusion.ipynb: (1) đo nhiễu của oracle bằng ảnh lật ngang,
(2) gộp dự đoán nhiều phiên bản (WBF) và chọn theo độ tự tin của detector.
Chạy: python make_analysis_nb.py <out.ipynb>"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Phân tích: oracle có bao nhiêu phần là nhiễu? Gộp nhiều phiên bản có hơn ảnh gốc không?

Dùng lại model cuối `full.pt` và dự đoán test đã lưu của notebook K-fold (version full). Không train gì thêm.

**Phần A: mức nhiễu của oracle (noise floor)**
- Dự đoán thêm *ảnh gốc lật ngang* (`flip`) bằng `full.pt`, lật box về lại.
- `flip` có nội dung giống hệt ảnh gốc, không tăng cường gì. Oracle `{orig, flip}` vì vậy chỉ đo **nhiễu của detector theo từng ảnh**.
- So với oracle `{orig, scnet}`, `{orig, semiuir}`, `{orig, scnet, semiuir}`: nếu oracle của nhiễu xấp xỉ oracle của tăng cường thì khoảng oracle (kể cả 0,68 → 0,77 của Awad et al.) phần lớn là nhiễu đo lường.
- Kiểm tra thêm: dự đoán lại ảnh gốc (không lật) phải cho đúng mAP của `summary_full.json`.

**Phần B: dùng nhiều phiên bản mà không cần ground-truth**
- **WBF** (Weighted Box Fusion) gộp box của nhiều phiên bản cho từng ảnh. Tham số cố định, **không tinh chỉnh trên test**: `iou_thr = 0.6`, `skip_box_thr = 0.001`, tối đa 150 box/phiên bản.
  - `WBF(orig, flip)`: test-time augmentation thông thường, không có tăng cường ảnh → **mốc so sánh công bằng**.
  - `WBF(orig, scnet, semiuir)`, `WBF(orig, scnet)`, `WBF(orig, flip, scnet, semiuir)`.
- **Chọn theo độ tự tin của detector** (heuristic Awad et al. gợi ý nhưng chưa thử): mỗi ảnh chọn phiên bản có điểm tự tin cao nhất.
  - `conf_top5`: trung bình 5 score cao nhất.
  - `conf_sum`: tổng score ≥ 0,25.
- Mọi dòng có mAP cả tập, mAP từng ảnh, chênh lệch so với ảnh gốc kèm **bootstrap CI 95%**.

**Input (Add Input):**
- Dataset `phanlvnminh/ruod640` (ảnh test gốc + `ann_test640.json`).
- Output notebook K-fold **version full (v6)**: `preds_test_*.json.gz`, `per_image_test.csv`, `summary_full.json`, `weights/full.pt`.

**Cách chạy:** GPU **T4** (1 GPU), Internet **On** (cài `ensemble-boxes`). `SMOKE = True` chạy thử 200 ảnh; sau đó `SMOKE = False` → Save Version → Save & Run All (≈ 20–30 phút).
""")

md("## 1. Cấu hình")
code(r'''
SMOKE = False              # True = 200 ảnh test, 50 lần bootstrap
WBF_IOU = 0.6
WBF_SKIP = 0.001
TOPK = 150                 # số box tối đa mỗi phiên bản đưa vào WBF
N_BOOT = 1000
SEED = 0
IMGSZ = 640
import os
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
OUT = f"{WORK}/analysis"
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    N_BOOT = 50
print("SMOKE =", SMOKE)
''')

md("## 2. Thư viện")
code(r'''
import subprocess, sys
for mod, pkg in [("pycocotools", "pycocotools"), ("ensemble_boxes", "ensemble-boxes"), ("ultralytics", "ultralytics")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import glob, json, gzip, time, contextlib, io, warnings
warnings.filterwarnings("ignore", message="Zero area box")
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2, torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion
DEV = 0 if torch.cuda.is_available() else "cpu"
print("device:", DEV)
''')

md("## 3. Input")
code(r'''
def find_one(pattern, hint):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    assert hits, f"Không thấy {pattern} trong {INPUT} -> {hint}"
    return hits[0]

found = glob.glob(f"{INPUT}/**/orig/images/test", recursive=True)
assert found, "Không thấy ruod640/orig/images/test -> Add Input dataset ruod640"
TEST_DIR = found[0]
R640 = os.path.dirname(os.path.dirname(os.path.dirname(TEST_DIR)))
HINT = "Add Input output version full (v6) của notebook K-fold"
WEIGHTS = find_one("full.pt", HINT)
PER_TEST = find_one("per_image_test.csv", HINT)
SUMMARY = find_one("summary_full.json", HINT)
VERS = ["orig", "scnet", "semiuir"]
PREDS_FILE = {v: find_one(f"preds_test_{v}.json.gz", HINT) for v in VERS}
print("weights:", WEIGHTS)

with contextlib.redirect_stdout(io.StringIO()):
    GT = COCO(f"{R640}/ann_test640.json")
CATS = sorted(GT.getCatIds())
te = pd.read_csv(PER_TEST)
if SMOKE:
    te = te.sample(200, random_state=SEED).sort_values("image_id").reset_index(drop=True)
IDS = te.image_id.astype(int).tolist()
NAMES = te.file_name.tolist()
WH = {i: (GT.imgs[i]["width"], GT.imgs[i]["height"]) for i in IDS}
REF = json.load(open(SUMMARY))["policies"]

P = {}
for v in VERS:
    with gzip.open(PREDS_FILE[v], "rt") as f:
        P[v] = [d for d in json.load(f) if d["image_id"] in set(IDS)]
    print(f"{v}: {len(P[v])} detection")
''')

md("## 4. Dự đoán lại ảnh gốc (kiểm tra) và ảnh gốc lật ngang")
code(r'''
model = YOLO(WEIGHTS)

def predict(flip):
    dets, t = [], time.time()
    for b in range(0, len(NAMES), 64):
        names, ids = NAMES[b:b + 64], IDS[b:b + 64]
        imgs = [cv2.imread(f"{TEST_DIR}/{n}") for n in names]
        if flip:
            imgs = [cv2.flip(im, 1) for im in imgs]
        res = model.predict(imgs, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, device=DEV, verbose=False)
        for iid, im, r in zip(ids, imgs, res):
            W = im.shape[1]
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                if flip:
                    x1, x2 = W - x2, W - x1
                dets.append({"image_id": iid, "category_id": CATS[c],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(s, 5)})
    print(f"  {'flip' if flip else 'orig (dự đoán lại)'}: {len(dets)} detection, {time.time() - t:.0f}s")
    return dets

P["orig_rerun"] = predict(False)
P["flip"] = predict(True)
with gzip.open(f"{OUT}/preds_test_orig_flip.json.gz", "wt") as f:
    json.dump(P["flip"], f)
''')

md("## 5. Hàm chấm điểm: mAP cả tập, mAP từng ảnh, bootstrap")
code(r'''
def quiet():
    return contextlib.redirect_stdout(io.StringIO())

def overall(dets):
    with quiet():
        E = COCOeval(GT, GT.loadRes([dict(d) for d in dets]), "bbox")
        E.params.imgIds = IDS
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0]), float(E.stats[1])

def per_image(dets):
    with quiet():
        E = COCOeval(GT, GT.loadRes([dict(d) for d in dets]), "bbox")
    out = np.zeros(len(IDS))
    for j, i in enumerate(IDS):
        E.params.imgIds = [i]
        with quiet():
            E.evaluate(); E.accumulate(); E.summarize()
        out[j] = max(float(E.stats[0]), 0.0)
    return out

def boot(diff):
    r = np.random.default_rng(SEED); n = len(diff)
    m = np.array([diff[r.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(diff.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

def compose(sources, choice):
    """sources: list tên phiên bản; choice[j] = chỉ số nguồn cho ảnh IDS[j]."""
    pick = dict(zip(IDS, choice))
    return [d for k, s in enumerate(sources) for d in P[s] if pick[d["image_id"]] == k]

T0 = time.time()
M = {}
for v in ["orig", "scnet", "semiuir", "flip", "orig_rerun"]:
    M[v] = per_image(P[v])
    if v in VERS and not SMOKE:     # phải trùng với per_image_test.csv của notebook K-fold
        dmax = float(np.abs(M[v] - te[f"map50_95_{v}"].to_numpy()).max())
        print(f"  {v}: lệch tối đa so với per_image_test.csv = {dmax:.2e}")
print(f"mAP từng ảnh xong ({time.time() - T0:.0f}s)")

m_rerun = overall(P["orig_rerun"])
m_orig = overall(P["orig"])
m_flip = overall(P["flip"])
print(f"Kiểm tra: orig đã lưu {m_orig[0]:.4f} | dự đoán lại {m_rerun[0]:.4f} | summary_full {REF['always_orig']['map50_95']:.4f}")
print(f"Ảnh lật ngang: mAP50:95 {m_flip[0]:.4f} (phải gần ảnh gốc; nếu ≈ 0 là lật box sai)")
assert m_flip[0] > 0.5 * m_orig[0], "mAP ảnh lật quá thấp -> kiểm tra phép lật box"
''')

md(r"""
## 6. Phần A: oracle của nhiễu so với oracle của tăng cường

`oracle(S)`: mỗi ảnh lấy phiên bản trong tập S có mAP từng ảnh cao nhất (cần ground-truth, chỉ là cận trên).
""")
code(r'''
SETS = {
    "{orig, orig chạy lại}": ["orig", "orig_rerun"],
    "{orig, flip}  ← chỉ có nhiễu": ["orig", "flip"],
    "{orig, scnet}": ["orig", "scnet"],
    "{orig, semiuir}": ["orig", "semiuir"],
    "{orig, scnet, semiuir}  ← oracle của dự án": ["orig", "scnet", "semiuir"],
    "{orig, flip, scnet, semiuir}": ["orig", "flip", "scnet", "semiuir"],
}
rows = []
for name, S in SETS.items():
    A = np.stack([M[s] for s in S], 1)
    choice = A.argmax(1)
    best = A.max(1)
    d_all = overall(compose(S, choice))
    mean, lo, hi = boot(best - M["orig"])
    rows.append({"oracle": name, "n_candidates": len(S), "mAP50_95": d_all[0], "mAP50": d_all[1],
                 "per_image": float(best.mean()), "gain_per_image": mean, "ci95_low": lo, "ci95_high": hi,
                 "gain_dataset": d_all[0] - m_orig[0],
                 "share_not_orig": float((choice != 0).mean())})
NOISE = pd.DataFrame(rows)
NOISE.to_csv(f"{OUT}/oracle_noise.csv", index=False)
print(NOISE.round(4).to_string(index=False))

g = NOISE.set_index("oracle")["gain_per_image"]
ratio = g["{orig, flip}  ← chỉ có nhiễu"] / g["{orig, scnet}"] if g["{orig, scnet}"] > 0 else float("nan")
print(f"\nOracle 2 ứng viên: nhiễu (orig, flip) = {ratio:.0%} mức tăng của (orig, scnet)")

# Tăng cường có ích "thật" hay không: ảnh được lợi từ scnet có trùng với ảnh được lợi từ flip?
gain_flip = M["flip"] - M["orig"]; gain_scnet = M["scnet"] - M["orig"]
corr = float(np.corrcoef(gain_flip, gain_scnet)[0, 1])
print(f"Tương quan (lợi ích flip, lợi ích scnet) theo ảnh: {corr:.3f}")
''')

md(r"""
## 7. Phần B: gộp nhiều phiên bản (WBF) và chọn theo độ tự tin

Không dùng ground-truth, tham số cố định. Đây là các phương pháp **dùng được thật**.
""")
code(r'''
def by_image(dets):
    g = {}
    for d in dets:
        g.setdefault(d["image_id"], []).append(d)
    return g

G = {v: by_image(P[v]) for v in ["orig", "flip", "scnet", "semiuir"]}
CAT_IDX = {c: k for k, c in enumerate(CATS)}

def to_lists(iid, v):
    W, H = WH[iid]
    ds = sorted(G[v].get(iid, []), key=lambda d: -d["score"])[:TOPK]
    b, sc, lb = [], [], []
    for d in ds:
        x1, y1 = max(0.0, d["bbox"][0] / W), max(0.0, d["bbox"][1] / H)
        x2, y2 = min(1.0, (d["bbox"][0] + d["bbox"][2]) / W), min(1.0, (d["bbox"][1] + d["bbox"][3]) / H)
        if x2 > x1 and y2 > y1:                      # bỏ box rỗng sau khi cắt về khung ảnh
            b.append([x1, y1, x2, y2]); sc.append(d["score"]); lb.append(CAT_IDX[d["category_id"]])
    return b, sc, lb

def fuse_one(args):
    iid, srcs = args
    W, H = WH[iid]
    B, S, L = zip(*[to_lists(iid, v) for v in srcs])
    if not any(len(s) for s in S):
        return []
    B = [b if len(b) else np.zeros((0, 4)) for b in B]
    S = [s if len(s) else np.zeros(0) for s in S]
    L = [l if len(l) else np.zeros(0) for l in L]
    b, s, l = weighted_boxes_fusion(list(B), list(S), list(L), iou_thr=WBF_IOU, skip_box_thr=WBF_SKIP)
    order = np.argsort(-s)[:300]
    return [{"image_id": iid, "category_id": CATS[int(l[k])],
             "bbox": [float(b[k][0] * W), float(b[k][1] * H), float((b[k][2] - b[k][0]) * W), float((b[k][3] - b[k][1]) * H)],
             "score": float(s[k])} for k in order]

def wbf(srcs):
    t = time.time()
    with Pool(4) as pool:
        out = pool.map(fuse_one, [(i, srcs) for i in IDS], chunksize=64)
    dets = [d for o in out for d in o]
    print(f"  WBF{tuple(srcs)}: {len(dets)} box, {time.time() - t:.0f}s")
    return dets

def conf_score(dets_img, rule):
    s = np.sort([d["score"] for d in dets_img])[::-1]
    if rule == "conf_top5":
        return float(s[:5].mean()) if len(s) else 0.0
    return float(s[s >= 0.25].sum())

METHODS = {}
for name, srcs in {"WBF(orig, flip)  ← TTA, không tăng cường": ["orig", "flip"],
                   "WBF(orig, scnet)": ["orig", "scnet"],
                   "WBF(orig, scnet, semiuir)": ["orig", "scnet", "semiuir"],
                   "WBF(orig, flip, scnet, semiuir)": ["orig", "flip", "scnet", "semiuir"]}.items():
    P[name] = wbf(srcs)
    METHODS[name] = P[name]

for rule in ["conf_top5", "conf_sum"]:
    sc = np.array([[conf_score(G[v].get(i, []), rule) for v in VERS] for i in IDS])
    choice = sc.argmax(1)
    METHODS[f"chọn theo {rule}"] = compose(VERS, choice)
    print(f"  {rule}: chọn orig/scnet/semiuir = {np.bincount(choice, minlength=3).tolist()}")
''')

code(r'''
rows = [{"method": "Ảnh gốc", "mAP50_95": m_orig[0], "mAP50": m_orig[1], "per_image": float(M["orig"].mean()),
         "delta_dataset": 0.0, "delta_per_image": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}]
for name, dets in METHODS.items():
    t = time.time()
    o = overall(dets)
    pi = per_image(dets)
    mean, lo, hi = boot(pi - M["orig"])
    rows.append({"method": name, "mAP50_95": o[0], "mAP50": o[1], "per_image": float(pi.mean()),
                 "delta_dataset": o[0] - m_orig[0], "delta_per_image": mean, "ci95_low": lo, "ci95_high": hi})
    print(f"  {name}: {o[0]:.4f} ({time.time() - t:.0f}s)")
FUSION = pd.DataFrame(rows)
FUSION.to_csv(f"{OUT}/fusion.csv", index=False)
print(FUSION.round(4).to_string(index=False))
''')

md("## 8. Tóm tắt")
code(r'''
summary = {"smoke": SMOKE, "n_images": len(IDS), "wbf": {"iou_thr": WBF_IOU, "skip_box_thr": WBF_SKIP, "topk": TOPK},
           "check": {"orig_saved": m_orig[0], "orig_rerun": m_rerun[0], "flip": m_flip[0],
                     "summary_full_orig": REF["always_orig"]["map50_95"]},
           "noise_vs_scnet_ratio": ratio, "corr_gain_flip_scnet": corr,
           "oracle": NOISE.to_dict("records"), "fusion": FUSION.to_dict("records")}
with open(f"{OUT}/summary_analysis.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("### Phần A: oracle (cần ground-truth)")
print("| Tập ứng viên | mAP@0.5:0.95 | mAP từng ảnh | Tăng từng ảnh (CI 95%) |")
print("|---|---|---|---|")
for r in NOISE.itertuples():
    print(f"| {r.oracle} | {r.mAP50_95:.4f} | {r.per_image:.4f} | +{r.gain_per_image:.4f} [{r.ci95_low:.4f}; {r.ci95_high:.4f}] |")
print("\n### Phần B: phương pháp dùng được (không cần ground-truth)")
print("| Phương pháp | mAP@0.5 | mAP@0.5:0.95 | Δ cả tập | Δ từng ảnh (CI 95%) |")
print("|---|---|---|---|---|")
for r in FUSION.itertuples():
    print(f"| {r.method} | {r.mAP50:.4f} | {r.mAP50_95:.4f} | {r.delta_dataset:+.4f} | "
          f"{r.delta_per_image:+.4f} [{r.ci95_low:+.4f}; {r.ci95_high:+.4f}] |")
print(f"\nOutput trong {OUT}:", sorted(os.listdir(OUT)))
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
