"""Sinh notebook domain_detectors.ipynb: so sánh detector 1 miền (thiết kế của Awad et al.) với detector đa miền.
Chạy: python make_domain_nb.py <out.ipynb>"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Detector một miền (Awad et al.) so với detector đa miền: tăng cường ảnh nên dùng lúc train hay lúc suy luận?

Awad et al. train **mỗi miền ảnh một detector riêng** ("domain detector") và kết luận detector ảnh gốc luôn thắng. Nhóm đề xuất **một detector train chung trên nhiều miền** (gốc + SCNet + Semi-UIR), khi suy luận chỉ dùng ảnh gốc.

| Detector | Dữ liệu train | Nguồn |
|---|---|---|
| **A** chỉ ảnh gốc | 9.800 ảnh gốc, 50 epoch | notebook baseline tuần 1 (`runs/baseline/weights/last.pt`) |
| **B** chỉ Semi-UIR | 9.800 ảnh Semi-UIR, 50 epoch | **train trong notebook này** |
| **C** đa miền (của nhóm) | 3 × 9.800 ảnh, 50 epoch | notebook K-fold, version full (`weights/full.pt`) |
| A150 *(tùy chọn)* | 9.800 ảnh gốc, **150 epoch** = cùng số lượt ảnh với C | train trong notebook này nếu `TRAIN_A150 = True` |

Cả bốn dùng cùng cấu hình: YOLO11n, `imgsz=640`, `batch=32`, `seed=0`, 2 GPU, dùng **`last.pt`** (không chọn checkpoint theo tập test).

Mỗi detector dự đoán cả 3 phiên bản của 4.200 ảnh test (conf 0,001, IoU 0,7) → ma trận mAP (pycocotools) + chênh lệch mAP từng ảnh so với **A trên ảnh gốc** kèm bootstrap CI 95%.

A150 trả lời câu hỏi: C thắng A là nhờ **đa dạng miền** hay chỉ nhờ **được nhìn nhiều ảnh hơn**?

**Input (Add Input):**
- Dataset `phanlvnminh/ruod640` (**hoặc**, nếu dataset hay lỗi mount, output notebook `pack_ruod640` chứa `ruod640_orig.zip`)
- Output notebook `phanlvnminh/computer-vision-implement-enhancers` (`enhanced_scnet.zip`, `enhanced_semiuir.zip`)
- Output notebook `phanlvnminh/notebook-ruod640` (baseline tuần 1 → detector A)
- Output notebook K-fold **version full (v6)** (`weights/full.pt` → detector C)

**Chạy lại để thêm A150 mà không train lại B:** Add Input output của chính notebook này (version đã train B). Notebook tự tìm `domain/weights/B_semiuir.pt` và dùng lại.

**Cách chạy:** GPU **T4 x2**, Internet **On**. `SMOKE = True` chạy thử (≈ 10 phút) → `SMOKE = False` → Save Version (≈ 2 giờ; thêm ≈ 3,5 giờ nếu bật `TRAIN_A150`).
""")

md("## 1. Cấu hình")
code(r'''
TRAIN_A150 = True           # detector A 150 epoch (≈ 3 giờ); False để bỏ qua
EPOCHS = 50
BATCH = 32
IMGSZ = 640
SEED = 0
DEVICE = "0,1"              # chỉ có 1 GPU thì đặt "0"
N_BOOT = 1000
SMOKE = False               # True = 1 epoch, 100 ảnh train, 100 ảnh test

VERSIONS = ["orig", "scnet", "semiuir"]
import os
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
OUT = f"{WORK}/domain"
os.makedirs(f"{OUT}/weights", exist_ok=True)
if SMOKE:
    N_BOOT = 50
print(f"TRAIN_A150={TRAIN_A150} | SMOKE={SMOKE}")
''')

md("## 2. Thư viện")
code(r'''
import subprocess, sys
for mod, pkg in [("ultralytics", "ultralytics"), ("pycocotools", "pycocotools")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import glob, json, gzip, time, shutil, zipfile, contextlib, io
import numpy as np, pandas as pd, torch, ultralytics
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
DEV0 = 0 if torch.cuda.is_available() else "cpu"
print("ultralytics", ultralytics.__version__, "| GPU:", torch.cuda.device_count())
''')

md("## 3. Dữ liệu và trọng số A, C")
code(r'''
def find_one(pattern, hint):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    assert hits, f"Không thấy {pattern} trong {INPUT} -> {hint}"
    return hits[0]

found = glob.glob(f"{INPUT}/**/orig/images/train", recursive=True)
if found:                                   # dataset ruod640 mount bình thường
    R640 = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
else:                                       # dự phòng: bản nén ruod640_orig.zip (notebook pack_ruod640)
    z = find_one("ruod640_orig.zip", "Add Input dataset ruod640, hoặc output notebook pack_ruod640 (ruod640_orig.zip)")
    R640 = "/tmp/data/ruod640"
    if not os.path.isdir(f"{R640}/orig/images/test"):
        t = time.time()
        with zipfile.ZipFile(z) as f:
            f.extractall(R640)
        print(f"Giải nén {os.path.basename(z)} trong {time.time() - t:.0f}s")
ROOTS = {"orig": f"{R640}/orig"}
for v in ("scnet", "semiuir"):
    z = find_one(f"enhanced_{v}.zip", "Add Input output notebook computer-vision-implement-enhancers")
    ROOTS[v] = f"/tmp/data/{v}"
    if not os.path.isdir(f"{ROOTS[v]}/images/test"):
        t = time.time()
        with zipfile.ZipFile(z) as f:
            f.extractall("/tmp/data")
        print(f"Giải nén {os.path.basename(z)} trong {time.time() - t:.0f}s")

W = {"A_orig": find_one("baseline/weights/last.pt", "Add Input output notebook notebook-ruod640 (baseline tuần 1)"),
     "C_mixed": find_one("weights/full.pt", "Add Input output notebook K-fold version full (v6)")}
for k, p in W.items():
    print(f"{k}: {p}")

with contextlib.redirect_stdout(io.StringIO()):
    GT = COCO(f"{R640}/ann_test640.json")
    GT_TRAIN = COCO(f"{R640}/ann_train640.json")
CATS = sorted(GT.getCatIds())
NAMES = [GT.cats[c]["name"] for c in CATS]
TEST = sorted(os.path.basename(im["file_name"]) for im in GT.dataset["images"])
TRAIN = sorted(os.path.basename(im["file_name"]) for im in GT_TRAIN.dataset["images"])
NAME2ID = {os.path.basename(im["file_name"]): im["id"] for im in GT.dataset["images"]}
for v, r in ROOTS.items():
    for s, names in (("train", TRAIN), ("test", TEST)):
        assert set(os.listdir(f"{r}/images/{s}")) == set(names), f"Ảnh {v}/{s} không khớp json"
        assert len(os.listdir(f"{r}/labels/{s}")) == len(names), f"Thiếu nhãn {v}/{s}"
if SMOKE:
    TRAIN, TEST = TRAIN[:100], TEST[:100]
IDS = [NAME2ID[n] for n in TEST]
print(f"train {len(TRAIN)} | test {len(TEST)} | lớp {NAMES}")
''')

md("## 4. Train detector B (chỉ Semi-UIR) và A150 (tùy chọn)")
code(r'''
TIMES = {}

def train(name, version, epochs):
    dst = f"{OUT}/weights/{name}.pt"
    if os.path.exists(dst):
        print("Đã có", dst); return dst
    prev = sorted(glob.glob(f"{INPUT}/**/domain/weights/{name}.pt", recursive=True))
    if prev and not SMOKE:                  # dùng lại detector đã train ở lần chạy trước (Add Input output notebook này)
        shutil.copy(prev[0], dst)
        for extra in glob.glob(prev[0].replace(".pt", "_results.csv")):
            shutil.copy(extra, f"{OUT}/weights/")
        print(f"[{name}] dùng lại {prev[0]}"); return dst
    d = f"{WORK}/lists/{name}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/train.txt", "w") as f:
        f.write("\n".join(f"{ROOTS[version]}/images/train/{n}" for n in TRAIN) + "\n")
    with open(f"{d}/val.txt", "w") as f:
        f.write("\n".join(f"{ROOTS[version]}/images/test/{n}" for n in TEST[:50]) + "\n")
    with open(f"{d}/data.yaml", "w") as f:
        f.write(f"train: {d}/train.txt\nval: {d}/val.txt\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print(f"[{name}] train {len(TRAIN)} ảnh {version}, {epochs} epoch")
    t = time.time()
    YOLO("yolo11n.pt").train(data=f"{d}/data.yaml", epochs=epochs, batch=BATCH, imgsz=IMGSZ, device=DEVICE,
                             seed=SEED, val=False, plots=False, workers=4, exist_ok=True,
                             project=f"{WORK}/runs", name=name)
    TIMES[name] = round(time.time() - t)
    shutil.copy(f"{WORK}/runs/{name}/weights/last.pt", dst)
    if os.path.exists(f"{WORK}/runs/{name}/results.csv"):
        shutil.copy(f"{WORK}/runs/{name}/results.csv", f"{OUT}/weights/{name}_results.csv")
    print(f"[{name}] xong sau {TIMES[name] / 3600:.2f} giờ")
    return dst

W["B_semiuir"] = train("B_semiuir", "semiuir", 1 if SMOKE else EPOCHS)
if TRAIN_A150:
    W["A150_orig"] = train("A150_orig", "orig", 1 if SMOKE else 3 * EPOCHS)
DETS_ORDER = ["A_orig", "B_semiuir", "C_mixed"] + (["A150_orig"] if TRAIN_A150 else [])
''')

md("## 5. Dự đoán: mỗi detector × mỗi phiên bản ảnh test")
code(r'''
def predict(weights, version):
    model, dets = YOLO(weights), []
    ps = [f"{ROOTS[version]}/images/test/{n}" for n in TEST]
    for i in range(0, len(ps), 256):
        for r in model.predict(ps[i:i + 256], imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300,
                               device=DEV0, verbose=False, stream=True):
            iid = NAME2ID[os.path.basename(r.path)]
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                dets.append({"image_id": iid, "category_id": CATS[c],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(s, 5)})
    return dets

P = {}
for d in DETS_ORDER:
    for v in VERSIONS:
        t = time.time()
        P[(d, v)] = predict(W[d], v)
        if d != "C_mixed":
            with gzip.open(f"{OUT}/preds_test_{d}_{v}.json.gz", "wt") as f:
                json.dump(P[(d, v)], f)
        print(f"  {d:10s} × {v:8s}: {len(P[(d, v)])} detection ({time.time() - t:.0f}s)")
''')

md("## 6. Chấm điểm: ma trận mAP, AP theo lớp, bootstrap so với A trên ảnh gốc")
code(r'''
def quiet():
    return contextlib.redirect_stdout(io.StringIO())

def evaluate(dets):
    with quiet():
        E = COCOeval(GT, GT.loadRes([dict(x) for x in dets]), "bbox")
        E.params.imgIds = IDS
        E.evaluate(); E.accumulate(); E.summarize()
    Pr = E.eval["precision"][:, :, :, 0, 2]
    per_cls = [float(Pr[:, :, k][Pr[:, :, k] > -1].mean()) if (Pr[:, :, k] > -1).any() else float("nan")
               for k in range(Pr.shape[2])]
    return float(E.stats[0]), float(E.stats[1]), per_cls

def per_image(dets):
    with quiet():
        E = COCOeval(GT, GT.loadRes([dict(x) for x in dets]), "bbox")
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

RES, PI = {}, {}
for key, dets in P.items():
    t = time.time()
    m95, m50, pc = evaluate(dets)
    PI[key] = per_image(dets)
    RES[key] = {"mAP50_95": m95, "mAP50": m50, "per_image": float(PI[key].mean()), "per_class": pc}
    print(f"  {key[0]:10s} × {key[1]:8s}: mAP50:95 {m95:.4f} | mAP50 {m50:.4f} ({time.time() - t:.0f}s)")

ref_file = glob.glob(f"{INPUT}/**/summary_full.json", recursive=True)
if ref_file and not SMOKE:
    ref = json.load(open(ref_file[0]))["policies"]["always_orig"]["map50_95"]
    print(f"Kiểm tra: C × orig = {RES[('C_mixed', 'orig')]['mAP50_95']:.4f} | summary_full = {ref:.4f}")

REF = PI[("A_orig", "orig")]
rows = []
for (d, v), r in RES.items():
    mean, lo, hi = boot(PI[(d, v)] - REF)
    rows.append({"detector": d, "test": v, "mAP50": r["mAP50"], "mAP50_95": r["mAP50_95"], "per_image": r["per_image"],
                 "delta_vs_A_orig": r["mAP50_95"] - RES[("A_orig", "orig")]["mAP50_95"],
                 "delta_per_image": mean, "ci95_low": lo, "ci95_high": hi})
TABLE = pd.DataFrame(rows)
TABLE.to_csv(f"{OUT}/domain_matrix.csv", index=False)
pd.set_option("display.width", 200)
print(TABLE.round(4).to_string(index=False))

MAT = TABLE.pivot(index="detector", columns="test", values="mAP50_95").reindex(index=DETS_ORDER, columns=VERSIONS)
print("\nMa trận mAP@0.5:0.95 (hàng = detector, cột = phiên bản ảnh test):")
print(MAT.round(4))

PC = pd.DataFrame({f"{d}×orig": RES[(d, "orig")]["per_class"] for d in DETS_ORDER}, index=NAMES)
PC.to_csv(f"{OUT}/per_class_orig.csv")
print("\nAP@0.5:0.95 theo lớp trên ảnh test gốc:"); print(PC.round(4))
''')

md("## 7. Tóm tắt")
code(r'''
summary = {"smoke": SMOKE, "epochs": EPOCHS, "train_a150": TRAIN_A150, "train_seconds": TIMES,
           "weights": W, "matrix": TABLE.drop(columns=[]).to_dict("records")}
with open(f"{OUT}/summary_domain.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

LABEL = {"A_orig": "A: chỉ ảnh gốc", "B_semiuir": "B: chỉ Semi-UIR", "C_mixed": "**C: đa miền (nhóm)**",
         "A150_orig": "A150: chỉ ảnh gốc, 150 epoch"}
print("| Detector | Test gốc | Test SCNet | Test Semi-UIR |")
print("|---|---|---|---|")
for d in DETS_ORDER:
    print(f"| {LABEL[d]} | " + " | ".join(f"{MAT.loc[d, v]:.4f}" for v in VERSIONS) + " |")
print("\n| So với A trên ảnh gốc | Δ mAP@0.5:0.95 | Δ từng ảnh [CI 95%] |")
print("|---|---|---|")
for r in TABLE.itertuples():
    if r.test == "orig" and r.detector != "A_orig" or (r.detector == "B_semiuir" and r.test == "semiuir"):
        print(f"| {LABEL[r.detector]} × {r.test} | {r.delta_vs_A_orig:+.4f} | {r.delta_per_image:+.4f} [{r.ci95_low:+.4f}; {r.ci95_high:+.4f}] |")

for p in [f"{WORK}/runs", f"{WORK}/lists", f"{WORK}/yolo11n.pt", "/tmp/data"]:  # /tmp/data gồm cả ruod640 giải nén
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
    elif os.path.exists(p): os.remove(p)
print("\nOutput:", sorted(os.listdir(OUT)))
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
