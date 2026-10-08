"""Sinh notebook kfold_yolo.ipynb theo phương pháp trong paper (1 detector chung, train trên cả 3 phiên bản).
Chạy: python make_kfold_nb.py <out.ipynb>"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# K-fold YOLO11n + mAP từng ảnh → nhãn Selector (theo paper, mục 3)

Ba phiên bản của mỗi ảnh: `v0 = orig`, `v1 = scnet`, `v2 = semiuir` (cùng tên file, cùng nhãn).

**STAGE = "folds"** (≈ 4 giờ): tạo nhãn cho Selector
1. Chia 9.800 ảnh train thành 3 fold (seed 0).
2. Mỗi fold k: **một** YOLO11n train **30 epoch** trên **cả 3 phiên bản** của 2 fold còn lại (≈ 19.600 ảnh), dùng `last.pt` dự đoán **cả 3 phiên bản** của fold k (conf 0,001).
3. mAP@0.5:0.95 từng ảnh (pycocotools) cho từng phiên bản → nhãn
   `y = 0` nếu `max_k m_k − m_0 < 0,01`, ngược lại `y = argmax_k m_k`.

**STAGE = "full"** (≈ 3 giờ): đánh giá trên test
1. **Một** YOLO11n train **50 epoch** trên cả 3 phiên bản của toàn bộ train (≈ 29.400 ảnh).
2. Dự đoán mỗi phiên bản của 4.200 ảnh test **một lần**, lưu lại; mọi chiến lược (luôn gốc / luôn SCNet / luôn Semi-UIR / ngẫu nhiên / oracle, sau này là Selector) được chấm bằng cách **ghép dự đoán đã lưu**, không chạy lại detector.

Hai stage độc lập, có thể chạy song song ở 2 phiên bản notebook hoặc 2 tài khoản.

**Input cần thêm (Add Input):**
- Dataset `phanlvnminh/ruod640`
- Output notebook `phanlvnminh/computer-vision-implement-enhancers` (`enhanced_scnet.zip`, `enhanced_semiuir.zip`)

**Cách chạy:** Settings → GPU **T4 x2**, Internet **On**. Chạy thử `SMOKE = True` → Run All (≈ 10 phút). Sau đó `SMOKE = False`, chọn `STAGE`, **Save Version → Save & Run All**.

*Lưu ý:* COCOeval chỉ tính AP trên các lớp có ground-truth trong ảnh; ảnh không có ground-truth được gán mAP = 0 (theo paper).
""")

md("## 1. Cấu hình")
code(r'''
STAGE = "folds"            # "folds" (nhãn Selector, ~4 giờ) | "full" (model cuối + test, ~3 giờ)
FOLDS_TO_RUN = [0, 1, 2]   # có thể tách, ví dụ [0, 1] ở phiên này và [2] ở phiên khác
K = 3
SEED = 0
EPOCHS_FOLD = 30           # theo paper
EPOCHS_FULL = 50           # theo paper
DELTA = 0.01               # ưu tiên ảnh gốc khi tăng cường không giúp đáng kể
BATCH = 32
IMGSZ = 640
DEVICE = "0,1"             # 2×T4 (DDP). Chỉ có 1 GPU thì đặt "0"
SMOKE = False              # True = chạy thử (1 epoch, 100 ảnh/fold, 100 ảnh test)

VERSIONS = ["orig", "scnet", "semiuir"]     # thứ tự = chỉ số nhãn 0, 1, 2
WORK = "/kaggle/working"
OUT = f"{WORK}/kfold"
assert STAGE in ("folds", "full"), STAGE
if SMOKE:
    EPOCHS_FOLD = EPOCHS_FULL = 1
print(f"STAGE={STAGE} | folds={FOLDS_TO_RUN} | epochs fold/full={EPOCHS_FOLD}/{EPOCHS_FULL} | SMOKE={SMOKE}")
''')

md("## 2. Cài đặt thư viện")
code(r'''
!pip install -q ultralytics pycocotools
import os, glob, json, gzip, time, shutil, zipfile, hashlib, contextlib, io, subprocess
import numpy as np, pandas as pd, torch, ultralytics
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import KFold
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| GPU:", torch.cuda.device_count())
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout)
''')

md("## 3. Dữ liệu: 3 phiên bản ảnh")
code(r'''
found = glob.glob("/kaggle/input/**/orig/images/train", recursive=True)
assert found, "Không thấy ruod640/orig/images/train -> Add Input dataset ruod640"
R640 = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
ANN = {s: f"{R640}/ann_{s}640.json" for s in ("train", "test")}
for p in ANN.values():
    assert os.path.exists(p), f"Thiếu {p}"

ROOTS = {"orig": f"{R640}/orig"}
for v in ("scnet", "semiuir"):
    zips = glob.glob(f"/kaggle/input/**/enhanced_{v}.zip", recursive=True)
    assert zips, f"Không thấy enhanced_{v}.zip -> Add Input notebook computer-vision-implement-enhancers"
    ROOTS[v] = f"/tmp/data/{v}"              # /tmp: ghi được, không bị lưu vào output
    if not os.path.isdir(f"{ROOTS[v]}/images/test"):
        t = time.time()
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall("/tmp/data")
        print(f"Giải nén {os.path.basename(zips[0])} trong {time.time() - t:.0f}s")

EXPECT = {"train": 9800, "test": 4200}
for v, root in ROOTS.items():
    for s, n in EXPECT.items():
        ni = len(os.listdir(f"{root}/images/{s}")); nl = len(os.listdir(f"{root}/labels/{s}"))
        assert ni == n and nl == n, f"{v}/{s}: {ni} ảnh, {nl} nhãn, cần {n}"
print("ROOTS:", ROOTS)

with contextlib.redirect_stdout(io.StringIO()):
    GT = {s: COCO(ANN[s]) for s in ANN}
CATS = sorted(GT["train"].getCatIds())                       # chỉ số lớp YOLO -> category_id COCO
NAMES = [GT["train"].cats[c]["name"] for c in CATS]
NAME2ID = {s: {os.path.basename(im["file_name"]): im["id"] for im in GT[s].dataset["images"]} for s in GT}
for v, root in ROOTS.items():
    for s in GT:
        assert set(NAME2ID[s]) == set(os.listdir(f"{root}/images/{s}")), f"Tên ảnh {v}/{s} không khớp json"
print("Lớp:", NAMES)
''')

md("## 4. Chia fold (cố định)")
code(r'''
train_names = sorted(NAME2ID["train"])
fold_of = np.zeros(len(train_names), dtype=int)
for k, (_, va) in enumerate(KFold(K, shuffle=True, random_state=SEED).split(train_names)):
    fold_of[va] = k
FOLDS = pd.DataFrame({"file_name": train_names,
                      "image_id": [NAME2ID["train"][n] for n in train_names],
                      "fold": fold_of})
fold_hash = hashlib.md5(FOLDS.to_csv(index=False).encode()).hexdigest()[:12]
TEST_NAMES = sorted(NAME2ID["test"])
if SMOKE:
    FOLDS = FOLDS.groupby("fold").head(100).reset_index(drop=True)
    TEST_NAMES = TEST_NAMES[:100]

os.makedirs(f"{OUT}/weights", exist_ok=True)
FOLDS.to_csv(f"{OUT}/folds.csv", index=False)
print(FOLDS.fold.value_counts().sort_index().to_dict(), "| folds hash:", fold_hash, "(full run: ea44b385539c)")
''')

md("## 5. Hàm train / dự đoán / chấm điểm")
code(r'''
TIMES = {}

def paths(names, split, versions=VERSIONS):
    return [f"{ROOTS[v]}/images/{split}/{n}" for v in versions for n in names]

def train(name, tr_names, va_names, va_split, epochs):
    """Một YOLO11n train trên cả 3 phiên bản; trả về last.pt (không chọn best theo tập val -> không rò rỉ)."""
    dst = f"{OUT}/weights/{name}.pt"
    if os.path.exists(dst):
        print("Đã có", dst); return dst
    d = f"{WORK}/lists/{name}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/train.txt", "w") as f: f.write("\n".join(paths(tr_names, "train")) + "\n")
    with open(f"{d}/val.txt", "w") as f: f.write("\n".join(paths(va_names, va_split, ["orig"])) + "\n")
    with open(f"{d}/data.yaml", "w") as f:
        f.write(f"train: {d}/train.txt\nval: {d}/val.txt\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print(f"[{name}] train {len(tr_names)} ảnh × {len(VERSIONS)} phiên bản, {epochs} epoch")
    t = time.time()
    YOLO("yolo11n.pt").train(data=f"{d}/data.yaml", epochs=epochs, batch=BATCH, imgsz=IMGSZ, device=DEVICE,
                             seed=SEED, val=False, plots=False, workers=4, exist_ok=True,
                             project=f"{WORK}/runs", name=name)
    TIMES[name] = round(time.time() - t)
    run = f"{WORK}/runs/{name}"
    shutil.copy(f"{run}/weights/last.pt", dst)
    if os.path.exists(f"{run}/results.csv"):
        shutil.copy(f"{run}/results.csv", f"{OUT}/weights/{name}_results.csv")
    print(f"[{name}] xong sau {TIMES[name] / 3600:.2f} giờ -> {dst}")
    return dst

def predict(weights, names, split, version):
    """Dự đoán như khi đo mAP (conf=0.001, iou=0.7), trả về list detection COCO."""
    model = YOLO(weights)
    ps = paths(names, split, [version])
    dets = []
    for i in range(0, len(ps), 256):
        for r in model.predict(ps[i:i + 256], imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300,
                               device=0, verbose=False, stream=True):
            iid = NAME2ID[split][os.path.basename(r.path)]
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.tolist(), b.cls.int().tolist(), b.conf.tolist()):
                dets.append({"image_id": iid, "category_id": CATS[c],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(s, 5)})
    return dets

def quiet():
    return contextlib.redirect_stdout(io.StringIO())

def overall_map(split, dets, img_ids):
    if not dets: return {"map50_95": 0.0, "map50": 0.0}
    with quiet():
        E = COCOeval(GT[split], GT[split].loadRes([dict(d) for d in dets]), "bbox")
        E.params.imgIds = [int(i) for i in img_ids]
        E.evaluate(); E.accumulate(); E.summarize()
    return {"map50_95": float(E.stats[0]), "map50": float(E.stats[1])}

def per_image_map(split, dets, img_ids, tag=""):
    """mAP từng ảnh; ảnh không có ground-truth (stats = -1) được gán 0."""
    gt, img_ids = GT[split], [int(i) for i in img_ids]
    if not dets:
        return pd.DataFrame({"image_id": img_ids, "map50_95": 0.0, "map50": 0.0})
    with quiet():
        dt = gt.loadRes([dict(d) for d in dets])
    E1, rows, t = COCOeval(gt, dt, "bbox"), [], time.time()
    for j, i in enumerate(img_ids):
        E1.params.imgIds = [i]
        with quiet():
            E1.evaluate(); E1.accumulate(); E1.summarize()
        rows.append({"image_id": i, "map50_95": max(float(E1.stats[0]), 0.0), "map50": max(float(E1.stats[1]), 0.0)})
        if (j + 1) % 3000 == 0:
            print(f"  {tag} {j + 1}/{len(img_ids)} ảnh ({time.time() - t:.0f}s)")
    return pd.DataFrame(rows)

def label_of(m):
    """m: mảng (N, 3) mAP của orig/scnet/semiuir -> nhãn theo paper."""
    best = m.argmax(1)
    return np.where(m.max(1) - m[:, 0] < DELTA, 0, best)

def save_gz(obj, path):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)
''')

md("## 6. STAGE = folds: K-fold → mAP từng ảnh → nhãn Selector")
code(r'''
if STAGE == "folds":
    oof = {v: [] for v in VERSIONS}
    for k in FOLDS_TO_RUN:
        tr = FOLDS.loc[FOLDS.fold != k, "file_name"].tolist()
        va = FOLDS.loc[FOLDS.fold == k, "file_name"].tolist()
        print(f"===== fold {k}: train {len(tr)} ảnh/phiên bản, dự đoán {len(va)} ảnh × 3 =====")
        w = train(f"fold{k}", tr, va, "train", EPOCHS_FOLD)
        for v in VERSIONS:
            t = time.time(); d = predict(w, va, "train", v); oof[v] += d
            print(f"  fold {k} / {v}: {len(d)} detection ({time.time() - t:.0f}s)")
    tag = "" if FOLDS_TO_RUN == list(range(K)) else "_f" + "".join(map(str, FOLDS_TO_RUN))
    for v in VERSIONS:
        save_gz(oof[v], f"{OUT}/preds_oof_{v}{tag}.json.gz")

    sub = FOLDS[FOLDS.fold.isin(FOLDS_TO_RUN)].reset_index(drop=True)
    table = sub.copy()
    summary = {"stage": "folds", "folds": FOLDS_TO_RUN, "fold_hash": fold_hash, "epochs": EPOCHS_FOLD,
               "smoke": SMOKE, "ultralytics": ultralytics.__version__, "train_seconds": TIMES, "oof_overall": {}}
    for v in VERSIONS:
        t = time.time()
        df = per_image_map("train", oof[v], sub.image_id, tag=v)
        table[f"map50_95_{v}"] = df["map50_95"].values
        table[f"map50_{v}"] = df["map50"].values
        summary["oof_overall"][v] = overall_map("train", oof[v], sub.image_id)
        print(f"{v}: per-image xong ({time.time() - t:.0f}s), mAP tổng {summary['oof_overall'][v]}")

    M = table[[f"map50_95_{v}" for v in VERSIONS]].to_numpy()
    table["label"] = label_of(M)
    table["oracle_gain"] = M.max(1) - M[:, 0]
    table.to_csv(f"{OUT}/labels_train{tag}.csv", index=False)

    counts = table.label.value_counts().reindex(range(3), fill_value=0)
    summary["label_counts"] = {VERSIONS[i]: int(c) for i, c in counts.items()}
    summary["mean_per_image"] = {v: float(table[f"map50_95_{v}"].mean()) for v in VERSIONS}
    summary["mean_per_image"]["oracle"] = float(M.max(1).mean())
    with open(f"{OUT}/summary_folds{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
''')

md("## 7. STAGE = full: model cuối → dự đoán test → chấm các chiến lược cố định / ngẫu nhiên / oracle")
code(r'''
if STAGE == "full":
    w = train("full", FOLDS.file_name.tolist(), TEST_NAMES, "test", EPOCHS_FULL)
    test_ids = [NAME2ID["test"][n] for n in TEST_NAMES]
    dets = {}
    for v in VERSIONS:
        t = time.time(); dets[v] = predict(w, TEST_NAMES, "test", v)
        save_gz(dets[v], f"{OUT}/preds_test_{v}.json.gz")
        print(f"test / {v}: {len(dets[v])} detection ({time.time() - t:.0f}s)")

    table = pd.DataFrame({"file_name": TEST_NAMES, "image_id": test_ids})
    for v in VERSIONS:
        df = per_image_map("test", dets[v], test_ids, tag=v)
        table[f"map50_95_{v}"] = df["map50_95"].values
        table[f"map50_{v}"] = df["map50"].values
    M = table[[f"map50_95_{v}" for v in VERSIONS]].to_numpy()
    table["label"] = label_of(M)
    table.to_csv(f"{OUT}/per_image_test.csv", index=False)

    def compose(choice):
        """Ghép dự đoán đã lưu: ảnh i lấy detection của phiên bản choice[i]."""
        pick = dict(zip(test_ids, choice))
        return [d for v_i, v in enumerate(VERSIONS) for d in dets[v] if pick[d["image_id"]] == v_i]

    rng = np.random.default_rng(SEED)
    policies = {f"always_{v}": np.full(len(test_ids), i) for i, v in enumerate(VERSIONS)}
    policies["random"] = rng.integers(0, 3, len(test_ids))
    policies["oracle"] = table["label"].to_numpy()
    results = {}
    for p, choice in policies.items():
        results[p] = overall_map("test", compose(choice), test_ids)
        results[p]["mean_per_image"] = float(M[np.arange(len(choice)), choice].mean())
        print(f"{p:16s} {results[p]}")
    summary = {"stage": "full", "epochs": EPOCHS_FULL, "smoke": SMOKE, "ultralytics": ultralytics.__version__,
               "train_seconds": TIMES, "policies": results,
               "oracle_label_counts": {VERSIONS[i]: int((table.label == i).sum()) for i in range(3)}}
    with open(f"{OUT}/summary_full.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
''')

md("## 8. Dọn output")
code(r'''
for p in [f"{WORK}/runs", f"{WORK}/lists", f"{WORK}/yolo11n.pt", "/tmp/data"]:
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
    elif os.path.exists(p): os.remove(p)
for f in sorted(glob.glob(f"{OUT}/**", recursive=True)):
    if os.path.isfile(f):
        print(f"{os.path.getsize(f) / 2**20:8.1f} MiB  {f}")
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
