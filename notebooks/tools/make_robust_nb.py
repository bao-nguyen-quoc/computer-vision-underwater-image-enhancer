"""Sinh notebook robustness_unseen.ipynb: detector đa miền có bền trên các kiểu suy giảm CHƯA THẤY khi train không?
Chạy: python make_robust_nb.py <out.ipynb>"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Độ bền trên suy giảm chưa thấy: detector đa miền có tổng quát hoá không?

Notebook `domain_detectors` cho thấy detector đa miền **C** (train trộn gốc + SCNet + Semi-UIR) ngang detector chỉ học ảnh gốc **A150** (cùng số lượt ảnh) trên ảnh gốc, nhưng bền hơn khi ảnh test là SCNet hay Semi-UIR. Câu hỏi còn lại: C chỉ **quen** hai miền đã học, hay thật sự **bền với thay đổi miền nói chung**?

Ở đây ta tạo các kiểu suy giảm **không detector nào được thấy khi train**, áp lên 4.200 ảnh test gốc (nhãn hộp giữ nguyên vì chỉ đổi màu/độ sáng, không đổi hình học). Mỗi kiểu có 3 mức nặng:

| Kiểu | Mô phỏng | Mức 1 → 3 |
|---|---|---|
| `dark` | Thiếu sáng ở độ sâu lớn: giảm độ sáng, gamma, nhiễu cảm biến | sáng còn 45% → 18% |
| `turbid` | Nước đục: tán xạ ngược phủ màn xanh xám, I = J·t + B·(1−t), kèm mờ | t = 0,60 → 0,30 |
| `cast` | Ám màu mạnh: mất kênh đỏ theo độ sâu, t_c = exp(−β_c·d) | d = 2 → 7 m |
| `blur` | Mờ chuyển động + nhiễu + nén JPEG mạnh | kernel 5 → 15 px |

Mô hình `turbid` và `cast` là dạng đơn giản của mô hình tạo ảnh dưới nước (Jaffe–McGlamery, Akkaynak & Treibitz).

**Không train gì cả**: chỉ dùng lại trọng số đã có. So sánh chính: **C so với A150** (cùng chi phí train). A (50 epoch) và B (chỉ Semi-UIR) để tham khảo.

**Chỉ số:**
- mAP@0.5:0.95 (pycocotools) cho mỗi detector × điều kiện.
- **Tỉ lệ giữ lại** = mAP điều kiện / mAP ảnh sạch.
- Chênh lệch từng ảnh C − A150 và **chênh lệch mức tụt** (C tụt ít hơn A150 bao nhiêu), kèm bootstrap CI 95%.

**Input (Add Input):**
- Dataset `phanlvnminh/ruod640` (**hoặc** output notebook `pack_ruod640` chứa `ruod640_orig.zip`)
- Output notebook `phanlvnminh/notebook-ruod640` (baseline tuần 1 → detector A)
- Output notebook K-fold **version full (v6)** (`weights/full.pt` → detector C)
- Output notebook `domain-detectors` **version có A150** (`domain/weights/A150_orig.pt`, `B_semiuir.pt`)

**Cách chạy:** GPU **T4** (1 GPU là đủ), Internet **On**. `SMOKE = True` chạy thử (≈ 5 phút) → `SMOKE = False` → Save Version (≈ 1–1,5 giờ).
""")

md("## 1. Cấu hình")
code(r'''
SMOKE = False               # True = 100 ảnh test, bootstrap 50 lần
IMGSZ = 640
SEED = 0
N_BOOT = 1000
BATCH = 64                  # số ảnh suy giảm tạo và dự đoán mỗi lượt
FAMILIES = ["dark", "turbid", "cast", "blur"]
LEVELS = [1, 2, 3]

import os
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
OUT = f"{WORK}/robust"
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    N_BOOT = 50
CONDS = ["clean"] + [f"{f}{l}" for f in FAMILIES for l in LEVELS]
print(f"SMOKE={SMOKE} | {len(CONDS)} điều kiện: {CONDS}")
''')

md("## 2. Thư viện")
code(r'''
import subprocess, sys
for mod, pkg in [("ultralytics", "ultralytics"), ("pycocotools", "pycocotools")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import glob, json, time, zipfile, contextlib, io
import numpy as np, pandas as pd, cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
''')

md("## 3. Hàm tạo suy giảm")
DEGRADE = r'''
# Mọi phép toán trên ảnh RGB float32 trong [0, 1]; ảnh vào/ra là BGR uint8 (chuẩn OpenCV).
def _noise(x, sigma, rng):
    return x + rng.normal(0.0, sigma, x.shape).astype(np.float32)

def deg_dark(x, l, rng):            # thiếu sáng: gamma + giảm độ sáng + nhiễu cảm biến
    gamma = {1: 1.3, 2: 1.6, 3: 2.0}[l]
    light = {1: 0.45, 2: 0.30, 3: 0.18}[l]
    sigma = {1: 0.010, 2: 0.020, 3: 0.030}[l]
    return _noise(np.power(x, gamma) * light, sigma, rng)

VEIL = np.array([0.35, 0.55, 0.50], np.float32)          # màu tán xạ ngược (R, G, B): xanh xám
def deg_turbid(x, l, rng):          # nước đục: I = J·t + B·(1 − t), rồi mờ nhẹ
    t = {1: 0.60, 2: 0.45, 3: 0.30}[l]
    s = {1: 0.8, 2: 1.5, 3: 2.5}[l]
    y = x * t + VEIL * (1.0 - t)
    return cv2.GaussianBlur(y, (0, 0), s)

BETA = np.array([0.40, 0.10, 0.07], np.float32)          # hệ số suy hao (1/m) cho R, G, B: đỏ mất nhanh nhất
BACK = np.array([0.05, 0.35, 0.45], np.float32)          # tán xạ ngược xanh lam
def deg_cast(x, l, rng):            # ám màu theo độ sâu: t_c = exp(−β_c·d)
    d = {1: 2.0, 2: 4.0, 3: 7.0}[l]
    t = np.exp(-BETA * d)
    return x * t + 0.5 * BACK * (1.0 - t)

def deg_blur(x, l, rng):            # mờ chuyển động (hướng ngẫu nhiên) + nhiễu + nén JPEG
    k = {1: 5, 2: 9, 3: 15}[l]
    sigma = {1: 0.02, 2: 0.04, 3: 0.06}[l]
    quality = {1: 60, 2: 40, 3: 25}[l]
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k)); ker /= max(ker.sum(), 1e-6)
    y = _noise(cv2.filter2D(x, -1, ker), sigma, rng)
    bgr = (np.clip(y, 0, 1)[:, :, ::-1] * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0

DEG = {"dark": deg_dark, "turbid": deg_turbid, "cast": deg_cast, "blur": deg_blur}

def degrade(bgr, cond, img_idx):
    """Ảnh BGR uint8 -> ảnh suy giảm BGR uint8. Cố định seed theo (điều kiện, ảnh) để chạy lại ra y hệt."""
    if cond == "clean":
        return bgr
    fam, lvl = cond[:-1], int(cond[-1])
    rng = np.random.default_rng([SEED, CONDS.index(cond), img_idx])
    x = bgr[:, :, ::-1].astype(np.float32) / 255.0
    y = DEG[fam](x, lvl, rng)
    return (np.clip(y, 0, 1)[:, :, ::-1] * 255 + 0.5).astype(np.uint8)
'''
code(DEGRADE)

md("## 4. Dữ liệu và trọng số")
code(r'''
def find_one(pattern, hint, required=True):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    if not hits:
        assert not required, f"Không thấy {pattern} trong {INPUT} -> {hint}"
        print(f"BỎ QUA: không thấy {pattern} ({hint})")
        return None
    return hits[0]

found = glob.glob(f"{INPUT}/**/orig/images/test", recursive=True)
if found:                                   # dataset ruod640 mount bình thường
    R640 = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
else:                                       # dự phòng: ruod640_orig.zip (notebook pack_ruod640)
    z = find_one("ruod640_orig.zip", "Add Input dataset ruod640, hoặc output notebook pack_ruod640")
    R640 = "/tmp/data/ruod640"
    if not os.path.isdir(f"{R640}/orig/images/test"):
        t = time.time()
        with zipfile.ZipFile(z) as f:
            f.extractall(R640)
        print(f"Giải nén {os.path.basename(z)} trong {time.time() - t:.0f}s")
TEST_DIR = f"{R640}/orig/images/test"

W = {"A150_orig": find_one("domain/weights/A150_orig.pt", "Add Input output notebook domain-detectors (version có A150)"),
     "C_mixed": find_one("weights/full.pt", "Add Input output notebook K-fold version full (v6)"),
     "A_orig": find_one("baseline/weights/last.pt", "output notebook notebook-ruod640", required=False),
     "B_semiuir": find_one("domain/weights/B_semiuir.pt", "output notebook domain-detectors", required=False)}
W = {k: v for k, v in W.items() if v}
DETS = [d for d in ["A_orig", "A150_orig", "B_semiuir", "C_mixed"] if d in W]
for d in DETS:
    print(f"{d}: {W[d]}")

with contextlib.redirect_stdout(io.StringIO()):
    GT = COCO(f"{R640}/ann_test640.json")
CATS = sorted(GT.getCatIds())
NAME2ID = {os.path.basename(im["file_name"]): im["id"] for im in GT.dataset["images"]}
TEST = sorted(NAME2ID)
assert set(os.listdir(TEST_DIR)) >= set(TEST), "Ảnh test không khớp ann_test640.json"
if SMOKE:
    TEST = TEST[:100]
IDS = [NAME2ID[n] for n in TEST]
print(f"test {len(TEST)} ảnh | {len(CATS)} lớp | detector: {DETS}")
''')

md("## 5. Ảnh minh hoạ các kiểu suy giảm")
code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ex = cv2.imread(f"{TEST_DIR}/{TEST[3]}")
fig, ax = plt.subplots(len(FAMILIES), 4, figsize=(16, 3.2 * len(FAMILIES)))
for r, fam in enumerate(FAMILIES):
    for c, cond in enumerate(["clean"] + [f"{fam}{l}" for l in LEVELS]):
        ax[r, c].imshow(degrade(ex, cond, 3)[:, :, ::-1]); ax[r, c].axis("off")
        ax[r, c].set_title("ảnh gốc" if cond == "clean" else cond, fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/degradation_examples.jpg", dpi=80); plt.show()
''')

md("## 6. Dự đoán: mỗi điều kiện tạo ảnh một lần, chạy đủ các detector")
code(r'''
import torch
from ultralytics import YOLO
DEV0 = 0 if torch.cuda.is_available() else "cpu"
MODELS = {d: YOLO(W[d]) for d in DETS}
P = {(d, c): [] for d in DETS for c in CONDS}

for cond in CONDS:
    t = time.time()
    for i in range(0, len(TEST), BATCH):
        names = TEST[i:i + BATCH]
        imgs = [degrade(cv2.imread(f"{TEST_DIR}/{n}"), cond, i + j) for j, n in enumerate(names)]
        for d, model in MODELS.items():
            res = model.predict(imgs, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, device=DEV0, verbose=False)
            for n, r in zip(names, res):
                iid = NAME2ID[n]
                for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    P[(d, cond)].append({"image_id": iid, "category_id": CATS[c],
                                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                         "score": round(s, 5)})
    print(f"{cond:8s}: " + " | ".join(f"{d} {len(P[(d, cond)])}" for d in DETS) + f"  ({time.time() - t:.0f}s)")
''')

md(r"""
## 7. Chấm điểm

`per_image_ap` tính mAP@0.5:0.95 **từng ảnh** từ một lần `COCOeval.evaluate()` duy nhất (thay vì gọi pycocotools 4.200 lần), theo đúng quy tắc của pycocotools: nội suy 101 điểm recall, bỏ các lớp ảnh đó không có ground-truth, ảnh không có lớp hợp lệ tính 0. Ô kiểm tra bên dưới so với cách chậm trên 30 ảnh đầu.
""")
EVAL = r'''
def quiet():
    return contextlib.redirect_stdout(io.StringIO())

REC = np.linspace(0, 1, 101)

EMPTY = {"image_id": -1, "category_id": -1, "bbox": [0, 0, 1, 1], "score": 0.0}   # chỗ giữ khi không có detection

def run_eval(dets, img_ids):
    with quiet():
        if not dets:                        # loadRes không nhận danh sách rỗng
            GT.dataset["images"].append({"id": -1}); GT.createIndex()
            dt = GT.loadRes([dict(EMPTY)])
            GT.dataset["images"].pop(); GT.createIndex()
        else:
            dt = GT.loadRes([dict(x) for x in dets])
        E = COCOeval(GT, dt, "bbox")
        E.params.imgIds = list(img_ids)
        E.evaluate()
    return E

def _ap_one(dt_scores, dt_match, dt_ignore, n_gt):
    """AP theo quy tắc accumulate() của pycocotools cho một tập detection (mảng T × D)."""
    order = np.argsort(-dt_scores, kind="mergesort")
    m, ig = dt_match[:, order], dt_ignore[:, order]
    tps = np.logical_and(m, ~ig).cumsum(1).astype(float)
    fps = np.logical_and(~m, ~ig).cumsum(1).astype(float)
    aps = np.zeros(m.shape[0])
    for t in range(m.shape[0]):
        tp, fp = tps[t], fps[t]
        if tp.size == 0:
            continue
        rc = tp / n_gt
        pr = tp / np.maximum(tp + fp, np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC, side="left")
        q = np.zeros(len(REC))
        ok = idx < len(pr)
        q[ok] = pr[idx[ok]]
        aps[t] = q.mean()
    return aps

def per_image_ap(E, img_ids):
    """mAP@0.5:0.95 từng ảnh (vùng 'all', maxDets 100) từ kết quả E.evaluate()."""
    p = E.params; T = len(p.iouThrs); maxdet = p.maxDets[-1]
    by_img = {}
    for e in E.evalImgs:
        if e is None or e["aRng"] != p.areaRng[0]:
            continue
        by_img.setdefault(e["image_id"], []).append(e)
    out = np.zeros(len(img_ids))
    for j, iid in enumerate(img_ids):
        vals = []
        for e in by_img.get(iid, []):
            n_gt = int(np.count_nonzero(~np.asarray(e["gtIgnore"], bool)))
            if n_gt == 0:
                continue
            sc = np.asarray(e["dtScores"][:maxdet], float)
            if sc.size == 0:
                vals.append(np.zeros(T)); continue
            dm = np.asarray(e["dtMatches"])[:, :maxdet] > 0
            di = np.asarray(e["dtIgnore"])[:, :maxdet].astype(bool)
            vals.append(_ap_one(sc, dm, di, n_gt))
        out[j] = float(np.mean(vals)) if vals else 0.0
    return out

def dataset_map(E):
    with quiet():
        E.accumulate(); E.summarize()
    return float(E.stats[0]), float(E.stats[1])

def slow_per_image(dets, img_ids):          # cách cũ (gọi pycocotools từng ảnh) để đối chiếu
    out = []
    for i in img_ids:
        E = run_eval(dets, [i])
        with quiet():
            E.accumulate(); E.summarize()
        out.append(max(float(E.stats[0]), 0.0))
    return np.array(out)

def boot(diff):
    r = np.random.default_rng(SEED); n = len(diff)
    m = np.array([diff[r.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(diff.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
'''
code(EVAL)
code(r'''
# Đối chiếu cách tính nhanh với cách chậm trên 30 ảnh đầu (C_mixed, ảnh sạch)
chk = IDS[:30]
fast = per_image_ap(run_eval(P[("C_mixed", "clean")], chk), chk)
slow = slow_per_image(P[("C_mixed", "clean")], chk)
print("chênh lớn nhất:", float(np.abs(fast - slow).max()))
assert np.allclose(fast, slow, atol=1e-6), "per_image_ap lệch pycocotools"
''')
code(r'''
RES, PI = {}, {}
for (d, c), dets in P.items():
    t = time.time()
    E = run_eval(dets, IDS)
    PI[(d, c)] = per_image_ap(E, IDS)
    RES[(d, c)] = dataset_map(E)
    print(f"  {d:10s} × {c:8s}: mAP50:95 {RES[(d, c)][0]:.4f} | mAP50 {RES[(d, c)][1]:.4f} ({time.time() - t:.0f}s)")

rows = []
for (d, c), (m95, m50) in RES.items():
    clean = RES[(d, "clean")][0]
    rows.append({"detector": d, "condition": c, "family": "clean" if c == "clean" else c[:-1],
                 "level": 0 if c == "clean" else int(c[-1]), "mAP50": m50, "mAP50_95": m95,
                 "drop": m95 - clean, "retention": m95 / clean if clean > 0 else float("nan")})
MATRIX = pd.DataFrame(rows)
MATRIX.to_csv(f"{OUT}/robust_matrix.csv", index=False)
pd.set_option("display.width", 220)
TAB = MATRIX.pivot(index="condition", columns="detector", values="mAP50_95").reindex(index=CONDS, columns=DETS)
print("\nmAP@0.5:0.95 (hàng = điều kiện test, cột = detector):"); print(TAB.round(4))
RET = MATRIX.pivot(index="condition", columns="detector", values="retention").reindex(index=CONDS, columns=DETS)
print("\nTỉ lệ giữ lại so với ảnh sạch:"); print(RET.round(3))
''')

md(r"""
## 8. So sánh chính: C (đa miền) với A150 (chỉ ảnh gốc, cùng chi phí train)

- `delta`: mAP từng ảnh C − A150 trong cùng điều kiện.
- `drop_gap`: (mức tụt của C) − (mức tụt của A150) so với ảnh sạch, tính trên từng ảnh. **Dương = C tụt ít hơn = bền hơn.**
""")
code(r'''
assert "A150_orig" in DETS and "C_mixed" in DETS, "Cần cả A150_orig và C_mixed"
pair = []
for c in CONDS:
    d_mean, d_lo, d_hi = boot(PI[("C_mixed", c)] - PI[("A150_orig", c)])
    drop_c = PI[("C_mixed", c)] - PI[("C_mixed", "clean")]
    drop_a = PI[("A150_orig", c)] - PI[("A150_orig", "clean")]
    g_mean, g_lo, g_hi = boot(drop_c - drop_a)
    pair.append({"condition": c, "map_C": RES[("C_mixed", c)][0], "map_A150": RES[("A150_orig", c)][0],
                 "delta_dataset": RES[("C_mixed", c)][0] - RES[("A150_orig", c)][0],
                 "delta": d_mean, "delta_lo": d_lo, "delta_hi": d_hi,
                 "drop_gap": g_mean, "drop_gap_lo": g_lo, "drop_gap_hi": g_hi,
                 "C_wins": d_lo > 0, "A150_wins": d_hi < 0})
PAIR = pd.DataFrame(pair)
PAIR.to_csv(f"{OUT}/paired_C_vs_A150.csv", index=False)
print(PAIR.round(4).to_string(index=False))

deg = PAIR[PAIR.condition != "clean"]
print(f"\nC thắng có ý nghĩa ở {int(deg.C_wins.sum())}/{len(deg)} điều kiện suy giảm, "
      f"A150 thắng ở {int(deg.A150_wins.sum())}/{len(deg)}.")
print(f"Tỉ lệ giữ lại trung bình trên 12 điều kiện: C {RET.loc[CONDS[1:], 'C_mixed'].mean():.3f} | "
      f"A150 {RET.loc[CONDS[1:], 'A150_orig'].mean():.3f}")
''')

md("## 9. Biểu đồ mAP theo mức suy giảm")
code(r'''
STYLE = {"A_orig": ("A: gốc, 50 ep", "#9AA9AE", "--"), "A150_orig": ("A150: gốc, 150 ep", "#1F4E79", "-"),
         "B_semiuir": ("B: chỉ Semi-UIR", "#B8B8B8", ":"), "C_mixed": ("C: đa miền", "#E08A3C", "-")}
fig, ax = plt.subplots(1, len(FAMILIES), figsize=(4.2 * len(FAMILIES), 3.8), sharey=True)
for k, fam in enumerate(FAMILIES):
    for d in DETS:
        lab, col, ls = STYLE[d]
        ys = [TAB.loc["clean", d]] + [TAB.loc[f"{fam}{l}", d] for l in LEVELS]
        ax[k].plot([0] + LEVELS, ys, ls, color=col, marker="o", lw=2.2 if d in ("C_mixed", "A150_orig") else 1.2, label=lab)
    ax[k].set_title(fam); ax[k].set_xticks([0] + LEVELS); ax[k].set_xticklabels(["sạch", "1", "2", "3"])
    ax[k].grid(alpha=0.3)
ax[0].set_ylabel("mAP@0.5:0.95"); ax[-1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/robustness_curves.png", dpi=110); plt.show()
''')

md("## 10. Tóm tắt (copy vào results/robustness_unseen.md)")
code(r'''
summary = {"smoke": SMOKE, "n_test": len(TEST), "conditions": CONDS, "weights": W,
           "matrix": MATRIX.to_dict("records"), "paired_C_vs_A150": PAIR.to_dict("records")}
with open(f"{OUT}/summary_robust.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

print("| Điều kiện | " + " | ".join(DETS) + " | C − A150 [CI 95%] | Chênh mức tụt [CI 95%] |")
print("|---|" + "---|" * (len(DETS) + 2))
for c, r in zip(CONDS, PAIR.itertuples()):
    print(f"| {c} | " + " | ".join(f"{TAB.loc[c, d]:.4f}" for d in DETS)
          + f" | {r.delta:+.4f} [{r.delta_lo:+.4f}; {r.delta_hi:+.4f}] | {r.drop_gap:+.4f} [{r.drop_gap_lo:+.4f}; {r.drop_gap_hi:+.4f}] |")

if os.path.isdir("/tmp/data"):
    import shutil; shutil.rmtree("/tmp/data", ignore_errors=True)
print("\nOutput:", sorted(os.listdir(OUT)))
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
if __name__ == "__main__":
    json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    print("wrote", sys.argv[1], len(cells), "cells")
