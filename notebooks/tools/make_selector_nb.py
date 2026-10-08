"""Sinh notebook selector_resnet18.ipynb: train Selector trên nhãn OOF (version folds), đánh giá trên test
bằng cách ghép dự đoán YOLO đã lưu (version full). Chạy: python make_selector_nb.py <out.ipynb>"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Selector ResNet18: chọn ảnh gốc / SCNet / Semi-UIR cho từng ảnh → YOLO11n

Không train lại YOLO. Notebook dùng lại kết quả của notebook K-fold:
- **Version folds** → `labels_train.csv`: mAP từng ảnh (out-of-fold) của 3 phiên bản và nhãn cho 9.800 ảnh train.
- **Version full** → `per_image_test.csv`, `preds_test_{orig,scnet,semiuir}.json.gz`: dự đoán của model cuối trên 4.200 ảnh test.

Các bước:
1. Selector chỉ nhìn **ảnh gốc** (đúng như khi triển khai), ResNet18 pre-train ImageNet, 3 đầu ra.
2. Chia 80% train / 20% val (phân tầng theo nhãn, seed 0). Chọn checkpoint theo **chosen-mAP trên val** (mAP trung bình của phiên bản được chọn), không theo accuracy.
3. Hai cách học:
   - `ce`: phân loại, cross-entropy có trọng số lớp (theo paper / plan).
   - `reg`: hồi quy trực tiếp mAP của 3 phiên bản, chọn phiên bản dự đoán cao nhất. Dùng được cả độ lớn lợi ích, không chỉ nhãn.
4. Ngưỡng `tau`: chỉ tăng cường khi lợi thế dự đoán của phiên bản tăng cường so với ảnh gốc lớn hơn `tau`. `tau` chọn trên **val**.
5. Biến thể Selector chính = biến thể có chosen-mAP cao nhất **trên val**; tập test chỉ dùng để báo cáo, không dùng để chọn.
6. Ghép dự đoán test đã lưu theo lựa chọn từng ảnh → mAP cả tập (pycocotools), mAP từng ảnh, **gap closed**, bootstrap CI, tỷ lệ chọn, ma trận nhầm lẫn, AP theo lớp, tốc độ.

**Input cần thêm (Add Input):**
- Dataset `phanlvnminh/ruod640` (ảnh gốc + `ann_test640.json`).
- Output notebook `phanlvnminh/k-fold-yolo11n-map-t-ng-nh-t-o-nh-n-cho-select` **version full (v6)**.
- `labels_train.csv` của **version folds (v4)**: nếu Kaggle cho chọn version khi Add Input thì thêm v4; nếu không, tải `kfold/labels_train.csv` từ trang Output của v4 rồi tạo dataset nhỏ (ví dụ `selector-labels`) và Add Input dataset đó. Notebook tự tìm file trong `/kaggle/input`.

**Cách chạy:** Settings → GPU **T4** (1 GPU là đủ), Internet **On** (tải trọng số ResNet18). Chạy thử `SMOKE = True` → Run All (≈ 5 phút). Sau đó `SMOKE = False` → **Save Version → Save & Run All** (≈ 40–60 phút).
""")

md("## 1. Cấu hình")
code(r'''
MODES = ["ce", "reg"]      # chạy cả hai, biến thể chính chọn theo val
EPOCHS = 15
BATCH = 64
LR = 3e-4
IMG = 224                  # cạnh ảnh đưa vào ResNet18 (cache ở IMG + 16, crop ngẫu nhiên khi train)
VAL_FRAC = 0.2
SEED = 0
N_BOOT = 1000              # số lần bootstrap cho khoảng tin cậy
SMOKE = False              # True = chạy thử (300 ảnh train, 200 ảnh test, 1 epoch)

VERSIONS = ["orig", "scnet", "semiuir"]     # chỉ số lựa chọn 0, 1, 2
import os
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
OUT = f"{WORK}/selector"
if SMOKE:
    EPOCHS, N_BOOT = 1, 50
print(f"MODES={MODES} | epochs={EPOCHS} | SMOKE={SMOKE}")
''')

md("## 2. Thư viện")
code(r'''
import subprocess, sys
try:
    import pycocotools
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
import glob, json, gzip, time, hashlib, contextlib, io, copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| device", DEV)
''')

md("## 3. Tìm input và kiểm tra")
code(r'''
def find_one(pattern, hint):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    assert hits, f"Không thấy {pattern} trong {INPUT} -> {hint}"
    if len(hits) > 1:
        print(f"  Nhiều file {pattern}, dùng {hits[0]}:", hits)
    return hits[0]

found = glob.glob(f"{INPUT}/**/orig/images/train", recursive=True)
assert found, "Không thấy ruod640/orig/images/train -> Add Input dataset ruod640"
R640 = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
IMG_DIR = {s: f"{R640}/orig/images/{s}" for s in ("train", "test")}
ANN_TEST = f"{R640}/ann_test640.json"
LABELS = find_one("labels_train.csv", "thêm output version folds (v4) hoặc dataset chứa labels_train.csv")
PER_TEST = find_one("per_image_test.csv", "thêm output version full (v6) của notebook K-fold")
PREDS = {v: find_one(f"preds_test_{v}.json.gz", "thêm output version full (v6) của notebook K-fold") for v in VERSIONS}
print("ruod640 :", R640); print("labels  :", LABELS); print("test    :", PER_TEST)

tr_all = pd.read_csv(LABELS)
te_all = pd.read_csv(PER_TEST)
MCOLS = [f"map50_95_{v}" for v in VERSIONS]
M50COLS = [f"map50_{v}" for v in VERSIONS]
for name, df in (("labels_train.csv", tr_all), ("per_image_test.csv", te_all)):
    miss = {"file_name", "image_id", "label", *MCOLS, *M50COLS} - set(df.columns)
    assert not miss, f"{name} thiếu cột {miss}"

fold_hash = hashlib.md5(tr_all[["file_name", "image_id", "fold"]].to_csv(index=False).encode()).hexdigest()[:12] \
    if "fold" in tr_all else "?"
print(f"train: {len(tr_all)} ảnh | test: {len(te_all)} ảnh | folds hash {fold_hash} (cần ea44b385539c)")
assert len(tr_all) == 9800 and len(te_all) == 4200 or "SMOKE_FAKE" in os.environ, \
    "Số ảnh không đúng: có thể đang dùng output của lần chạy SMOKE"
print("nhãn train:", tr_all.label.value_counts().sort_index().to_dict(),
      "| nhãn oracle test:", te_all.label.value_counts().sort_index().to_dict())

with contextlib.redirect_stdout(io.StringIO()):
    GT = COCO(ANN_TEST)
CATS = sorted(GT.getCatIds())
CAT_NAMES = [GT.cats[c]["name"] for c in CATS]
assert set(te_all.image_id) <= set(GT.getImgIds()), "image_id trong per_image_test.csv không khớp ann_test640.json"

if SMOKE:
    tr_all = tr_all.sample(300, random_state=SEED).reset_index(drop=True)
    te_all = te_all.sample(200, random_state=SEED).sort_values("image_id").reset_index(drop=True)
''')

md("## 4. Nạp ảnh gốc vào RAM (resize một lần)")
code(r'''
CACHE = IMG + 16
def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((CACHE, CACHE), Image.BILINEAR), dtype=np.uint8)

def load_all(names, split):
    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(load, [f"{IMG_DIR[split]}/{n}" for n in names])))
    print(f"  {split}: {arr.shape} trong {time.time() - t:.0f}s")
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()     # N, 3, H, W  uint8

X_tr_all = load_all(tr_all.file_name, "train")
X_te = load_all(te_all.file_name, "test")

idx = np.arange(len(tr_all))
i_tr, i_va = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED,
                              stratify=tr_all.label if tr_all.label.value_counts().min() >= 2 else None)
Y = tr_all.label.to_numpy()
M_tr = tr_all[MCOLS].to_numpy(np.float32)
M_te, M50_te = te_all[MCOLS].to_numpy(), te_all[M50COLS].to_numpy()
print(f"train {len(i_tr)} | val {len(i_va)} | nhãn val {np.bincount(Y[i_va], minlength=3).tolist()}")
''')

md(r"""
## 5. Model, train, chọn checkpoint theo chosen-mAP

Tăng cường dữ liệu chỉ dùng crop ngẫu nhiên và lật ngang. **Không** dùng color jitter: màu và độ tương phản chính là tín hiệu để quyết định có tăng cường hay không.
""")
code(r'''
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

def prep(xb, train):
    xb = xb.to(DEV, non_blocking=True).float().div_(255)
    if train:
        oy, ox = np.random.randint(0, CACHE - IMG + 1, 2)
        xb = xb[:, :, oy:oy + IMG, ox:ox + IMG]
        flip = torch.rand(len(xb), device=DEV) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
    else:
        o = (CACHE - IMG) // 2
        xb = xb[:, :, o:o + IMG, o:o + IMG]
    return (xb - MEAN) / STD

def make_net():
    try:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:      # không có Internet
        raise RuntimeError("Không tải được trọng số ResNet18 -> bật Internet trong Settings") from e
    net.fc = nn.Linear(512, 3)
    return net.to(DEV)

@torch.no_grad()
def scores_of(net, X, mode, bs=256):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            o = net(prep(X[i:i + bs], False)).float()
        out.append(o.softmax(1) if mode == "ce" else o)
    return torch.cat(out).cpu().numpy()

def decide(s, tau=0.0):
    """s: (N, 3) xác suất (ce) hoặc mAP dự đoán (reg). Tăng cường chỉ khi lợi thế so với gốc > tau."""
    best_enh = 1 + s[:, 1:].argmax(1)
    margin = s[:, 1:].max(1) - s[:, 0]
    return np.where(margin > tau, best_enh, 0)

def chosen_map(M, choice):
    return float(M[np.arange(len(choice)), choice].mean())

def train_selector(mode):
    torch.manual_seed(SEED); np.random.seed(SEED)
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(i_tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    cnt = np.bincount(Y[i_tr], minlength=3)
    w = torch.tensor(len(i_tr) / (3 * np.maximum(cnt, 1)), dtype=torch.float, device=DEV)
    Xv, Mv = X_tr_all[i_va], M_tr[i_va]
    best, best_state, hist = -1.0, None, []
    for ep in range(EPOCHS):
        net.train(); t = time.time(); tot = 0.0
        perm = np.random.permutation(i_tr)
        for b in range(0, len(perm), BATCH):
            j = perm[b:b + BATCH]
            xb = prep(X_tr_all[j], True)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                o = net(xb).float()
            if mode == "ce":
                loss = F.cross_entropy(o, torch.as_tensor(Y[j], device=DEV), weight=w)
            else:
                loss = F.mse_loss(o, torch.as_tensor(M_tr[j], device=DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(j)
        s = scores_of(net, Xv, mode)
        c = decide(s)
        score = chosen_map(Mv, c)
        acc = float((c == Y[i_va]).mean())
        hist.append({"epoch": ep, "loss": tot / len(i_tr), "val_acc": acc, "val_chosen_map": score,
                     "val_pick": np.bincount(c, minlength=3).tolist()})
        flag = ""
        if score > best:
            best, best_state, flag = score, copy.deepcopy(net.state_dict()), "  *"
        print(f"[{mode}] ep {ep:2d} loss {tot / len(i_tr):.4f} | val acc {acc:.3f} | val chosen-mAP {score:.4f} "
              f"| chọn {hist[-1]['val_pick']} | {time.time() - t:.0f}s{flag}")
    net.load_state_dict(best_state)
    torch.save(best_state, f"{OUT}/selector_{mode}.pt")
    return net, pd.DataFrame(hist)

def tune_tau(s, M):
    """Chọn tau trên val: quét các phân vị của margin, giữ tau cho chosen-mAP cao nhất."""
    margin = s[:, 1:].max(1) - s[:, 0]
    cands = np.unique(np.concatenate([[0.0], np.quantile(margin, np.linspace(0, 1, 101))]))
    vals = [chosen_map(M, decide(s, t)) for t in cands]
    k = int(np.argmax(vals))
    return float(cands[k]), float(vals[k])

VAL_REF = {"orig": chosen_map(M_tr[i_va], np.zeros(len(i_va), int)),
           "oracle": float(M_tr[i_va].max(1).mean())}
print("val: luôn gốc", round(VAL_REF["orig"], 4), "| oracle", round(VAL_REF["oracle"], 4))

NETS, HIST, VARIANTS = {}, {}, {}
for mode in MODES:
    t0 = time.time()
    NETS[mode], HIST[mode] = train_selector(mode)
    HIST[mode].to_csv(f"{OUT}/history_{mode}.csv", index=False)
    s_va = scores_of(NETS[mode], X_tr_all[i_va], mode)
    tau, v_tuned = tune_tau(s_va, M_tr[i_va])
    VARIANTS[f"{mode}"] = {"mode": mode, "tau": 0.0, "val_chosen_map": chosen_map(M_tr[i_va], decide(s_va))}
    VARIANTS[f"{mode}+tau"] = {"mode": mode, "tau": tau, "val_chosen_map": v_tuned}
    print(f"[{mode}] xong {time.time() - t0:.0f}s | tau=0: {VARIANTS[mode]['val_chosen_map']:.4f} "
          f"| tau={tau:.4f}: {v_tuned:.4f}")

for v in VARIANTS.values():
    v["val_gap_closed"] = (v["val_chosen_map"] - VAL_REF["orig"]) / max(VAL_REF["oracle"] - VAL_REF["orig"], 1e-9)
MAIN = max(VARIANTS, key=lambda k: VARIANTS[k]["val_chosen_map"])
print(pd.DataFrame(VARIANTS).T)
print("Biến thể Selector chính (chọn theo val):", MAIN)
''')

md("## 6. Dự đoán lựa chọn trên test + đo tốc độ Selector")
code(r'''
S_TE = {mode: scores_of(NETS[mode], X_te, mode) for mode in MODES}
CHOICE = {k: decide(S_TE[v["mode"]], v["tau"]) for k, v in VARIANTS.items()}

sel = te_all[["file_name", "image_id"]].copy()
for k, c in CHOICE.items():
    sel[f"choice_{k}"] = c
sel["choice"] = CHOICE[MAIN]
for mode in MODES:
    for j, v in enumerate(VERSIONS):
        sel[f"score_{mode}_{v}"] = S_TE[mode][:, j].round(5)
sel.to_csv(f"{OUT}/selector_test.csv", index=False)

# Tốc độ: 1 ảnh/lần (đọc file + resize + ResNet18), giống khi triển khai
net = NETS[MAIN.split("+")[0]].eval()
names = te_all.file_name.tolist()[:200]
with torch.no_grad():
    for n in names[:10]:
        net(prep(torch.from_numpy(load(f"{IMG_DIR['test']}/{n}")).permute(2, 0, 1)[None], False))
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.time()
    for n in names:
        net(prep(torch.from_numpy(load(f"{IMG_DIR['test']}/{n}")).permute(2, 0, 1)[None], False))
    if DEV == "cuda": torch.cuda.synchronize()
SEL_MS = (time.time() - t) / len(names) * 1000
print(f"Selector: {SEL_MS:.1f} ms/ảnh ({DEV})")
print("Tỷ lệ chọn trên test:", {k: np.bincount(c, minlength=3).tolist() for k, c in CHOICE.items()})
''')

md("## 7. Ghép dự đoán YOLO đã lưu và chấm điểm")
code(r'''
def quiet():
    return contextlib.redirect_stdout(io.StringIO())

DETS = {}
for v in VERSIONS:
    with gzip.open(PREDS[v], "rt") as f:
        DETS[v] = json.load(f)
    print(f"{v}: {len(DETS[v])} detection")
TEST_IDS = te_all.image_id.astype(int).tolist()

def compose(choice):
    pick = dict(zip(TEST_IDS, choice))
    return [d for j, v in enumerate(VERSIONS) for d in DETS[v] if pick.get(d["image_id"], -1) == j]

def evaluate(choice):
    with quiet():
        E = COCOeval(GT, GT.loadRes(compose(choice)), "bbox")
        E.params.imgIds = TEST_IDS
        E.evaluate(); E.accumulate(); E.summarize()
    P = E.eval["precision"][:, :, :, 0, 2]                 # IoU, recall, lớp | area=all, maxDets=100
    per_cls = [float(P[:, :, k][P[:, :, k] > -1].mean()) if (P[:, :, k] > -1).any() else float("nan")
               for k in range(P.shape[2])]
    return {"map50_95": float(E.stats[0]), "map50": float(E.stats[1]), "per_class": per_cls}

n = len(TEST_IDS)
rng = np.random.default_rng(SEED)
POL = {f"always_{v}": np.full(n, j) for j, v in enumerate(VERSIONS)}
POL["random"] = rng.integers(0, 3, n)
POL.update({f"selector_{k}": c for k, c in CHOICE.items()})
POL["oracle"] = te_all.label.to_numpy()

RES = {}
for p, c in POL.items():
    t = time.time()
    RES[p] = evaluate(c)
    RES[p]["mean_per_image"] = chosen_map(M_te, c)
    RES[p]["mean_per_image_50"] = chosen_map(M50_te, c)
    RES[p]["pick"] = np.bincount(c, minlength=3).tolist()
    print(f"{p:22s} mAP50:95 {RES[p]['map50_95']:.4f} | mAP50 {RES[p]['map50']:.4f} | "
          f"từng ảnh {RES[p]['mean_per_image']:.4f} | chọn {RES[p]['pick']} ({time.time() - t:.0f}s)")

# Đối chiếu với summary_full.json của version full: các dòng cố định / oracle phải trùng
hits = glob.glob(f"{INPUT}/**/summary_full.json", recursive=True)
if hits and not SMOKE:
    ref = json.load(open(hits[0]))["policies"]
    for p in ["always_orig", "always_scnet", "always_semiuir", "oracle"]:
        d = abs(ref[p]["map50_95"] - RES[p]["map50_95"])
        print(f"  kiểm tra {p:15s} summary_full {ref[p]['map50_95']:.4f} vs ở đây {RES[p]['map50_95']:.4f}"
              + ("  OK" if d < 1e-4 else "  LỆCH!"))
''')

md("## 8. Gap closed, bootstrap CI, ma trận nhầm lẫn, AP theo lớp")
code(r'''
def gap(p, key):
    o, top = RES["always_orig"][key], RES["oracle"][key]
    return (RES[p][key] - o) / (top - o) if top > o else float("nan")

def boot(c, n_boot=N_BOOT):
    """Paired bootstrap trên hiệu mAP từng ảnh (chiến lược − luôn gốc)."""
    d = M_te[np.arange(n), c] - M_te[:, 0]
    r = np.random.default_rng(SEED)
    means = np.array([d[r.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float((means <= 0).mean())

rows = []
for p, c in POL.items():
    m, lo, hi, pv = boot(c)
    rows.append({"policy": p, "mAP50": RES[p]["map50"], "mAP50_95": RES[p]["map50_95"],
                 "per_image_mAP50_95": RES[p]["mean_per_image"],
                 "gap_closed_dataset": gap(p, "map50_95"), "gap_closed_per_image": gap(p, "mean_per_image"),
                 "delta_vs_orig": m, "ci95_low": lo, "ci95_high": hi, "p_boot(delta<=0)": pv,
                 "pick_orig/scnet/semiuir": "/".join(map(str, RES[p]["pick"]))})
TABLE = pd.DataFrame(rows)
TABLE.to_csv(f"{OUT}/results_table.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 20)
print(TABLE.round(4).to_string(index=False))

def confusion(true, pred):
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (true, pred), 1)
    return cm

CM = {"val": confusion(Y[i_va], decide(scores_of(NETS[VARIANTS[MAIN]["mode"]], X_tr_all[i_va],
                                                  VARIANTS[MAIN]["mode"]), VARIANTS[MAIN]["tau"])),
      "test": confusion(te_all.label.to_numpy(), CHOICE[MAIN])}
for k, cm in CM.items():
    print(f"\nMa trận nhầm lẫn {k} (hàng = nhãn oracle, cột = Selector {MAIN}), acc {np.trace(cm) / cm.sum():.3f}")
    print(pd.DataFrame(cm, index=VERSIONS, columns=VERSIONS))

PER_CLASS = pd.DataFrame({p: RES[p]["per_class"] for p in ["always_orig", "always_scnet", "always_semiuir",
                                                            f"selector_{MAIN}", "oracle"]}, index=CAT_NAMES)
PER_CLASS.to_csv(f"{OUT}/per_class_ap.csv")
print("\nAP50:95 theo lớp:"); print(PER_CLASS.round(4))
''')

md("## 9. Lưu tóm tắt và bảng cho báo cáo")
code(r'''
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (k, cm) in zip(ax, CM.items()):
    a.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, cm[i, j], ha="center", va="center")
    a.set_xticks(range(3), VERSIONS); a.set_yticks(range(3), VERSIONS)
    a.set_xlabel("Selector chọn"); a.set_ylabel("Nhãn oracle"); a.set_title(f"{k} (acc {np.trace(cm) / cm.sum():.3f})")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion.png", dpi=150); plt.close()

summary = {"smoke": SMOKE, "modes": MODES, "epochs": EPOCHS, "img": IMG, "folds_hash": fold_hash,
           "main_variant": MAIN, "variants_val": VARIANTS, "val_ref": VAL_REF,
           "selector_ms_per_image": SEL_MS, "device": DEV,
           "test": {p: {k: RES[p][k] for k in ("map50_95", "map50", "mean_per_image", "pick")} for p in RES},
           "gap_closed_main": {"dataset": gap(f"selector_{MAIN}", "map50_95"),
                               "per_image": gap(f"selector_{MAIN}", "mean_per_image")},
           "confusion": {k: v.tolist() for k, v in CM.items()}}
with open(f"{OUT}/summary_selector.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

NAMES_VI = {"always_orig": "Ảnh gốc", "always_scnet": "SCNet toàn bộ", "always_semiuir": "Semi-UIR toàn bộ",
            "random": "Chọn ngẫu nhiên", f"selector_{MAIN}": f"**Selector ({MAIN})**", "oracle": "Oracle (cận trên)"}
print("| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | mAP từng ảnh | Gap closed (cả tập / từng ảnh) |")
print("|---|---|---|---|---|")
for p, name in NAMES_VI.items():
    r = TABLE.set_index("policy").loc[p]
    print(f"| {name} | {r.mAP50:.4f} | {r.mAP50_95:.4f} | {r.per_image_mAP50_95:.4f} | {r.gap_closed_dataset:.1%} / {r.gap_closed_per_image:.1%} |")
print(f"\nSelector {SEL_MS:.1f} ms/ảnh. Output trong {OUT}:")
for f in sorted(os.listdir(OUT)):
    print(f"  {os.path.getsize(f'{OUT}/{f}') / 2**20:7.2f} MiB  {f}")
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
