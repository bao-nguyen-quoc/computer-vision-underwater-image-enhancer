"""Sinh notebook pack_ruod640.ipynb: nén ảnh gốc + nhãn + json của ruod640 thành 1 file zip (CPU, không tốn GPU).
Lý do: dataset ruod640 có ~28.000 file nhỏ nên Kaggle hay lỗi ERRORED_MOUNTING_DATASET; output 1 file zip lớn mount ổn định hơn.
Chạy: python make_pack_nb.py <out.ipynb>"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Đóng gói ruod640 (ảnh gốc) thành 1 file zip

Dataset `ruod640` có khoảng 28.000 file nhỏ nên Kaggle thỉnh thoảng không mount được (`ERRORED_MOUNTING_DATASET`). Notebook này nén `orig/` (ảnh + nhãn YOLO) và `ann_train640.json`, `ann_test640.json` thành **`ruod640_orig.zip`** trong output. Các notebook sau Add Input output này thay cho dataset, và tự giải nén vào `/tmp`.

**Cách chạy:** Accelerator **None** (CPU, không tốn quota GPU), Add Input dataset `ruod640` → **Save Version → Save & Run All** (≈ 5–10 phút). Nếu vẫn lỗi mount thì chạy lại sau vài phút: chỉ cần mount thành công một lần.
""")
code(r'''
import glob, os, zipfile, time
found = glob.glob("/kaggle/input/**/orig/images/train", recursive=True)
assert found, "Không thấy ruod640/orig/images/train -> Add Input dataset ruod640"
R640 = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
OUT = "/kaggle/working/ruod640_orig.zip"
t, n = time.time(), 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:          # ảnh jpg đã nén sẵn, không nén lại
    for name in ("ann_train640.json", "ann_test640.json"):
        z.write(f"{R640}/{name}", name, compress_type=zipfile.ZIP_DEFLATED); n += 1
    for dp, _, fs in os.walk(f"{R640}/orig"):
        for f in sorted(fs):
            z.write(f"{dp}/{f}", os.path.relpath(f"{dp}/{f}", R640)); n += 1
print(f"{n} file, {os.path.getsize(OUT) / 2**30:.2f} GiB, {time.time() - t:.0f}s")
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
for s, k in (("train", 9800), ("test", 4200)):
    ni = sum(x.startswith(f"orig/images/{s}/") for x in names); nl = sum(x.startswith(f"orig/labels/{s}/") for x in names)
    print(f"{s}: {ni} ảnh, {nl} nhãn"); assert ni == k and nl == k
''')
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False,
                              "language": "python", "sourceType": "notebook", "dataSources": []}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
