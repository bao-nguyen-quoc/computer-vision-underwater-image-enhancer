# Selective Enhancement + YOLO11 — Kế hoạch theo issue (tháng 10/2026)

**Mục tiêu đến 31/10/2026:** có pipeline *selector* (ResNet18) chọn cho từng ảnh một trong 3 phiên bản **ảnh gốc / SCNet / Semi-UIR** để đưa vào YOLO11n, đánh giá trên RUOD, kèm báo cáo.

- Dataset: https://www.kaggle.com/datasets/phanlvnminh/ruod640
- Code: https://github.com/bao-nguyen-quoc/computer-vision-underwater-image-enhancer

---

## 1. Timeline tổng quan và các mốc chặn

Mỗi mốc chặn (🔒) là đầu vào của tuần sau. Trễ mốc = cả nhóm bị kẹt, nên xong đúng hạn là ưu tiên số 1.

| Tuần | Thời gian | Kết quả cuối tuần | Mốc chặn 🔒 |
|---|---|---|---|
| 1 | 5–10/10 | RUOD chuẩn hóa 640; 2 bộ ảnh enhance; mAP baseline | 🔒 **6/10**: `ruod640/orig` lên Kaggle/Drive<br>🔒 **10/10**: `scnet/` và `semiuir/` (ảnh + nhãn) |
| 2 | 11–17/10 | `labels_selector.csv` (9.800 ảnh train); code selector sẵn sàng | 🔒 **17/10**: `labels_selector.csv` |
| 3 | 18–24/10 | Selector train xong; bảng kết quả đầy đủ trên 4.200 ảnh test | 🔒 **24/10**: `selector_test.csv` + bảng kết quả |
| 4 | 25–31/10 | Báo cáo hoàn chỉnh (+ ACDC nếu còn thời gian) | **31/10**: nộp báo cáo |

---

## 2. Yêu cầu chung cho toàn bộ project

Mọi issue đều phải tuân thủ các điểm dưới đây.

### 2.1 Dữ liệu và cấu trúc thư mục
- **Một repo GitHub chung cho code**; dữ liệu và trọng số chia sẻ qua Kaggle Dataset hoặc Google Drive, không commit vào repo.
- **Resize cạnh dài = 640 trước mọi bước khác.** Nhãn YOLO là tọa độ chuẩn hóa nên không đổi.
- **Ba phiên bản ảnh dùng cùng tên file**, mỗi phiên bản một thư mục, nhãn copy vào cả 3 (Ultralytics tìm nhãn bằng cách đổi `images` → `labels` trong đường dẫn):

```
ruod640/
  orig/{images,labels}/{train,test}/
  scnet/{images,labels}/{train,test}/      # labels copy từ orig/labels
  semiuir/{images,labels}/{train,test}/    # labels copy từ orig/labels
  ann_train640.json                        # COCO đã scale về 640, dùng tính mAP
  ann_test640.json
```

### 2.2 Quy tắc thực nghiệm
- **Seed = 0** cho mọi lần chia dữ liệu, shuffle, random.
- **Enhancer dùng nguyên trọng số pre-train, không train lại**; chạy thẳng trên ảnh 640, không resize thêm.
- **Chống rò rỉ test:** tập test chỉ dùng để đánh giá cuối (tuần 3). Không chọn checkpoint theo test, không sinh nhãn selector từ test. YOLO các fold dùng `last.pt`. (`val` trong file yaml chỉ để theo dõi, không dùng để chọn model.)
- **Nhãn selector** = phiên bản có mAP@0.5:0.95 từng ảnh cao nhất; nếu chênh < 0.01 so với ảnh gốc thì chọn **gốc**. Nhãn phải sinh từ model K-fold (model chưa từng thấy ảnh đó), nếu không mAP bị cao giả tạo.
- **Chọn checkpoint selector theo mAP của phiên bản được chọn**, không theo accuracy (accuracy cao có thể chỉ vì luôn đoán "gốc").

### 2.3 Quy trình làm việc
- Mỗi issue = 1 branch + 1 PR nhỏ; tên branch dạng `w1-02-prepare-ruod`.
- **Definition of Done (chung):** (1) chạy lại được từ đầu bằng lệnh ghi trong README; (2) output đúng tên/đường dẫn quy định; (3) số lượng file khớp (9.800 train / 4.200 test); (4) số liệu đã ghi vào `results.md` (bảng kết quả chung).
- **Khi bàn giao:** báo vào kênh chung kèm đường dẫn, số lượng file đã kiểm tra, và lệnh/phiên bản thư viện đã dùng.
- **GPU:** Kaggle T4 hoặc Colab, dùng 2 tài khoản để nhân đôi hạn mức. Lưu checkpoint ra output/Drive trước khi session hết hạn.
- Script mẫu (`prepare_ruod.py`, `enhance_semiuir.py`, `kfold_yolo.py`, `per_image_map.py`, `train_selector.py`, `eval_selection.py`) nằm ở phần Code mẫu của file kế hoạch gốc. Đường dẫn và tên file RUOD cần sửa theo bản thực tế.

---

## 3. Tài nguyên

| Thành phần | Link | Ghi chú |
|---|---|---|
| RUOD | https://github.com/xiaoDetection/RUOD (Drive 3.4 GB hoặc 2 file Releases) | 14.000 ảnh: 9.800 train / 4.200 test, 10 lớp. **Kiểm tra định dạng nhãn ngay khi giải nén** |
| SCNet | https://github.com/zhenqifu/SCNet (trọng số + UIEB đã chia có trong README) | Chạy `eval.py` để enhance, `get_performance.py` để ra PSNR/SSIM/LPIPS |
| Semi-UIR | https://github.com/Huang-ShiRui/Semi-UIR (`pretrained/model.pth`) | Chạy ở độ phân giải gốc; cần bản đồ chiếu sáng (LA), script mẫu tự tính |
| YOLO11 | https://docs.ultralytics.com/models/yolo11/ · `pip install ultralytics` | Dùng `yolo11n.pt` pre-train COCO |
| Selector | `torchvision.models.resnet18` | Pre-train ImageNet, đổi lớp cuối ra 3 lớp |
| mAP từng ảnh | `pip install pycocotools` | Theo giao thức của Awad et al. |
| RUOD đã enhance sẵn | https://github.com/RSSL-MTU/Enhancement-Detection-Analysis | Có **Semi-UIR** (dự phòng) và **ACDC** (enhancer thứ 3 tùy chọn); tải về rồi resize về 640 |

---

## 4. Danh sách issue theo timeline

Quy ước: `Phụ thuộc` = issue phải xong trước. Nhãn `data/detector` hoặc `enhancer/selector` chỉ để phân loại công việc, dùng làm label GitHub nếu cần.

### Tuần 1 (5–10/10): Dữ liệu, enhancer, YOLO baseline

**W1-01 · Tải RUOD và kiểm tra định dạng nhãn** · `data/detector` · Hạn: 5/10 (làm ngay)
- Việc: tải RUOD, giải nén, xác định cấu trúc thư mục, tên file và khóa JSON (thường là COCO JSON).
- Done khi: biết chính xác đường dẫn ảnh, tên file nhãn, tên khóa; đã sửa các biến `SRC_IMG`, `SRC_ANN` trong script.

**W1-02 · Chuẩn hóa RUOD về 640 và xuất nhãn** · `data/detector` · Hạn: 6/10 · Phụ thuộc: W1-01
- Việc: chạy `prepare_ruod.py`: resize cạnh dài 640, xuất nhãn YOLO và `ann_train640.json`, `ann_test640.json`.
- Done khi: `orig/` có đủ 9.800 + 4.200 ảnh và nhãn tương ứng; đã in và ghi lại **tên 10 lớp theo thứ tự** (dùng cho `names` trong yaml).

**W1-03 · Upload `ruod640/orig` và chia sẻ** 🔒 · `data/detector` · Hạn: **6/10** · Phụ thuộc: W1-02
- Done khi: dataset lên Kaggle/Drive, cả nhóm truy cập được, báo kèm số ảnh.

**W1-04 · Train YOLO11n baseline (chỉ ảnh gốc)** · `data/detector` · Hạn: 10/10 · Phụ thuộc: W1-02
- Việc: tạo `data_orig.yaml`; train 50 epoch, `imgsz=640`, `batch=32`.
- Done khi: có `best/last.pt` được lưu lại và mAP@0.5, mAP@0.5:0.95 trên test đã ghi vào `results.md` (dòng "YOLO baseline").

**W1-05 · Cài môi trường SCNet và Semi-UIR** · `enhancer/selector` · Hạn: 6/10 (không phụ thuộc dữ liệu, làm ngay)
- Việc: clone 2 repo, tải trọng số, cài môi trường. Semi-UIR viết cho PyTorch 1.8, có thể lỗi `adamp` / `deform_conv` → `pip install adamp`, thử PyTorch cũ hơn.
- Done khi: cả 2 model chạy được trên 1 ảnh bất kỳ. Nếu Semi-UIR mất quá nửa ngày → dùng bản Semi-UIR có sẵn trong Drive của Awad et al. (xem Rủi ro).

**W1-06 · Kiểm tra nhanh chất lượng enhancer trên UIEB** · `enhancer/selector` · Hạn: 7/10 · Phụ thuộc: W1-05
- Việc: chạy trọng số gốc trên tập test UIEB, so PSNR/SSIM với số trong bài (SCNet 22.08 / 0.8625; Semi-UIR 24.59 / 0.901 trên 90 ảnh test UIEB).
- Done khi: kết quả đã ghi lại để dùng cho chương Tăng cường.

**W1-07 · Viết script enhance hàng loạt cho cả 2 enhancer** · `enhancer/selector` · Hạn: 7/10 · Phụ thuộc: W1-05
- Việc: script nhận `ruod640/orig/images/{train,test}`, xuất `scnet/images/...` và `semiuir/images/...`, **giữ nguyên tên file**, không resize. Semi-UIR tự tính bản đồ chiếu sáng trong script.
- Done khi: chạy thử 20 ảnh, kích thước ảnh ra = ảnh vào, tên file khớp.

**W1-08 · Enhance toàn bộ tập train (9.800 ảnh) bằng SCNet và Semi-UIR** · `enhancer/selector` · Hạn: nên xong sớm (≤ 8/10) · Phụ thuộc: W1-03, W1-07
- Lý do tách riêng: tuần 2 chỉ cần ảnh train; bàn giao train trước giúp bên detector bắt đầu K-fold sớm.

**W1-09 · Enhance toàn bộ tập test (4.200 ảnh) và bàn giao** 🔒 · `enhancer/selector` · Hạn: **10/10** · Phụ thuộc: W1-08
- Việc: copy `orig/labels` sang `scnet/labels` và `semiuir/labels`; upload; báo.
- Done khi: mỗi phiên bản có đủ 9.800 + 4.200 ảnh và nhãn, tên file khớp 100% với `orig/`.

### Tuần 2 (11–17/10): YOLO 3-fold và nhãn cho selector

Tuần tốn GPU nhất: chia 2 tài khoản chạy song song nếu cần.

**W2-01 · Chia 3 fold và chuẩn bị file train** · `data/detector` · Hạn: 11/10 · Phụ thuộc: W1-09 (hoặc ít nhất ảnh train đã có)
- Việc: chia 9.800 ảnh train thành 3 fold (seed 0); tạo `train_k{0,1,2}.txt` (gồm **cả 3 phiên bản** của 2 fold còn lại, ≈19.600 ảnh) và `data_k{0,1,2}.yaml`.

**W2-02 · Train 3 model fold** · `data/detector` · Hạn: 15/10 · Phụ thuộc: W2-01
- Việc: với mỗi fold k, train YOLO11n 30 epoch trên 2 fold còn lại. Nếu thiếu GPU: giảm còn 20 epoch, bật `cache=True`.
- Done khi: có `fold{0,1,2}/weights/last.pt`.

**W2-03 · Dự đoán out-of-fold cho 3 phiên bản** · `data/detector` · Hạn: 16/10 · Phụ thuộc: W2-02
- Việc: model fold k dự đoán trên ảnh fold k, cho cả 3 phiên bản, `conf=0.001`, lưu COCO JSON.
- Done khi: có `pred_k{k}_{orig,scnet,semiuir}.json`.

**W2-04 · Tính mAP@0.5:0.95 từng ảnh** · `data/detector` · Hạn: 16/10 · Phụ thuộc: W2-03
- Việc: chạy `per_image_map.py` (chạy trên CPU trong lúc GPU làm việc khác, mất vài chục phút).
- Lưu ý: ảnh không có nhãn thật sẽ cho mAP = -1 → script ép về 0; cần kiểm tra số ảnh loại này.

**W2-05 · Sinh `labels_selector.csv` và kiểm tra phân bố nhãn** 🔒 · `data/detector` · Hạn: **17/10** · Phụ thuộc: W2-04
- Output: `file, map_orig, map_scnet, map_semiuir, label` cho đủ 9.800 ảnh.
- Done khi: gửi file kèm **phân bố nhãn** (bao nhiêu ảnh chọn gốc/SCNet/Semi-UIR). Nếu phần lớn là nhãn 0, báo ngay.

**W2-06 · Tính UIQM và UCIQE trên 1.000 ảnh test ngẫu nhiên** · `enhancer/selector` · Hạn: 17/10 · Phụ thuộc: W1-09
- Việc: random 1.000 ảnh test (seed 0), tính cho cả 3 phiên bản. Dùng cho chương Tăng cường.

**W2-07 · Viết `train_selector.py` và chạy thử bằng nhãn giả** · `enhancer/selector` · Hạn: 17/10
- Việc: ResNet18 pre-train ImageNet, lớp cuối 3 lớp, trọng số lớp chống mất cân bằng, chọn checkpoint theo chosen-mAP (xem 2.2).
- Done khi: chạy hết 1–2 epoch với nhãn giả không lỗi; sẵn sàng nhận nhãn thật.

### Tuần 3 (18–24/10): Selector và bảng kết quả

**W3-01 · Train YOLO11n cuối cùng** · `data/detector` · Hạn: 21/10
- Việc: train trên toàn bộ 9.800 ảnh train × 3 phiên bản, 50 epoch. Issue này **không phụ thuộc nhãn selector**, có thể bắt đầu sớm nếu còn hạn mức GPU.

**W3-02 · Dự đoán trên test cho 3 phiên bản** · `data/detector` · Hạn: 22/10 · Phụ thuộc: W3-01
- Output: `pred_test_{orig,scnet,semiuir}.json`.

**W3-03 · Tính mAP từng ảnh trên test (cho Oracle)** · `data/detector` · Hạn: 22/10 · Phụ thuộc: W3-02
- Chạy `per_image_map.py` với `ann_test640.json`; áp dụng **cùng quy tắc ngưỡng 0.01** như khi sinh nhãn.

**W3-04 · Train selector** · `enhancer/selector` · Hạn: 20/10 · Phụ thuộc: W2-05, W2-07
- Việc: chia 80% train / 20% val (seed 0), ~15 epoch, lưu `selector.pt` theo chosen-mAP.

**W3-05 · Báo cáo chất lượng selector trên val** · `enhancer/selector` · Hạn: 21/10 · Phụ thuộc: W3-04
- Output: accuracy + ma trận nhầm lẫn + phân bố dự đoán.

**W3-06 · Dự đoán lựa chọn cho 4.200 ảnh test** 🔒 · `enhancer/selector` · Hạn: **22/10** · Phụ thuộc: W3-04
- Output: `selector_test.csv` (`file, choice`).

**W3-07 · Chạy `eval_selection.py` và điền bảng kết quả** 🔒 · `data/detector` · Hạn: **24/10** · Phụ thuộc: W3-02, W3-03, W3-06
- Việc: chỉ ghép dự đoán đã có theo lựa chọn từng ảnh, **không chạy lại YOLO**.
- Done khi: bảng ở mục 5 điền đủ mAP@0.5 và mAP@0.5:0.95; tính tỉ lệ khoảng cách thu hẹp.

**W3-08 · Đo tốc độ toàn pipeline (ms/ảnh)** · `enhancer/selector` · Hạn: 24/10 · Phụ thuộc: W3-06
- Việc: đo selector + enhancer được chọn + YOLO trên **một GPU**, ghi rõ loại GPU và batch size.

### Tuần 4 (25–31/10): Báo cáo và mở rộng

**W4-00 · Khóa số liệu** · Hạn: 25/10
- Sau mốc này không chạy lại thí nghiệm; mọi số trong báo cáo lấy từ `results.md`.

**W4-01 · Chuẩn bị hình minh họa** · Hạn: 27/10
- Ví dụ trực quan: ảnh gốc / SCNet / Semi-UIR và lựa chọn của selector cho cùng một ảnh; một vài trường hợp selector chọn đúng/sai.

**W4-02 · Viết các chương báo cáo** · Hạn: 29/10 (bản nháp), 31/10 (bản cuối)

| Chương | Nội dung |
|---|---|
| 1. Giới thiệu | Bài toán, động lực, đóng góp |
| 2. Tổng quan | UIE (SCNet, Semi-UIR), UOD (YOLO), Awad et al. 2026 |
| 3. Phương pháp | Pipeline, sinh nhãn K-fold, selector |
| 4. Thực nghiệm | Dữ liệu RUOD, thiết lập, metrics |
| 5. Kết quả | Bảng chất lượng ảnh, bảng mAP, oracle, ví dụ trực quan |
| 6. Kết luận | Hạn chế, hướng phát triển |

**W4-03 · Đọc chéo và hoàn thiện báo cáo** · Hạn: 31/10
- Kiểm tra số liệu khớp `results.md`, định dạng, tài liệu tham khảo.

**W4-04 (tùy chọn, nếu còn thời gian) · Thêm ACDC làm enhancer thứ 3**
1. Tải bản RUOD đã enhance bằng ACDC từ Drive của Awad et al., resize về 640, đổi tên khớp file gốc.
2. Dùng lại 3 model fold đã có để dự đoán trên ảnh ACDC, tính mAP từng ảnh (không train lại YOLO).
3. Sinh lại nhãn với 4 lựa chọn, train lại selector, thêm 2 dòng vào bảng kết quả.

---

## 5. Bảng kết quả cần điền (`results.md`)

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | ms/ảnh |
|---|---|---|---|
| YOLO baseline (chỉ train ảnh gốc) | | | |
| Ảnh gốc | | | |
| SCNet toàn bộ | | | |
| Semi-UIR toàn bộ | | | |
| Chọn ngẫu nhiên | | | |
| **Selector (của nhóm)** | | | |
| Oracle (cận trên) | | | |

Tỉ lệ khoảng cách được thu hẹp = (mAP selector − mAP ảnh gốc) / (mAP oracle − mAP ảnh gốc).

---

## 6. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Định dạng nhãn RUOD khác dự kiến | `prepare_ruod.py` báo lỗi khóa | Sửa đường dẫn/tên khóa ngay ngày đầu (W1-01) |
| Semi-UIR lỗi môi trường | Lỗi import `adamp` / `deform_conv` | `pip install adamp`, thử PyTorch cũ hơn; nếu mất > nửa ngày thì dùng bản Semi-UIR có sẵn của Awad et al. |
| Enhance trễ làm kẹt tuần 2 | Chưa xong 14.000 ảnh × 2 enhancer vào 8–10/10 | Bàn giao train trước, test sau; chia 2 tài khoản chạy 2 enhancer song song |
| Hết hạn mức GPU tuần 2 | 3 fold × 30 epoch không xong | Chạy song song 2 tài khoản; giảm còn 20 epoch; `cache=True` |
| Nhãn selector lệch về "gốc" | Phần lớn nhãn = 0 | Đã có trọng số lớp + chọn checkpoint theo chosen-mAP; báo cáo phân bố nhãn |
| Selector không vượt "ảnh gốc" | Dòng Selector ≈ dòng Ảnh gốc | Vẫn là kết quả hợp lệ: báo Oracle, phân tích ảnh nào được lợi, so với Awad et al. |
| Rò rỉ dữ liệu test | Dùng test để chọn checkpoint hoặc sinh nhãn | Tuân thủ mục 2.2; test chỉ đụng tới ở tuần 3 |
