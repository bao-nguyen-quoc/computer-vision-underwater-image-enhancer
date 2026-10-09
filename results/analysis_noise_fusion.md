# Phân tích: oracle có bao nhiêu phần là nhiễu, và gộp nhiều phiên bản có hơn ảnh gốc không

Notebook: [`notebooks/analysis_noise_fusion.ipynb`](../notebooks/analysis_noise_fusion.ipynb). Lần chạy: [Kaggle analysis-noise-fusion, version 356354794](https://www.kaggle.com/code/phanlvnminh/analysis-noise-fusion), GPU T4, khoảng 25 phút. Bảng số trong `results/analysis/` được làm tròn 4 chữ số, chép từ output notebook.

## Thiết lập

- **Dữ liệu:** dùng lại model cuối `full.pt` của notebook K-fold (version full) và các dự đoán test đã lưu của 3 phiên bản: gốc, SCNet, Semi-UIR (4.200 ảnh RUOD).
- **Dự đoán mới:** `full.pt` dự đoán thêm **ảnh gốc lật ngang** (`flip`), rồi lật box về lại. Cấu hình giống lúc đo mAP: conf 0,001, IoU 0,7, tối đa 300 box.
- **Tự kiểm tra (đều đạt):**
  - Dự đoán lại ảnh gốc cho 0,6155, trùng `summary_full.json`.
  - mAP từng ảnh lệch so với `per_image_test.csv` khoảng 1e-16.
  - Ảnh lật đạt 0,6169, ngang ảnh gốc, chứng tỏ phép lật box đúng.
- **Oracle:** mỗi ảnh lấy phiên bản có mAP từng ảnh cao nhất trong tập ứng viên. Cách này dùng ground-truth nên chỉ là cận trên.
- **WBF:** gộp box của nhiều phiên bản cho từng ảnh (thư viện `ensemble-boxes`).
  - Tham số cố định, **không tinh chỉnh trên test**: `iou_thr = 0,6`, `skip_box_thr = 0,001`, tối đa 150 box mỗi phiên bản.
- **Chọn theo độ tự tin của detector:** mỗi ảnh chọn phiên bản có điểm tự tin cao nhất, theo một trong hai cách:
  - `conf_top5`: trung bình 5 score cao nhất.
  - `conf_sum`: tổng các score ≥ 0,25.
- **Khoảng tin cậy:** paired bootstrap 1.000 lần trên hiệu mAP từng ảnh so với ảnh gốc.

## Phần A: oracle của nhiễu so với oracle của tăng cường

| Tập ứng viên | Số ứng viên | mAP@0.5 | mAP@0.5:0.95 | Tăng cả tập | mAP từng ảnh | Tăng từng ảnh [CI 95%] | Ảnh không chọn gốc |
|---|---|---|---|---|---|---|---|
| {gốc, gốc chạy lại} | 2 | 0,8472 | 0,6159 | +0,0004 | 0,7190 | +0,0005 [+0,0003; +0,0006] | 2,3% |
| **{gốc, gốc lật}: chỉ có nhiễu** | 2 | 0,8529 | **0,6300** | **+0,0146** | 0,7393 | **+0,0207** [+0,0193; +0,0221] | 38,5% |
| {gốc, SCNet} | 2 | 0,8504 | 0,6250 | +0,0095 | 0,7329 | +0,0144 [+0,0133; +0,0154] | 31,6% |
| {gốc, Semi-UIR} | 2 | 0,8505 | 0,6250 | +0,0095 | 0,7350 | +0,0164 [+0,0150; +0,0179] | 31,7% |
| {gốc, SCNet, Semi-UIR}: oracle của dự án | 3 | 0,8523 | 0,6300 | +0,0146 | 0,7420 | +0,0234 [+0,0219; +0,0250] | 44,2% |
| {gốc, lật, SCNet, Semi-UIR} | 4 | 0,8560 | 0,6386 | +0,0231 | 0,7539 | +0,0353 [+0,0336; +0,0371] | 58,6% |

- Với 2 ứng viên, oracle {gốc, lật} tăng **144%** so với mức tăng của {gốc, SCNet} (tính theo mAP từng ảnh).
- Theo mAP cả tập, oracle {gốc, lật} bằng đúng oracle 3 phiên bản của dự án (0,6300).
- Tương quan theo ảnh giữa lợi ích của lật và lợi ích của SCNet là **0,306**.
- Chạy lại cùng ảnh gần như không tạo khác biệt (+0,0005). Nhiễu ở đây là độ nhạy của detector với một thay đổi nhỏ của ảnh đầu vào, không phải do quá trình suy luận ngẫu nhiên.

## Phần B: phương pháp dùng được (không cần ground-truth)

| Phương pháp | mAP@0.5 | mAP@0.5:0.95 | Δ cả tập | mAP từng ảnh | Δ từng ảnh [CI 95%] |
|---|---|---|---|---|---|
| Ảnh gốc | 0,8468 | 0,6155 | — | 0,7186 | — |
| WBF(gốc, lật): TTA thông thường, không tăng cường | 0,8466 | 0,6145 | −0,0009 | 0,7183 | −0,0002 [−0,0019; +0,0014] |
| WBF(gốc, SCNet) | 0,8442 | 0,6085 | −0,0070 | 0,7136 | −0,0050 [−0,0065; −0,0035] |
| WBF(gốc, SCNet, Semi-UIR) | 0,8468 | 0,6111 | −0,0044 | 0,7160 | −0,0025 [−0,0044; −0,0007] |
| **WBF(gốc, lật, SCNet, Semi-UIR)** | **0,8512** | **0,6177** | **+0,0023** | **0,7218** | **+0,0032 [+0,0014; +0,0051]** |
| Chọn theo `conf_top5` | 0,8448 | 0,6129 | −0,0025 | 0,7165 | −0,0021 [−0,0035; −0,0006] |
| Chọn theo `conf_sum` | 0,8461 | 0,6136 | −0,0018 | 0,7167 | −0,0018 [−0,0034; −0,0003] |

Tỷ lệ chọn gốc/SCNet/Semi-UIR: `conf_top5` là 1482/1351/1367, `conf_sum` là 1423/1417/1360.

## Kết luận

1. **Khoảng oracle của tăng cường có chọn lọc gần như hoàn toàn là nhiễu đo lường.**
   - Lật ngang ảnh không thay đổi nội dung nhưng cho mức tăng oracle bằng hoặc lớn hơn SCNet và Semi-UIR.
   - Oracle tăng theo số ứng viên, bất kể ứng viên đó là gì.
   - Vì vậy con số 0,68 → 0,77 của Awad et al. (chọn trên 10 phiên bản, mỗi phiên bản một detector riêng) không chứng minh được rằng tăng cường ảnh giúp phát hiện vật thể.
2. **Điều này giải thích vì sao Selector ResNet18 không vượt ảnh gốc** (xem [`selector_results.md`](selector_results.md)). Phần lớn "nhãn tốt nhất" là nhiễu, không có quy luật hình ảnh nào để học.
3. **Heuristic "chọn theo độ tự tin của detector"** mà Awad et al. gợi ý làm mAP giảm có ý nghĩa thống kê. Detector tự tin hơn trên một phiên bản không có nghĩa là nó đúng hơn.
4. **Gộp 4 góc nhìn bằng WBF là phương pháp duy nhất vượt ảnh gốc có ý nghĩa thống kê:** +0,0023 mAP@0.5:0.95 và +0,0044 mAP@0.5, khoảng tin cậy nằm hẳn trên 0.
   - Gộp 2 hoặc 3 phiên bản thì không giúp, hoặc làm kém đi.
   - Đổi lại phải chạy 4 lần YOLO cộng SCNet và Semi-UIR (khoảng 2 s/ảnh).
5. **Giá trị thực tế của tăng cường ảnh** nằm ở:
   - làm dữ liệu train để detector **bền khi miền ảnh thay đổi**. Trên ảnh gốc, detector đa miền chỉ ngang detector chỉ train ảnh gốc với cùng số lượt ảnh (0,6155 so với 0,6150). Nhưng trên ảnh SCNet / Semi-UIR, nó hơn +0,035 / +0,062. Xem [`domain_detectors.md`](domain_detectors.md);
   - làm thêm góc nhìn khi gộp.

   Tăng cường không có giá trị khi dùng để chọn một phiên bản cho mỗi ảnh lúc suy luận, và cũng không làm detector chính xác hơn trên ảnh gốc.

## Hạn chế

- Chỉ dùng một detector (YOLO11n) và một lần train. Mức nhiễu có thể khác với detector lớn hơn.
- Phép biến đổi "nhiễu" duy nhất là lật ngang. Có thể kiểm tra thêm bằng các thay đổi nhỏ khác như dịch ảnh vài pixel hay đổi kích thước.
- Tham số WBF không được tinh chỉnh. Tinh chỉnh trên một tập riêng có thể tăng thêm, nhưng không được dùng tập test để chỉnh.
