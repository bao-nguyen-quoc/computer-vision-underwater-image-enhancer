# Detector một miền (Awad et al.) so với detector đa miền

Notebook: [`notebooks/domain_detectors.ipynb`](../notebooks/domain_detectors.ipynb).

Hai lần chạy trên Kaggle (T4 × 2):
- [domain-detectors-v2](https://www.kaggle.com/code/phanlvnminh/domain-detectors-v2): train B, chấm A/B/C.
- [domain-detectors-v3](https://www.kaggle.com/code/phanlvnminh/domain-detectors-v3): dùng lại B, train thêm A150, chấm cả 4 detector.

Ảnh đầu vào lấy từ `ruod640_orig.zip`, do [`notebooks/pack_ruod640.ipynb`](../notebooks/pack_ruod640.ipynb) tạo, vì dataset `ruod640` hay lỗi mount trên Kaggle. Các bảng số trong `results/domain/` được làm tròn 4 chữ số, chép từ output notebook.

## Câu hỏi

Awad et al. train **mỗi miền ảnh một detector** (domain detector) và kết luận detector ảnh gốc luôn tốt nhất. Notebook này kiểm tra hai câu hỏi:
- Một detector **train chung trên nhiều miền** (gốc + SCNet + Semi-UIR) có tốt hơn không?
- Nếu tốt hơn, là nhờ **đa dạng miền ảnh** hay chỉ nhờ **được train trên nhiều ảnh hơn**?

## Thiết lập

| Detector | Dữ liệu train | Epoch | Số lượt ảnh | Nguồn |
|---|---|---|---|---|
| A | 9.800 ảnh gốc | 50 | 1× | baseline tuần 1 (`notebook-ruod640`, `runs/baseline/weights/last.pt`) |
| B | 9.800 ảnh Semi-UIR | 50 | 1× | train trong notebook (0,98 giờ) |
| A150 | 9.800 ảnh gốc | 150 | 3× | train trong notebook (2,98 giờ) |
| C | 3 × 9.800 ảnh (gốc, SCNet, Semi-UIR) | 50 | 3× | model cuối của notebook K-fold (`weights/full.pt`) |

- **Cấu hình chung:** YOLO11n pre-train COCO, `imgsz=640`, `batch=32`, `seed=0`, 2 GPU. Dùng **`last.pt`**, không chọn checkpoint theo tập test.
- **A150** nhìn đúng bằng số lượt ảnh của C. Đây là nhóm đối chứng để tách hiệu ứng "train nhiều hơn" khỏi hiệu ứng "đa dạng miền".
- **Chấm điểm:** mỗi detector dự đoán cả 3 phiên bản của 4.200 ảnh test (conf 0,001, IoU 0,7). Tính mAP bằng pycocotools; khoảng tin cậy 95% bằng paired bootstrap 1.000 lần trên hiệu mAP từng ảnh so với **A trên ảnh gốc**.
- **Tự kiểm tra:** C trên ảnh gốc ra 0,6155, trùng `summary_full.json` của notebook K-fold.

## Ma trận mAP@0.5:0.95

| Detector | Lượt ảnh | Test gốc | Test SCNet | Test Semi-UIR | Chênh lệch giữa 3 miền |
|---|---|---|---|---|---|
| A: chỉ ảnh gốc | 1× | 0,5990 | 0,5687 | 0,5376 | 0,0614 |
| B: chỉ Semi-UIR | 1× | 0,5475 | 0,5828 | 0,5951 | 0,0476 |
| A150: chỉ ảnh gốc, 150 epoch | 3× | 0,6150 | 0,5777 | 0,5469 | 0,0681 |
| **C: đa miền** | 3× | **0,6155** | **0,6129** | **0,6084** | **0,0071** |

mAP@0.5:

| Detector | Test gốc | Test SCNet | Test Semi-UIR |
|---|---|---|---|
| A | 0,8367 | 0,8077 | 0,7719 |
| B | 0,7749 | 0,8214 | 0,8345 |
| A150 | 0,8470 | 0,8116 | 0,7760 |
| C | 0,8468 | 0,8449 | 0,8422 |

## Chênh lệch so với A trên ảnh gốc

| Detector × test | Δ mAP@0.5:0.95 | mAP từng ảnh | Δ từng ảnh [CI 95%] |
|---|---|---|---|
| A × gốc (mốc) | — | 0,7065 | — |
| A × SCNet | −0,0303 | 0,6849 | −0,0216 [−0,0245; −0,0190] |
| A × Semi-UIR | −0,0614 | 0,6629 | −0,0436 [−0,0473; −0,0401] |
| B × gốc | −0,0515 | 0,6580 | −0,0485 [−0,0528; −0,0442] |
| B × SCNet | −0,0162 | 0,6933 | −0,0131 [−0,0165; −0,0097] |
| B × Semi-UIR | −0,0038 | 0,7034 | −0,0030 [−0,0061; +0,0002] |
| A150 × gốc | +0,0160 | 0,7157 | +0,0092 [+0,0062; +0,0121] |
| A150 × SCNet | −0,0213 | 0,6913 | −0,0152 [−0,0186; −0,0117] |
| A150 × Semi-UIR | −0,0521 | 0,6690 | −0,0374 [−0,0418; −0,0335] |
| C × gốc | +0,0165 | 0,7186 | +0,0121 [+0,0090; +0,0150] |
| C × SCNet | +0,0139 | 0,7154 | +0,0090 [+0,0059; +0,0119] |
| C × Semi-UIR | +0,0094 | 0,7130 | +0,0065 [+0,0034; +0,0096] |

## AP@0.5:0.95 theo lớp trên ảnh test gốc

| Lớp | A | B | A150 | C |
|---|---|---|---|---|
| holothurian | 0,4675 | 0,3600 | **0,5007** | 0,4966 |
| echinus | 0,5116 | 0,4248 | **0,5379** | 0,5305 |
| scallop | 0,4627 | 0,4016 | **0,5042** | 0,4957 |
| starfish | 0,5397 | 0,3922 | **0,5591** | 0,5508 |
| fish | 0,5075 | 0,4899 | 0,5149 | **0,5198** |
| corals | 0,5196 | 0,4921 | 0,5379 | **0,5386** |
| diver | 0,7309 | 0,7092 | 0,7315 | **0,7456** |
| cuttlefish | 0,8291 | 0,8141 | 0,8296 | **0,8360** |
| turtle | 0,8292 | 0,8101 | 0,8347 | **0,8383** |
| jellyfish | 0,5919 | 0,5809 | 0,5991 | **0,6026** |

## Kết luận

1. **Tái hiện được kết quả của Awad et al.** Mỗi detector một miền tốt nhất trên miền của nó: A trên ảnh gốc, B trên ảnh Semi-UIR. Detector ảnh gốc (A × gốc, 0,5990) nhỉnh hơn detector Semi-UIR (B × Semi-UIR, 0,5951).
2. **Trên ảnh gốc, phần tăng của C so với A chủ yếu do train nhiều hơn, không phải do đa dạng miền.**
   - A150 (chỉ ảnh gốc, cùng số lượt ảnh với C) đạt 0,6150, gần bằng C (0,6155).
   - Khoảng tin cậy của A150 và C so với A chồng lên nhau, nên không kết luận được C hơn A150 trên ảnh gốc.
   - Baseline A (50 epoch) chưa train đủ.
3. **Ưu điểm thật của train đa miền là bền khi miền ảnh thay đổi, với cùng chi phí train.**
   - So với A150: C hơn +0,0352 trên ảnh SCNet và +0,0615 trên ảnh Semi-UIR.
   - Chênh lệch giữa 3 miền của C chỉ 0,0071, so với 0,0681 của A150 và 0,0614 của A.
   - Ở cả 3 cột, C vượt A có ý nghĩa thống kê.
4. **Theo lớp:** trên ảnh gốc, A150 và C ngang nhau. A150 nhỉnh hơn ở holothurian, echinus, scallop, starfish; C nhỉnh hơn ở fish, corals, diver, cuttlefish, turtle, jellyfish. Mọi chênh lệch đều trong khoảng 0,001–0,014.
5. **Hệ quả thực tế:** tăng cường ảnh dưới nước **không làm detector chính xác hơn trên ảnh gốc**. Giá trị của nó là làm dữ liệu train để detector **bền với thay đổi màu sắc và độ tương phản**, mà không tốn thêm thời gian khi suy luận.

## Hạn chế

- Mỗi cấu hình chỉ train một lần (seed 0). Chênh lệch nhỏ như A150 so với C trên ảnh gốc (0,0005) nằm trong mức nhiễu giữa các lần train.
- "Miền khác" ở đây là chính các phiên bản tăng cường. Chưa kiểm tra trên dữ liệu dưới nước từ nguồn khác (ví dụ ảnh của một bộ dữ liệu khác).
- Chỉ dùng YOLO11n. Detector lớn hơn có thể cho kết quả khác.
