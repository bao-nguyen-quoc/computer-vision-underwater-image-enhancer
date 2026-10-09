# Kết quả YOLO11n K-fold và đánh giá trên test (RUOD)

Ghi lại kết quả để dùng khi cập nhật slide và báo cáo. Số liệu gốc nằm trong `results/kfold/summary_folds.json` và `results/kfold/summary_full.json`.

## Nguồn

| Lần chạy | Notebook | Kaggle version | Thời gian |
|---|---|---|---|
| `STAGE = "folds"`: K-fold, tạo nhãn cho Selector | [`notebooks/kfold_yolo.ipynb`](../notebooks/kfold_yolo.ipynb) | [v4](https://www.kaggle.com/code/phanlvnminh/k-fold-yolo11n-map-t-ng-nh-t-o-nh-n-cho-select/notebook?scriptVersionId=356097637) | ≈ 4 giờ (train 3 fold: 3929 s, 3957 s, 4061 s) |
| `STAGE = "full"`: model cuối, đánh giá trên test | [`notebooks/kfold_yolo.ipynb`](../notebooks/kfold_yolo.ipynb) | [v6](https://www.kaggle.com/code/phanlvnminh/k-fold-yolo11n-map-t-ng-nh-t-o-nh-n-cho-select?scriptVersionId=356235920) | 3,1 giờ (train 10.732 s) |

Môi trường: Kaggle, 2 × T4, Ultralytics 8.4.174.

## Thiết lập

- Ba phiên bản của mỗi ảnh: `orig`, `scnet`, `semiuir`. Cùng tên file, cùng nhãn, ảnh cạnh dài 640.
- **Folds:** chia 9.800 ảnh train thành 3 fold (KFold, seed 0, hash `ea44b385539c`).
  - Mỗi fold: một YOLO11n train 30 epoch trên cả 3 phiên bản của 2 fold còn lại (≈ 19.600 ảnh).
  - Dùng `last.pt` dự đoán cả 3 phiên bản của fold giữ lại (conf 0,001, IoU NMS 0,7).
- **Full:** một YOLO11n train 50 epoch trên cả 3 phiên bản của toàn bộ train (≈ 29.400 ảnh).
  - Dự đoán mỗi phiên bản của 4.200 ảnh test một lần.
  - Mọi chiến lược được chấm bằng cách ghép các dự đoán đã lưu, không chạy lại detector.
- **mAP từng ảnh:** pycocotools, mAP@0.5:0.95 tính riêng cho từng ảnh. Ảnh không có ground-truth được gán 0.
- **Nhãn Selector:** `y = 0` (gốc) nếu `max_k m_k − m_0 < 0,01`, ngược lại `y = argmax_k m_k`.

## 1. Nhãn Selector trên train (out-of-fold, 9.800 ảnh)

| Phiên bản | mAP@0.5 (cả tập) | mAP@0.5:0.95 (cả tập) | mAP@0.5:0.95 trung bình từng ảnh | Số nhãn |
|---|---|---|---|---|
| Ảnh gốc | 0,8200 | 0,5808 | 0,6873 | 5.704 (58,2%) |
| SCNet | 0,8180 | 0,5785 | 0,6859 | 2.208 (22,5%) |
| Semi-UIR | 0,8138 | 0,5729 | 0,6817 | 1.888 (19,3%) |
| Oracle | — | — | 0,7164 (+0,029) | |

## 2. Đánh giá trên test (4.200 ảnh, model cuối)

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5:0.95 trung bình từng ảnh |
|---|---|---|---|
| Ảnh gốc | **0,8468** | **0,6155** | **0,7186** |
| SCNet toàn bộ | 0,8449 | 0,6129 | 0,7154 |
| Semi-UIR toàn bộ | 0,8422 | 0,6084 | 0,7130 |
| Chọn ngẫu nhiên | 0,8444 | 0,6120 | 0,7148 |
| Selector `ce+tau` (của nhóm) | 0,8474 | 0,6159 | 0,7182 |
| Oracle (cận trên) | 0,8524 | 0,6296 | 0,7417 |

- Nhãn oracle trên test: ảnh gốc 2.620 (62,4%), SCNet 846 (20,1%), Semi-UIR 734 (17,5%).
- Khoảng có thể giành được: +0,0142 mAP@0.5:0.95 cả tập, +0,0231 mAP từng ảnh.
- Công thức gap closed: (Selector − Ảnh gốc) / (Oracle − Ảnh gốc).

### Detector baseline và detector train trên 3 phiên bản

Cả hai đều chấm trên ảnh test gốc bằng validator của Ultralytics. Validator này nội suy khác pycocotools, nên chỉ so sánh với nhau, không so với bảng trên.

| Detector | Dữ liệu train | Epoch | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLO11n baseline (tuần 1) | chỉ ảnh gốc, 9.800 ảnh | 50 | 0,8446 | 0,6037 |
| YOLO11n model cuối | 3 phiên bản, 29.400 ảnh | 50 | 0,854 | 0,621 |

Train trên cả 3 phiên bản không làm hại ảnh gốc; ngược lại còn tăng +0,018 mAP@0.5:0.95, nhiều khả năng do nhiều dữ liệu hơn và ảnh tăng cường đóng vai trò augmentation.

### AP theo lớp của model cuối trên ảnh test gốc (validator Ultralytics)

| Lớp | Ảnh | Đối tượng | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| holothurian | 756 | 2.382 | 0,863 | 0,702 | 0,800 | 0,501 |
| echinus | 806 | 3.425 | 0,881 | 0,846 | 0,897 | 0,532 |
| scallop | 437 | 2.373 | 0,834 | 0,686 | 0,798 | 0,501 |
| starfish | 780 | 2.373 | 0,864 | 0,824 | 0,889 | 0,555 |
| fish | 966 | 4.001 | 0,785 | 0,679 | 0,762 | 0,527 |
| corals | 798 | 2.751 | 0,769 | 0,675 | 0,740 | 0,552 |
| diver | 932 | 1.997 | 0,910 | 0,915 | 0,947 | 0,751 |
| cuttlefish | 810 | 1.607 | 0,949 | 0,948 | 0,972 | 0,842 |
| turtle | 915 | 1.296 | 0,951 | 0,944 | 0,970 | 0,843 |
| jellyfish | 325 | 764 | 0,738 | 0,705 | 0,769 | 0,608 |
| **all** | 4.200 | 22.969 | 0,854 | 0,792 | 0,854 | 0,621 |

## 3. So với Awad et al., J. Imaging 2026 (Bảng 6, RUOD)

| | Awad et al. | Nhóm |
|---|---|---|
| Detector | YOLO-NAS-L, 800×600, 300 epoch; mỗi phiên bản ảnh một detector riêng | YOLO11n, 640, 50 epoch; một detector chung cho 3 phiên bản |
| Số phiên bản ảnh | 10 (gốc + 9 phương pháp UIE) | 3 (gốc, SCNet, Semi-UIR) |
| mAP từng ảnh, ảnh gốc | 0,68 | **0,719** |
| mAP từng ảnh, Semi-UIR | 0,67 | 0,713 |
| Mixed / oracle | 0,77 (+0,09) | 0,742 (+0,023) |
| Bộ chọn thật (không biết GT) | không có | Selector ResNet18 (đang làm) |

- Kết luận giống nhau: tăng cường đồng loạt làm giảm mAP, chọn cho từng ảnh thì tăng. Ảnh gốc chiếm phần lớn các lựa chọn tốt nhất.
- Đường cơ sở của nhóm cao hơn (0,719 so với 0,68). Cách tính mAP từng ảnh có thể khác nhau: nhóm dùng pycocotools, bài báo tự cài đặt.
- Oracle của bài báo lớn hơn vì hai lý do:
  - Chọn từ 10 phiên bản: lấy max trên nhiều ứng viên hơn thì max cao hơn.
  - Mỗi phiên bản có detector riêng: lấy max còn gồm cả hiệu ứng đa dạng giữa các model, không chỉ hiệu ứng của tăng cường ảnh.
- Mixed / oracle là cận trên chọn bằng ground-truth, không phải kết quả đạt được. Bài báo không có bộ chọn học được; đây là phần đóng góp của nhóm.

## 4. Nhận xét dùng cho slide và báo cáo

1. Tăng cường đồng loạt không giúp detection: SCNet −0,0026, Semi-UIR −0,0071 mAP@0.5:0.95 so với ảnh gốc. Chọn ngẫu nhiên cũng thấp hơn ảnh gốc.
2. Chọn đúng cho từng ảnh có thể tăng: oracle +0,0142 cả tập, +0,0231 từng ảnh. Đây là trần cho Selector.
3. Phân bố nhãn train (58/23/19%) và nhãn oracle test (62/20/17%) gần nhau. Selector học trên train có cơ sở áp dụng được cho test.
4. Khoảng có thể giành được nhỏ, nên báo cáo Selector cần có gap closed và khoảng tin cậy bootstrap, không chỉ một con số mAP.

## 5. File output trên Kaggle

| Version | File |
|---|---|
| v4 (folds) | `kfold/folds.csv`, `kfold/labels_train.csv`, `kfold/summary_folds.json`, `kfold/preds_oof_{orig,scnet,semiuir}.json.gz`, `kfold/weights/fold{0,1,2}.pt` |
| v6 (full) | `kfold/per_image_test.csv`, `kfold/summary_full.json`, `kfold/preds_test_{orig,scnet,semiuir}.json.gz`, `kfold/weights/full.pt`, `kfold/weights/full_results.csv` |

Trang Output của Kaggle hiển thị thư mục `weights` là "empty" ở cả hai version. Log của v6 cho thấy `weights/full.pt` (5,2 MB) đã được lưu. Các file `fold{0,1,2}.pt` của v4 được code lưu cùng chỗ nhưng chưa kiểm tra; cần kiểm tra trước khi dùng lại, ví dụ để thêm ACDC.

## 6. Selector

Đã chạy [`notebooks/selector_resnet18.ipynb`](../notebooks/selector_resnet18.ipynb); chi tiết trong [`selector_results.md`](selector_results.md).
- Selector `ce+tau` đạt 0,6159 mAP@0.5:0.95, ngang ảnh gốc: gap closed +3,2% theo mAP cả tập, −1,5% theo mAP từng ảnh. Khoảng tin cậy 95% của chênh lệch chứa 0.
- Selector an toàn hơn tăng cường đồng loạt và chọn ngẫu nhiên.

## 7. Bước tiếp theo

- Đã chạy `analysis_noise_fusion` (xem [`analysis_noise_fusion.md`](analysis_noise_fusion.md)):
  - oracle {gốc, gốc lật} tăng bằng hoặc hơn oracle của tăng cường, tức khoảng oracle chủ yếu là nhiễu;
  - WBF 4 góc nhìn +0,0023 là phương pháp duy nhất vượt ảnh gốc có ý nghĩa thống kê.
- Thêm dòng YOLO baseline chấm bằng pycocotools trên cùng tập test, để bảng 2 dùng một cách tính thống nhất.
