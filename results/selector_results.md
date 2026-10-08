# Kết quả Selector ResNet18 (RUOD test)

Notebook: [`notebooks/selector_resnet18.ipynb`](../notebooks/selector_resnet18.ipynb). Lần chạy: [Kaggle selector-resnet18, version 356318687](https://www.kaggle.com/code/phanlvnminh/selector-resnet18?scriptVersionId=356318687), GPU T4. Bảng chi tiết nằm trong `results/selector/` (làm tròn 4 chữ số, chép từ output notebook).

## Thiết lập

- **Input:**
  - nhãn out-of-fold `labels_train.csv` của notebook K-fold, version folds (9.800 ảnh, folds hash `ea44b385539c`);
  - dự đoán test đã lưu của version full (4.200 ảnh).
- **Selector:**
  - ResNet18 pre-train ImageNet, 3 đầu ra. Chỉ nhìn ảnh gốc, resize 224 (cache 240, crop ngẫu nhiên và lật ngang khi train, không color jitter).
  - Chia 80/20 phân tầng theo nhãn (7.840 train / 1.960 val), 15 epoch, AdamW, lr 3e-4.
- **Hai cách học:**
  - `ce`: cross-entropy có trọng số lớp.
  - `reg`: hồi quy mAP của 3 phiên bản.
- **Ngưỡng `tau`:** chỉ tăng cường khi lợi thế dự đoán so với ảnh gốc lớn hơn `tau`. `tau` chọn trên val.
- **Chọn checkpoint và biến thể chính:** theo chosen-mAP trên val. Biến thể chính là **`ce+tau`** (`tau = 0,625`). Tập test chỉ dùng để báo cáo.
- **Chấm điểm:** ghép dự đoán YOLO đã lưu theo lựa chọn từng ảnh, rồi tính pycocotools trên test.
  - Các dòng luôn gốc / luôn SCNet / luôn Semi-UIR / oracle khớp đúng `summary_full.json` (notebook tự kiểm tra, cả 4 dòng đều OK).

## Kết quả chính trên test (4.200 ảnh)

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | mAP từng ảnh | Gap closed (cả tập / từng ảnh) | Δ từng ảnh so với gốc [CI 95%] | Chọn gốc/SCNet/Semi-UIR |
|---|---|---|---|---|---|---|
| Ảnh gốc | 0,8468 | 0,6155 | 0,7186 | 0% / 0% | — | 4200/0/0 |
| SCNet toàn bộ | 0,8449 | 0,6129 | 0,7154 | −18,1% / −13,6% | −0,0031 [−0,0049; −0,0013] | 0/4200/0 |
| Semi-UIR toàn bộ | 0,8422 | 0,6084 | 0,7130 | −50,0% / −24,2% | −0,0056 [−0,0077; −0,0034] | 0/0/4200 |
| Chọn ngẫu nhiên | 0,8444 | 0,6120 | 0,7148 | −24,7% / −16,3% | −0,0038 [−0,0054; −0,0021] | 1387/1401/1412 |
| Selector `ce` | 0,8472 | 0,6146 | 0,7173 | −6,4% / −5,4% | −0,0012 [−0,0025; 0,0000] | 2210/1101/889 |
| **Selector `ce+tau` (chính)** | **0,8474** | **0,6159** | 0,7182 | **+3,2%** / −1,5% | −0,0004 [−0,0010; +0,0003] | 3593/319/288 |
| Selector `reg` | 0,8457 | 0,6142 | 0,7182 | −8,8% / −1,7% | −0,0004 [−0,0018; +0,0009] | 2133/1373/694 |
| Selector `reg+tau` | 0,8462 | 0,6146 | 0,7183 | −5,9% / −1,4% | −0,0003 [−0,0014; +0,0008] | 3019/828/353 |
| Oracle (cận trên) | 0,8524 | 0,6296 | 0,7417 | 100% / 100% | +0,0231 [+0,0216; +0,0247] | 2620/846/734 |

Tốc độ Selector: **7,7 ms/ảnh** trên T4 (đọc file, resize và ResNet18, mỗi lần 1 ảnh).

## Trên tập val (1.960 ảnh)

| Biến thể | tau | chosen-mAP | Gap closed |
|---|---|---|---|
| Luôn gốc | — | 0,6924 | 0% |
| `ce` | 0 | 0,6923 | −0,4% |
| `ce+tau` | 0,625 | 0,6931 | +2,5% |
| `reg` | 0 | 0,6918 | −2,4% |
| `reg+tau` | 0,040 | 0,6931 | +2,2% |
| Oracle | — | 0,7202 | 100% |

Diễn biến khi train `ce`: loss train giảm từ 1,15 xuống 0,28, nhưng chosen-mAP trên val chỉ dao động trong khoảng 0,687–0,692. Mô hình học thuộc tập train chứ không học được quy luật tổng quát.

## Ma trận nhầm lẫn của `ce+tau` (hàng: nhãn oracle, cột: Selector chọn)

| Test (acc 0,569) | gốc | SCNet | Semi-UIR |
|---|---|---|---|
| gốc | 2259 | 176 | 185 |
| SCNet | 708 | 83 | 55 |
| Semi-UIR | 626 | 60 | 48 |

So sánh: đoán "luôn gốc" có accuracy 0,624 trên test.

## AP@0.5:0.95 theo lớp

| Lớp | Luôn gốc | Luôn SCNet | Luôn Semi-UIR | Selector `ce+tau` | Oracle |
|---|---|---|---|---|---|
| holothurian | 0,4966 | 0,4910 | 0,4877 | 0,4966 | 0,5102 |
| echinus | 0,5305 | 0,5250 | 0,5187 | 0,5298 | 0,5446 |
| scallop | 0,4957 | 0,4941 | 0,4791 | 0,4958 | 0,5058 |
| starfish | 0,5508 | 0,5409 | 0,5431 | 0,5506 | 0,5652 |
| fish | 0,5198 | 0,5196 | 0,5181 | 0,5191 | 0,5337 |
| corals | 0,5386 | 0,5367 | 0,5338 | 0,5377 | 0,5519 |
| diver | 0,7456 | 0,7467 | 0,7397 | 0,7462 | 0,7638 |
| cuttlefish | 0,8360 | 0,8320 | 0,8315 | 0,8361 | 0,8487 |
| turtle | 0,8383 | 0,8390 | 0,8330 | 0,8384 | 0,8518 |
| jellyfish | 0,6026 | 0,6041 | 0,5991 | 0,6089 | 0,6208 |

## Nhận xét dùng cho slide và báo cáo

1. **Selector ngang ảnh gốc, chưa vượt có ý nghĩa thống kê:**
   - mAP cả tập chỉ cao hơn 0,0004 (gap closed 3,2%).
   - Khoảng tin cậy 95% của chênh lệch mAP từng ảnh so với ảnh gốc ([−0,0010; +0,0003]) chứa 0.
2. **Selector an toàn hơn hẳn tăng cường đồng loạt và chọn ngẫu nhiên:** ba cách đó đều kém ảnh gốc có ý nghĩa (khoảng tin cậy nằm hẳn dưới 0). Với `tau` chọn trên val, Selector giữ ảnh gốc cho 86% ảnh.
3. **Selector gần như không học được tín hiệu từ ảnh:**
   - accuracy thấp hơn đoán "luôn gốc";
   - với ảnh mà oracle chọn SCNet, Selector chỉ đoán đúng 83/846;
   - mô hình học thuộc tập train nhưng val không cải thiện.
4. **Vì sao Selector không học được (đã kiểm chứng bằng [`analysis_noise_fusion`](analysis_noise_fusion.md)):**
   - Oracle {gốc, gốc lật} tăng +0,0207 từng ảnh, bằng 144% mức tăng của {gốc, SCNet}.
   - Khoảng oracle (+0,023 từng ảnh) vì vậy chủ yếu là nhiễu của detector theo từng ảnh, không phải do đặc điểm hình ảnh.
   - Bộ chọn chỉ dựa trên hình ảnh không thể khai thác khoảng này. Con số 0,68 → 0,77 của Awad et al. phóng đại lợi ích của tăng cường có chọn lọc.
5. **Chi phí:** Selector chỉ thêm 7,7 ms/ảnh. Nếu chọn Semi-UIR (khoảng 2 s/ảnh) thì chi phí lớn nằm ở bước tăng cường, nên giữ ảnh gốc cho phần lớn ảnh còn tiết kiệm thời gian.
