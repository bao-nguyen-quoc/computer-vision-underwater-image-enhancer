# Kết quả review:

## Điểm đã làm đúng:

> Bản này đã sửa được phần lớn lỗi lần trước: giọng văn hết khẳng định quá mức, bỏ được "Proposed UIE Oracle", có thêm câu thừa nhận chỉ chạy 1 detector, 1 lần train, và mục V-F đã nói rõ detector đa miền được train trên nhiều ảnh hơn.

## Điểm đã làm sai:

1. Sai format. Format yêu cầu là chuẩn **springer template**. Hiện tại đang dùng **IEEE format conference**.
2. Sai số liệu:
    1. **Hai giá trị oracle không khớp nhau:** 
        Bảng III ghi oracle 0,6296 (+0,0231 theo từng ảnh).
        Bảng VI ghi 0,6300 (+0,0234) cho cùng bộ {gốc, SCNet, Semi-UIR}. 
        Tương tự, Bảng III cho thấy 1.580 ảnh không chọn bản gốc (37,6%), còn Bảng VI ghi 44,2%. 
        *Nguyên nhân:* Bảng III dùng nhãn có ngưỡng δ = 0,01, Bảng VI lấy max thuần.
        *Cách sửa:* thêm một câu hoặc chú thích ở bảng giải thích điểm khác này.
    2. **Mục V-C-3 hiểu sai con số 846 và 734:**
        Bài viết "SCNet improved mAP on 846 test images", nhưng 846 là số ảnh có nhãn oracle là SCNet, tức SCNet tốt nhất trong 3 bản và hơn bản gốc ít nhất δ. Số ảnh SCNet hơn bản gốc thực tế nhiều hơn: theo Bảng VI, oracle {gốc, SCNet} chọn SCNet ở 31,6% số ảnh. Câu về Semi-UIR (734) bị lỗi tương tự.
    3. **Bảng I ghi sai SSIM của oracle:**
        Bài ghi **0,9010**, trùng với số Semi-UIR công bố. Mình tính lại từ **uieb_metrics.csv** thì được **0,9006**. PSNR **24,95** thì đúng.
    4. **Weight decay của ResNet18 sai.**
        Bài ghi 10⁻², nhưng code make_selector_nb.py dùng weight_decay=1e-4. AdamW, lr 3e-4 và 15 epoch thì khớp.
    5. **YOLO11n có thể không dùng AdamW.**
        Điểm này tạm không kiểm chứng được nên đừng đề cập tới AdamW trong file nữa.
    6. **Bảng VIII chấm bằng thước đo khác.**
        Bảng VIII dùng Ultralytics validator, nên cùng một model full.pt có 0,6210 ở Bảng VIII nhưng 0,6155 ở Bảng III. Người đọc sẽ thấy mâu thuẫn. Nên chuyển sang số pycocotools cho thống nhất: A = 0,5995, C = 0,6155.
3. Bổ sung về nội dung:
    1. **Nên thêm A150 vào Bảng VIII.**
        Repo đã có số A150 = **0,6150**, gần bằng C = **0,6155**. Số này trả lời luôn câu hỏi mà mục V-F đang bỏ ngỏ: nếu cùng số lượt ảnh, trên ảnh gốc thì C không hơn. Nhưng nếu thêm A150 mà không có phần độ bền thì câu kết "enhancement was more useful as training data" trong abstract và mục V-F sẽ bị yếu đi.
    2. **Thiếu phần độ bền (robustness).**
        Đây là bằng chứng mạnh nhất cho câu kết trên: C hơn A150 ở 12/12 điều kiện suy giảm, CI > 0, và giữ được 68,5% mAP so với 65,0%. Bạn nên thêm một mục ngắn với 1 bảng hoặc hình. Nếu không thêm thì nên viết nhẹ câu kết lại.
    3. **Bảng III, dòng baseline vẫn ghi p_boot = 1.000.**
        Nên đổi thành "–"
    4. Gạch đầu dòng cuối ở phần Introduction (0,6037 → 0,6210) chưa có câu lưu ý "but sees 3× more images" như trong mục V-F.
4. Vấn đề với tài liệu tham khảo:
    1. **trích nguồn YOLO11** ghi sai phiên bản. Ghi "Version 8.0.0, 2024", trong khi bài dùng 8.4.174 và YOLO11 ra năm 2024. Nên ghi phiên bản 8.3 hoặc bản thực dùng.
5. Vấn đề với trình bày:
    1. Bảng VIII bị đẩy xuống trang 6, nằm sau phần tài liệu tham khảo. Trang 6 vì thế gần như trống. Nên đặt [t] cho bảng này gần mục V-F, hoặc gộp nó vào Bảng VII.
    2. Hình 1 không được nhắc đến trong phần thân bài. Nếu không có tác dụng gì thì nên bỏ.
    3. Abstract ghi "reaches the accuracy of raw images". Nên đổi thành "matches the mAP of raw images" vì chữ accuracy dễ nhầm với độ chính xác phân loại 56,9%.

## Yêu cầu bổ sung:

Để thuận tiện cho một số vấn đề kiểm soát content. Yêu cầu cập nhật file tex, đồng thời tạo thêm 2 file khác là:

- full_raw_vn.md (bản raw dịch tiếng Anh sang tiếng Việt)
- full_raw_en.md (bản raw tiếng Anh)

Cả hai chứa full raw content của LaTeX main.tex để trong 1 số trường hợp có thể ngay lập tức lấy raw content ra để chỉnh sửa trong các yêu cầu xuất định dạng khác pdf hoặc LaTeX.

## Ràng buộc khi chỉnh sửa content:

- **Format** Chắc chắn phải cập nhật sang chuẩn springer template thay vì IEEE conference template.
- **Ngôn ngữ:**
    - Không dùng các trạng từ thể hiện sự khẳng định mức độ cao: "significantly", "greatly", "dramatically", "remarkably", "substantially" và các từ tương tự.
    - Không dùng các từ có hàm ý công kích các bài báo hoặc tác giả khác.
    - Câu dùng trong bài nên có độ dài ngắn tương đối hỗn loạn. Tránh các câu có độ dài quá đều nhau.
- **Các số liệu và thông tin mã nguồn:**
    - Đang có nếu đúng thì tiếp tục sử dụng, nếu sai thì sửa chữa.
    - Cần đảm bảo phù hợp với thực tế của project. Bạn được đọc các folder `results/`, `notebooks/` để có thêm thông tin hoặc kiểm tra nếu cần.

## Yêu cầu:

Chỉnh sửa lại bài báo dựa trên các yêu cầu và ràng buộc trên

---

# Kết quả review:

Các vấn đề cũ đã được fix, vấn đề mới được cải thiện đúng.

## Yêu cầu bổ sung:

Tại thư mục `paper/figures`, đã có lần lượt 4 ảnh sau:
| Tên file ảnh | Chú thích hình ảnh tương ứng trong paper (bản en) | Vị trí kỳ vọng xuất hiện |
| --- | --- | --- |
| sample_enhancement.png | Sample image before and after enhancement. From left to right: raw input, Semi-UIR and SCNet. | Ở cuối phần Methodology > System Overview |
| pipeline_en.png | Pipeline. Each training image is enhanced by SCNet and Semi-UIR; out-of-fold YOLO11n detectors score the three versions per image; the best version (with margin δ) becomes the label of a ResNet18 selector that sees only the raw image. | Ở cuối phần Methodology > System Overview |
| robustness_en.png | mAP@0.5:0.95 of the four detectors as severity increases for each degradation type (clean = undegraded test images).  | Ở trong phần 5.7 |
| degradation_grid.jpg | Synthetic degradations applied to one RUOD test image. Rows: dark, turbid, color cast, blur + noise; columns: severity levels 1–3. | Ở trong phần 5.7 |

## Yêu cầu:

- Cập nhật nội dung file LaTeX ở các phần bị ảnh hưởng sao cho phù hợp với các thông tin và xuất hiện của các ảnh mới.
- Cập nhật lại các file `full_raw_en.md` và `full_raw_vn.md`, nhất quán với nội dung được cập nhật ở file tex.
- Chỉ sau khi việc cập nhật hoàn tất, tạo file main_vn.tex, chứa nội dung là bản dịch VN của paper. Các thuật ngữ trong phần tiếng việt nếu có thể thì giữ nguyên thuật ngữ tiếng Anh tương ứng, đảm bảo nội dung nhất quán giữa 4 file: `main.tex`, `main_vn.tex`, `full_raw_en.md` và `full_raw_vn.md`.
- Một số hình ảnh chỉ đang có version tiếng Anh, hãy sử dụng luôn cho bản tex tiếng Việt.