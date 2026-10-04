# VN30 Portfolio Analysis & Forecasting System

Ứng dụng web phân tích dữ liệu thị trường chứng khoán Việt Nam, áp dụng mô hình **CAPM** (Capital Asset Pricing Model) để ước lượng rủi ro hệ thống (Beta) và mô hình **ARIMA** để dự báo giá cổ phiếu, minh họa qua cổ phiếu ACB (Ngân hàng TMCP Á Châu) và rổ VN30.

Dự án thực hiện trong học phần **Gói phần mềm 1**, GVHD: ThS. Ngô Phú Thanh — Đặng Mai Hoa (K234141706).

## Tính năng chính

- **Thu thập & tiền xử lý dữ liệu**: đọc dữ liệu giá 30 cổ phiếu thuộc rổ VN30 và chỉ số VNINDEX từ file CSV (nguồn Investing.com), xử lý dữ liệu thiếu bằng phương pháp Forward Fill, tính lợi suất ngày và lợi suất tháng.
- **Mô hình CAPM**: hồi quy tuyến tính lợi suất từng cổ phiếu theo lợi suất thị trường (VNINDEX) để ước lượng hệ số Beta, Alpha, R², đánh giá mức độ rủi ro hệ thống của từng mã.
- **Dự báo ARIMA**: áp dụng mô hình ARIMA(5,1,0) để dự báo lợi suất/giá cổ phiếu ACB trong 30 ngày tới, kèm khoảng tin cậy 95%.
- **Xây dựng danh mục đầu tư**: tự động tạo 2 danh mục chiến lược — Ổn định (5 mã có Beta thấp nhất) và Mạo hiểm (5 mã có Beta cao nhất) — và so sánh hiệu suất kỳ vọng giữa hai danh mục.
- **Trực quan hóa tương tác**: biểu đồ phân phối lợi suất, biểu đồ phân tán Beta, biểu đồ Risk-Return Trade-off, biểu đồ phân bổ danh mục, dựng bằng Plotly.

## Phát hiện nổi bật

Phân tích trên dữ liệu 5 năm (2020–2025) cho thấy kết quả đi ngược lại giả định "rủi ro cao đi kèm lợi nhuận cao": danh mục Ổn định (Beta thấp, gồm FPT, DGC, GVR...) đạt lợi suất kỳ vọng vượt trội (+58.02%/năm), trong khi danh mục Mạo hiểm (Beta cao, gồm nhóm Ngân hàng/Bất động sản) lại cho lợi suất kỳ vọng âm sâu (-120.74%/năm) — phản ánh sự thiếu hiệu quả của thị trường Việt Nam trong giai đoạn nghiên cứu.

## Công nghệ sử dụng

- **Python**: ngôn ngữ xử lý và phân tích chính
- **Streamlit**: xây dựng giao diện web tương tác
- **Pandas / NumPy**: xử lý và biến đổi dữ liệu
- **statsmodels**: mô hình ARIMA
- **SciPy**: hồi quy tuyến tính cho mô hình CAPM
- **Plotly**: trực quan hóa dữ liệu

## Cấu trúc dự án

```
├── app.py              # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt     # Danh sách thư viện cần cài đặt
└── README.md            # Tài liệu mô tả dự án (file này)
```

## Hướng dẫn chạy ứng dụng

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. **Lưu ý quan trọng**: ứng dụng đọc dữ liệu giá cổ phiếu từ các file CSV đặt trong biến `data_folder` ở đầu file `app.py`. Trước khi chạy, cần cập nhật đường dẫn này trỏ tới thư mục chứa dữ liệu CSV (giá các mã VN30 và VNINDEX, định dạng tải từ Investing.com) trên máy của bạn.

3. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Báo cáo phân tích đầy đủ

Báo cáo chi tiết (phương pháp luận, kết quả, nhận xét, khuyến nghị mở rộng mô hình với GARCH/Fama-French 3 yếu tố) được trình bày trong file PDF đính kèm repo này.

## Tác giả

**Đặng Mai Hoa** — K234141706
Sinh viên ngành Công nghệ Tài chính (Fintech), Trường Đại học Kinh tế – Luật (UEL)
