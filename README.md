# Motor Imagery Classification

## Giới thiệu

Dự án này sử dụng các phương pháp học máy và học sâu vào việc phân loại tín hiệu điện não đồ (EEG) trong các bài tập Motor Imagery để áp dụng cho các ứng dụng BCI. Mục tiêu là phát triển một mô hình có khả năng phân loại chính xác các tín hiệu EEG thu được trong quá trình người tham gia tưởng tượng các động tác.

## Yêu cầu hệ thống

- Python 3.x

## Tải và cài đặt

### Clone Repository

```sh
git clone https://github.com/longluv1605/motor-imagery-classification.git
cd motor-imagery-classification
```

### Cài đặt thư viện

Cài đặt các thư viện cần thiết bằng pip:

```sh
pip install -r requirements.txt
```

## Chạy chương trình

### Chạy notebook

Mở và chạy các notebook trong thư mục để thực hiện các bước tiền xử lý và phân loại

### Chạy scripts

- Chạy lệnh sau để huấn luyện Spectral CNN

    ```sh
    python train_cnn.py
    ```

- Chạy lệnh sau để huấn luyện EEGNet

    ```sh
    python train_eeg.py
    ```

- Chạy lệnh sau để tinh chỉnh siêu tham số cho EEGNet

    ```sh
    python tuning_eeg.py
    ```

## Cấu trúc thư mục

- `data/`: Thư mục chứa dữ liệu EEG.
- `models/`: Thư mục chứa source code của mô hình.
- `utils/`: Thư mục chứa các package hỗ trợ cho việc thực hiện dự án
- `save/`: chứa model, tham số sau khi huấn luyện

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc muốn đóng góp, vui lòng liên hệ qua [longtrong53@gmail.com](mailto:longtrong53@gmail.com).
