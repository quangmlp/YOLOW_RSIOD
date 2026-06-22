# YOLO-IOD: Phát hiện Vật thể Lũy tiến Dựa trên YOLO-World

## 🌟 Giới thiệu

**YOLO-IOD** là một khung cấu trúc tiên tiến dành cho bài toán **Phát hiện Vật thể Lũy tiến (Incremental Object Detection - IOD)**, được phát triển dựa trên khả năng nhận diện tập từ vựng mở (open-vocabulary) mạnh mẽ của [YOLO-World](https://github.com/AILab-CVC/YOLO-World). 

Khi học thêm các nhóm vật thể mới một cách lũy tiến, các bộ phát hiện truyền thống thường gặp phải hiện tượng "quên lãng thảm khốc" (catastrophic forgetting) đối với các lớp đã học trước đó. **YOLO-IOD** giải quyết thách thức này bằng cách giới thiệu:
- **Chưng cất tri thức đa phương thức (Multimodal Knowledge Distillation)**: Một phương pháp chưng cất chéo toàn diện (kết hợp chưng cất phân loại và chưng cất hồi quy hộp bao) giúp căn chỉnh các đặc trưng hình ảnh và ngôn ngữ.
- **Gán nhãn giả mạnh mẽ (Robust Pseudo-Labeling)**: Tận dụng khả năng zero-shot mạnh mẽ của YOLO-World để tạo ra các nhãn giả có độ tin cậy cao cho các lớp cũ mà không cần sử dụng các hình ảnh gốc từ tập dữ liệu cũ.

Kho lưu trữ này tập trung vào các kịch bản ảnh viễn thám phức tạp và cung cấp cấu hình đầy đủ cho các tập dữ liệu **DIOR** và **DOTA**.

---

## 📂 Cấu trúc Kho lưu trữ

- `yolo_world/`: Mã nguồn cốt lõi của YOLO-IOD bao gồm các mạng xương sống đa phương thức (multimodal backbones), nhánh cổ PAFPN được tùy chỉnh, các hàm mất mát chưng cất chéo (cross-distillation loss), và bộ tiền xử lý dữ liệu.
- `third_party/mmyolo/`: Khung cấu trúc MMYolo được tùy chỉnh để làm hệ sinh thái nền tảng.
- `configs/`: Các cấu hình hoàn chỉnh cho các tác vụ học lũy tiến (ví dụ: DIOR `10+10` và DOTA `5+5+5`).
- `tools/`: Các script huấn luyện và kiểm thử (`train.py`, `test.py`).
- `script/`: Các công cụ tiện ích để tạo nhãn giả, định dạng tập dữ liệu COCO và phân chia dữ liệu.
- `Colab_Notebooks/`: Tập hợp đầy đủ các Jupyter notebook được thiết kế cho Google Colab nhằm trực quan hóa phân phối dữ liệu, kết quả phát hiện và phân tích các trường hợp phát hiện lỗi (failure cases).
- `assets/`: Các sơ đồ và tài nguyên trực quan.

---

## 🛠️ Cài đặt & Thiết lập

Chúng tôi khuyến nghị sử dụng Anaconda để quản lý môi trường.

### 1. Tạo Môi trường Conda
```bash
conda create -n yoloiod python=3.10 -y
conda activate yoloiod
```

### 2. Cài đặt các Thư viện Phụ thuộc
```bash
pip install setuptools==69.5.1
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim

mim install "mmengine>=0.10.3"
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

pip install -r requirements/basic_requirements.txt
pip install "numpy<2" transformers==4.30.2 scikit-learn prettytable albumentations
```

### 3. Cài đặt MMYolo và YOLO-IOD
Cài đặt thư viện `mmyolo` cục bộ và dự án chính:
```bash
# Biên dịch MMYolo
cd third_party/mmyolo 
pip install -v -e . --no-build-isolation

# Biên dịch YOLO-IOD
cd ../..
pip install -v -e . --no-build-isolation
```

---

## 📊 Chuẩn bị Dữ liệu

Tải xuống tập dữ liệu **DIOR** và **DOTA**, sau đó chuyển đổi các nhãn của chúng sang định dạng COCO JSON tiêu chuẩn.
Cấu trúc các tập dữ liệu bên trong thư mục `data/` như sau:

```
data/
├── DIOR/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotation/
│       └── annotation/
│           ├── train_task_1.json
│           ├── test_task_1.json
│           ├── ...
└── DOTA/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotation/
        └── annotation/
            ├── train_task_0.json
            ├── test_task_0.json
            ├── ...
```

*(Lưu ý: Đảm bảo các đường dẫn khớp với tệp cấu hình của bạn. Kiểm tra thư mục `script/` để tìm các công cụ chuyển đổi nhãn).*

---

## 🚀 Huấn luyện & Đánh giá

YOLO-IOD được huấn luyện lũy tiến qua nhiều giai đoạn (tác vụ - tasks).
Ví dụ: trong thiết lập **DIOR 10+10**, trước tiên bạn huấn luyện trên 10 lớp cơ sở (Task 0), sau đó huấn luyện tuần tự trên 10 lớp mới (Task 1).

### DIOR (10 + 10)
```bash
# Giai đoạn 1: Huấn luyện Task 0 (Các lớp cơ sở)
python tools/train.py configs/dior_10_10/yolo_iod_dior_10_10_task0.py

# Giai đoạn 2: Huấn luyện Task 1 (Các lớp lũy tiến mới)
python tools/train.py configs/dior_10_10/yolo_iod_dior_10_10_task1.py
```

### DOTA (5 + 5 + 5)
```bash
# Giai đoạn 1: Huấn luyện Task 0 (Các lớp cơ sở)
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task0.py

# Giai đoạn 2: Huấn luyện Task 1 (Các lớp lũy tiến mới)
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task1.py

# Giai đoạn 3: Huấn luyện Task 2 (Các lớp lũy tiến mới tiếp theo)
python tools/train.py configs/dota_5_5_5/yolo_iod_dota_5_5_5_task2.py
```

### Kiểm thử (Testing)
Để kiểm thử trọng số (checkpoint) đã lưu của một tác vụ cụ thể:
```bash
python tools/test.py \
    configs/dior_10_10/yolo_iod_dior_10_10_task1.py \
    work_dirs/yolo_iod_dior_10_10_task1/epoch_20.pth
```

---

## 🔬 Trực quan hóa & Suy luận (Jupyter Notebooks)

Thư mục `Colab_Notebooks/` chứa các notebook tự chạy phục vụ cho việc trực quan hóa tương tác. Các notebook này đã được cấu hình để chạy hiệu quả trên môi trường Google Colab:

1. **`Notebook_1_DataStats.ipynb`**: Phân tích phân phối lớp học và thống kê kích thước vật thể trên các tập dữ liệu DIOR và DOTA.
2. **`Notebook_2_Detection_Results.ipynb`**: Thực hiện suy luận và trực quan hóa kết quả dự đoán của mô hình cạnh nhau bằng công cụ trực quan hóa `MMDet`.
3. **`Notebook_3_Failure_Cases.ipynb`**: Được thiết kế để phân lập và phân tích các trường hợp phát hiện lỗi cụ thể: *Phân loại sai (Misclassification)*, *Bỏ sót vật thể (Missing objects)*, và *Phát hiện nhầm (False Detections)*.
4. **`Notebook_4_Ablation_and_Arch.ipynb`**: Vẽ biểu đồ, theo dõi nhật ký huấn luyện chi tiết và các nghiên cứu thực nghiệm cắt bỏ (ablation studies) qua các lượt chạy.

---

## 🧬 Nghiên cứu Thực nghiệm Cắt bỏ (Ablation Studies)

Để tái tạo các nghiên cứu thực nghiệm cắt bỏ tự động được mô tả trong báo cáo, hãy thực thi lệnh:
```bash
python run_ablation_v5.py
```
Script này sẽ huấn luyện tuần tự nhiều cấu hình khác nhau (Baseline, Full Model, +Pseudo-Labeling, +Distillation), trích xuất các chỉ số đánh giá (mAP, mAP_50) và tạo bảng so sánh hiệu năng chi tiết giữa các tác vụ.

---

## 📄 Giấy phép
Dự án này được phát hành theo các điều khoản trong tệp [LICENSE](LICENSE) đi kèm.

## 🎓 Trích dẫn & Ghi nhận
YOLO-IOD tích hợp kiến trúc từ dự án [YOLO-World](https://github.com/AILab-CVC/YOLO-World) và [MMDetection](https://github.com/open-mmlab/mmdetection). Chúng tôi vô cùng trân trọng đóng góp mã nguồn mở của các tác giả.
