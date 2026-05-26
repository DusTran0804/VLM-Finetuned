# License Plate Recognition (LPR) với YOLOv8 & Fine-Tuned VLM (Qwen2-VL)

![LPR Demo UI Mockup](assets/demo_ui.png)

Dự án này là một hệ thống **Nhận diện Biển số xe (LPR - License Plate Recognition)** kết hợp sức mạnh của mô hình phát hiện vật thể **YOLOv8** và mô hình ngôn ngữ thị giác lớn (VLM) **Qwen2-VL-2B-Instruct** được tinh chỉnh (fine-tuned) bằng **Unsloth** để thực hiện OCR biển số xe với độ chính xác cao.

---

## Tổng Quan Hệ Thống

Hệ thống hoạt động theo quy trình 2 giai đoạn (Two-stage Pipeline):

1. **Giai đoạn 1: Phát hiện Biển số (Object Detection)**
   - Sử dụng mô hình **YOLOv8** (`license_plate_detector_yolov8.pt`) để phát hiện và khoanh vùng biển số từ khung hình/ảnh đầu vào.
   - Ảnh biển số được cắt (crop) ra từ ảnh gốc để chuẩn bị cho giai đoạn OCR.
2. **Giai đoạn 2: Nhận diện Ký tự (VLM OCR)**
   - Sử dụng mô hình **Qwen2-VL-2B-Instruct** đã được tinh chỉnh bằng kỹ thuật **LoRA** trên **Unsloth** (`unsloth_finetune`).
   - VLM sẽ đọc ảnh biển số đã crop và đưa ra văn bản biển số chính xác, hạn chế tối đa việc nhầm lẫn giữa các ký tự tương tự nhau (như `O` và `0`, `I` và `1`).

---

## Yêu Cầu Hệ Thống & Cài Đặt

### 1. Yêu Cầu Phần Cứng
* **Huấn luyện (Training):** Khuyên dùng GPU hỗ trợ **CUDA** (như Google Colab, Server Linux) vì thư viện `unsloth` được tối ưu hóa tối đa cho các GPU NVIDIA.
* **Suy luận (Inference/Chạy Streamlit):**
  * Hỗ trợ **CUDA GPU** (chạy cực nhanh bằng Unsloth).
  * Hỗ trợ **macOS (Apple Silicon M-Series)**: Ứng dụng tự động tận dụng **MPS (Metal Performance Shaders)** để gia tốc phần cứng trên các dòng Mac M1/M2/M3...
  * Hỗ trợ **CPU**: Tự động fallback chạy trên CPU thông thường nếu không phát hiện được GPU CUDA hoặc Apple Silicon.

### 2. Cài Đặt Môi Trường
Để chạy ứng dụng trên máy cục bộ hoặc máy chủ, cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

Nếu bạn chạy ứng dụng trên **macOS** hoặc chạy **chỉ bằng CPU** (không có card đồ họa NVIDIA CUDA), bạn cần cài đặt thêm thư viện `peft` và `accelerate` để nạp adapter LoRA cục bộ mà không cần cài đặt `unsloth`:
```bash
pip install peft accelerate
```

> [!NOTE]
> * **Khi chạy trên CUDA:** Ứng dụng sẽ sử dụng `unsloth` để tải mô hình 4-bit giúp tiết kiệm tối đa VRAM.
> * **Khi chạy trên macOS/CPU:** Ứng dụng sẽ tự động chuyển sang sử dụng `transformers` + `peft` để tải mô hình gốc `Qwen/Qwen2-VL-2B-Instruct` và nạp adapter LoRA từ thư mục `Models/unsloth_finetune/`. Lần đầu chạy trên macOS/CPU, hệ thống sẽ tự động tải mô hình gốc (~4.5 GB) từ Hugging Face.

---

## Cấu Trúc Thư Mục Dự Án

```text
VLM-FineTuned/
├── Models/
│   ├── license_plate_detector_yolov8.pt  # Mô hình YOLOv8 phát hiện biển số
│   └── unsloth_finetune/                 # Thư mục lưu adapter LoRA của Qwen2-VL sau khi fine-tune
├── Image/                                # Thư mục chứa hình ảnh đầu vào (tùy chọn)
├── Result/                               # Thư mục lưu kết quả xử lý (tự động lưu ảnh vẽ khung, ảnh crop, CSV log)
├── main.py                               # Ứng dụng Streamlit UI chạy LPR (hỗ trợ CUDA, macOS MPS, và CPU)
├── requirements.txt                      # Các thư viện phụ thuộc của dự án
└── src.ipynb                             # Notebook hướng dẫn cài đặt & Fine-tune Qwen2-VL trên Google Colab
```

---

## Huấn Luyện Mô Hình (Fine-Tuning)

Mô hình VLM OCR được tinh chỉnh thông qua file notebook `src.ipynb` trên Google Colab. Các bước chính:

1. **Mô hình gốc:** `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit` (đã được lượng tử hóa 4-bit giúp tiết kiệm VRAM tối đa).
2. **Dataset huấn luyện:** `EZCon/taiwan-license-plate-recognition` trên Hugging Face.
3. **Cấu hình LoRA (Unsloth):**
   - Tinh chỉnh các tầng thị giác (vision layers) và ngôn ngữ (language layers).
   - Thiết lập `r = 16`, `lora_alpha = 16`, sử dụng `SFTTrainer` từ thư viện `trl`.
4. **Lưu mô hình:** Kết quả sau khi huấn luyện được lưu vào thư mục `Models/unsloth_finetune`.

---

## Giao Diện Trực Quan với Streamlit

Giao diện Web được xây dựng bằng **Streamlit** hỗ trợ nhận diện đa dạng chế độ đầu vào. Chạy ứng dụng bằng lệnh:
```bash
streamlit run main.py
```

### Các Tính Năng & Chế Độ Hỗ Trợ:

- **Image Upload**: Tải lên ảnh đơn (`.jpg`, `.png`). Hệ thống sẽ phát hiện biển số, vẽ khung chữ nhật (bounding box) kèm biển số nhận diện trực tiếp lên ảnh nguồn, hiển thị ảnh biển số phóng to và tự động lưu kết quả (ảnh vẽ khung, ảnh biển số crop và file báo cáo CSV) vào thư mục `Result/`.
- **Video Upload**: Tải lên video bài test (`.mp4`, `.avi`). Hệ thống tự động trích xuất các biển số xe xuất hiện trong video, hiển thị dưới dạng **Grid Gallery** trực quan, đồng thời lưu các ảnh biển số đã cắt và nhật ký vào thư mục `Result/`.
- **Webcam (local)**: Nhận diện biển số thời gian thực qua webcam của máy tính. *(Lưu ý: Không hoạt động khi deploy trên Google Colab do giới hạn quyền truy cập camera cục bộ)*.
- **RTSP / IP Camera**: Hỗ trợ luồng stream RTSP (từ IP camera, đầu ghi hình) hoặc HTTP MJPEG stream để giám sát giao thông trực tiếp.

### Tự Động Lưu Kết Quả:
Sau mỗi lượt xử lý ảnh hoặc video thành công, hệ thống tự động lưu kết quả vào thư mục `Result/`:
* **`Result/[tên_file]_annotated.jpg`**: Ảnh gốc đã được vẽ khung xanh xung quanh biển số và chèn text biển số xe màu đỏ.
* **`Result/[tên_file]_plate_[stt]_[biển_số].jpg`**: Ảnh crop cận cảnh biển số xe.
* **`Result/results_log.csv`**: File log tổng hợp lịch sử nhận diện (gồm cột mốc thời gian, tên file nguồn, biển số xe nhận diện được, tọa độ bounding box và đường dẫn lưu ảnh).

### Tinh Chỉnh Cài Đặt (Sidebar):
- **Show FPS**: Hiển thị tốc độ khung hình xử lý thực tế trên màn hình stream.
- **Show bounding boxes & text**: Vẽ hộp bao màu xanh lá quanh biển số xe kèm văn bản OCR trực tiếp trên video stream.
- **Max plates to display**: Giới hạn số lượng biển số xe hiển thị/phân tích trên mỗi khung hình (từ 1 - 10).
- **Process every N-th frame**: Bước nhảy khung hình khi phân tích video giúp giảm tải GPU mà vẫn giữ được độ chính xác (mặc định xử lý mỗi 5 khung hình).

---

## Lưu Ý Quan Trọng
- **Hiệu Năng trên Mac M-Series**: Khi chạy trên macOS (MPS GPU), tốc độ suy luận sẽ nhanh hơn đáng kể so với CPU thông thường. Trong lần chạy đầu tiên, hãy kiên nhẫn để Hugging Face tải mô hình gốc `Qwen2-VL-2B-Instruct` từ internet về máy (dung lượng khoảng 4.5 GB).
- **Sử dụng GPU**: Qwen2-VL cần xử lý tính toán lớn. Nếu chạy hoàn toàn bằng CPU thông thường, tốc độ xử lý (FPS) sẽ bị giảm sút.
- **Lỗi Luồng RTSP**: Nếu luồng camera IP bị gián đoạn, hãy kiểm tra lại thông tin tài khoản đăng nhập trong URL RTSP hoặc độ trễ mạng của Camera.

