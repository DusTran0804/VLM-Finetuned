# Sử dụng image Python 3.10 cơ bản
FROM python:3.10-slim

# Cài đặt thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho OpenCV (libgl1-mesa-glx, libglib2.0-0) và Git (cho việc tải repo)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt requirements trước (tận dụng cache của Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào image
COPY . .

# Thiết lập thư mục config cho Ultralytics (tránh lỗi permission denied khi dùng thư mục mặc định của hệ thống)
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"

# Mở cổng mặc định của Hugging Face Spaces là 7860
EXPOSE 7860

# Lệnh khởi chạy FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
