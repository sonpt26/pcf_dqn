# Sử dụng image Python 3.10.14 làm base
FROM python:3.10.14-bookworm

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cập nhật danh sách gói và cài đặt các gói bổ sung
RUN apt-get update && apt-get install -y \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ nội dung của thư mục hiện tại vào thư mục làm việc trong container
COPY . .

# Xác định lệnh mặc định khi khởi động container
CMD ["sleep", "infinity"]

