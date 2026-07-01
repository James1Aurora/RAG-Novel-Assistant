FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Sistem deps (minimal) dan pemasangan dependensi Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
