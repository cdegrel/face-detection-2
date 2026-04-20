FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    libopus-dev \
    libvpx-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5005

ENV PYTHONPATH=/app

CMD ["python", "-m", "src"]
