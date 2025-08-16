FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN pip3 install --upgrade pip && pip3 install .
COPY . .
EXPOSE 8000 9100
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
