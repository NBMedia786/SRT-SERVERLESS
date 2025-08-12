# Base image with CUDA + PyTorch. You can change GPU/CUDA version to match your pod type.
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Workdir
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    torchvision\
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    librosa \
    scipy \
    webrtcvad-wheels \
    soundfile \
    faster-whisper \
    streamlit \
    runpod

COPY handler.py ./
COPY transcription_system.py ./

# Expose the worker
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL_SIZE=large-v3
ENV WHISPER_COMPUTE_TYPE=float16
ENV WHISPER_LANGUAGE=en
ENV WHISPER_VAD_FILTER=false

# RunPod serverless worker entrypoint
#CMD ["python", "-m", "runpod.serverless.worker", "--handler", "handler.run"]
CMD ["python", "handler.py"]