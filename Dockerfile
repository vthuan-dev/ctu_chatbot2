FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Xóa sạch torch cũ (nếu có)
RUN pip uninstall -y torch torchvision torchaudio || true

# Install Python dependencies
COPY runpod_requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install --no-cache-dir bitsandbytes==0.42.0
RUN pip install --no-cache-dir unsloth_zoo
RUN pip uninstall -y xformers && \
    pip install --no-cache-dir xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118

COPY runpod_handler.py /handler.py

ENV MODEL_NAME="thuanhero1/llama3-8b-finetuned-ctu"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/handler.py"]