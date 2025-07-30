FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install Python dependencies
COPY runpod_requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install bitsandbytes phù hợp CUDA 11.8
RUN pip install --no-cache-dir bitsandbytes==0.42.0

# Fix xformers: gỡ bản PyPI, cài bản phù hợp CUDA 11.8
RUN pip uninstall -y xformers && \
    pip install --no-cache-dir xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy handler
COPY runpod_handler.py /handler.py

# Set environment variable for your HuggingFace model
ENV MODEL_NAME="thuanhero1/llama3-8b-finetuned-ctu"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/handler.py"]