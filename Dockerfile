FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install Python dependencies
COPY runpod_requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY runpod_handler.py /handler.py

# Set environment variable for your HuggingFace model
ENV MODEL_NAME="thuanhero1/llama3-8b-finetuned-ctu"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/handler.py"]