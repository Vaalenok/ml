FROM seemeai/paddleocr:3.2.0-3.3.1

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

RUN pip uninstall -y paddlepaddle

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
RUN cp /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.8 && \
    cp /opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12 /usr/lib/libcublas.so.11 && \
    cp /opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12 /usr/lib/libcublasLt.so.11 && \
    cp /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so && \
    ln -s /usr/lib/libcublas.so.11 /usr/lib/libcublas.so && \
    ldconfig

RUN python3 -c "from paddleocr import PaddleOCR; import numpy as np; ocr = PaddleOCR(lang='ru', use_gpu=False); ocr.ocr(np.zeros((1, 1, 3), dtype='uint8'))"

COPY paddle_api.py .

RUN mkdir -p /tmp/uploads

EXPOSE 8000

ENTRYPOINT []

CMD ["python3", "-m", "uvicorn", "paddle_api:app", "--host", "0.0.0.0", "--port", "8000"]