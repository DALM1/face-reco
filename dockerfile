FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    qt5-qmake \
    qtbase5-dev \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir opencv-python

RUN pip install --no-cache-dir PyQt5==5.15.6

CMD ["python", "face_detection.py"]
