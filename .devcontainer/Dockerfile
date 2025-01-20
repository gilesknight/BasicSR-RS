FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
    apt-get install -y \
    curl \
    git \
    pkg-config \
    libhdf5-dev

WORKDIR /tmp

COPY .devcontainer/requirements.txt /tmp/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir