ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

LABEL maintainer='zhaoshu@iie.ac.cn'

RUN apt-get update \
    && apt-get install -y tmux git vim wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

RUN pip install loguru wandb

WORKDIR /workspace