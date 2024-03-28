ARG CUDA_VERSION="12.1.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

RUN apt-get update && apt-get install -y wget git build-essential

WORKDIR /workspace
RUN git clone --depth=1 https://github.com/google-research/bleurt.git && pip install ./bleurt
RUN git clone --depth=1 https://github.com/Y-IAB/BARTScore && pip install -e ./BARTScore
RUN git clone --depth=1 https://github.com/Y-IAB/lm-evaluation-harness && pip install -e ./lm-evaluation-harness[openai]

WORKDIR /workspace/lm-evaluation-harness