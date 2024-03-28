ARG CUDA_VERSION="12.1.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG PYTHON_VERSION="3.10"

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y wget git build-essential && rm -rf /var/lib/apt/lists/* \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace
RUN git clone --depth=1 https://github.com/google-research/bleurt.git && pip install ./bleurt
RUN git clone --depth=1 https://github.com/Y-IAB/BARTScore && pip install -e ./BARTScore
RUN git clone --depth=1 https://github.com/Y-IAB/lm-evaluation-harness && pip install -e ./lm-evaluation-harness[openai]

WORKDIR /workspace/lm-evaluation-harness