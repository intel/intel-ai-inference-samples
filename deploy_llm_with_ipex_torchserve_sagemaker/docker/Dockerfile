# Copyright (c) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG PYTHON=python3
ARG PYTHON_VERSION=3.10.13
ARG MINIFORGE3_VERSION=24.3.0-0

FROM ubuntu:20.04 AS sagemaker

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

ARG PYTHON
ARG PYTHON_VERSION
ARG MINIFORGE3_VERSION

ENV TORCHSERVE_VERSION="0.11.0"
ENV SM_TOOLKIT_VERSION="2.0.22"
ENV SAGEMAKER_SERVING_MODULE sagemaker_pytorch_serving_container.serving:main

# Set Env Variables for the images
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"
ENV PYTHONIOENCODING=UTF-8
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV TEMP=/home/model-server/tmp
ENV MKL_THREADING_LAYER=GNU

ENV DLC_CONTAINER_TYPE=inference

RUN apt-get -y update \
 && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    jq \
    libcurl4-openssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjemalloc-dev \
    google-perftools \
    libsm6 \
    libssl-dev \
    libxext6 \
    libxrender-dev \
    numactl \
    openjdk-17-jdk \
    openssl \
    unzip \
    vim \
    wget \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Install CondaForge miniconda
RUN curl -L -o ~/miniforge3.sh https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE3_VERSION}/Miniforge3-${MINIFORGE3_VERSION}-Linux-x86_64.sh \
 && chmod +x ~/miniforge3.sh \
 && ~/miniforge3.sh -b -p /opt/conda \
 && rm ~/miniforge3.sh \
 && /opt/conda/bin/conda install -c conda-forge -y \
    python=${PYTHON_VERSION} \
    cython \
    "mkl>=2023.2.0" \
    mkl-include \
    parso \
    "scipy==1.10.1" \
    typing \
    h5py \
    requests \
    libgcc \
    cmake \
    packaging \
    awscli \
    "boto3==1.28.60" \
    pyyaml \
    packaging \
    conda-content-trust \
    charset-normalizer \
    scikit-learn \
    pandas \
    "numpy==1.24.4" \
    botocore \
    s3transfer || true \
 && /opt/conda/bin/conda clean -afy \
 && rm -rf /etc/apt/sources.list.d/*

# symlink pip for OS use
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
 && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

RUN pip install --no-cache-dir sagemaker-pytorch-inference==${SM_TOOLKIT_VERSION}

# Install Common python packages
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -U \
    opencv-python \
    pyopenssl \
    "cryptography>41.0.6" \
    "ipython>=8.10.0,<9.0" \
    "urllib3>=1.26.18,<2" \
    "prompt-toolkit<3.0.39"

# Install Pytorch
RUN pip install --no-cache-dir -U torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Intel® Extension for PyTorch*
RUN pip install --no-cache-dir -U \
    intel-openmp \
    intel-extension-for-pytorch==2.3.100

RUN python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

RUN pip install --no-cache-dir -U \
    transformers==4.33.2 \
    accelerate \
    tiktoken

# Install TorchServe pypi dependencies directly from their requirements.txt file
# NOTE: This also brings in unnecessary cpu dependencies like nvgpu
RUN pip install --no-cache-dir -U -r https://raw.githubusercontent.com/pytorch/serve/v${TORCHSERVE_VERSION}/requirements/common.txt \
 && pip install --no-cache-dir -U \
    torchserve==${TORCHSERVE_VERSION} \
    torch-model-archiver==${TORCHSERVE_VERSION}

# create user and folders
RUN useradd -m model-server \
 && mkdir -p /home/model-server/tmp /opt/ml/model \
 && chown -R model-server /home/model-server /opt/ml/model

# add TS entrypoint
COPY torchserve-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
COPY config.properties /home/model-server
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 && rm -rf ${HOME_DIR}/oss_compliance*

## Cleanup ##
RUN pip cache purge \
 && rm -rf /tmp/tmp* \
 && rm -iRf /root/.cache

# INTEL specific ENVs
ENV KMP_SETTINGS=1
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so:/opt/conda/lib/libiomp5.so:${LD_PRELOAD}"

EXPOSE 8080 8081
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["torchserve", "--start", "--ts-config", "/home/model-server/config.properties", "--model-store", "/home/model-server/"]

