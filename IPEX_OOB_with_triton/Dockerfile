# Copyright (c) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

FROM nvcr.io/nvidia/tritonserver:23.05-py3

# this installs utils such as numactl and libjemalloc 
RUN sed -i '50d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y numactl \
                       libjemalloc-dev && \
    apt-get clean

# DOWNLOAD IPEX
# The versions of the IPEX C++ library can be found here:
#   https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html#install-c-sdk
# The version of IPEX needs to align with the version of PyTorch on
# the tritonserver Docker image that you're using. For example,
# the Docker image nvcr.io/nvidia/tritonserver:23.05-py3 comes with PyTorch 2.0.0,
# as found here:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2023
RUN BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && \
	curl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip -o ${BASEDIR}/libtorch.zip && \
	unzip -o libtorch.zip -d ${BASEDIR} && \
	curl https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/cpu/libintel-ext-pt-cxx11-abi-2.0.0%2Bcpu.run -o ${BASEDIR}/ipex.run && \
	bash ${BASEDIR}/ipex.run install ${BASEDIR}/libtorch && \
	cp ${BASEDIR}/libtorch/lib/libintel-ext-pt-cpu.so /opt/tritonserver/backends/pytorch/
# When in the Docker container, you can now run tritonserver like this:
# LD_PRELOAD="/opt/tritonserver/backends/pytorch/libintel-ext-pt-cpu.so ${LD_PRELOAD}" tritonserver --model-repository=/models

# you can append ", dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" to malloc conf for optimal performance but these can sometimes cause OOM crash
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto" 
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so /opt/tritonserver/backends/pytorch/libintel-ext-pt-cpu.so ${LD_PRELOAD}"
ENV DNNL_PRIMITIVE_CACHE_CAPACITY=1024
ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"
