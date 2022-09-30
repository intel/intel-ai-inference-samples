# Copyright (c) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

FROM nvcr.io/nvidia/tritonserver:22.03-py3
ENV WORK_DIR=/workspace

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN sed -i '50d' /etc/apt/sources.list && \
    apt-get update && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    apt-get install --no-install-recommends -y numactl=2.0.12-1 \
                       libjemalloc-dev=5.2.1-1ubuntu1 \
                       google-perftools=2.7-1ubuntu2 && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/lib/x86_64-linux-gnu/libjemalloc.so /usr/lib/x86_64-linux-gnu/libtcmalloc.so && \
    apt-get clean

ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so:/usr/local/lib/libiomp5.so":${LD_PRELOAD}
ENV DNNL_PRIMITIVE_CACHE_CAPACITY=1024
ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"
ENV TOKENIZERS_PARALLELISM=true
