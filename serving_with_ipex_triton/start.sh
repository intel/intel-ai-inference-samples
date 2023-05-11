#!/bin/bash

# Copyright (c) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

print_help() {
    echo ""
    echo "Usage: $0 [arg]"
    echo ""
    echo "   This script starts Docker container for host, client or localhost scenario."
    echo "   If no arguments are specified script will start Docker containers for localhost."
    echo ""
    echo "   Available arguments:"
    echo "   client              - starts client container."
    echo "   host                - starts host container."
    echo "   localhost (default) - starts client and host containers on the same instance."
    echo ""
    exit 2
}

start_host() {
    # Prepare Triton Host Server backend
	model_name="densenet"
    backend_dir="$(pwd)/backend/${model_name}" 
    [ ! -d "${backend_dir}" ] && mkdir -p "$(pwd)/backend/" && cp -fr "$(pwd)/model_repository/densenet" "${backend_dir}" && cp "$(pwd)/model_repository/densenet/config.pbtxt" "${backend_dir}"
    
    # Run Triton Host Server for specified model
    docker run -it --read-only --rm --shm-size=1g -p${1}:8000 -p8001:8001 -p8002:8002 --tmpfs /tmp:rw,noexec,nosuid,size=1g --tmpfs /root/.cache/:rw,noexec,nosuid,size=4g -v$(pwd)/backend:/models --name ai_inference_host tritonserver_custom:latest tritonserver --model-repository=/models --log-verbose 1 --log-error 1 --log-info 1	
}

start_host 8000