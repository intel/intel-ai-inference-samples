## Serving models with IPEX® and PyTorch backend on Triton Server

## Description
This readme provides a methodology to run Intel® Extension for PyTorch (IPEX) optimized model on triton server.

## Preparation
- Docker installed on host instance.
- Sample images from ImageNet dataset. 

### Execution on localhost

#### 1 Copy the IPEX model at desired directory 

Place the ipex optimized model at the /model_repository

#### 2 Create and Run Triton container 

`$ docker build -t tritonserver_ipex -f Dockerfile .` 

`$ docker run -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/model_repository:/models --name ai_inference_host tritonserver_ipex:latest tritonserver --model-repository=/models`

#### 3 Run inference with a client script

`$ python3 client_imagenet.py --dataset /home/ubuntu/ImageNet/imagenet_images `  - sends requests to Triton Server Host for sample model. This file uses ImagesNet images for inference. 

## Additional info
Downloading and loading models take some time, so please wait until you run client_imagenet.py.
Model loading progress can be tracked by following Triton Server Host docker container logs.

## Support
Please submit your questions, feature requests, and bug reports on the [GitHub issues page](https://github.com/intel/intel-ai-inference-samples/issues).

## License 
AI Inference samples project is licensed under Apache License Version 2.0. Refer to the [LICENSE](../LICENSE) file for the full license text and copyright notice.

This third party software, even if included with the distribution of the Intel software, may be governed by separate license terms, including without limitation, third party license terms, other Intel software license terms, and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the [THIRD-PARTY-PROGRAMS](./THIRD-PARTY-PROGRAMS) file.

## Trademark Information
Intel, the Intel logo, OpenVINO, the OpenVINO logo and Intel Xeon are trademarks of Intel Corporation or its subsidiaries.
* Other names and brands may be claimed as the property of others.

&copy;Intel Corporation


