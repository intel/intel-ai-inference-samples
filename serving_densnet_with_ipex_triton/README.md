# Serving DenseNet models with IPEX and Triton Server

## Description
This sample provide code to integrate Intel® Extension for PyTorch (IPEX) with Triton Inference Server framework. This project provides custom python backend for IPEX and can be used as performance benchmark for DenseNet.

## Preparation
Make sure that Docker is installed on both host and client instance.
Sample images from ImageNet dataset. 

## Supported models
Currently AI Inference samples support following Bert models finetuned on Squad dataset:
- DenseNet121        - PyTorch+IPEX [DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/ "DenseNet121")

## Possible run scenarios
AI Inference samples allow user to run inference on localhost or on remote Triton Server Host. 
By default config.properties is filled with localhost run option. 

### Execution on localhost
To build, start Docker containers, run tests, stop and do cleanup on localhost execute scripts in following order:

`$ docker build -t tritonserver_custom -f Dockerfile.ipex .`  - builds Docker image for Triton Server.

`$ bash start.sh`  - runs Docker containers for Triton Server Host.

`$ python3 client_imagenet.py --dataset /home/ubuntu/ImageNet/imagenet_images `  - sends requests to Triton Server Host for DenseNet model. This file uses ImagesNet images for inference. 

## Additional info
Downloading and loading models take some time, so please wait until you run client_imagenet.py.
Model loading progress can be tracked by following Triton Server Host docker container logs.

## Support
Please submit your questions, feature requests, and bug reports on the [GitHub issues page](https://github.com/intel/intel-ai-inference-samples/issues).

## License 
AI Inference samples project is licensed under Apache License Version 2.0. Refer to the [LICENSE](../LICENSE) file for the full license text and copyright notice.

This distribution includes third party software governed by separate license terms.

3-clause BSD license:
- [model.py](./model_repository/densenet/1/model.py) -  for PyTorch (IPEX)

This third party software, even if included with the distribution of the Intel software, may be governed by separate license terms, including without limitation, third party license terms, other Intel software license terms, and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the [THIRD-PARTY-PROGRAMS](./THIRD-PARTY-PROGRAMS) file.

## Trademark Information
Intel, the Intel logo, OpenVINO, the OpenVINO logo and Intel Xeon are trademarks of Intel Corporation or its subsidiaries.
* Other names and brands may be claimed as the property of others.

&copy;Intel Corporation

