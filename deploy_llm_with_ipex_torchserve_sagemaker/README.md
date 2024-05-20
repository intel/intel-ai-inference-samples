# Deploy optimized LLM model with Intel® Extension for PyTorch (IPEX) and Torchserve on AWS Sagemaker

## Description
This sample provide code to deploy a LLM model with Intel® Extension for PyTorch (IPEX) with Torchserve on AWS Sagemaker on instances supporting 4th Gen Intel® Xeon® Scalable Processors. The notebook provided builds a Docker container, pushes it to AWS ECR, creates a torchserve file and puts in on AWS S3 bucket and creates and invokes an AWS Sagemaker endpoint. The pipeline was tested on [Codegen model](https://huggingface.co/Salesforce/codegen25-7b-multi_P).

## Execution
Please refer to [E2E-LLM-Sagemaker-IPEX.ipynb](E2E-LLM-Sagemaker-IPEX.ipynb) for more information.

## Support
Please submit your questions, feature requests, and bug reports on the [GitHub issues page](https://github.com/intel/intel-ai-inference-samples/issues).

## License 
AI Inference samples project is licensed under Apache License Version 2.0. Refer to the [LICENSE](../LICENSE) file for the full license text and copyright notice.

This distribution includes third party software governed by separate license terms.

This third party software, even if included with the distribution of the Intel software, may be governed by separate license terms, including without limitation, third party license terms, other Intel software license terms, and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the [THIRD-PARTY-PROGRAMS](./THIRD-PARTY-PROGRAMS) file.

## Trademark Information
Intel, the Intel logo, OpenVINO, the OpenVINO logo and Intel Xeon are trademarks of Intel Corporation or its subsidiaries.
* Other names and brands may be claimed as the property of others.

## Human Rights Disclaimer
Intel is committed to respecting human rights and avoiding causing or directly contributing to adverse impacts on human rights. See [Intel's Global Human Rights Policy](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html "Intel's Global Human Rights Policy"). The software licensed from Intel is intended for socially responsible applications and should not be used to cause or contribute to a violation of internationally recognized human rights.

&copy;Intel Corporation
