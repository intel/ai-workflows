
# TensorFlow BERT Base Inference - AWS SageMaker

## Overview
This is a workflow to demonstrate how users utilize the Intel’s hardware (Cascade Lake or above) and related optimized software to perform cloud inference on the Amazon Sagemaker platform. For detailed information about the workflow, go to [Cloud Training and Cloud Inference on Amazon Sagemaker/Elastic Kubernetes Service](https://github.com/intel/NLP-Workflow-with-AWS) GitHub repository.

## How it Works
Takes a pretrained BERT Base model and utilizes the AWS infrastructure to perform inference. The diagram below shows the architecture for both training and inference but for this workflow, only the inference diagram is of interest.

### Architecture
![sagemaker_architecture](https://user-images.githubusercontent.com/43555799/207917598-ec21b0c5-0915-4a3b-a5e2-33458051f286.png)

### Model Spec
The uncased BERT base model is used to demonstrate this workflow.

```python
bert-base-uncased-config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 128,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.21.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

## Get Started

### **Prerequisites**

#### Download the repo
```
git clone https://github.com/intel/NLP-Workflow-with-AWS.git
cd NLP-Workflow-with-AWS.git
git checkout v0.2.0
```

### **Docker**
Below setup and how-to-run sessions are for users who want to use provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).
#### Setup 
Docker is required to start this workflow. You will also need AWS credentials and the related AWS CLI installed on the machine to push data/docker image to the Amazon ECR.

Set up an [AWS Credential Account](https://aws.amazon.com/account/) and [configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) it.

##### Pull Docker Image

```
docker pull intel/ai-workflows:nlp-aws-sagemaker
```

#### How to run 

(Optional) Export related proxy into docker environment.
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the pipeline, follow below instructions outside of docker instance. 
```
docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  --privileged --rm --init -it \
  --net=host \
  intel/ai-workflows:nlp-aws-sagemaker \ 
  /bin/bash
```

##### Sagemaker Inference
After starting the container, execute the following command in the interactive shell.
```
cd /root/notebooks
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```
Start the notebook with "intel-sagemaker-inference" in the filename.

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 
TBD
#### How to run 
TBD

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation.   
| **Name**                          | **Description**
| :---                              | :---
| CPU                               | Intel® Xeon® processor family, 2nd Gen or newer
| Usable RAM                        | 16 GB
| Disk Size                         | 256 GB

## Useful Resources
[Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

## Support
Cloud inference on AWS Sagemaker tracks both bugs and enhancement requests using Github. We welcome input, however, before filing a request, please make sure you do the following: Search the Github issue database.
