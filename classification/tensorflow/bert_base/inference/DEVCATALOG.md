
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

#### Output
```
$ AWS_CSV_FILE=./aws_config.csv S3_MODEL_URI="s3://model.tar.gz" ROLE="role" make nlp-sagemaker
[+] Building 0.1s (9/9) FINISHED
 => [internal] load build definition from Dockerfile                                                                                                                                                        0.0s
 => => transferring dockerfile: 32B                                                                                                                                                                         0.0s
 => [internal] load .dockerignore                                                                                                                                                                           0.0s
 => => transferring context: 2B                                                                                                                                                                             0.0s
 => [internal] load metadata for docker.io/library/ubuntu:20.04                                                                                                                                             0.0s
 => [1/5] FROM docker.io/library/ubuntu:20.04                                                                                                                                                               0.0s
 => CACHED [2/5] RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata=2022c-0ubuntu0.20.04.0 --no-install-recommends && rm -rf /var/lib/apt/lists/*                    0.0s
 => CACHED [3/5] RUN apt-get -y update && apt-get install -y --no-install-recommends          wget=1.20.3-1ubuntu2          nginx=1.18.0-0ubuntu1.3          cmake=3.16.3-1ubuntu1          software-prope  0.0s
 => CACHED [4/5] RUN pip install --no-cache-dir boto3==1.24.15 &&    pip install --no-cache-dir sagemaker==2.96.0 &&    pip install --no-cache-dir tensorflow-cpu==2.9.1 &&    pip install --no-cache-dir   0.0s
 => CACHED [5/5] RUN pip install --no-cache-dir virtualenv==20.14.1 &&     virtualenv intel_neural_compressor_venv &&     . intel_neural_compressor_venv/bin/activate &&     pip install --no-cache-dir Cy  0.0s
 => exporting to image                                                                                                                                                                                      0.0s
 => => exporting layers                                                                                                                                                                                     0.0s
 => => writing image sha256:91b43c6975feab4db06cf34a9635906d2781102a05d406b93c5bf2eb87c30a94                                                                                                                0.0s
 => => naming to docker.io/library/intel_amazon_cloud_trainandinf:inference-ubuntu-20.04                                                                                                                    0.0s
[+] Running 1/0
 ⠿ Container bert_uncased_base-aws-sagemaker-1  Created                                                                                                                                                     0.0s
Attaching to bert_uncased_base-aws-sagemaker-1
bert_uncased_base-aws-sagemaker-1  | [NbConvertApp] Converting notebook 1.0-intel-sagemaker-inference.ipynb to python
bert_uncased_base-aws-sagemaker-1  | [NbConvertApp] Writing 3597 bytes to 1.0-intel-sagemaker-inference.py
bert_uncased_base-aws-sagemaker-1  | update_endpoint is a no-op in sagemaker>=2.
bert_uncased_base-aws-sagemaker-1  | See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
bert_uncased_base-aws-sagemaker-1  | ---!/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (1.26.12) or chardet (3.0.4) doesn't match a supported version!
bert_uncased_base-aws-sagemaker-1  |   warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
bert_uncased_base-aws-sagemaker-1 exited with code 0
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
