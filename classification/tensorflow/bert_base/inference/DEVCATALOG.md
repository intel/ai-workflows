# TensorFlow BERT Base Inference - AWS SageMaker

Run inference on the BERT base model with TensorFlow on Intel's hardware on Amazon Sagemaker.

Check out more workflow examples and reference implementations in the
[Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
This workflow demonstrates how you can use Intel’s CPU hardware (Cascade Lake or above) and related optimized software to perform cloud inference on the Amazon Sagemaker platform. A step-by-step Jupyter notebook is provided to perform the following:

1. Specify AWS information
2. Build a custom docker image for inference
3. Deploy the TensorFlow model using Sagemaker, with options to change the instance type and number of nodes
4. Preprocess the input data and send it to the endpoint

For detailed information about the workflow, go to the [Cloud Training and Cloud Inference on Amazon Sagemaker/Elastic Kubernetes Service](https://github.com/intel/NLP-Workflow-with-AWS) GitHub repository and follow the instructions for AWS inference.

## Hardware Requirements
We recommend you use the following hardware for this reference implementation.
| **Name**                          | **Description**
| :---                              | :---
| CPU                               | Intel® Xeon® processor family, 2nd Gen or newer
| Usable RAM                        | 16 GB
| Disk Size                         | 256 GB

## How it Works
In this workflow, we'll use a pretrained BERT Base model and perform inference with the Amazon Sagemaker infrastructure. This diagram shows the architecture for both training and inference but only the inference path is demonstrated in this workflow.

### Architecture
![sagemaker_architecture](https://user-images.githubusercontent.com/43555799/207917598-ec21b0c5-0915-4a3b-a5e2-33458051f286.png)

### Model Spec
In this workflow, we use the uncased BERT base model. Parameters can be changed depending on your requirements.  

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

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Main Repository](https://github.com/intel/NLP-Workflow-with-AWS) repository into your working directory.
```
mkdir ~/work && cd ~/work
git clone https://github.com/intel/NLP-Workflow-with-AWS.git
cd NLP-Workflow-with-AWS.git
git checkout main
```
### Download the Datasets
No Intel-supplied dataset is needed, but you will need your own input for inference.

## Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.

### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

Because the Docker image is run on a cloud service, you will need Azure credentials to perform inference related operations:
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)


### Set Up AWS Credentials
You will need AWS credentials and the related AWS CLI installed on the machine to push data/docker image to the Amazon Elastic Container Registry (Amazon ECR).

Set up an [AWS Credential Account](https://aws.amazon.com/account/) and [configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) it.

### Set Up Docker Image
Pull the provided docker image.
```
docker pull intel/ai-workflows:nlp-aws-sagemaker
```
If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

### Run Docker Image
Provide your own dataset and a value for ``path to dataset`` and
run the workflow using the ``docker run`` command, as shown. 
```
export DATASET_DIR=<path to dataset>
export OUTPUT_DIR=/output
docker run -a stdout $DOCKER_RUN_ENVS \
  --env DATASET=${DATASET} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:/output \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm --pull always \
  intel/ai-workflows:nlp-aws-sagemaker
```

### Sagemaker Inference
After starting the container, execute the following command in the interactive shell.
```
cd /root/notebooks
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```
Start the notebook with "intel-sagemaker-inference" in the filename.

---

## Run Using Bare Metal
This workflow requires Docker and cannot be run using bare metal.  

## Expected Output
Running the Docker image should give the following output:  
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

After running inference, the Jupyter notebook will print out the prediction value as a number which represents the confidence or probability the two input sentences have the same meaning. A value of 0 means the sentences do not have the same meaning. A value of 1 means the sentences should have the same meaning.  

## Summary and Next Steps
In this workflow, you loaded a Docker image and performed inference on a TensorFlow BERT base model on Amazon Sagemaker using Intel® Xeon® Scalable Processors. The [GitHub repository](https://github.com/intel/NLP-Workflow-with-AWS/tree/main) also contains workflows for training on Sagemaker and training and inference on Elastic Kubernetes Service (EKS).  

## Learn More
For more information or to read about other relevant workflow
examples, see these guides and software resources:

- Put ref links and descriptions here, for example
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- links to other similar or related items from the dev catalog

## Troubleshooting
Issues, problems, and their workarounds if possible, will be listed here.

## Support
We track bugs and enhancement requests using [GitHub issues](https://github.com/intel/NLP-Workflow-with-AWS/issues).  Search through these issues before submitting your own bug or enhancement request.
