# **TensorFlow BERT Base INFERENCE - AWS SageMaker**

## **Description**

This pipeline provides instructions on how to run inference using BERT Base model on infrastructure provided by AWS SageMaker with make and docker compose.

## **Project Structure**
```
├── aws_sagemaker @ v0.2.0
├── Makefile
├── README.md
└── docker-compose.yml
```

[*Makefile*](Makefile)
```
AWS_CSV_FILE ?= credentials.csv
AWS_DATA=$$(pwd)/aws_data
FINAL_IMAGE_NAME ?= nlp-sagemaker
OUTPUT_DIR ?= /output
ROLE ?= role
S3_MODEL_URI ?= link

export AWS_PROFILE := $(shell cat ${AWS_CSV_FILE} | awk -F',' 'NR==2{print $$1}')
export REGION ?= us-west-2

nlp-sagemaker:
	./aws_sagemaker/scripts/setup.sh aws_sagemaker/
	mkdir -p ${AWS_DATA} && cp -r ${HOME}/.aws ${AWS_DATA}/.aws/
	@AWS_PROFILE=${AWS_PROFILE} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 docker compose up --build nlp-sagemaker		
clean:
	if [ -d ${AWS_DATA} ]; then \
		rm -rf ${AWS_DATA}; \
	fi; \
	if [ -d aws/ ]; then \
		rm -rf aws/; \
	fi; \
	if [ -d aws-cli/ ]; then \
		rm -rf aws-cli/; \
	fi; \
	if [ -f awscliv2.zip ]; then \
		rm -f awscliv2.zip; \
	fi
	docker compose down
```

[*docker-compose.yml*](docker-compose.yml)
```
services:
  nlp-sagemaker:
    build:
      context: ./aws_sagemaker/
      args:
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: ./Dockerfile
    command: sh -c "jupyter nbconvert --to python 1.0-intel-sagemaker-inference.ipynb && python3 1.0-intel-sagemaker-inference.py"
    environment:
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
      - AWS_PROFILE=${AWS_PROFILE}
    image: ${FINAL_IMAGE_NAME}:inference-ubuntu-20.04
    network_mode: "host"
    privileged: true
    volumes:
      - ${OUTPUT_DIR}:${OUTPUT_DIR}
      - ./aws_sagemaker/notebooks:/root/notebooks
      - ./aws_sagemaker/src:/root/src
      - ./aws_data/.aws:/root/.aws
    working_dir: /root/notebooks
```

# **AWS SageMaker**

End-to-End AI workflow using the AWS SageMaker Cloud Infrastructure for inference of the BERT base model. More Information [here](https://github.com/intel/NLP-Workflow-with-AWS.git). The pipeline runs the `1.0-intel-sagemaker-inference.ipynb` of the [Intel's AWS SageMaker Workflow](https://github.com/intel/NLP-Workflow-with-AWS/blob/v0.2.0/notebooks/1.0-intel-sagemaker-inference.ipynb) project.

## **Quick Start**

* Pull and configure the dependent repo submodule ```git submodule update --init --recursive ```.

* Install [Pipeline Repository Dependencies](../../../../README.md).
 
* Setup your pipeline specific variables
  * Please, create a key pair using the instructions from this [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-creds-create). NOTE: Please, download the csv file as described in the 7th step of the instructions.
  * Before you start, you need to add execution role for AWS SageMaker. For more information on how to do it follow instructions from "Create execution role" section of this [link](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html). **Note down the role name or the ARN for the role.**
  * You need to have the S3 link to the quantized model generated from training.

* Variables in Makefile: Use variables from the table below to run make command.

  Variable Name | Default | Notes |
  :----------------:|:------------------: | :--------------------------------------:|
  AWS_CSV_FILE | `credentials.csv` | Location of the AWS account credentials file |
  FINAL_IMAGE_NAME  | `nlp-sagemaker` | Final Docker Image Name |
  OUTPUT_DIR | `./output` | Output directory |
  REGION | `us-west-2` | Region of yor S3 bucket and profile |
  ROLE |  `role` | Name or ARN of the role you created for SageMaker |
  S3_MODEL_URI | `link` | URI of the trained and quantized model checkpoint |

## **Build and Run**
Build and run with defaults:

```make nlp-sagemaker```

## **Build and Run Example**
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

