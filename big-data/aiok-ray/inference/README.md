# Ray DLRM INFERENCE - Recommendation-Ray
## Description
This document contains instructions on how to run recommendation-ray pipeline with make and docker compose.
## Project Structure 
```
├── AIOK_Ray @ aiok-ray-v0.2
├── DEVCATALOG.md
├── Makefile
├── README.md
└── docker-compose.yml
```
[_Makefile_](Makefile)
```
DATASET_DIR ?= ./data
FINAL_IMAGE_NAME ?= recommendation-ray
CHECKPOINT_DIR ?= /output
RUN_MODE ?= kaggle
DOCKER_NETWORK_NAME = ray-inference

recommendation-ray:
	if [ ! -d "AIOK_Ray/dlrm_all/dlrm/dlrm" ]; then \
		CWD=${PWD}; \
		cd AIOK_Ray/; \
		sh dlrm_all/dlrm/patch_dlrm.sh; \
		cd ${CWD}; \
	fi
	@wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
	-P AIOK_Ray/Dockerfile-ubuntu18.04/ \
	-O AIOK_Ray/Dockerfile-ubuntu18.04/miniconda.sh
	if [ ! "$(shell docker network ls | grep ${DOCKER_NETWORK_NAME})" ]; then \
		docker network create --driver=bridge ${DOCKER_NETWORK_NAME} ; \
	fi
	@DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 CHECKPOINT_DIR=${CHECKPOINT_DIR} \
	 RUN_MODE=${RUN_MODE} \
 	 docker compose up recommendation-ray --build

clean: 
	docker network rm ${DOCKER_NETWORK_NAME}
	DATASET_DIR=${DATASET_DIR} OUTPUT_DIR=${OUTPUT_DIR} docker compose down
```
[_docker-compose.yml_](docker-compose.yml)
```
networks: 
  ray-inference:
    external: true
services:
  recommendation-ray:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: DockerfilePytorch
      context: AIOK_Ray/Dockerfile-ubuntu18.04
    command: 
      - /bin/bash
      - -c
      - |
        bash $$APP_DIR/scripts/run_inference_docker.sh $RUN_MODE
    container_name: ray-inference
    hostname: ray
    networks:
      - ray-inference
    environment:
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
      - RUN_MODE=${RUN_MODE}
      - APP_DIR=/home/vmagent/app/e2eaiok
      - OUTPUT_DIR=/output
    image: ${FINAL_IMAGE_NAME}:training-inference-ubuntu-18.04
    privileged: true
    devices:
      - /dev/dri
    volumes: 
      - ${DATASET_DIR}:/home/vmagent/app/dataset/criteo
      - ./AIOK_Ray:/home/vmagent/app/e2eaiok
      - ${CHECKPOINT_DIR}:/output
    working_dir: /home/vmagent/app/e2eaiok/dlrm_all/dlrm/
    shm_size: 300g
```


# Ray Recommendation System
End2End AI Workflow utilizing Ray framework for simplifying the end-to-end process at large scale. More information [here](https://github.com/intel/e2eAIOK/tree/aiok-ray-v0.2)

## Quick Start
* Pull and configure the dependent repo submodule `git submodule update --init --recursive`.

* Install [Pipeline Repository Dependencies](../../../README.md)

* The model supports following three datasets: kaggle, criteo_small, criteo_full. The instructions to download each of them is provided at [README.md](https://github.com/intel/e2eAIOK/tree/aiok-ray-v0.2/README.md#dataset)

* This pipeline requires the pre-trained model. Please run the [training pipeline](../training/README.md) before running inference to get the trained model.

* Other variables:

| Variable Name | Default | Notes |
| --- | --- | --- |
| DATASET_DIR | `./data` | Dataset directory |
| RUN_MODE | `kaggle` | Dataset run mode from `kaggle`, `criteo_small`, `criteo_full` |
| FINAL_IMAGE_NAME | `recommendation-ray` | Final Docker image name |
| CHECKPOINT_DIR | `/output` | Checkpoint directory. Should contain a directory result/ with checkoints saved from training |

## Build and Run
Build and Run with defaults:
```
make recommendation-ray
```

## Build and Run Example
```
$ DATASET_DIR=/localdisk/sharvils/data/criteo_kaggle/ CHECKPOINT_DIR=.output/ RUN_MODE=kaggle make recommendation-ray

 => [internal] load build definition from DockerfilePytorch                                                                            0.0s
 => => transferring dockerfile: 39B                                                                                                    0.0s
 => [internal] load .dockerignore                                                                                                      0.0s
 => => transferring context: 2B                                                                                                        0.0s
 => [internal] load metadata for docker.io/library/ubuntu:18.04                                                                        0.3s
 => [internal] load build context                                                                                                      0.0s
 => => transferring context: 68B                                                                                                       0.0s
 => [ 1/40] FROM docker.io/library/ubuntu:18.04@sha256:daf3e62183e8aa9a56878a685ed26f3af3dd8c08c8fd11ef1c167a1aa9bd66a3                0.0s
 => CACHED [ 2/40] WORKDIR /root/                                                                                                      0.0s
 => CACHED [ 3/40] RUN apt-get update -y && apt-get upgrade -y && apt-get install -y openjdk-8-jre build-essential cmake wget curl gi  0.0s
 => CACHED [ 4/40] COPY miniconda.sh .                                                                                                 0.0s
 => CACHED [ 5/40] COPY spark-env.sh .                                                                                                 0.0s
 => CACHED [ 6/40] RUN ls ~/                                                                                                           0.0s
 => CACHED [ 7/40] RUN /bin/bash ~/miniconda.sh -b -p /opt/intel/oneapi/intelpython/latest                                             0.0s
 => CACHED [ 8/40] RUN yes | conda create -n pytorch_mlperf python=3.7                                                                 0.0s
 => CACHED [ 9/40] RUN conda install gxx_linux-64==8.4.0                                                                               0.0s
 => CACHED [10/40] RUN cp /opt/intel/oneapi/intelpython/latest/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py /opt/intel  0.0s
 => CACHED [11/40] RUN cp /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_lin  0.0s
 => CACHED [12/40] RUN cp -r /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/* /opt/intel/oneapi/intelpython/latest/envs  0.0s
 => CACHED [13/40] RUN python -m pip install sklearn onnx tqdm lark-parser pyyaml                                                      0.0s
 => CACHED [14/40] RUN conda install ninja cffi typing --no-update-deps                                                                0.0s
 => CACHED [15/40] RUN conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps                                      0.0s
 => CACHED [16/40] RUN conda install -c conda-forge gperftools                                                                         0.0s
 => CACHED [17/40] RUN git clone https://github.com/pytorch/pytorch.git && cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3 &&   0.0s
 => CACHED [18/40] RUN git clone https://github.com/intel/intel-extension-for-pytorch.git && cd intel-extension-for-pytorch && git ch  0.0s
 => CACHED [19/40] RUN cd intel-extension-for-pytorch && cp torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patc  0.0s
 => CACHED [20/40] RUN cp -r /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/* /opt/intel/oneapi/intelpython/latest/envs  0.0s
 => CACHED [21/40] RUN cd pytorch && python setup.py install                                                                           0.0s
 => CACHED [22/40] RUN cd intel-extension-for-pytorch && python setup.py install                                                       0.0s
 => CACHED [23/40] RUN git clone https://github.com/oneapi-src/oneCCL.git && cd oneCCL && git checkout 2021.1-beta07-1 && mkdir build  0.0s
 => CACHED [24/40] RUN git clone https://github.com/intel/torch-ccl.git && cd torch-ccl && git checkout 2021.1-beta07-1                0.0s
 => CACHED [25/40] RUN source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh && cd torch-ccl && pytho  0.0s
 => CACHED [26/40] RUN python -m pip install --no-cache-dir --ignore-installed sigopt==7.5.0 pandas pytest prefetch_generator tensorb  0.0s
 => CACHED [27/40] RUN python -m pip install "git+https://github.com/mlperf/logging.git@1.0.0"                                         0.0s
 => CACHED [28/40] RUN pip install ray==2.1.0 raydp-nightly pyrecdp pandas scikit-learn "pyarrow<7.0.0"                                0.0s
 => CACHED [29/40] RUN apt-get update -y && apt-get install -y openssh-server pssh sshpass vim                                         0.0s
 => CACHED [30/40] RUN sed -i 's/#Port 22/Port 12346/g' /etc/ssh/sshd_config                                                           0.0s
 => CACHED [31/40] RUN sed -i 's/#   Port 22/    Port 12346/g' /etc/ssh/ssh_config                                                     0.0s
 => CACHED [32/40] RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config                                                              0.0s
 => CACHED [33/40] RUN conda init bash                                                                                                 0.0s
 => CACHED [34/40] RUN echo "source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh" >> /etc/bash.bash  0.0s
 => CACHED [35/40] RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/pyt  0.0s
 => CACHED [36/40] RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/lib/pyt  0.0s
 => CACHED [37/40] RUN echo "source ~/spark-env.sh" >> /etc/bash.bashrc                                                                0.0s
 => CACHED [38/40] RUN echo "KMP_BLOCKTIME=1" >> /etc/bash.bashrc                                                                      0.0s
 => CACHED [39/40] RUN echo "KMP_AFFINITY="granularity=fine,compact,1,0"" >> /etc/bash.bashrc                                          0.0s
 => CACHED [40/40] RUN echo "root:docker" | chpasswd                                                                                   0.0s
 => exporting to image                                                                                                                 0.0s
 => => exporting layers                                                                                                                0.0s
 => => writing image sha256:740242ef084e164945902d271a7edf9291015b5a3ad9fa79b8527e452ece03b3                                           0.0s
 => => naming to docker.io/library/recommendation-ray:training-inference-ubuntu-18.04                                                  0.0s
[+] Running 1/1
 ⠿ Container ray-inference  Created                                                                                                    0.1s
Attaching to ray-inference
ray-inference  | check cmd
ray-inference  | check dataset
ray-inference  | check data path: /home/vmagent/app/dataset/criteo
ray-inference  | check kaggle dataset
```
...
```
ray-inference  | [1] Start inference==========================:
ray-inference  | [0] Start inference==========================:
ray-inference  | [1]  loss 0.462588, auc 0.7900 accuracy 78.372 %
ray-inference  | [1] Test time:1.955101728439331
ray-inference  | [1] Total results length:3274330
ray-inference  | [0]  loss 0.462588, auc 0.7900 accuracy 78.372 %
ray-inference  | inference time is 27 seconds.
```
