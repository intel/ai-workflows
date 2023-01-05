# Distributed training and inference on Ray and Spark with Intel® End-to-End Optimization Kit 

## Overview
Modern recommendation systems require a complex pipeline to handle both data processing and feature engineering at a tremendous scale, while promising high service level agreements for complex deep learning models. Usually, this leads to two separate clusters for data processing and training: a data process cluster, usually CPU based to process huge dataset (terabytes or petabytes) stored in distributed storage system, and a training cluster, usually GPU based for training. This separate data processing and training cluster results a complex software stack, heavy data movement cost.
Meanwhile, Deep learning models were commonly used for recommendation systems, quite often, those models are over-parameterized. It takes up to several days or even weeks to training a heavy model on CPU. 
This workflow trying to address those pain points: unifies both data processing and training on Ray – the open source project that make it simple to scale any compute-intensive python workloads, and then optimize the E2E process, especially the training part, shorten training time weight lighter models, maintaining same accuracy, and delivers high-throughput models for inference with Intel® End-to-End Optimization Kit.

> Important: original source disclose - [Deep Learning Recommendation Model](https://github.com/facebookresearch/dlrm)

## How it works 
![image](https://user-images.githubusercontent.com/18349036/209234932-12100303-16d7-4352-9d7b-ed23f4cf7028.png)

## Get Started
NOTE: Before you get started please make sure you have the trained model available from training pipeline.

### Prerequisites
```bash
git clone https://github.com/intel/e2eAIOK/ AIOK_Ray
cd AIOK_Ray
git checkout tags/aiok-ray-v0.2
git submodule update --init --recursive
sh dlrm_all/dlrm/patch_dlrm.sh
cd ..
```

### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image to run the entire pipeline.

##### Pull Docker Image
```
docker pull intel/ai-workflows:recommendation-ray-inference
```

#### How to run 

The code snippet below runs the inference session. The model files will be generated and stored in the `/output/result` folder.

(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

```bash
if [ ! "$(docker network ls | grep ray-inference)" ]; then
    docker network create --driver=bridge ray-inference;
fi
export OUTPUT_DIR=/output
export DATASET_DIR=<Path to Dataset>
export RUN_MODE=<Pick from kaggle/criteo_small/criteo_full>

docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --env RUN_MODE=${RUN_MODE} \
  --env APP_DIR=/home/vmagent/app/e2eaiok \
  --env OUTPUT_DIR=/output \
  --volume ${DATASET_DIR}:/home/vmagent/app/dataset/criteo \
  --volume $(pwd)/AIOK_Ray:/home/vmagent/app/e2eaiok \
  --volume ${OUTPUT_DIR}:/output \
  --workdir /home/vmagent/app/e2eaiok/dlrm_all/dlrm/ \
  --privileged --init -it \
  --shm-size=300g --network ray-inference \
  --device=/dev/dri \
  --hostname ray \
  --name ray-inference \
  intel/ai-workflows:recommendation-ray-inference \
  bash -c 'bash $APP_DIR/scripts/run_inference_docker.sh $RUN_MODE'
```
------

## Useful Resources

## Recommended Hardware and OS

Operating System: Ubuntu 20.04
Memory Requirement: Minimum 250G

| Dataset Name | Recommended Disk Size |
| --- | --- |
| Kaggle | 90G |
| Criteo Small | 800G |
| Criteo Full | 4500G |

### Dataset
> Note: For kaggle run, train.txt and test.txt are required.

kaggle: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz

> Note: For criteo small run, day_0, day_1, day_2, day_3, day_23 are required.

> Note: For criteo full test, day_0-day_23 are required

criteo small and criteo full: https://labs.criteo.com/2013/12/download-terabyte-click-logs/

### Step by Step Guide

[option1] Build docker for single or multiple node and enter docker with click-to-run script
```
python3 scripts/start_e2eaiok_docker.py
sshpass -p docker ssh ${local_host_name} -p 12346
# If you met any network/package not found error, please follow log output to do the fixing and re-run above cmdline.

# If you are behind proxy, use below cmd
# python3 scripts/start_e2eaiok_docker.py --proxy "http://ip:port"
# sshpass -p docker ssh ${local_host_name} -p 12346

# If you disk space is limited, you can specify spark_shuffle_dir and dataset_path to different mounted volumn
# python3 scripts/start_e2eaiok_docker.py --spark_shuffle_dir "" --dataset_path ""
# sshpass -p docker ssh ${local_host_name} -p 12346
```

[option2] Build docker manually
```
# prepare a folder for dataset
cd frameworks.bigdata.AIOK
cur_path=`pwd`
mkdir -p ../e2eaiok_dataset

# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -P Dockerfile-ubuntu18.04/ -O Dockerfile-ubuntu18.04/miniconda.sh

# build docker from dockerfile
docker build -t e2eaiok-ray-pytorch Dockerfile-ubuntu18.04 -f Dockerfile-ubuntu18.04/DockerfilePytorch
# if you are behine proxy
# docker build -t e2eaiok-ray-pytorch Dockerfile-ubuntu18.04 -f Dockerfile-ubuntu18.04/DockerfilePytorch --build-arg http_proxy={ip}:{port} --build-arg https_proxy=http://{ip}:{port}

# run docker
docker run -it --shm-size=300g --privileged --network host --device=/dev/dri -v ${cur_path}/../e2eaiok_dataset/:/home/vmagent/app/dataset -v ${cur_path}:/home/vmagent/app/e2eaiok -v ${cur_path}/../spark_local_dir/:/home/vmagent/app/spark_local_dir -w /home/vmagent/app/ --name e2eaiok-ray-pytorch e2eaiok-ray-pytorch /bin/bash
```

### Test with other dataset (run cmd inside docker)
```
# active conda
conda activate pytorch_mlperf

# if behind proxy, please set proxy firstly
# export https_proxy=http://{ip}:{port}

# criteo test
cd /home/vmagent/app/e2eaiok/dlrm_all/dlrm/; bash run_aiokray_dlrm.sh criteo_small ${current_node_ip}
```

### Test full pipeline manually (run cmd inside docker)
```
# active conda
conda activate pytorch_mlperf

# if behind proxy, please set proxy firstly
# export https_proxy=http://{ip}:{port}

# prepare env
bash run_prepare_env.sh ${run_mode} ${current_node_ip}

# data process
bash run_data_process.sh ${run_mode} ${current_node_ip}

# train
bash run_train.sh ${run_mode} ${current_node_ip}

# inference
bash run_inference.sh ${run_mode} ${current_node_ip}
```

## Support

For questions and support, please contact Jian Shang at jian.zhang@intel.com.

