# Distributed training and inference on Ray and Spark with Intel® End-to-End Optimization Kit 

## Overview
Modern recommendation systems require a complex pipeline to handle both data processing and feature engineering at a tremendous scale, while promising high service level agreements for complex deep learning models. Usually, this leads to two separate clusters for data processing and training: a data process cluster, usually CPU based to process huge dataset (terabytes or petabytes) stored in distributed storage system, and a training cluster, usually GPU based for training. This separate data processing and training cluster results a complex software stack, heavy data movement cost.
Meanwhile, Deep learning models were commonly used for recommendation systems, quite often, those models are over-parameterized. It takes up to several days or even weeks to training a heavy model on CPU. 
This workflow trying to address those pain points: unifies both data processing and training on Ray – the open source project that make it simple to scale any compute-intensive python workloads, and then optimize the E2E process, especially the training part, shorten training time weight lighter models, maintaining same accuracy, and delivers high-throughput models for inference with Intel® End-to-End Optimization Kit.

> Important: original source disclose - [Deep Learning Recommendation Model](https://github.com/facebookresearch/dlrm)

## How it works 
![image](https://user-images.githubusercontent.com/18349036/209234932-12100303-16d7-4352-9d7b-ed23f4cf7028.png)

## Get Started

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
docker pull intel/ai-workflows:recommendation-ray-training
```

#### How to run 

The code snippet below runs the training session. The model files will be generated and stored in the `/output/result` folder.

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
if [ ! "$(docker network ls | grep ray-training)" ]; then
    docker network create --driver=bridge ray-training;
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
  --shm-size=300g --network ray-training \
  --device=/dev/dri \
  --hostname ray \
  --name ray-training \
  intel/ai-workflows:recommendation-ray-training \
  bash -c 'bash $APP_DIR/scripts/run_train_docker.sh $RUN_MODE'
```
------

#### Output
```
$ DATASET_DIR=/localdisk/sharvils/data/criteo_kaggle/ OUTPUT_DIR=.output/ RUN_MODE=kaggle make recommendation-ray

[+] Building 1.8s (45/45) FINISHED                                                                                                          
 => [internal] load build definition from DockerfilePytorch                                                                            0.0s
 => => transferring dockerfile: 39B                                                                                                    0.0s
 => [internal] load .dockerignore                                                                                                      0.1s
 => => transferring context: 2B                                                                                                        0.0s
 => [internal] load metadata for docker.io/library/ubuntu:18.04                                                                        1.6s
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
 => => writing image sha256:c3033b3a534c74733f213b58511352a94fcba0aa3b465abb5232efeecac933f7                                           0.0s
 => => naming to docker.io/library/recommendation-ray:training-inference-ubuntu-18.04                                                  0.0s
[+] Running 1/1
 ⠿ Container ray-training  Created                                                                                                     0.1s
Attaching to ray-training
ray-training  | check cmd
ray-training  | check dataset
ray-training  | check data path: /home/vmagent/app/dataset/criteo
ray-training  | check kaggle dataset
```
...
```
ray-training  | [1] Finished training it 16/150 of epoch 0, 439.75 ms/it, loss 0.561402, accuracy 74.972 %
ray-training  | [0] Finished training it 16/150 of epoch 0, 439.65 ms/it, loss 0.552633, accuracy 75.560 %
ray-training  | [0] Finished training it 32/150 of epoch 0, 460.74 ms/it, loss 0.515942, accuracy 75.517 %[0] 
ray-training  | [1] Finished training it 32/150 of epoch 0, 460.43 ms/it, loss 0.518215, accuracy 75.598 %
ray-training  | [1] Finished training it 48/150 of epoch 0, 459.48 ms/it, loss 0.495488, accuracy 76.734 %
ray-training  | [0] Finished training it 48/150 of epoch 0, 459.12 ms/it, loss 0.499906, accuracy 76.189 %
ray-training  | [0] Finished training it 64/150 of epoch 0, 459.54 ms/it, loss 0.478464, accuracy 77.671 %
ray-training  | [1] Finished training it 64/150 of epoch 0, 459.67 ms/it, loss 0.484237, accuracy 77.321 %
ray-training  | [0] Finished training it 80/150 of epoch 0, 463.88 ms/it, loss 0.472614, accuracy 77.830 %
ray-training  | [1] Finished training it 80/150 of epoch 0, 463.85 ms/it, loss 0.477289, accuracy 77.685 %
ray-training  | [1] Finished training it 96/150 of epoch 0, 460.16 ms/it, loss 0.464757, accuracy 78.387 %
ray-training  | [0] Finished training it 96/150 of epoch 0, 461.06 ms/it, loss 0.470029, accuracy 77.911 %
ray-training  | [0] Finished training it 112/150 of epoch 0, 463.29 ms/it, loss 0.461404, accuracy 78.553 %
ray-training  | [1] Finished training it 112/150 of epoch 0, 464.08 ms/it, loss 0.459267, accuracy 78.671 %
ray-training  | [0] Finished training it 128/150 of epoch 0, 459.62 ms/it, loss 0.459535, accuracy 78.636 %
ray-training  | [1] Finished training it 128/150 of epoch 0, 459.50 ms/it, loss 0.455789, accuracy 78.833 %
ray-training  | [0] Finished training it 144/150 of epoch 0, 459.49 ms/it, loss 0.453435, accuracy 78.892 %
ray-training  | [1] Finished training it 144/150 of epoch 0, 459.80 ms/it, loss 0.452942, accuracy 78.996 %
ray-training  | [0] Finished training it 150/150 of epoch 0, 465.29 ms/it, loss 0.462829, accuracy 78.277 %[0] 
ray-training  | [1] Finished training it 150/150 of epoch 0, 465.18 ms/it, loss 0.454790, accuracy 78.785 %
ray-training  | [0] :::MLLOG {"namespace": "", "time_ms": 1671571837234, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/home/vmagent/app/e2eaiok/dlrm_all/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1482, "epoch_num": 1.0000076351505636}}
ray-training  | [1] Testing at - 150/150 of epoch 0, loss 0.463755, auc 0.7926, best auc 0.7926, accuracy 78.221 %, best accuracy 0.000 %
ray-training  | [0] :::MLLOG {"namespace": "", "time_ms": 1671571839204, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.7925731168846557, "metadata": {"file": "/home/vmagent/app/e2eaiok/dlrm_all/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1606, "epoch_num": 1.0000076351505636}}
ray-training  | [0] Testing at - 150/150 of epoch 0, loss 0.463755, auc 0.7926, best auc 0.7926, accuracy 78.221 %, best accuracy 0.000 %
ray-training  | [0] :::MLLOG {"namespace": "", "time_ms": 1671571839205, "event_type": "INTERVAL_END", "key": "eval_stop", "value": null, "metadata": {"file": "/home/vmagent/app/e2eaiok/dlrm_all/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1630, "epoch_num": 1.0000076351505636}}
ray-training  | [1] Test time:1.9714336395263672
ray-training  | [0] Test time:1.97495436668396
ray-training  | [0] :::MLLOG {"namespace": "", "time_ms": 1671571839205, "event_type": "INTERVAL_END", "key": "epoch_stop", "value": null, "metadata": {"file": "/home/vmagent/app/e2eaiok/dlrm_all/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1673, "epoch_num": 1}}
ray-training  | [1] Total Time:71.28883194923401
ray-training  | [1] Saving best model to /output/result/
ray-training  | [0] :::MLLOG {"namespace": "", "time_ms": 1671571839205, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/home/vmagent/app/e2eaiok/dlrm_all/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1676, "first_epoch_num": 1}}
ray-training  | [0] Total Time:71.2883768081665
ray-training  | [0] Saving best model to /output/result/
ray-training  | [1] Saved beat model to /output/result/, cost 0.33439135551452637s
ray-training  | [0] Saved beat model to /output/result/, cost 0.48244571685791016s
ray-training  | training time is 97 seconds.
ray-training exited with code 0
```

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

