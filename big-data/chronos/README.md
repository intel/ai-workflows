# **BigDL Chronos TRAINING - Time Series Forecasting**

## **Description**

This pipeline provides instructions on how to train a Temporal Convolution Neural Network on BigDL Chronos framework using time series dataset with make and docker compose. For more information on the workload visit [BigDL](https://github.com/intel-analytics/BigDL/tree/main) repository.

## **Project Structure**
```
├── BigDL @ ai-workflow
├── DEVCATALOG.md
├── Dockerfile.chronos
├── Makefile
└── docker-compose.yml
```
[*Makefile*](Makefile)

```
FINAL_IMAGE_NAME ?= chronos

chronos:
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	docker compose up chronos --build

clean:
	docker compose down

```
[*docker-compose.yml*](docker-compose.yml)

```
services:
  chronos:
    build:
      args:
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.chronos
    command: sh -c "jupyter nbconvert --to python chronos_nyc_taxi_tsdataset_forecaster.ipynb && \
                    sed '26,40d' chronos_nyc_taxi_tsdataset_forecaster.py > chronos_taxi_forecaster.py && \
                    python chronos_taxi_forecaster.py"
    environment:
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    network_mode: "host"
    privileged: true
    volumes:
    - ./BigDL:/workspace/BigDL
    working_dir: /workspace/BigDL/python/chronos/colab-notebook

```
# **Time Series Forcasting**

Training pipeline that uses the BigDL Chronos framework for time series forecasting using a Temporal Convlutional Neural Network. More information [here](https://github.com/intel-analytics/BigDL/tree/main).

## **Quick Start**

* Make sure that the enviroment setup pre-requisites are satisfied per the document [here](../../README.md)

* Pull and configure the dependent repo submodule ```git submodule update --init --recursive ```

* Install [Pipeline Repository Dependencies](../../README.md)

* Other Variables:

Variable Name    | Default             |Notes                                   |
:---------------:|:-------------------: | :------------------------------------: |
FINAL_IMAGE_NAME | chronos | Final Docker Image Name             |

## **Build and Run**

Build and run with defaults:

```$ make chronos```

## **Build and Run Example**

```
#1 [internal] load build definition from Dockerfile.chronos
#1 transferring dockerfile: 55B done
#1 DONE 0.0s

#2 [internal] load .dockerignore
#2 transferring context: 2B done
#2 DONE 0.0s

#3 [internal] load metadata for docker.io/library/ubuntu:20.04
#3 DONE 0.0s

#4 [1/5] FROM docker.io/library/ubuntu:20.04
#4 DONE 0.0s

#5 [2/5] RUN apt-get update --fix-missing &&     apt-get install -y apt-utils vim curl nano wget unzip git &&     apt-get install -y gcc g++ make &&     apt-get install -y libsm6 libxext6 libxrender-dev &&     apt-get install -y openjdk-8-jre &&     rm /bin/sh &&     ln -sv /bin/bash /bin/sh &&     echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su &&     chgrp root /etc/passwd && chmod ug+rw /etc/passwd &&     wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh &&     chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh &&     ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /usr/local &&     rm Miniconda3-py37_4.12.0-Linux-x86_64.sh
#5 CACHED

#6 [4/5] RUN echo "source activate chronos" > ~/.bashrc
#6 CACHED

#7 [3/5] RUN conda create -y -n chronos python=3.7 setuptools=58.0.4 && source activate chronos &&     pip install --no-cache-dir --pre --upgrade bigdl-chronos[pytorch,automl] matplotlib notebook==6.4.12 &&     pip uninstall -y torchtext
#7 CACHED

#8 [5/5] RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc
#8 CACHED

#9 exporting to image
#9 exporting layers done
#9 writing image sha256:329995e99da4001c6d57e243085145acfce61f5bddabd9459aa598b846eae331 done
#9 naming to docker.io/library/time-series-chronos:training-ubuntu-20.04 done
#9 DONE 0.0s
Attaching to training-time-series-chronos-1
training-time-series-chronos-1  | [NbConvertApp] Converting notebook chronos_nyc_taxi_tsdataset_chronos.ipynb to python
training-time-series-chronos-1  | [NbConvertApp] Writing 10692 bytes to chronos_nyc_taxi_tsdataset_chronos.py
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | /usr/local/envs/chronos/lib/python3.7/site-packages/bigdl/chronos/forecaster/utils.py:157: UserWarning: 'batch_size' cannot be divided with no remainder by 'self.num_processes'. We got 'batch_size' = 32 and 'self.num_processes' = 7
training-time-series-chronos-1  |   format(batch_size, num_processes))
training-time-series-chronos-1  | GPU available: False, used: False
training-time-series-chronos-1  | TPU available: False, using: 0 TPU cores
training-time-series-chronos-1  | IPU available: False, using: 0 IPUs
training-time-series-chronos-1  | HPU available: False, using: 0 HPUs
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/7
```
...

```
 98%|█████████▊| 287/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0157]
Epoch 2:  98%|█████████▊| 289/294 [00:09<00:00, 31.56it/s, loss=0.0157]
Epoch 2:  98%|█████████▊| 289/294 [00:09<00:00, 31.56it/s, loss=0.0162]
Epoch 2:  99%|█████████▊| 290/294 [00:09<00:00, 31.57it/s, loss=0.0162]
Epoch 2:  99%|█████████▊| 290/294 [00:09<00:00, 31.57it/s, loss=0.0165]
Epoch 2:  99%|█████████▉| 291/294 [00:09<00:00, 31.58it/s, loss=0.0165]
Epoch 2:  99%|█████████▉| 291/294 [00:09<00:00, 31.58it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2: 100%|█████████▉| 293/294 [00:09<00:00, 31.58it/s, loss=0.0164]
Epoch 2: 100%|█████████▉| 293/294 [00:09<00:00, 31.58it/s, loss=0.0175]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.0175]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.017] 
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.017]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.60it/s, loss=0.017]
training-time-series-forecaster-1  | Global seed set to 1
training-time-series-forecaster-1  | GPU available: False, used: False
training-time-series-forecaster-1  | TPU available: False, using: 0 TPU cores
training-time-series-forecaster-1  | IPU available: False, using: 0 IPUs
training-time-series-forecaster-1  | HPU available: False, using: 0 HPUs
training-time-series-forecaster-1 exited with code 0
```
