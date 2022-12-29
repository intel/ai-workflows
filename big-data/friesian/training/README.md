# BigDL Friesian - Training

## Description
This document contains instructions on how to train the Wide and Deep model with make and docker compose.
## Project Structure 
```
├── BigDL @ ai-workflow
├── DEVCATALOG.md
├── Dockerfile.friesian-training
├── Makefile
├── README.md
└── docker-compose.yml
```
[_Makefile_](Makefile)
```
DATASET_DIR ?= /dataset
FINAL_IMAGE_NAME ?= friesian-training
MODEL_OUTPUT ?= /model_output

friesian-training:
	wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
	tar -xvzf dac_sample.tar.gz
	mkdir -p ${DATASET_DIR}/data-csv
	mv dac_sample.txt ${DATASET_DIR}/data-csv/day_0.csv
	rm dac_sample.tar.gz
	@DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL_OUTPUT=${MODEL_OUTPUT} \
 	docker compose up friesian-training --build

clean: 
	@DATASET_DIR=${DATASET_DIR} \
	 OUTPUT_DIR=${MODEL_OUTPUT} \
	docker compose down
```
[_docker-compose.yml_](docker-compose.yml)
```
services:
  csv-to-parquet:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.friesian-training
    command: conda run -n bigdl --no-capture-output conda run -n bigdl --no-capture-output python3 csv_to_parquet.py --input /dataset/data-csv/day_0.csv --output /dataset/data-parquet/day_0.parquet
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - MODEL_OUTPUT=${MODEL_OUTPUT}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - ${MODEL_OUTPUT}:/model
      - $PWD:/workspace
    working_dir: /workspace/BigDL/python/friesian/example/wnd
  preprocessing:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.friesian-training
    command: conda run -n bigdl --no-capture-output python wnd_preprocessing.py --executor_cores 36 --executor_memory 50g --days 0-0 --input_folder /dataset/data-parquet --output_folder /dataset/data-processed --frequency_limit 15 --cross_sizes 10000,10000
    depends_on:
      csv-to-parquet:
        condition: service_completed_successfully
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - MODEL_OUTPUT=${MODEL_OUTPUT}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - ${MODEL_OUTPUT}:/model
      - $PWD:/workspace
    working_dir: /workspace/BigDL/python/friesian/example/wnd
  friesian-training:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.friesian-training
    command: conda run -n bigdl --no-capture-output python wnd_train.py --executor_cores 36 --executor_memory 50g --data_dir /dataset/data-processed --model_dir /model
    depends_on:
      preprocessing:
        condition: service_completed_successfully
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - MODEL_OUTPUT=${MODEL_OUTPUT}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - ${MODEL_OUTPUT}:/model
      - $PWD:/workspace
    working_dir: /workspace/BigDL/python/friesian/example/wnd
```

# Train a WideAndDeep Model on the Criteo Dataset
This example demonstrates how to use BigDL Friesian to preprocess the Criteo dataset and train the WideAndDeep model in a distributed fashion.

End-to-End Recommendation Systems AI Workflow utilizing BigDL - Friesian. More information at [Intel Analytics - BigDL](https://github.com/intel-analytics/BigDL)

## Quick Start
* Pull and configure the dependent repo submodule 
```
git submodule update --init --recursive BigDL
```

* Install [Pipeline Repository Dependencies](../../../README.md)

* Other variables:

| Variable Name | Default | Notes |
| --- | --- | --- |
| DATASET_DIR | `/dataset` | Default directory for dataset to be downloaded. Dataset will be downloaded when running with `make` command |
| FINAL_IMAGE_NAME | `friesian-training` | Final image name |
| MODEL_OUTPUT | `/model_output` | Trained model will be produced in this directory |

## Build and Run
Build and Run with defaults:
```
make friesian-training
```

## Build and Run Example
```
$ DATASET_DIR=/localdisk/criteo-dataset MODEL_OUTPUT=/locadisk/model make friesian-training
wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
--2022-12-13 10:06:29--  https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
Resolving proxy-dmz.intel.com (proxy-dmz.intel.com)... 10.7.211.16
Connecting to proxy-dmz.intel.com (proxy-dmz.intel.com)|10.7.211.16|:912... connected.
Proxy request sent, awaiting response... 200 OK
Length: 8787154 (8.4M) [application/x-gzip]
Saving to: ‘dac_sample.tar.gz’

dac_sample.tar.gz                 100%[============================================================>]   8.38M  6.27MB/s    in 1.3s    

2022-12-13 10:06:31 (6.27 MB/s) - ‘dac_sample.tar.gz’ saved [8787154/8787154]

tar -xvzf dac_sample.tar.gz
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.creationtime'
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
dac_sample.txt
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
./._readme.txt
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
readme.txt
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
license.txt
mkdir -p /localdisk/criteo-dataset/data-csv
mv dac_sample.txt /localdisk/criteo-dataset/data-csv/day_0.csv
rm dac_sample.tar.gz
[+] Building 0.5s (10/10) FINISHED                                                                                                     
 => [internal] load build definition from Dockerfile.friesian-training                                                            0.0s
 => => transferring dockerfile: 50B                                                                                               0.0s
 => [internal] load .dockerignore                                                                                                 0.0s
 => => transferring context: 2B                                                                                                   0.0s
 => [internal] load metadata for docker.io/library/ubuntu:20.04                                                                   0.4s
 => [1/6] FROM docker.io/library/ubuntu:20.04@sha256:0e0402cd13f68137edb0266e1d2c682f217814420f2d43d300ed8f65479b14fb             0.0s
 => CACHED [2/6] RUN apt-get update && apt-get install --no-install-recommends --fix-missing -y     ca-certificates     vim       0.0s
 => CACHED [3/6] RUN wget --no-check-certificate -q https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz &  0.0s
 => CACHED [4/6] RUN apt-get update &&     wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O  0.0s
 => CACHED [5/6] RUN conda create -yn bigdl python=3.7.5 &&     source activate bigdl &&     conda update -y -n base -c defaults  0.0s
 => CACHED [6/6] RUN mkdir -p /workspace                                                                                          0.0s
 => exporting to image                                                                                                            0.0s
 => => exporting layers                                                                                                           0.0s
 => => writing image sha256:3a788bfb21c562aacdb45eed2eb9232c05a7394ec4c9588016654f57f99e1232                                      0.0s
 => => naming to docker.io/library/friesian-training:training-ubuntu-20.04                                                        0.0s
[+] Running 3/3
 ⠿ Container training-csv-to-parquet-1     Recreated                                                                              0.1s
 ⠿ Container training-preprocessing-1      Recreated                                                                              0.2s
 ⠿ Container training-friesian-training-1  Recreated                                                                              0.1s
Attaching to training-friesian-training-1
```
...
```
training-friesian-training-1  | 
63/79 [======================>.......] - ETA: 1s - loss: 0.5001 - binary_accuracy: 0.7751 - binary_crossentropy: 0.5001 - auc: 0.6836
65/79 [=======================>......] - ETA: 1s - loss: 0.5003 - binary_accuracy: 0.7749 - binary_crossentropy: 0.5003 - auc: 0.6842
66/79 [========================>.....] - ETA: 0s - loss: 0.5003 - binary_accuracy: 0.7750 - binary_crossentropy: 0.5003 - auc: 0.6839
67/79 [========================>.....] - ETA: 0s - loss: 0.5002 - binary_accuracy: 0.7751 - binary_crossentropy: 0.5002 - auc: 0.6840
69/79 [=========================>....] - ETA: 0s - loss: 0.5001 - binary_accuracy: 0.7749 - binary_crossentropy: 0.5001 - auc: 0.6854
70/79 [=========================>....] - ETA: 0s - loss: 0.4999 - binary_accuracy: 0.7750 - binary_crossentropy: 0.4999 - auc: 0.6855
72/79 [==========================>...] - ETA: 0s - loss: 0.4997 - binary_accuracy: 0.7752 - binary_crossentropy: 0.4997 - auc: 0.6857
73/79 [==========================>...] - ETA: 0s - loss: 0.4997 - binary_accuracy: 0.7752 - binary_crossentropy: 0.4997 - auc: 0.6859
74/79 [===========================>..] - ETA: 0s - loss: 0.4997 - binary_accuracy: 0.7749 - binary_crossentropy: 0.4997 - auc: 0.6862
76/79 [===========================>..] - ETA: 0s - loss: 0.4995 - binary_accuracy: 0.7749 - binary_crossentropy: 0.4995 - auc: 0.6873
77/79 [============================>.] - ETA: 0s - loss: 0.4995 - binary_accuracy: 0.7749 - binary_crossentropy: 0.4995 - auc: 0.6874
79/79 [==============================] - ETA: 0s - loss: 0.4993 - binary_accuracy: 0.7750 - binary_crossentropy: 0.4993 - auc: 0.6875
training-friesian-training-1  | Training time is:  26.327171802520752
79/79 [==============================] - 6s 81ms/step - loss: 0.4993 - binary_accuracy: 0.7750 - binary_crossentropy: 0.4993 - auc: 0.6875 - val_loss: 0.5382 - val_binary_accuracy: 0.7730 - val_binary_crossentropy: 0.5382 - val_auc: 0.6826
training-friesian-training-1  | Stopping orca context
training-friesian-training-1 exited with code 0
```

## Cleanup
Remove containers, copied files, and special configurations
```
make clean
```
