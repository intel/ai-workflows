# BigDL Friesian - Training

## Overview
[BigDL Friesian](https://bigdl.readthedocs.io/en/latest/doc/Friesian/index.html) is an application framework for building optimized large-scale recommender solutions optimized on Intel Xeon. This example workflow demonstrates how to use Friesian to easily build an end-to-end [Wide & Deep Learning](https://arxiv.org/abs/1606.07792) recommennder system on a real-world large dataset provided by Twitter.

This example demonstrates how to use BigDL Friesian to preprocess the Criteo dataset and train the WideAndDeep model in a distributed fashion.

## How it Works
Friesian provides various built-in distributed feature engineering operations and the distributed training of popular recommendation algorithms based on [BigDL Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html) and Spark. 

The overall architecture of Friesian is shown in the following diagram:

<img src="https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/images/architecture.png" width="100%" />

## Get Started

### **Prerequisites**
#### Download the repo 
Clone [BigDL](https://github.com/intel-analytics/BigDL) repo
```
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
git checkout ai-workflow
```

### Docker
Below setup and how-to-run sessions are for users who want to use provided docker image running on a sample dataset.
For bare metal environment, as well as running instructions on running on full dataset, please go to [bare metal session](#bare-metal).

### Dataset Preparation
Sample dataset is available at [Criteo Engineering](https://labs.criteo.com/2014/02/download-dataset/). 

Downloading data to `/dataset` directory. The following commands will download the dataset, unzip and place it appropriate sub directory for data preprocessing.
```bash
export DATASET_DIR=/dataset
```
```bash
wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
tar -xvzf dac_sample.tar.gz
mkdir -p ${DATASET_DIR}/data-csv
mv dac_sample.txt ${DATASET_DIR}/data-csv/day_0.csv
```

##### Pull Docker Image
```
docker pull intel/ai-workflows:friesian-training
```

#### How to run 
(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the pipeline, follow below instructions outside of docker instance. 
- Convert sample csv file to parquet file
```bash
export DATASET_DIR=/dataset
export MODEL_OUTPUT=/model
```
```bash
docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --env DATASET_DIR=${DATASET_DIR} \
  --env MODEL_OUTPUT=${MODEL_OUTPUT} \
  --volume ${DATASET_DIR}:/dataset \
  --volume ${MODEL_OUTPUT}:/output \
  --volume $(pwd):/workspace \
  --workdir /workspace/python/friesian/example/wnd \
  --privileged --init -it --rm \
  intel/ai-workflows:friesian-training \
  conda run -n bigdl --no-capture-output conda run -n bigdl --no-capture-output python3 csv_to_parquet.py --input /dataset/data-csv/day_0.csv --output /dataset/data-parquet/day_0.parquet
```
- Data processing
```bash
docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --env DATASET_DIR=${DATASET_DIR} \
  --env MODEL_OUTPUT=${MODEL_OUTPUT} \
  --volume ${DATASET_DIR}:/dataset \
  --volume ${MODEL_OUTPUT}:/output \
  --volume $(pwd):/workspace \
  --workdir /workspace/python/friesian/example/wnd \
  --privileged --init -it --rm \
  intel/ai-workflows:friesian-training \
  conda run -n bigdl --no-capture-output python wnd_preprocessing.py --executor_cores 36 --executor_memory 50g --days 0-0 --input_folder /dataset/data-parquet --output_folder /dataset/data-processed --frequency_limit 15 --cross_sizes 10000,10000
```
- Model training 
```bash
docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --env DATASET_DIR=${DATASET_DIR} \
  --env MODEL_OUTPUT=${MODEL_OUTPUT} \
  --volume ${DATASET_DIR}:/dataset \
  --volume ${MODEL_OUTPUT}:/output \
  --volume $(pwd):/workspace \
  --workdir /workspace/python/friesian/example/wnd \
  --privileged --init -it --rm \
  intel/ai-workflows:friesian-training \
  conda run -n bigdl --no-capture-output python wnd_train.py --executor_cores 36 --executor_memory 50g --data_dir /dataset/data-processed --model_dir /model
```

#### Output
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

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster.
```bash
conda create -n bigdl python=3.7  # "bigdl" is the conda environment name, you can use any name you like.
conda activate bigdl
pip install tensorflow==2.6.0
pip install --pre --upgrade bigdl-friesian[train]
```
#### Prepare the data
You can download the full __1TB__ Click Logs dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/), which includes data of 24 days (day_0 to day_23) with 4,373,472,329 records in total.

After you download the files, convert them to parquet files with the name `day_x.parquet` (x=0-23), and put all parquet files in one folder. You may use the script `csv_to_parquet.py` provided in this directory to convert the data of each day to parquet.
- The first 23 days (day_0 to day_22) are used for WND training with 4,195,197,692 records in total.
- The first half (89,137,319 records in total) of the last day (day_23) is used for test. To prepare the test dataset, you need to split the first half of day_23 into a new file (e.g. using command `head -n 89137319 day_23 > day_23_test`) and finally convert to parquet files with the name `day_23_test.parquet` under the same folder with the train parquet files.

If you want to use some sample data for test, you can download `dac_sample` from [Criteo Engineering](https://labs.criteo.com/2014/02/download-dataset/), unzip and convert `dac_sample.txt` to parquet with name `day_0.parquet`.
#### How to run 

#### Data Preprocessing
* Spark local, we can use the first (several) day(s) or the sample data to have a trial, example command:
```bash
python wnd_preprocessing.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --days 0-0 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

* Spark standalone, example command to run on the full Criteo dataset:
```bash
python wnd_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

* Spark yarn client mode, example command to run on the full Criteo dataset:
```bash
python wnd_preprocessing.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_folder /path/to/the/folder/to/save/preprocessed/parquet_files \
    --frequency_limit 15 \
    --cross_sizes 10000,10000
```

__Options:__
* `input_folder`: The path to the folder of parquet files, either a local path or an HDFS path.
* `output_folder`: The path to save the preprocessed data to parquet files and meta data. HDFS path is recommended for yarn cluster_mode.
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each executor. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each executor. Default to be 160g.
* `num_executors`: The number of executors to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `days`: The day range for data preprocessing, such as 0-23 for the full Criteo dataset, 0-0 for the first day, 0-1 for the first two days, etc. Default to be 0-23.
* `frequency_limit`: Categories with frequency below this value will be omitted from encoding. We recommend using 15 when you preprocess the full 1TB dataset. Default to be 15.
* `cross_sizes`: The bucket sizes for cross columns (`c14-c15` and `c16-c17`) separated by comma. Default to be 10000,10000. Please pay attention that there must NOT be a blank space between the two numbers.

#### Model training
* Spark local, example command:
```bash
python wnd_train.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model_dir ./wnd_model
```

* Spark standalone, example command:
```bash
python wnd_train.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model /path/to/save/the/trained/model
```

* Spark yarn client mode, example command:
```bash
python wnd_train.py \
    --cluster_mode yarn \
    --executor_cores 56 \
    --executor_memory 240g \
    --num_executors 8 \
    --data_dir /path/to/the/folder/of/preprocessed/parquet_files \
    --model /path/to/save/the/trained/model
```

__Options:__
* `data_dir`: The path to the folder of preprocessed parquet files and meta data, either a local path or an HDFS path.
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn, standalone or spark-submit. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each executor. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each executor. Default to be 30g.
* `num_executors`: The number of executors to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `model_dir`: The path to saved the trained model, either a local path or an HDFS path. Default to be "./wnd_model".
* `batch_size`: The batch size to train the model. Default to be 1024.
* `epoch`: The number of epochs to train the model. Default to be 2.
* `learning_rate`: The learning rate to train the model. Default to be 0.0001.

## Recommended Hardware
The hardware below is recommended for use with this reference implementation.

- Intel® 4th Gen Xeon® Scalable Performance processors

## Learn More
- For more detailed descriptions for distributed feature engineering and training, check the [notebooks](https://github.com/intel-analytics/BigDL/tree/main/apps/wide-deep-recommendation).
- For more reference use cases, visit [Use Cases Page](https://bigdl.readthedocs.io/en/latest/doc/Friesian/examples.html).
- For more detailed API documentations [Friesian API Page](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Friesian/index.html).

## Known Issues
NA

## Troubleshooting
NA

## Support Forum
For any issues, please submit a ticket at [BigDL issues page](https://github.com/intel-analytics/BigDL/issues).