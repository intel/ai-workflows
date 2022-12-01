# Vision-based Transfer Learning - Training - **DRAFT**

## Overview
This guide contains instruction on how to run reference end-to-end pipeline for transfer learning with Docker container. For detailed information about the workflow, go to [End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) GitHub repository.

## How it Works
The goal of this vision-based workflow is to do transfer learning on images to accomplish different classification tasks that range from binary classification to multiclass classification giving best performance on Intel Hardware utilizing the optimizations that could be done.

The pipeline showcases how transfer learning enabled with Intel optimized TensorFlow could be used for image classification on three domains: sports , medical imaging and remote sensing .The workflow showcases AMX  BF16 in SPR which speeds up the training time significantly, without loss in accuracy.

The workflow uses pretrained SOTA models (RESNET V1.5) from TF hub and transfers the knowledge from a pretrained domain to a different custom domain achieving required accuracy.

<br><img src="https://user-images.githubusercontent.com/52259352/202562899-d2867491-f08b-4393-be27-d8db28931bd6.png"><br>
<br><img src="https://user-images.githubusercontent.com/52259352/202562891-5b065c21-9ea5-427d-b555-8cc3419c8a39.png"><br>

## Get Started

### **Prerequisites**

#### Download the repo
```
git clone https://github.com/intel/vision-based-transfer-learning-and-inference.git .
git checkout v1.0.1
```
#### Download the datasets
Medical Imaging dataset is downloaded from TensorFlow website when the code is run for the first time. The dataset used for this domain is `colorectal_histology`. More details can be found at [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/colorectal_histology). 

Remote Sensing dataset used for this domain is [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45).  
[Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate ) the dataset and unzip the folder. To use this dataset, it should be split into `validation` and `train` directories. Use the script [resisc_dataset.py](https://github.com/intel/vision-based-transfer-learning-and-inference/blob/v1.0.1/resisc_dataset.py). Follow the example below:
```
pip install -r requirements.txt
export INPUT_DIR=<path_to_NWPU-RESISC45_dir>
export OUTPUT_DIR=<path_to_split_dataset>
python3 resisc_dataset.py --INDIR=${INPUT_DIR} --OUTDIR=${OUTPUT_DIR}
mv ${OUTPUT_DIR}/val ${OUTPUT_DIR}/validation
```

### **Docker**
Below setup and how-to-run sessions are for users who want to use provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).

#### **Pull Docker Image**
```
docker pull intel/ai-workflows:vision-transfer-learning-training
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

For example, this is how to run single instance using the following options: `PLATFORM=None`, `PRECISION=FP32` and `SCRIPT=colorectal`.
```
export DATASET_DIR=/data
export OUTPUT_DIR=/output
export PLATFORM=None
export PRECISION=FP32
export SCRIPT=colorectal
docker run \
  $DOCKER_RUN_ENVS \
  --env DATASET_DIR=/workspace/data \
  --env OUTPUT_DIR=${OUTPUT_DIR}/${SCRIPT} \
  --env PLATFORM=${PLATFORM} \
  --env PRECISION=${PRECISION} \
  --volume /${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  --privileged --init -it \
  intel/ai-workflows:vision-transfer-learning-training \
  conda run --no-capture-output -n transfer_learning ./${SCRIPT}.sh
```

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 


#### How to run 


## Recommended Hardware 


## Useful Resources 


## Support  
[End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) tracks both bugs and enhancement requests using Github. We welcome input, however, before filing a request, please make sure you do the following: Search the Github issue database.
