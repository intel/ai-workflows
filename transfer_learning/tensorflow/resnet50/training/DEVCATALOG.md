# Vision-based Transfer Learning - Training

## Overview
This guide contains instructions on how to run a reference end-to-end pipeline for transfer learning with Docker container. For detailed information about the workflow, go to [End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) GitHub repository.

## How it Works
The goal of this vision-based workflow is to perform transfer learning on images in order to accomplish different classification tasks that range from binary classification to multiclass classification, giving the best possible performance on Intel Hardware utilizing the available optimizations.

The pipeline showcases how transfer learning enabled by Intel optimized TensorFlow could be used for image classification in three domains: sports , medical imaging, and remote sensing. The workflow showcases AMX  BF16 in SPR which speeds up the training time significantly, without loss in accuracy.

The workflow uses pretrained SOTA models (RESNET V1.5) from TF hub and transfers the knowledge from a pretrained domain to a different custom domain, achieving the required accuracy.

<br><img src="https://user-images.githubusercontent.com/52259352/202562899-d2867491-f08b-4393-be27-d8db28931bd6.png"><br>
<br><img src="https://user-images.githubusercontent.com/52259352/202562891-5b065c21-9ea5-427d-b555-8cc3419c8a39.png"><br>

## Get Started

### **Prerequisites**

#### Download the Repo
```
git clone https://github.com/intel/vision-based-transfer-learning-and-inference.git
cd vision-based-transfer-learning-and-inference
git checkout v1.0.1
```
#### Download the Datasets
The Medical Imaging dataset is downloaded from TensorFlow website when the code is run for the first time. The dataset used for this domain is `colorectal_histology`. More details can be found at [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/colorectal_histology). 

The Remote Sensing dataset used for this domain is [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45).  
[Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate ) the dataset and unzip the folder. To use this dataset, it should be split into `validation` and `train` directories. Use the script [resisc_dataset.py](https://github.com/intel/vision-based-transfer-learning-and-inference/blob/v1.0.1/resisc_dataset.py). Follow the example below:
```
pip install -r requirements.txt
export INPUT_DIR=<path_to_NWPU-RESISC45_dir>
export OUTPUT_DIR=<path_to_split_dataset>
python3 resisc_dataset.py --INDIR=${INPUT_DIR} --OUTDIR=${OUTPUT_DIR}
mv ${OUTPUT_DIR}/val ${OUTPUT_DIR}/validation
```

### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).

#### **Pull Docker Image**
```
docker pull intel/ai-workflows:vision-transfer-learning-training
```

#### How to Run 

(Optional) Export related proxy into docker environment.
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

For example, you can run single instance using the following options: `PLATFORM=None`, `PRECISION=FP32` and `SCRIPT=colorectal`.
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

#### Install conda and create new environment

##### Download Miniconda and install

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Install conda following the steps.

##### Create environment:

```
conda create -n transfer_learning python=3.8 --yes
conda activate transfer_learning
```

##### Install TCMalloc

```
conda install -c conda-forge gperftools -y
Set conda path and LD_PRELOAD path
eg :
CONDA_PREFIX=/home/sdp/miniconda3/envs/inc/
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so"
```
##### Install Required packages

```
pip install -r requirements.txt
```
#### How to run 
##### Command Line Arguments

```
--PRECISION - whether to use Mixed_Precision or FP32 precision Options : [FP32(default),Mixed Precision]"
              For Mixed Precion , BF16 is used if supported by hardware , if FP16 is supported it is chosen, if none is supported falls back to FP32
--PLATFORM - To optimize for SPR : [None(default),SPR]"
--inference - whether to run only inference"
--cp  - Specify checkpoint directory for inference"
--OUTPUT_DIR  - Specify output Directory where training checkpoints. graphs need to be saved"
--DATASET_DIR  - Specify dataset Directory; if using custom dataset please have train,val,test folders in dataset directory. 
                 If test dataset is not present validation dataset is used"
 --BATCH_SIZE - Batch Size for training[32(default)]"
 --NUM_EPOCHS  - Num epochs for training[100(default)]"

These options can also be set via export variable

ex : export OUTPUT_DIR="logs/fit/trail" 
```
 

#### To run in SPR 


   ##### 1) Remote Sensing Dataset Training
        a) FP32 : bash resisc.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/resiscFP32/" --DATASET_DIR datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256
        b) BF16: bash resisc.sh --PRECISION Mixed_Precision  --OUTPUT_DIR "logs/fit/resiscBF16/" --DATASET_DIR  datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256

   ##### 2) Medical Imaging Dataset Training
        a) FP32 : bash colorectal.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/colorectalFP32/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
        b) BF16: bash colorectal.sh --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/colorectalBF16/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
   
## Recommended Hardware 
### Operating Systems

| Name | Version | 
| ------ | ------ |
| RHEL | 8.2 or higher |
| CentOS | 8.2 or higher |
| Ubuntu | 18.04<br>20.04 |

### Processor

| Name | Version | 
| ------ | ------ |
| x86 | x86-64 |

### Software Dependencies

| Name | Version | 
| ------ | ------ |
| numactl | N/A |
| scikit-learn | 1.1.2 |
| tensorflow-datasets | 4.6.0 |
| tensorflow-hub | 0.12.0|
| tensorflow | 2.9.0 |
| numpy | 1.23.2 |
| matplotlib | 3.5.2 |
|tensorflow | 2.10.0|

## Useful Resources 
[Intel?? AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

## Support  
[End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) tracks both bugs and enhancement requests using Github. We welcome input, however, before filing a request, please make sure you do the following: Search the Github issue database.
