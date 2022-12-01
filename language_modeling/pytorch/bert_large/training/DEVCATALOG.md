# **Document Level Sentiment Analysis Using Hugging Face Transformers - Fine-Tuning**

## Overview
DLSA is Intel optimized representative End-to-end Fine-Tuning & Inference pipeline for Document level sentiment analysis using BERT model implemented with Hugging Face transformer API. For detailed information about the workflow, go to [Document Level Sentiment Analysis](https://github.com/intel/document-level-sentiment-analysis) GitHub repository.


## How it Works

* Uses Hugging face Tokenizer API, Intel PCL Optimization with Hugging Face for Fine-Tuning and Intel Extension for PyTorch for inference optimizations and quantization.

* Classifies the sentiment of any input English paragraph as positive or negative sentiment.

* Uses HF’s BERT model with Masked-Language-Modeling task pretrained using large corpus of English data, to fine tune a new BERT model with sentiment analysis task using SST-2 or IMDB dataset.

* The workflow uses BF16 precision in SPR which speeds up the training time using Intel® AMX, without noticeable loss in accuracy when compared to FP32 precision using (Intel®  AVX-512).

<img width="939" alt="DLSA_workflow" src="https://user-images.githubusercontent.com/43555799/204886026-8c2e1540-3ff7-42ac-9a61-f0a25553b563.png">

## Get Started

### **Prerequisites**

#### Download the repo
```
git clone https://github.com/intel/document-level-sentiment-analysis.git
cd document-level-sentiment-analysis/profiling-transformers
git checkout v1.0.0
```
#### Download the datasets
```
mkdir datasets
cd datasets
#download and extract SST-2 dataset
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip && unzip SST-2.zip && mv SST-2 sst
#download and extract IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && tar -zxf aclImdb_v1.tar.gz
#back to profiling-transformers folder
cd ..
```
**Make sure the network connections work well for downloading the datasets.**

### **Docker**
Below setup and how-to-run sessions are for users who want to use provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).

#### Setup 

##### Pull Docker Image
```
docker pull intel/ai-workflows:document-level-sentiment-analysis
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
Run different fine-tuning pipeline by replacing "Bash Command" according to the pipeline.
```
docker run -a stdout  $DOCKER_RUN_ENVS  --volume $(pwd):/workspace --workdir /workspace \
--privileged --init -it intel/ai-workflows:document-level-sentiment-analysis <Bash Command>
```
For example, here is how to run the single node fine-tuning pipeline with stock pyTorch.
```
export DATASET=sst2
export MODEL=bert-large-uncased
export OUTPUT_DIR=/output
docker run -a stdout $DOCKER_RUN_ENVS \
  --env DATASET=${DATASET} \
  --env MODEL_NAME_OR_PATH=${MODEL} \
  --env ${OUTPUT_DIR}:${OUTPUT_DIR}/fine_tuned \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${PWD}:/workspace \
  --workdir /workspace/profiling-transformers \
  --privileged --init -it \
  intel/ai-workflows:document-level-sentiment-analysis \
  fine-tuning/run_dist.sh -np 1 -ppn 1 fine-tuning/run_ipex_native.sh
```

##### Fine-Tuning Pipeline

|  Implementations                               | Model    | Instance | API         | Framework       | Precision  |
| ---------------------------------- | -------- | -------- | ----------- | ----------------------- | ---------- |
| [Run with IPEX (Single Instance)](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/single-node-ipex.md) | HF Model  | Single   | Non-trainer | PyTorch + IPEX          | FP32,BF16  |
| [Run with IPEX (Multi Instance)](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/multi-nodes-ipex.md) | HF Model  | Multiple | Non-trainer | PyTorch + IPEX          | FP32,BF16  |

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 
```
conda create -n dlsa python=3.8 --yes
conda activate dlsa
sh install.sh
```
#### How to run 

##### Fine-Tuning Pipeline


|  Implementations                               | Model    | Instance | API         | Framework       | Precision  |
| ---------------------------------- | -------- | -------- | ----------- | ----------------------- | ---------- |
| [Run with HF Transformers + IPEX ](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/single-node-trainer.md)   | HF Model | Single   | Trainer     | PyTorch + IPEX          | FP32, BF16 |
| [Run with Stock Pytorch](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/single-node-stock-pytorch.md) | HF Model  | Single   | Non-trainer | PyTorch                 | FP32       |
| [Run with IPEX (Single Instance)](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/single-node-ipex.md) | HF Model  | Single   | Non-trainer | PyTorch + IPEX          | FP32,BF16  |
| [Run with IPEX (Multi Instance)](https://github.com/intel/document-level-sentiment-analysis/blob/main/docs/fine-tuning/multi-nodes-ipex.md) | HF Model  | Multiple | Non-trainer | PyTorch + IPEX          | FP32,BF16  |

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation. 
| Recommended Hardware         |  Precision  |
| ---------------------------------- | ---------- |
|* Intel® 4th Gen Xeon® Scalable Performance processors|BF16 |
|* Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |

## Useful Resources 
[Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

## Support  
E2E DLSA tracks both bugs and enhancement requests using Github. We welcome input, however, before filing a request, please make sure you do the following: Search the Github issue database.

