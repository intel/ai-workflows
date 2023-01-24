# **Document Level Sentiment Analysis Using Hugging Face Transformers - Fine-Tuning**

## Overview
DLSA is Intel optimized representative End-to-end Fine-Tuning & Inference pipeline for Document level sentiment analysis using BERT model implemented with Hugging Face transformer API. For detailed information about the workflow, go to [Document Level Sentiment Analysis](https://github.com/intel/document-level-sentiment-analysis) GitHub repository.


## How it Works

* Uses Hugging Face Tokenizer API, Intel PCL Optimization with Hugging Face for Fine-Tuning and Intel Extension for PyTorch for inference optimizations and quantization.

* Classifies the sentiment of any input paragraph of English text as either a positive or negative sentiment.

* Uses HF’s BERT model with Masked-Language-Modeling task pretrained using large corpus of English language data, to fine tune a new BERT model with sentiment analysis task using SST-2 or IMDB dataset.

* The workflow uses BF16 precision in SPR which speeds up the training time using Intel® AMX, without noticeable loss in accuracy when compared to FP32 precision using (Intel®  AVX-512).

<img width="939" alt="DLSA_workflow" src="https://user-images.githubusercontent.com/43555799/204886026-8c2e1540-3ff7-42ac-9a61-f0a25553b563.png">

## Get Started

### **Prerequisites**

#### Download the Repo
```
git clone https://github.com/intel/document-level-sentiment-analysis.git
cd document-level-sentiment-analysis/profiling-transformers
git checkout v1.0.0
```
#### Download the Datasets
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
Below setup and how-to-run sessions are for users who want to use the provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).

#### Setup 

##### Pull Docker Image
```
docker pull intel/ai-workflows:document-level-sentiment-analysis
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
Run different fine-tuning pipeline by replacing "Bash Command" according to the pipeline.
```
docker run -a stdout  $DOCKER_RUN_ENVS  --volume $(pwd):/workspace --workdir /workspace \
--privileged --init -it intel/ai-workflows:document-level-sentiment-analysis <Bash Command>
```
For example, here is how to run the single node fine-tuning pipeline with stock PyTorch.
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
  --workdir /workspace \
  --privileged --init -it \
  intel/ai-workflows:document-level-sentiment-analysis \
  fine-tuning/run_dist.sh -np 1 -ppn 1 fine-tuning/run_ipex_native.sh
```
#### Output

```
$ make hugging-face-dlsa
[+] Building 97.7s (9/9) FINISHED                                                                                                                                                                                          
 => [internal] load build definition from Dockerfile.hugging-face-dlsa                                                                                                                                                0.0s
 => => transferring dockerfile: 1.05kB                                                                                                                                                                                0.0s
 => [internal] load .dockerignore                                                                                                                                                                                     0.0s
 => => transferring context: 2B                                                                                                                                                                                       0.0s
 => [internal] load metadata for docker.io/intel/intel-optimized-pytorch:latest                                                                                                                                       1.2s
 => [auth] intel/intel-optimized-pytorch:pull token for registry-1.docker.io                                                                                                                                          0.0s
 => [1/4] FROM docker.io/intel/intel-optimized-pytorch:latest@sha256:37fd3ab2e8e40272780aeeb99631e9168f0cbecf38c397bda584ca7ec645e359                                                                                33.7s
 => => resolve docker.io/intel/intel-optimized-pytorch:latest@sha256:37fd3ab2e8e40272780aeeb99631e9168f0cbecf38c397bda584ca7ec645e359                                                                                 0.0s
 => => sha256:95e46843f5d3a46cc9eeb35c27d6f0f58d75f4d7f7fd0f463c6c5ff75918aee0 4.00kB / 4.00kB                                                                                                                        0.0s
 => => sha256:4f419499a9a5bad041fbf20b19ebf7aa21b343d62665947f9be58f2ca9019d65 184B / 184B                                                                                                                            0.2s
 => => sha256:37fd3ab2e8e40272780aeeb99631e9168f0cbecf38c397bda584ca7ec645e359 1.78kB / 1.78kB                                                                                                                        0.0s
 => => sha256:8e5c1b329fe39c318c0d49821b339fb94a215c5dc0a2898c8030b5a4d091bcba 28.57MB / 28.57MB                                                                                                                      0.7s
 => => sha256:7acca15cf7e304377750d2d34d63130f7af645384659f6502b27eef5da37a048 145.31MB / 145.31MB                                                                                                                    4.9s
 => => sha256:d25748f32da6f2ed3e3ed10d063ee06df0b25a594081fe9caeec13ee3ce58516 256B / 256B                                                                                                                            0.5s
 => => sha256:9a9f79fe09de72088aba7c5d904807b40af2f4078a17aa589bcbf4654b0adad2 705.19MB / 705.19MB                                                                                                                   19.4s
 => => sha256:4507af6d9549d9ba35b46f19157ff5591f0d16f2fb18f760a0d122f0641b940d 192B / 192B                                                                                                                            0.9s
 => => extracting sha256:8e5c1b329fe39c318c0d49821b339fb94a215c5dc0a2898c8030b5a4d091bcba                                                                                                                             0.5s
 => => sha256:08036a96427d59e64f289f6d420a8a056ea1a3d419985e8e8aa2bb8c5e014c88 62.13kB / 62.13kB                                                                                                                      1.2s
 => => extracting sha256:7acca15cf7e304377750d2d34d63130f7af645384659f6502b27eef5da37a048                                                                                                                             2.9s
 => => extracting sha256:4f419499a9a5bad041fbf20b19ebf7aa21b343d62665947f9be58f2ca9019d65                                                                                                                             0.0s
 => => extracting sha256:d25748f32da6f2ed3e3ed10d063ee06df0b25a594081fe9caeec13ee3ce58516                                                                                                                             0.0s
 => => extracting sha256:9a9f79fe09de72088aba7c5d904807b40af2f4078a17aa589bcbf4654b0adad2                                                                                                                            13.3s
 => => extracting sha256:4507af6d9549d9ba35b46f19157ff5591f0d16f2fb18f760a0d122f0641b940d                                                                                                                             0.0s
 => => extracting sha256:08036a96427d59e64f289f6d420a8a056ea1a3d419985e8e8aa2bb8c5e014c88                                                                                                                             0.0s
 => [2/4] RUN apt-get update && apt-get install --no-install-recommends --fix-missing -y     ca-certificates     git     libgomp1     numactl     patch     wget     mpich                                           22.6s
 => [3/4] RUN mkdir -p /workspace                                                                                                                                                                                     0.6s
 => [4/4] RUN pip install --upgrade pip &&     pip install astunparse                 cffi                 cmake                 dataclasses                 datasets==2.3.2                 intel-openmp            33.6s 
 => exporting to image                                                                                                                                                                                                5.9s 
 => => exporting layers                                                                                                                                                                                               5.9s 
 => => writing image sha256:87b75ae09f3b6ac3f04b245c0aaa0288f9064593fcffb57f9389bf6e59a82f30                                                                                                                          0.0s 
 => => naming to docker.io/library/hugging-face-dlsa:training-intel-optimized-pytorch-latest                                                                                                                          0.0s 
WARN[0097] Found orphan containers ([training-vision-transfer-learning-1]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.                                                                                                                                                                                                                     
[+] Running 1/1
 ⠿ Container training-hugging-face-dlsa-1  Created                                                                                                                                                                    0.2s
Attaching to training-hugging-face-dlsa-1
training-hugging-face-dlsa-1  | Running 2 tasks on 1 nodes with ppn=2
training-hugging-face-dlsa-1  | /opt/conda/bin/python
training-hugging-face-dlsa-1  | /usr/bin/gcc
training-hugging-face-dlsa-1  | /usr/bin/mpicc
training-hugging-face-dlsa-1  | /usr/bin/mpiexec.hydra
training-hugging-face-dlsa-1  | #### INITIAL ENV ####
training-hugging-face-dlsa-1  | Using CCL_WORKER_AFFINITY=0,28
training-hugging-face-dlsa-1  | Using CCL_WORKER_COUNT=1
training-hugging-face-dlsa-1  | Using I_MPI_PIN_DOMAIN=[0xFFFFFFE,0xFFFFFFE0000000]
training-hugging-face-dlsa-1  | Using KMP_BLOCKTIME=1
training-hugging-face-dlsa-1  | Using KMP_HW_SUBSET=1T
training-hugging-face-dlsa-1  | Using OMP_NUM_THREADS=27
training-hugging-face-dlsa-1  | Using LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so:/opt/conda/lib/libiomp5.so:
training-hugging-face-dlsa-1  | Using PYTORCH_MPI_THREAD_AFFINITY=0,28
training-hugging-face-dlsa-1  | Using DATALOADER_WORKER_COUNT=0
training-hugging-face-dlsa-1  | Using ARGS_NTASKS=2
training-hugging-face-dlsa-1  | Using ARGS_PPN=2
training-hugging-face-dlsa-1  | #### INITIAL ENV ####
training-hugging-face-dlsa-1  | PyTorch version: 1.11.0+cpu
training-hugging-face-dlsa-1  | MASTER_ADDR=e7930b8a9eae
training-hugging-face-dlsa-1  | [0] e7930b8a9eae
training-hugging-face-dlsa-1  | [1] e7930b8a9eae
```
...
```
training-hugging-face-dlsa-1  | [0] *********** TEST_METRICS ***********
training-hugging-face-dlsa-1  | [0] Accuracy: 0.5091743119266054
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Inference' took 66.178s (66,177,509,482ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] ##############################
training-hugging-face-dlsa-1  | [0] Benchmark Summary:
training-hugging-face-dlsa-1  | [0] ##############################
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Load Data' took 0.064s (63,980,232ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '----Training data encoding' took 1.625s (1,625,314,718ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '----Training tensor data convert' took 0.000s (86,070ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '----PyTorch test data encoding' took 0.030s (30,427,632ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '----PyTorch test tensor data convert' took 0.000s (69,433ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '----Init tokenizer' took 6.438s (6,438,305,050ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Pre-process' took 6.438s (6,438,316,023ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Load Model' took 25.581s (25,581,075,968ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Process int8 model' took 0.000s (1,322ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Process bf16 model' took 0.000s (755ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '--------Init Fine-Tuning' took 0.007s (7,005,553ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '--------Training Loop' took 6400.528s (6,400,528,084,071ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] '--------Save Fine-Tuned Model' took 1.846s (1,845,884,212ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Fine-Tune' took 6402.381s (6,402,381,151,290ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 'Inference' took 66.178s (66,177,509,482ns)
training-hugging-face-dlsa-1  | [0] **********************************************************************
training-hugging-face-dlsa-1  | [0] 
training-hugging-face-dlsa-1  | [0] 
Train Step: 100%|██████████| 2105/2105 [1:48:26<00:00,  3.09s/it][1] [1] 
Epoch: 100%|██████████| 1/1 [1:48:26<00:00, 6506.01s/it][1] s/it][1] 
Test Step:   0%|          | 0/109 [00:00<?, ?it/s][1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '--------Training Loop' took 6506.010s (6,506,009,911,526ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '--------Save Fine-Tuned Model' took 5.175s (5,175,177,382ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Fine-Tune' took 6511.189s (6,511,189,374,535ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
Test Step: 100%|██████████| 109/109 [01:08<00:00,  1.58it/s][1] 
training-hugging-face-dlsa-1  | [1] *********** TEST_METRICS ***********
training-hugging-face-dlsa-1  | [1] Accuracy: 0.5091743119266054
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Inference' took 68.940s (68,939,695,151ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] ##############################
training-hugging-face-dlsa-1  | [1] Benchmark Summary:
training-hugging-face-dlsa-1  | [1] ##############################
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Load Data' took 0.061s (60,809,048ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '----Training data encoding' took 1.639s (1,638,570,780ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '----Training tensor data convert' took 0.000s (74,039ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '----PyTorch test data encoding' took 0.035s (35,348,035ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '----PyTorch test tensor data convert' took 0.000s (62,836ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '----Init tokenizer' took 6.566s (6,566,305,202ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Pre-process' took 6.566s (6,566,314,800ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Load Model' took 25.585s (25,584,507,153ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Process int8 model' took 0.000s (1,046ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Process bf16 model' took 0.000s (1,156ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '--------Init Fine-Tuning' took 0.004s (4,075,527ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '--------Training Loop' took 6506.010s (6,506,009,911,526ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] '--------Save Fine-Tuned Model' took 5.175s (5,175,177,382ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Fine-Tune' took 6511.189s (6,511,189,374,535ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 'Inference' took 68.940s (68,939,695,151ns)
training-hugging-face-dlsa-1  | [1] **********************************************************************
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | [1] 
training-hugging-face-dlsa-1  | End Time:    Wed Aug 31 17:40:30 UTC 2022
training-hugging-face-dlsa-1  | Total Time: 110 min and 14 sec
training-hugging-face-dlsa-1 exited with code 0
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
#### How to Run 

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

