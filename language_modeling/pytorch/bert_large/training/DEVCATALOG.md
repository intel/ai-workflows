# Document Level Sentiment Analysis Using Hugging Face Transformers - Fine-Tuning

Learn to deploy E2E DLSA pipeline and Intel optimized software for fine-tuning using Hugging Face Transformers with PyTorch*, Hugging Face, and Intel® Neural Compressor.

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
DLSA is an Intel optimized representative End-to-end Fine-Tuning and Inference pipeline for Document level sentiment analysis (DLSA) using BERT model and implemented with Hugging Face transformer APIs. For detailed information about the workflow, go to the [Document Level Sentiment Analysis](https://github.com/intel/document-level-sentiment-analysis) GitHub repository.

## Recommended Hardware 
Depending on your required precision, we recommend you use the following hardware for this reference implementation: 
| Recommended Hardware         |  Precision  |
| ---------------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors| BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |


## How it Works
* Uses Hugging Face Tokenizer API, Intel PCL Optimization with Hugging Face for Fine-Tuning and Intel Extension for PyTorch for inference optimizations and quantization.

* Classifies the sentiment of any input paragraph of English text as either a positive or negative sentiment.

* Uses Hugging Faces BERT model with Masked-Language-Modeling task pretrained using large corpus of English language data, to fine tune a new BERT model with sentiment analysis task using SST-2 (the Stanford Sentiment Treebank) or IMDB (the Internet Movie database) dataset.

* The workflow uses BF16 precision in Saphire Rapids, which speeds up the training time using Intel® Advanced Matrix Extensions (Intel® AMX), without noticeable loss in accuracy when compared to FP32 precision using Intel® Advanced Vector Extensions 512 (Intel® AVX-512).

<img width="939" alt="DLSA_workflow" src="https://user-images.githubusercontent.com/43555799/204886026-8c2e1540-3ff7-42ac-9a61-f0a25553b563.png">

## Get Started


### Download the Workflow Repository
Clone the [Main Repository](https://github.com/intel/document-level-sentiment-analysis.git) repository into your working directory.
```
git clone https://github.com/intel/document-level-sentiment-analysis.git
cd document-level-sentiment-analysis/profiling-transformers
git checkout v1.0.0
```
### Download the Datasets
The datasets are not included in the workflow repo, for that you'll need to:
1. Create a ``datasets`` folder in the ``profiling-transformers/`` folder. 
2. Download and extract the dataset in the new datasets directory

Be sure to have a good network connection. In our example you'll download
two large datasets (about 88MB total): SST-2 (the Stanford Sentiment Treebank)
and IMDB (the Internet Movie database). 
```
mkdir datasets
cd datasets
# Download and extract SST-2 dataset
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip && unzip SST-2.zip && mv SST-2 sst
# Download and extract IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && tar -zxf aclImdb_v1.tar.gz
cd ..
```

---

## Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.

On the recommended hardware, it should take about two hours to run
this reference implementation example using Docker or bare metal.

### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.


### Set Up Docker Image
Pull the provided docker image.
```
docker pull intel/ai-workflows:document-level-sentiment-analysis
```

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

### Run Docker Image
Run different fine-tuning pipeline by replacing "Bash Command" according to the [pipeline](#fine-tuning-pipelines).
```
docker run -a stdout  $DOCKER_RUN_ENVS  --volume $(pwd):/workspace --workdir /workspace \
--privileged --init -it intel/ai-workflows:document-level-sentiment-analysis <Bash Command>
```
For example, here is how to run the single node fine-tuning pipeline with stock PyTorch & SST-2.
```
export DATASET=sst2
export MODEL=bert-large-uncased
export OUTPUT_DIR=$(pwd)/output
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

---

## Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running a provided Docker image with Docker, see the [Docker
instructions](#run-using-docker).

On the recommended hardware, it should take about two hours to run
this reference implementation example using Docker or bare metal.

### Set Up Workflow
```
conda create -n dlsa python=3.8 --yes
conda activate dlsa
sh install.sh
```

### Run Workflow
Run a specific fine-tuning pipeline by following the instructions in the following section.
```
<bash shell commands>
```

# Fine-Tuning Pipelines

## How to Run DLSA Single Node Fine-Tuning with Trainer(FP32, BF16)

### Single node (CPU)

```
./fine-tuning/train_trainer.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/train_native.sh -h`:

```markdown
Usage: ./fine-tuning/train_trainer.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --bf16 - whether using hf bf16 inference
   --use_ipex - whether using ipex
   -h | --help - displays this message
```
---

## How to Run DLSA Single Node Fine-Tuning Pipeline with Stock PyTorch

### Single node (CPU)

```
./fine-tuning/train_native.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/train_native.sh -h`:

```markdown
Usage: ./fine-tuning/train_native.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   ~~--bf16_ipex_ft - wether to use bf16_ipex_ft precision~~
   ~~--fp32_ipex_ft - wether to use fp32_ipex_ft precision~~
   -h | --help - displays this message
```
---

## How to Run DLSA Single Node Fine-Tuning with IPEX(FP32, BF16)

### Single node (CPU)

```
./fine-tuning/train_native.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/train_native.sh -h`:

```markdown
Usage: ./fine-tuning/train_native.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --bf16_ipex_ft - wether to use bf16_ipex_ft precision
   --fp32_ipex_ft - wether to use fp32_ipex_ft precision
   -h | --help - displays this message
```
---

## How to Run DLSA Multi Instance Fine-Tuning with IPEX (FP32, BF16)

### Install MPI library:

Install MPI from [here]( https://anaconda.org/intel/impi_rt )

MPI is included in the Intel OneAPI Toolkit. It's recommended to use the package manager to install.

>Note: This step should be operated on all the work nodes

### To run:

```
source /opt/intel/oneapi/mpi/latest/env/vars.sh
cd profiling-transformers
```

>Note:
>
>np: num process, means how many processes you will run on a cluster
>
>ppn: process per node, means how many processes you will run on 1 worker node.
>
>For example, if I want to run on 2 nodes, each node runs with 1 process, use the config `-np 2 -ppn 1`
>
>If I want to run on 4 nodes, each node runs with 2 processes, use the config `-np 8 -ppn 2`

### Running single process in single node

```
bash fine-tuning/run_dist.sh -np 1 -ppn 1 bash fine-tuning/run_ipex_native.sh
```

### Running multi instances in single node

```
# Run 2 instances in single node
bash fine-tuning/run_dist.sh -np 2 -ppn 2 bash fine-tuning/run_ipex_native.sh
```

### Running with IPEX BF16

>Before you run BF16 fine-tuning, you need to verify whether your server supports BF16. (Only Copper Lake & Sapphire Rapids CPUs support BF16)

add `--bf16_ipex_ft` at the end of the command:

```
bash fine-tuning/run_dist.sh -np 2 -ppn 2 bash fine-tuning/run_ipex_native.sh --bf16_ipex_ft 1
```
---


## Expected Output
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



## Learn More
For more information or to read about other relevant workflow
examples, see these guides and software resources:

- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Optimization for PyTorch*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html)
- [Optimizing End-to-End Artificial Intelligence Pipelines](https://www.intel.com/content/www/us/en/developer/articles/technical/optimizing-end-to-end-ai-pipelines.html#gs.pdcvha)



## Support  
The End-to-end Document Level Sentiment Analysis team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/document-level-sentiment-analysis/issues). Before submitting a suggestion or bug report,  search the [DLSA GitHub issues](https://github.com/intel/document-level-sentiment-analysis/issues) to see if your issue has already been reported.

