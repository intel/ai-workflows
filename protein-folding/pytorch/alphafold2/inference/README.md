# **PyTorch AlphaFold2 INFERENCE - AI Drug Design**

## **Description**
This document provides instructions on how to run inference pipeline of AlphaFold2 prediction of protein structures with make and docker compose. For bare metal instructions, please visit [Model Zoo for Intel® Architecture - AlphaFold2 Inference](https://github.com/IntelAI/models/tree/v2.9.0/quickstart/aidd/pytorch/alphafold2/inference)

## Project Structure 
```
├── protein-prediction @ v2.9.0
├── DEVCATALOG.md
├── docker-compose.yml
├── Dockerfile.protein-prediction
├── Makefile
└── README.md
```
[_Makefile_](Makefile)
```
.PHONY: protein-prediction
DATASET_DIR ?= /dataset
EXPERIMENT_NAME ?= testing
FINAL_IMAGE_NAME ?= protein-structure-prediction
MODEL ?= model_1
OUTPUT_DIR ?= /output

protein-prediction:
	mkdir -p '${OUTPUT_DIR}/weights/extracted' '${OUTPUT_DIR}/logs' '${OUTPUT_DIR}/samples' '${OUTPUT_DIR}/experiments/${EXPERIMENT_NAME}'
	curl -o ${OUTPUT_DIR}/samples/sample.fa https://rest.uniprot.org/uniprotkb/Q6UWK7.fasta
	@EXPERIMENT_NAME=${EXPERIMENT_NAME} \
	 DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL=${MODEL} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
 	docker compose up protein-prediction-inference --build

clean: 
	@DATASET_DIR=${DATASET_DIR} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	docker compose down
```
[_docker-compose.yml_](docker-compose.yml)
```
services:
  param:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.protein-prediction
    command: conda run -n alphafold2 --no-capture-output python extract_params.py --input /dataset/params/params_${MODEL}.npz --output_dir /output/weights/extracted/${MODEL}
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - EXPERIMENT_NAME=${EXPERIMENT_NAME}
      - MODEL=${MODEL}
      - OUTPUT_DIR=${OUTPUT_DIR}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:inference-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - $PWD:/workspace
      - ${OUTPUT_DIR}:/output
    working_dir: /workspace/protein-prediction/models/aidd/pytorch/alphafold2/inference
  protein-prediction-preproc:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.protein-prediction
    command: conda run -n alphafold2 --no-capture-output bash online_preproc_baremetal.sh /output /dataset /output/samples /output/experiments/${EXPERIMENT_NAME}
    depends_on:
      param:
        condition: service_completed_successfully
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - EXPERIMENT_NAME=${EXPERIMENT_NAME}
      - MODEL=${MODEL}
      - OUTPUT_DIR=${OUTPUT_DIR}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:inference-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - $PWD:/workspace
      - ${OUTPUT_DIR}:/output
    working_dir: /workspace/protein-prediction/quickstart/aidd/pytorch/alphafold2/inference
  protein-prediction-inference:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.protein-prediction
    command: conda run -n alphafold2 --no-capture-output bash online_inference_baremetal.sh /opt/conda/envs/alphafold2 /output /dataset /output/samples /output/experiments/${EXPERIMENT_NAME} ${MODEL}  
    depends_on:
      protein-prediction-preproc:
        condition: service_completed_successfully
    environment: 
      - DATASET_DIR=${DATASET_DIR}
      - EXPERIMENT_NAME=${EXPERIMENT_NAME}
      - MODEL=${MODEL}
      - OUTPUT_DIR=${OUTPUT_DIR}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:inference-ubuntu-20.04
    privileged: true
    volumes: 
      - ${DATASET_DIR}:/dataset
      - $PWD:/workspace
      - ${OUTPUT_DIR}:/output
    working_dir: /workspace/protein-prediction/quickstart/aidd/pytorch/alphafold2/inference
```

# **PyTorch AlphaFold2 Inference**
End-to-End AI Workflow utilizing Intel® Xeon and Intel® Optane® PMem by Intel® oneAPI. For more information, please visit [Model Zoo for Intel® Architecture - AlphaFold2 Inference repository](https://github.com/IntelAI/models/tree/v2.9.0/quickstart/aidd/pytorch/alphafold2/inference)

## Quick Start

* Install [Pipeline Repository Dependencies](../../../../README.md)

* Pull and configure the dependent repo submodule `git submodule update --init --recursive`.

* Download dataset:
All dataset can be downloaded using `download_all_data` script. Please specify dataset directory.
```
bash ./protein-prediction/models/aidd/pytorch/alphafold2/inference/alphafold/scripts/download_all_data.sh <DOWNLOAD_DIR>
```
To download individual dataset, seperate scripts are also available at [AlphaFold repository v2.0.1 script directory](https://github.com/deepmind/alphafold/tree/v2.0.1/scripts).
Note: The total size of the dataset is around 750G. This process could take a while depending on internet connection.

* Other variables:

| Variable Name | Default | Notes |
| --- | --- | --- |
| DATASET | `/dataset` | Dataset directory |
| EXPERIMENT_NAME | `testing` | User defined experienment name |
| FINAL_IMAGE_NAME | `protein-structure-prediction` | Final Docker image name |
| MODEL | `model_1` | Model name. Other models are available at [AlphaFold](https://github.com/deepmind/alphafold) repository |
| OUTPUT_DIR | `/output` | Output directory |

## Build and Run
Build and run with pre-processing script:
* This step will set up all directories in default output directory. Download a sample file to `/samples` directory. Extract model parameter based on the model selected. Then run preprocessing script and inferencing script. 
```
make protein-prediction
```

## Build and Run Example
The make command below shows user specified `DATASET_DIR` and `OUTPUT_DIR`. Make sure `OUTPUT_DIR` is an empty directory. 
```
$ DATASET_DIR=/localdisk/user/dataset OUTPUT_DIR=/localdisk/user/output make protein-prediction
mkdir -p 'weights/extracted' 'logs' 'samples' 'experiments/testing'
curl -o ./samples/sample.fa https://rest.uniprot.org/uniprotkb/Q6UWK7.fasta
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   167  100   167    0     0    275      0 --:--:-- --:--:-- --:--:--   275
[+] Building 0.4s (9/9) FINISHED                                                                   
 => [internal] load build definition from Dockerfile.protein-prediction                       0.0s
 => => transferring dockerfile: 51B                                                           0.0s
 => [internal] load .dockerignore                                                             0.0s
 => => transferring context: 2B                                                               0.0s
 => [internal] load metadata for docker.io/library/ubuntu:20.04                               0.3s
 => [1/5] FROM docker.io/library/ubuntu:20.04@sha256:450e066588f42ebe1551f3b1a535034b6aa46cd  0.0s
 => CACHED [2/5] RUN apt-get update && apt-get install --no-install-recommends --fix-missing  0.0s
 => CACHED [3/5] RUN apt-get update &&     wget --quiet https://repo.anaconda.com/miniconda/  0.0s
 => CACHED [4/5] RUN conda create -yn alphafold2 python=3.9.7 &&     source activate alphafo  0.0s
 => CACHED [5/5] RUN mkdir -p /workspace                                                      0.0s
 => exporting to image                                                                        0.0s
 => => exporting layers                                                                       0.0s
 => => writing image sha256:dcefbb1fa3f6ba08cc58fee6800314cb76d5b3b9e60b81c32b4151b20662b26c  0.0s
 => => naming to docker.io/library/protein-structure-prediction:inference-ubuntu-20.04        0.0s
[+] Running 4/4
 ⠿ Network inference_default                           Created                                0.0s
 ⠿ Container inference-param-1                         Creat...                               0.1s
 ⠿ Container inference-protein-prediction-preproc-1    Created                                0.1s
 ⠿ Container inference-protein-prediction-inference-1  Created                                0.1s
Attaching to inference-protein-prediction-inference-1
inference-protein-prediction-inference-1  | modelinfer /workspace/samples/sample.fa on core 0-55 of socket 0-1
inference-protein-prediction-inference-1  | /opt/conda/envs/alphafold2/lib/python3.9/site-packages/absl/flags/_validators.py:254: UserWarning: Flag --preset has a non-None default value; therefore, mark_flag_as_required will pass even if flag is not specified in the command line!
inference-protein-prediction-inference-1  |   mark_flag_as_required(flag_name, flag_values)
inference-protein-prediction-inference-1  | /opt/conda/envs/alphafold2/lib/python3.9/site-packages/intel_extension_for_pytorch/frontend.py:396: UserWarning: Conv BatchNorm folding failed during the optimize process.
inference-protein-prediction-inference-1  |   warnings.warn("Conv BatchNorm folding failed during the optimize process.")
inference-protein-prediction-inference-1  | /opt/conda/envs/alphafold2/lib/python3.9/site-packages/intel_extension_for_pytorch/frontend.py:401: UserWarning: Linear BatchNorm folding failed during the optimize process.
inference-protein-prediction-inference-1  |   warnings.warn("Linear BatchNorm folding failed during the optimize process.")
```
...
```
inference-protein-prediction-inference-1  | ### Validate preprocessed results.
inference-protein-prediction-inference-1  | ### [INFO] output_dir= /workspace/experiments/sample
inference-protein-prediction-inference-1  | ######### /workspace/experiments/sample/intermediates
inference-protein-prediction-inference-1  | ### [INFO] build evoformer network
inference-protein-prediction-inference-1  | ### [INFO] Execute model inference
inference-protein-prediction-inference-1  | ### [INFO] jit compilation
inference-protein-prediction-inference-1  | ### [INFO] start AlphaFold Iteration-1
inference-protein-prediction-inference-1  |   [INFO] atom37 -> torsion angles
inference-protein-prediction-inference-1  |   # [INFO] start evoformer iteration 0
inference-protein-prediction-inference-1  |   # [INFO] linear embedding of features
inference-protein-prediction-inference-1  |   # [INFO] embedding left/right single 
inference-protein-prediction-inference-1  |   # [INFO] embedding previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] recycle previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] cvt residue features to one-hot format 
inference-protein-prediction-inference-1  |  ## [INFO] execute template embedding
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_activations
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_iterations
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 1/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 2/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 3/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 4/4
inference-protein-prediction-inference-1  |  ## [INFO] execute template projection
inference-protein-prediction-inference-1  |   # [TIME] total embedding duration = 1.3425679206848145 sec
inference-protein-prediction-inference-1  |  ## [INFO] execute evoformer_iterations
inference-protein-prediction-inference-1  |   # [TIME] total evoformer duration = 8.5543217658996582 sec
inference-protein-prediction-inference-1  |   # [TIME] total heads duration = 29.62464928627014 sec
inference-protein-prediction-inference-1  |   # [INFO] save curr update as previous output.
inference-protein-prediction-inference-1  |   # [INFO] update to prev done.
inference-protein-prediction-inference-1  |   # [INFO] duration = 111.93s
inference-protein-prediction-inference-1  | ### [INFO] start AlphaFold Iteration-2
inference-protein-prediction-inference-1  |   [INFO] atom37 -> torsion angles
inference-protein-prediction-inference-1  |   # [INFO] start evoformer iteration 1
inference-protein-prediction-inference-1  |   # [INFO] linear embedding of features
inference-protein-prediction-inference-1  |   # [INFO] embedding left/right single 
inference-protein-prediction-inference-1  |   # [INFO] embedding previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] recycle previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] cvt residue features to one-hot format 
inference-protein-prediction-inference-1  |  ## [INFO] execute template embedding
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_activations
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_iterations
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 1/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 2/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 3/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 4/4
inference-protein-prediction-inference-1  |  ## [INFO] execute template projection
inference-protein-prediction-inference-1  |   # [TIME] total embedding duration = 1.1267759799957275 sec
inference-protein-prediction-inference-1  |  ## [INFO] execute evoformer_iterations
inference-protein-prediction-inference-1  |   # [TIME] total evoformer duration = 7.067838191986084 sec
inference-protein-prediction-inference-1  |   # [TIME] total heads duration = 0.056215763092041016 sec
inference-protein-prediction-inference-1  |   # [INFO] save curr update as previous output.
inference-protein-prediction-inference-1  |   # [INFO] update to prev done.
inference-protein-prediction-inference-1  |   # [INFO] duration = 116.39s
inference-protein-prediction-inference-1  | ### [INFO] start AlphaFold Iteration-3
inference-protein-prediction-inference-1  |   [INFO] atom37 -> torsion angles
inference-protein-prediction-inference-1  |   # [INFO] start evoformer iteration 2
inference-protein-prediction-inference-1  |   # [INFO] linear embedding of features
inference-protein-prediction-inference-1  |   # [INFO] embedding left/right single 
inference-protein-prediction-inference-1  |   # [INFO] embedding previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] recycle previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] cvt residue features to one-hot format 
inference-protein-prediction-inference-1  |  ## [INFO] execute template embedding
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_activations
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_iterations
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 1/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 2/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 3/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 4/4
inference-protein-prediction-inference-1  |  ## [INFO] execute template projection
inference-protein-prediction-inference-1  |   # [TIME] total embedding duration = 1.1419472694396973 sec
inference-protein-prediction-inference-1  |  ## [INFO] execute evoformer_iterations
inference-protein-prediction-inference-1  |   # [TIME] total evoformer duration = 7.0081231594085693 sec
inference-protein-prediction-inference-1  |   # [TIME] total heads duration = 0.056620121002197266 sec
inference-protein-prediction-inference-1  |   # [INFO] save curr update as previous output.
inference-protein-prediction-inference-1  |   # [INFO] update to prev done.
inference-protein-prediction-inference-1  |   # [INFO] duration = 8.24s
inference-protein-prediction-inference-1  | ### [INFO] start AlphaFold Iteration-4
inference-protein-prediction-inference-1  |   [INFO] atom37 -> torsion angles
inference-protein-prediction-inference-1  |   # [INFO] start evoformer iteration 3
inference-protein-prediction-inference-1  |   # [INFO] linear embedding of features
inference-protein-prediction-inference-1  |   # [INFO] embedding left/right single 
inference-protein-prediction-inference-1  |   # [INFO] embedding previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] recycle previous molecular graph 
inference-protein-prediction-inference-1  |   # [INFO] cvt residue features to one-hot format 
inference-protein-prediction-inference-1  |  ## [INFO] execute template embedding
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_activations
inference-protein-prediction-inference-1  |  ## [INFO] execute extra_msa_iterations
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 1/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 2/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 3/4
inference-protein-prediction-inference-1  |   # [INFO] execute extra_msa_iter 4/4
inference-protein-prediction-inference-1  |  ## [INFO] execute template projection
inference-protein-prediction-inference-1  |   # [TIME] total embedding duration = 1.1143763065338135 sec
inference-protein-prediction-inference-1  |  ## [INFO] execute evoformer_iterations
inference-protein-prediction-inference-1  |   # [TIME] total evoformer duration = 6.8662335872650146 sec
inference-protein-prediction-inference-1  |   # [TIME] total heads duration = 0.06540632247924805 sec
inference-protein-prediction-inference-1  |   # [INFO] duration = 8.07s
inference-protein-prediction-inference-1  | ### [INFO] post-assessment: plddt
inference-protein-prediction-inference-1  | ### [INFO] post-save: unrelaxed structure
inference-protein-prediction-inference-1 exited with code 0
```
