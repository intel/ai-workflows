# PyTorch BERT Large TRAINING - Hugging Face DLSA
## Description
This document contains instructions on how to run hugging face DLSA e2e pipelines with make using docker compose, argo, and helm.
## Project Structure 
```
├── dlsa @ v1.0.0
├── DEVCATALOG.md
├── Dockerfile.hugging-face-dlsa
├── Makefile
├── README.md
├── chart
│   ├── Chart.yaml
│   ├── charts
│   ├── README.md
│   ├── templates
│   │   ├── _helpers.tpl
│   │   └── workflowTemplate.yaml
│   └── values.yaml
└── docker-compose.yml

```
[_Makefile_](Makefile)
```
DATASET ?= sst2
DATASET_DIR ?= /data
FINAL_IMAGE_NAME ?= document-level-sentiment-analysis
MODEL ?= bert-large-uncased
NAMESPACE ?= argo
NUM_NODES ?= 2
OUTPUT_DIR ?= /output
PROCESS_PER_NODE ?= 2

hugging-face-dlsa:
	mkdir ./dlsa/profiling-transformers/datasets && cp -r ${DATASET_DIR} ./dlsa/profiling-transformers/datasets
	@DATASET=${DATASET} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL=${MODEL} \
	 NUM_NODES=${NUM_NODES} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 PROCESS_PER_NODE=${PROCESS_PER_NODE} \
 	 docker compose up hugging-face-dlsa --build
	rm -rf ./dlsa/profiling-transformers/datasets

argo-single-node:
	helm install \
	--namespace ${NAMESPACE} \
	--set proxy=${http_proxy} \
	--set workflow.dataset=${DATASET} \
	--set workflow.num_nodes=${NUM_NODES} \
	--set workflow.model=${MODEL} \
	--set workflow.process_per_node=${PROCESS_PER_NODE} \
	${FINAL_IMAGE_NAME} ./chart
	argo submit --from wftmpl/${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}

workflow-log:
	argo logs @latest -f

clean: 
	rm -rf ./dlsa/profiling-transformers/datasets
	docker compose down

helm-clean: 
	kubectl delete wftmpl ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
	helm uninstall ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
```
[_docker-compose.yml_](docker-compose.yml)
```
services:
  hugging-face-dlsa:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.hugging-face-dlsa
    command: fine-tuning/run_dist.sh -np ${PROCESSES_PER_NODE} -ppn ${PROCESS_PER_NODE} fine-tuning/run_ipex_native.sh
    environment: 
      - DATASET=${DATASET}
      - MODEL_NAME_OR_PATH=${MODEL}
      - OUTPUT_DIR=${OUTPUT_DIR}/fine_tuned
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-intel-optimized-pytorch-1.12.100-oneccl-inc
    privileged: true
    volumes: 
      - ${OUTPUT_DIR}:${OUTPUT_DIR}
      - ./dlsa:/workspace/dlsa
    working_dir: /workspace/dlsa/profiling-transformers
```
[_workflowTemplate.yml_](./chart/templates/workflowTemplate.yaml)
```
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: {{ .Values.metadata.name }}
  labels:
  {{- include "chart.labels" . | nindent 4 }}
spec:
  arguments:
    parameters:
    - name: ref
      value: {{ .Values.workflow.ref }}
    - name: repo
      value: {{ .Values.workflow.repo }}
    - name: http_proxy
      value: {{ .Values.proxy }}
    - enum:
      - sst2
      - imdb
      name: dataset
      value: {{ .Values.workflow.dataset }}
    - name: model
      value: {{ .Values.workflow.model }}
    - name: num_nodes
      value: {{ .Values.workflow.num_nodes }}
    - name: process_per_node
      value: {{ .Values.workflow.process_per_node }}
  entrypoint: main
  templates:
  - steps:
    - - name: git-clone
        template: git-clone
    - - name: hugging-face-dlsa-training
        template: hugging-face-dlsa-training
    name: main
  - container:
      args:
      - clone
      - -b
      - '{{"{{workflow.parameters.ref}}"}}'
      - '{{"{{workflow.parameters.repo}}"}}'
      - workspace
      command:
      - git
      env:
      - name: http_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      - name: https_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      image: intel/ai-workflows:document-level-sentiment-analysis
      volumeMounts:
      - mountPath: /workspace
        name: workspace
      workingDir: /
    name: git-clone
  - container:
      args:
      - '-c'
      - >-
        fine-tuning/run_dist.sh fine-tuning/run_ipex_native.sh
        # Keeping until multi-node support is added
        #-np '{{"{{workflow.parameters.num_nodes}}"}}' \
        #-ppn '{{"{{workflow.parameters.process_per_node}}"}}' \
        #fine-tuning/run_ipex_native.sh
      command:
      - sh
      env:
      - name: DATASET
        value: '{{"{{workflow.parameters.dataset}}"}}'
      - name: MODEL
        value: '{{"{{workflow.parameters.model}}"}}'
      - name: OUTPUT_DIR
        value: /output
      - name: http_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      - name: https_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      image: intel/ai-workflows:document-level-sentiment-analysis
      volumeMounts:
      - mountPath: /workspace
        name: workspace
      - mountPath: /output
        name: output-dir
      {{- if eq .Values.dataset.type "nfs" }}
      - mountPath: /workspace/profiling-transformers/datasets/{{ .Values.dataset.key }}
        name: dataset
        subPath: {{ .Values.dataset.nfs.subPath }}
      {{ end }}
      workingDir: /workspace/profiling-transformers
    {{- if eq .Values.dataset.type "s3" }}
    inputs: 
      artifacts:
      - name: dataset
        path: /workspace/profiling-transformers/datasets/{{ .Values.dataset.key }}
        s3:
          key: {{ .Values.dataset.datasetKey }}
    {{ end }}
    name: hugging-face-dlsa-training
    outputs:
      artifacts: 
      - name: checkpoint
        path: /output
        s3:
          key: {{ .Values.dataset.logsKey }}
    {{- if eq .Values.dataset.type "nfs" }}
    volumes:
    - name: dataset
      nfs:
        server: {{ .Values.dataset.nfs.server }}
        path: {{ .Values.dataset.nfs.path }}
        readOnly: {{ .Values.dataset.nfs.readOnly }}
    {{ end }}
  volumeClaimTemplates:
  - metadata:
      name: workspace
    name: workspace
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: {{ .Values.volumeClaimTemplates.workspace.resources.requests.storage }}
  - metadata:
      name: output-dir
    name: output-dir
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: {{ .Values.volumeClaimTemplates.output_dir.resources.requests.storage }}
```
# Hugging Face DLSA
End2End AI Workflow utilizing Hugging Face for Document-Level Sentiment Analysis

## Quick Start
* Pull and configure the dependent repo submodule `git submodule update --init --recursive`.

* Install [Pipeline Repository Dependencies](https://github.com/intel/ai-workflows/blob/main/pipelines/README.md)

* Acquire dataset files
```
# download and extract SST-2 dataset
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip && unzip SST-2.zip && mv SST-2 sst
# download and extract IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz && tar -zxf aclImdb_v1.tar.gz
```

* Other variables:

| Variable Name | Default | Notes |
| --- | --- | --- |
| DATASET | `sst2` | Name of dataset, `sst2` and `imdb` are supported |
| DATASET_DIR | `/data` | DLSA dataset directory, default is placeholder, `aclImdb` is for `DATASET=imdb` and `sst` is for `DATASET=sst2` |
| FINAL_IMAGE_NAME | `hugging-face-dlsa` | Final Docker image name |
| MODEL | `bert-large-uncased` | Name of model on [Huggingface](https://huggingface.co). |
| NAMESPACE | `argo` | For running with k8s |
| OUTPUT_DIR | `/output` | Output directory |
## Build and Run
Build and Run with defaults:
```
$ make hugging-face-dlsa
```
## Run with Argo using Helm
* Install [Helm](https://helm.sh/docs/intro/install/)
  * ```
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
    chmod 700 get_helm.sh && \
    ./get_helm.sh
    ```
* Install [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/) and [Argo CLI](https://github.com/argoproj/argo-workflows/releases)
* Configure your [Artifact Repository](https://argoproj.github.io/argo-workflows/configure-artifact-repository/)

Install Workflow Template with Helm and Submit Workflow
```
make argo-single-node
```
To view your last workflow's progress
```
make workflow-log
```
## Build and Run Example
With Docker Compose
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
With Helm
```
$ make argo-single-node
helm install \
--namespace argo \
--set proxy=... \
--set workflow.dataset=sst2 \
--set workflow.model=bert-large-uncased \
hugging-face-dlsa ./chart
NAME: hugging-face-dlsa
LAST DEPLOYED: Tue Nov 15 13:27:04 2022
NAMESPACE: argo
STATUS: deployed
REVISION: 1
TEST SUITE: None
argo submit --from wftmpl/hugging-face-dlsa --namespace=argo
Name:                hugging-face-dlsa-szdd5
Namespace:           argo
ServiceAccount:      unset
Status:              Pending
Created:             Tue Nov 15 13:27:04 -0800 (now)
Progress:
```
## Check Results
With Argo
```
$ make workflow-log
argo logs @latest -f
hugging-face-dlsa-szdd5-git-clone-3089467112: time="2022-11-15T21:27:07.829Z" level=info msg="capturing logs" argo=true
hugging-face-dlsa-szdd5-git-clone-3089467112: Cloning into 'workspace'...
hugging-face-dlsa-szdd5-git-clone-3089467112: Note: switching to 'f5b55b1afba669c42c10166570ce1ae6d2aebabd'.
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112: You are in 'detached HEAD' state. You can look around, make experimental
hugging-face-dlsa-szdd5-git-clone-3089467112: changes and commit them, and you can discard any commits you make in this
hugging-face-dlsa-szdd5-git-clone-3089467112: state without impacting any branches by switching back to a branch.
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112: If you want to create a new branch to retain commits you create, you may
hugging-face-dlsa-szdd5-git-clone-3089467112: do so (now or later) by using -c with the switch command. Example:
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112:   git switch -c <new-branch-name>
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112: Or undo this operation with:
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112:   git switch -
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112: Turn off this advice by setting config variable advice.detachedHead to false
hugging-face-dlsa-szdd5-git-clone-3089467112: 
hugging-face-dlsa-szdd5-git-clone-3089467112: time="2022-11-15T21:27:08.834Z" level=info msg="sub-process exited" argo=true error="<nil>"
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: time="2022-11-15T21:27:17.578Z" level=info msg="capturing logs" argo=true
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Running 1 tasks on 1 nodes with ppn=1
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: /opt/conda/bin/python
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: /usr/bin/gcc
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: /usr/bin/mpicc
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: /usr/bin/mpiexec.hydra
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: #### INITIAL ENV ####
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using CCL_WORKER_COUNT=0
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using I_MPI_PIN_DOMAIN=[0xFFFFFFFFFF]
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using KMP_BLOCKTIME=1
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using KMP_HW_SUBSET=1T
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using OMP_NUM_THREADS=40
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so:/opt/conda/lib/libiomp5.so:
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using PYTORCH_MPI_THREAD_AFFINITY=0
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Using DATALOADER_WORKER_COUNT=0
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: #### INITIAL ENV ####
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: PyTorch version: 1.11.0+cpu
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: MASTER_ADDR=hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Running mpiexec.hydra -np 1 -ppn 1 -l -genv I_MPI_PIN_DOMAIN=[0xFFFFFFFFFF] -genv CCL_WORKER_AFFINITY= -genv CCL_WORKER_COUNT=0 -genv OMP_NUM_THREADS=40 fine-tuning/run_ipex_native.sh
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: Start Time:  Tue Nov 15 21:27:18 UTC 2022
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] PyTorch: setting up devices
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] https://huggingface.co/bert-large-uncased/resolve/main/tokenizer_config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpqiivor12
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] 
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] **********************************************************************
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] 'Load Data' took 0.071s (71,258,769ns)
hugging-face-dlsa-szdd5-hugging-face-dlsa-training-2426560540: [0] **********************************************************************
```
...
## Cleanup
Remove containers, copied files, and special configurations
```
make clean
```
Remove Workflow Template
```
make argo-clean
```
Remove Helm Chart
```
make helm-clean
```