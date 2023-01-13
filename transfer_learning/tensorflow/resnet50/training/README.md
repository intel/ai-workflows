# Tensorflow ResNet50 TRAINING - Vision Transfer Learning
## Description
This document contains instructions on how to run Vision Transfer Learning e2e pipeline with make and docker compose.
## Project Structure 
```
├── transfer-learning-training @ v1.0.1
├── DEVCATALOG.md
├── Dockerfile.vision-transfer-learning
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
BATCH_SIZE ?= 32
DATASET_DIR ?= /workspace/data
FINAL_IMAGE_NAME ?= vision-transfer-learning
NAMESPACE ?= argo
NUM_EPOCHS ?= 100
OUTPUT_DIR ?= /output
PLATFORM ?= None
PRECISION ?= FP32
SCRIPT ?= colorectal

vision-transfer-learning:
	@BATCH_SIZE=${BATCH_SIZE} \
	 DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 NUM_EPOCHS=${NUM_EPOCHS} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 PLATFORM=${PLATFORM} \
	 PRECISION=${PRECISION} \
	 SCRIPT=${SCRIPT} \
	 docker compose up vision-transfer-learning --build

argo-single-node: 
	helm install \
	--namespace ${NAMESPACE} \
	--set proxy=${http_proxy} \
	--set workflow.batch_size=${BATCH_SIZE} \
	--set workflow.num_epochs=${NUM_EPOCHS} \
	--set workflow.platform=${PLATFORM} \
	--set workflow.precision=${PRECISION} \
	--set workflow.dataset_dir=${DATASET_DIR} \
	--set workflow.script=${SCRIPT} \
	${FINAL_IMAGE_NAME} ./chart
	argo submit --from wftmpl/${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}

workflow-log:
	argo logs @latest -f -c output-log

clean: 
	docker compose down

helm-clean: 
	kubectl delete wftmpl ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
	helm uninstall ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
```
[_docker-compose.yml_](docker-compose.yml)
```
services:
  vision-transfer-learning:
    build:
      args: 
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: Dockerfile.vision-transfer-learning
    command: conda run --no-capture-output -n transfer_learning ./${SCRIPT}.sh
    environment: 
      - BATCH_SIZE=${BATCH_SIZE}
      - DATASET_DIR=/workspace/data
      - OUTPUT_DIR=${OUTPUT_DIR}/${SCRIPT}
      - NUM_EPOCHS=${NUM_EPOCHS}
      - PLATFORM=${PLATFORM}
      - PRECISION=${PRECISION}
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    privileged: true
    volumes: 
      - /${DATASET_DIR}:/workspace/data
      - ${OUTPUT_DIR}:${OUTPUT_DIR}
      - ./transfer-learning-training:/workspace/transfer-learning
    working_dir: /workspace/transfer-learning
```
[_workflowTemplate.yml_](./chart/templates/workflowTemplate.yaml)
```
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: {{ .Values.metadata.name }}
  labels:
  {{- include "demo.labels" . | nindent 4 }}
spec:
  arguments:
    parameters:
    - name: ref
      value: {{ .Values.workflow.ref }}
    - name: repo
      value: {{ .Values.workflow.repo }}
    - name: dataset-dir
      value: {{ .Values.workflow.dataset_dir }}
    - enum:
      - SPR
      - None
      name: platform
      value: {{ .Values.workflow.platform }}
    - enum:
      - FP32
      - bf16
      name: precision
      value: {{ .Values.workflow.precision }}
    - enum:
      - colorectal
      - resisc
      - sports
      name: script
      value: {{ .Values.workflow.script }}
    - name: http_proxy
      value: {{ .Values.proxy }}
    - name: num-epochs
      value: {{ .Values.workflow.num_epochs }}
    - name: batch-size
      value: {{ .Values.workflow.batch_size }}
  entrypoint: main
  templates:
  - steps:
    - - name: git-clone
        template: git-clone
    - - name: vision-transfer-learning-training
        template: vision-transfer-learning-training
    - - name: vision-transfer-learning-inference
        template: vision-transfer-learning-inference
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
      image: intel/ai-workflows:vision-transfer-learning-training
      volumeMounts:
      - mountPath: /workspace
        name: workspace
      workingDir: /
    name: git-clone
  - container:
      args:
      - run
      - --no-capture-output
      - -n
      - transfer_learning
      - '{{"./{{workflow.parameters.script}}.sh"}}'
      command:
      - conda
      env:
      - name: DATASET_DIR
        value: /dataset
      - name: PRECISION
        value: '{{"{{workflow.parameters.precision}}"}}'
      - name: PLATFORM
        value: '{{"{{workflow.parameters.platform}}"}}'
      - name: OUTPUT_DIR
        value: /output
      - name: NUM_EPOCHS
        value: '{{"{{workflow.parameters.num-epochs}}"}}'
      - name: BATCH_SIZE
        value: '{{"{{workflow.parameters.batch-size}}"}}'
      - name: http_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      - name: https_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      image: intel/ai-workflows:vision-transfer-learning-training
      volumeMounts:
      - mountPath: /workspace
        name: workspace
      - mountPath: /output
        name: output-dir
      {{- if eq .Values.dataset.type "nfs" }}
      - mountPath: /dataset
        name: dataset-dir
      {{ end }}
      workingDir: /workspace
    {{- if eq .Values.dataset.type "s3" }}
    inputs: 
      artifacts: 
      - name: dataset
        path: /dataset
        s3:
          key: {{ .Values.dataset.s3.datasetKey }}
    {{ end }}
    name: vision-transfer-learning-training
    outputs:
      artifacts: 
      - name: checkpoint
        path: /output
        s3:
          key: {{ .Values.dataset.logsKey }}
    sidecars:
    - args:
      - -c
      - while ! tail -f /output/result.txt ; do sleep 5 ; done
      command:
      - sh
      container: null
      image: {{ .Values.sidecars.image }}
      mirrorVolumeMounts: true
      name: output-log
      workingDir: /output
    {{- if eq .Values.dataset.type "nfs" }}
    volumes:
    - name: dataset-dir
      nfs:
        server: {{ .Values.dataset.nfs.server }}
        path: {{ .Values.dataset.nfs.path }}
        readOnly: {{ .Values.dataset.nfs.readOnly }}
    {{ end }}
  - container:
      args:
      - run
      - --no-capture-output
      - -n
      - transfer_learning
      - '{{"./{{workflow.parameters.script}}.sh"}}'
      - --inference
      - -cp
      - /output
      command:
      - conda
      env:
      - name: DATASET_DIR
        value: /dataset
      - name: PRECISION
        value: '{{"{{workflow.parameters.precision}}"}}'
      - name: PLATFORM
        value: '{{"{{workflow.parameters.platform}}"}}'
      - name: OUTPUT_DIR
        value: /output/inference
      - name: http_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      - name: https_proxy
        value: '{{"{{workflow.parameters.http_proxy}}"}}'
      image: intel/ai-workflows:vision-transfer-learning-inference
      volumeMounts:
      - mountPath: /workspace
        name: workspace
      - mountPath: /output
        name: output-dir
      {{- if eq .Values.dataset.type "nfs" }}
      - mountPath: /dataset
        name: dataset-dir
      {{ end }}
      workingDir: /workspace
    {{- if eq .Values.dataset.type "s3" }}
    inputs: 
      artifacts: 
      - name: dataset
        path: /dataset
        s3: 
          key: {{ .Values.dataset.s3.datasetKey }}
    {{ end }}
    name: vision-transfer-learning-inference
    outputs: 
      artifacts: 
      - name: logs
        path: /output/inference
        s3:
          key: {{ .Values.dataset.logsKey }}
    sidecars:
    - args:
      - -c
      - while ! tail -f /output/result.txt ; do sleep 5 ; done
      command:
      - sh
      container: null
      image: {{ .Values.sidecars.image }}
      mirrorVolumeMounts: true
      name: output-log
      workingDir: /output
    {{- if eq .Values.dataset.type "nfs" }}
    volumes:
    - name: dataset-dir
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

# Vision Transfer Learning
End2End AI Workflow for transfer learning based image classification using ResNet50.

## Quick Start
* Pull and configure the dependent repo submodule `git submodule update --init --recursive`.

* Install [Pipeline Repository Dependencies](https://github.com/intel/ai-workflows#dependency-requirements)

* Other variables:

| Variable Name | Default | Notes |
| --- | --- | --- |
| BATCH_SIZE | `32` | Number of samples to process |
| DATASET_DIR | `/data` | Dataset directory, optional for `SCRIPT=colorectal`, default is placeholder |
| FINAL_IMAGE_NAME | `vision-transfer-learning` | Final Docker image name OR name of helm installation |
| NAMESPACE | `default` | For running with k8s |
| NUM_EPOCHS | `100` | Number of passes through training dataset |
| OUTPUT_DIR | `/output` | Output directory |
| PLATFORM | `None` | `SPR` and `None` are supported, Hyperthreaded SPR systems are not currently working |
| PRECISION | `FP32` | `bf16` and `FP32` are supported |
| SCRIPT | `colorectal` | `sports`, `resisc`, and `colorectal` are supported scripts that use different datasets/checkpoints |

## Build and Run
Build and Run with defaults:
```
make vision-transfer-learning
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
$ DATASET_DIR=/localdisk/aia_mlops_dataset/i5-transfer-learning/sports SCRIPT=sports make vision-transfer-learning
[+] Building 0.1s (9/9) FINISHED                                                                                                                                                                          
 => [internal] load build definition from Dockerfile.vision-transfer-learning                                                                                                                        0.0s
 => => transferring dockerfile: 57B                                                                                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                                                    0.0s
 => => transferring context: 2B                                                                                                                                                                      0.0s
 => [internal] load metadata for docker.io/library/ubuntu:20.04                                                                                                                                      0.0s
 => [1/5] FROM docker.io/library/ubuntu:20.04                                                                                                                                                        0.0s
 => CACHED [2/5] RUN apt-get update && apt-get install --no-install-recommends --fix-missing -y     build-essential     ca-certificates     git     gcc     numactl     wget                         0.0s
 => CACHED [3/5] RUN apt-get update &&     wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh &&     bash miniconda.sh -b -p /opt/conda &&     rm m  0.0s
 => CACHED [4/5] RUN conda create -y -n transfer_learning python=3.8 &&     source activate transfer_learning &&     conda install -y -c conda-forge gperftools &&     conda install -y intel-openm  0.0s
 => CACHED [5/5] RUN mkdir -p /workspace/transfer-learning                                                                                                                                           0.0s
 => exporting to image                                                                                                                                                                               0.0s
 => => exporting layers                                                                                                                                                                              0.0s
 => => writing image sha256:20fc21d79272d6af76735b20eb456bcf1a19019e8541e658292d3be60cb5b80f                                                                                                         0.0s
 => => naming to docker.io/library/vision-transfer-learning:training-ww23-2022-ubuntu-20.04                                                                                                          0.0s
WARN[0000] Found orphan containers ([hadoop-main]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
[+] Running 1/1
 ⠿ Container training-vision-transfer-learning-1  Recreated                                                                                                                                          0.1s
Attaching to training-vision-transfer-learning-1
training-vision-transfer-learning-1  | /usr/bin/bash: /opt/conda/envs/transfer_learning/lib/libtinfo.so.6: no version information available (required by /usr/bin/bash)
training-vision-transfer-learning-1  | INFERENCE Default value is zero
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.checkpoint_management has been moved to tensorflow.python.checkpoint.checkpoint_management. The old module will be deleted in version 2.9.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.resource has been moved to tensorflow.python.trackable.resource. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.util has been moved to tensorflow.python.checkpoint.checkpoint. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base_delegate has been moved to tensorflow.python.trackable.base_delegate. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.graph_view has been moved to tensorflow.python.checkpoint.graph_view. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.python_state has been moved to tensorflow.python.trackable.python_state. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.saving.functional_saver has been moved to tensorflow.python.checkpoint.functional_saver. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.saving.checkpoint_options has been moved to tensorflow.python.checkpoint.checkpoint_options. The old module will be deleted in version 2.11.
training-vision-transfer-learning-1  | 2022-08-30 16:18:31.273775: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
training-vision-transfer-learning-1 exited with code 0
```
With Helm
```
$ NUM_EPOCHS=2 NAMESPACE=argo make argo-single-node
helm install \
--namespace argo \
--set proxy=... \
--set wf.batch_size=32 \
--set wf.num_epochs=2 \
--set wf.platform=None \
--set wf.precision=FP32 \
--set wf.script=colorectal \
vision-transfer-learning ./chart
NAME: vision-transfer-learning
LAST DEPLOYED: Tue Nov  8 10:38:37 2022
NAMESPACE: argo
STATUS: deployed
REVISION: 1
TEST SUITE: None
argo submit --from wftmpl/vision-transfer-learning --namespace=argo
Name:                vision-transfer-learning-7jh2d
Namespace:           argo
ServiceAccount:      unset
Status:              Pending
Created:             Tue Nov 08 10:38:37 -0800 (now)
Progress:
```
## Check Results
With Docker Compose
```
$ tail -f /output/sports/result.txt 
Dataset directory is  /workspace/data
Setting Output Directory
Is Tf32 enabled ? :  False
Found 3752 files belonging to 2 classes.
Found 804 files belonging to 2 classes.
Found 805 files belonging to 2 classes.
Total classes =  2
Epoch 1/200
118/118 [==============================] - 62s 500ms/step - loss: 0.2118 - acc: 0.9166 - val_loss: 0.0951 - val_acc: 0.9664 - lr: 0.0010
Epoch 2/200
118/118 [==============================] - 49s 410ms/step - loss: 0.0696 - acc: 0.9827 - val_loss: 0.0608 - val_acc: 0.9813 - lr: 0.0010
Epoch 3/200
118/118 [==============================] - 48s 401ms/step - loss: 0.0461 - acc: 0.9901 - val_loss: 0.0484 - val_acc: 0.9863 - lr: 0.0010
Epoch 4/200
118/118 [==============================] - 42s 355ms/step - loss: 0.0353 - acc: 0.9917 - 
```
...
```
118/118 [==============================] - 46s 384ms/step - loss: 0.0011 - acc: 1.0000 - val_loss: 0.0182 - val_acc: 0.9925 - lr: 2.0000e-04
Epoch 37/200
118/118 [==============================] - 51s 428ms/step - loss: 0.0011 - acc: 1.0000 - val_loss: 0.0202 - val_acc: 0.9913 - lr: 2.0000e-04
Epoch 38/200
118/118 [==============================] - ETA: 0s - loss: 0.0011 - acc: 1.0000     
Epoch 38: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
118/118 [==============================] - 57s 483ms/step - loss: 0.0011 - acc: 1.0000 - val_loss: 0.0184 - val_acc: 0.9925 - lr: 2.0000e-04
Total elapsed time =  2522.6976577951573
Maximum validation accuracy =  0.9950248599052429
26/26 [==============================] - 10s 350ms/step - loss: 0.0255 - acc: 0.9901
Test accuracy : 0.9900621175765991
26/26 [==============================] - 12s 424ms/step
Classification report
              precision    recall  f1-score   support

           0       1.00      0.99      0.99       450
           1       0.98      0.99      0.99       355

    accuracy                           0.99       805
   macro avg       0.99      0.99      0.99       805
weighted avg       0.99      0.99      0.99       805
```
With Argo, note that this sidecar retries `tail` until it finds a log file.
```
$ make workflow-log
argo logs @latest -f -c output-log
ERRO[2022-11-08T10:40:08.074Z] container output-log is not valid for pod vision-transfer-learning-7jh2d-git-clone-245377324  namespace=argo podName=vision-transfer-learning-7jh2d-git-clone-245377324 workflow=vision-transfer-learning-7jh2d
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: time="2022-11-08T18:39:16.500Z" level=info msg="capturing logs" argo=true
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: tail: cannot open '/output/result.txt' for reading: No such file or directory
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: tail: no files remaining
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Is Tf32 enabled ? :  False
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Dataset directory is  datasets/
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Setting Output log Directory
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Downloading and preparing dataset 246.14 MiB (download: 246.14 MiB, generated: Unknown size, total: 246.14 MiB) to datasets/colorectal_histology/2.0.0...
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Dataset colorectal_histology downloaded and prepared to datasets/colorectal_histology/2.0.0. Subsequent calls will reuse this data.
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Since test directory files are not present so using validation files as test files
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Total classes =  8
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Normalizing
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Epoch 1/2
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: 125/125 - 351s - loss: 0.6346 - acc: 0.7840 - val_loss: 0.3608 - val_acc: 0.8810 - lr: 0.0010 - 351s/epoch - 3s/step
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Epoch 2/2
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: 125/125 - 142s - loss: 0.3711 - acc: 0.8740 - val_loss: 0.3100 - val_acc: 0.8950 - lr: 0.0010 - 142s/epoch - 1s/step
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Total elapsed Training time =  493.43984486255795
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Maximum validation accuracy =  0.8949999809265137
32/32 [==============================] - 29s 878ms/step - loss: 0.3100 - acc: 0.8950
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Accuracy of model on test dataset: 0.8949999809265137
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Total elapsed Test time =  28.700972772203386
32/32 [==============================] - 26s 809ms/step-training-2847134387: 
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Classification report
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:               precision    recall  f1-score   support
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: 
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            0       0.95      0.95      0.95       112
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            1       0.90      0.62      0.73       127
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            2       0.75      0.85      0.79       137
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            3       0.93      0.89      0.91       126
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            4       0.77      0.97      0.86       126
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            5       0.97      0.94      0.95       128
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            6       1.00      0.97      0.98       118
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:            7       0.98      1.00      0.99       126
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: 
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:     accuracy                           0.90      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387:    macro avg       0.91      0.90      0.90      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: weighted avg       0.90      0.90      0.89      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: 
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: time="2022-11-08T18:50:28.533Z" level=info msg="sub-process exited" argo=true error="<nil>"
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: time="2022-11-08T18:50:28.533Z" level=info msg="not saving outputs - not main container" argo=true
vision-transfer-learning-7jh2d-vision-transfer-learning-training-2847134387: Error: exit status 143
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: time="2022-11-08T18:50:40.581Z" level=info msg="capturing logs" argo=true
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:            3       0.93      0.89      0.91       126
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:            4       0.77      0.97      0.86       126
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:            5       0.97      0.94      0.95       128
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:            6       1.00      0.97      0.98       118
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:            7       0.98      1.00      0.99       126
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: 
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:     accuracy                           0.90      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117:    macro avg       0.91      0.90      0.90      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: weighted avg       0.90      0.90      0.89      1000
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: 
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: time="2022-11-08T18:51:49.606Z" level=info msg="sub-process exited" argo=true error="<nil>"
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: time="2022-11-08T18:51:49.606Z" level=info msg="not saving outputs - not main container" argo=true
vision-transfer-learning-7jh2d-vision-transfer-learning-inference-667320117: Error: exit status 143
```

## Cleanup
Remove containers, copied files, and special configurations
```
make clean
```
Remove Helm Chart
```
make helm-clean
```
