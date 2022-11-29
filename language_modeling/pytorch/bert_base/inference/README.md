# **PyTorch BERT Base INFERENCE - NLP Azure**

## **Description**

This pipeline provides instructions on how to run inference using BERT Base model on infrastructure provided by Azure Machine Learning with make and docker compose.

## **Project Structure**
```
├── azureml @ v1.0.1
├── Makefile
├── README.md
└── docker-compose.yml
```
[*Makefile*](Makefile)
```
AZURE_CONFIG_FILE ?= $$(pwd)/config.json
FINAL_IMAGE_NAME ?= 
FP32_TRAINED_MODEL ?= $$(pwd)/../training/azureml/notebooks/fp32_model_output

nlp-azure:
	mkdir -p ./azureml/notebooks/fp32_model_output && cp -r ${FP32_TRAINED_MODEL} ./azureml/notebooks/
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	AZURE_CONFIG_FILE=${AZURE_CONFIG_FILE} \
	docker compose up nlp-azure --build

clean:
	docker compose down
	rm -rf ./azureml/notebooks/fp32_model_output
```

[*docker-compose.yml*](docker-compose.yml)
```
services:
  nlp-azure:
    build:
      args:
        http_proxy: ${http_proxy}
        https_proxy: ${https_proxy}
        no_proxy: ${no_proxy}
      dockerfile: ./azureml/Dockerfile
    command: sh -c "jupyter nbconvert --to python 1.0-intel-azureml-inference.ipynb && python3 1.0-intel-azureml-inference.py"
    environment:
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:inference-ubuntu-20.04
    network_mode: "host"
    privileged: true
    volumes:
      - ./azureml/notebooks:/root/notebooks
      - ./azureml/src:/root/src
      - /${AZURE_CONFIG_FILE}:/root/notebooks/config.json
    working_dir: /root/notebooks
```

# **Azure Machine Learning**

End-to-End AI workflow using the Azure ML Cloud Infrastructure for executing inference using the BERT Base model. More Information [here](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources). The pipeline runs the `1.0-intel-azureml-inference.ipynb` of the [Azure ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML/blob/d96750561d88dc04ff20fc900eed44abbda94f0a/notebooks/1.0-intel-azureml-inference.ipynb) project. 

## **Quick Start**

* Make sure that the enviroment setup pre-requisites are satisfied per the document [here](../../../../README.md).

* Pull and configure the dependent repo submodule ```git submodule update --init --recursive ```.

* Install [Pipeline Repository Dependencies](../../../../README.md).

* Use the quickstart [link](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to setup your Azure ML resources. 

  * If required, create virtual networks and NAT gateway by following this [link](https://learn.microsoft.com/en-us/azure/virtual-network/nat-gateway/quickstart-create-nat-gateway-portal).

* Download the `config.json` file from your Azure ML Studio Workspace. 

* This pipeline requires the pre-trained FP32 model. Please run the [training pipeline](../training/README.md) before running inference to get the model. 

* Other Variables:

Variable Name | Default | Notes |
:----------------:|:------------------: | :--------------------------------------:|
AZURE_CONFIG_FILE | `$$(pwd)/config.json` | Azure Workspace Configuration file |
FINAL_IMAGE_NAME  | nlp-azure | Final Docker Image Name | 
FP32_TRAINED_MODEL  | `$$(pwd)/../training/azureml/notebooks/fp32_model_output` | FP32 model obtained from Training | 

## **Build and Run**
Build and run with defaults:

```make nlp-azure```

## **Build and Run Example**
```
#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 32B done
#1 DONE 0.0s

#2 [internal] load .dockerignore
#2 transferring context: 2B done
#2 DONE 0.0s

#3 [internal] load metadata for docker.io/library/ubuntu:20.04
#3 DONE 0.7s

#4 [1/3] FROM docker.io/library/ubuntu:20.04@sha256:9c2004872a3a9fcec8cc757ad65c042de1dad4da27de4c70739a6e36402213e3
#4 DONE 0.0s

#5 [2/3] RUN apt-get update &&     apt-get install --no-install-recommends curl=7.68.0-1ubuntu2.13 -y &&     apt-get install --no-install-recommends python3-pip=20.0.2-5ubuntu1.6 -y &&     rm -r /var/lib/apt/lists/*
#5 CACHED

#6 [3/3] RUN pip install --no-cache-dir azureml-sdk==1.45.0 && pip install --no-cache-dir notebook==6.4.12
#6 CACHED

#7 exporting to image
#7 exporting layers done
#7 writing image sha256:b4b0d17ff3f251644447a83a133d0d41a7f42129b05739ba4d843ecced862eeb done
#7 naming to docker.io/library/nlp-azure:inference-ubuntu-20.04 done
#7 DONE 0.0s
Attaching to inference-nlp-azure-1
inference-nlp-azure-1  | [NbConvertApp] Converting notebook 1.0-intel-azureml-inference.ipynb to python
inference-nlp-azure-1  | [NbConvertApp] Writing 9806 bytes to 1.0-intel-azureml-inference.py
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint hyperdrive = azureml.train.hyperdrive:HyperDriveRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint automl = azureml.train.automl.run:AutoMLRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.PipelineRun = azureml.pipeline.core.run:PipelineRun._from_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.ReusedStepRun = azureml.pipeline.core.run:StepRun._from_reused_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.StepRun = azureml.pipeline.core.run:StepRun._from_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.scriptrun = azureml.core.script_run:ScriptRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'pyopenssl', 'PyOpenSSL'}).
inference-nlp-azure-1  | Loaded existing workspace configuration
inference-nlp-azure-1  | Validating arguments.
inference-nlp-azure-1  | Arguments validated.
inference-nlp-azure-1  | Uploading file to /inc/ptq_config
inference-nlp-azure-1  | Uploading an estimated of 1 files
inference-nlp-azure-1  | Uploading ../src/inference_container/config/ptq.yaml
inference-nlp-azure-1  | Uploaded ../src/inference_container/config/ptq.yaml, 1 files out of an estimated total of 1
inference-nlp-azure-1  | Uploaded 1 files
inference-nlp-azure-1  | Creating new dataset
inference-nlp-azure-1  | Validating arguments.
inference-nlp-azure-1  | Arguments validated.
inference-nlp-azure-1  | Uploading file to /trained_fp32_hf_model
inference-nlp-azure-1  | Uploading an estimated of 11 files
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/training_args.bin
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/training_args.bin, 1 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/config.json
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/config.json, 2 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/training_args.bin
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/training_args.bin, 3 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/trainer_state.json
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/trainer_state.json, 4 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/scheduler.pt
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/scheduler.pt, 5 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/config.json
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/config.json, 6 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/rng_state_0.pth
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/rng_state_0.pth, 7 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/rng_state_1.pth
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/rng_state_1.pth, 8 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/pytorch_model.bin
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/pytorch_model.bin, 9 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/pytorch_model.bin
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/pytorch_model.bin, 10 files out of an estimated total of 11
inference-nlp-azure-1  | Uploading ./fp32_model_output/outputs/trained_model/checkpoint-500/optimizer.pt
inference-nlp-azure-1  | Uploaded ./fp32_model_output/outputs/trained_model/checkpoint-500/optimizer.pt, 11 files out of an estimated total of 11
inference-nlp-azure-1  | Uploaded 11 files
inference-nlp-azure-1  | Creating new dataset
inference-nlp-azure-1  | Found existing cluster, use it.
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Running
inference-nlp-azure-1  | RunId: INC_PTQ_1666128985_788b95f3
inference-nlp-azure-1  | Web View: https://ml.azure.com/runs/INC_PTQ_1666128985_788b95f3?wsid=/subscriptions/0a5dbdd4-ee35-483f-b248-93e05a52cd9f/resourcegroups/intel_azureml_resource/workspaces/cloud_t7_i9&tid=46c98d88-e344-4ed4-8496-4ed7712e255d
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Streaming user_logs/std_log.txt
inference-nlp-azure-1  | ===============================
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading builder script:   0%|          | 0.00/7.78k [00:00<?, ?B/s]
inference-nlp-azure-1  | Downloading builder script: 28.8kB [00:00, 12.3MB/s]                   
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading metadata:   0%|          | 0.00/4.47k [00:00<?, ?B/s]
inference-nlp-azure-1  | Downloading metadata: 28.7kB [00:00, 14.7MB/s]                   
inference-nlp-azure-1  | Downloading and preparing dataset glue/mrpc (download: 1.43 MiB, generated: 1.43 MiB, post-processed: Unknown size, total: 2.85 MiB) to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data files:   0%|          | 0/3 [00:00<?, ?it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data: 0.00B [00:00, ?B/s][A
inference-nlp-azure-1  | Downloading data: 6.22kB [00:00, 4.12MB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data files:  33%|███▎      | 1/3 [00:00<00:00,  2.24it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data: 0.00B [00:00, ?B/s][A
inference-nlp-azure-1  | Downloading data: 1.05MB [00:00, 18.6MB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data files:  67%|██████▋   | 2/3 [00:00<00:00,  2.30it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data: 0.00B [00:00, ?B/s][A
inference-nlp-azure-1  | Downloading data: 441kB [00:00, 13.3MB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading data files: 100%|██████████| 3/3 [00:01<00:00,  2.50it/s]
inference-nlp-azure-1  | Downloading data files: 100%|██████████| 3/3 [00:01<00:00,  2.43it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Generating train split:   0%|          | 0/3668 [00:00<?, ? examples/s]
inference-nlp-azure-1  | Generating train split:  38%|███▊      | 1390/3668 [00:00<00:00, 13893.19 examples/s]
inference-nlp-azure-1  | Generating train split:  78%|███████▊  | 2867/3668 [00:00<00:00, 14406.34 examples/s]
inference-nlp-azure-1  |                                                                                      
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Generating validation split:   0%|          | 0/408 [00:00<?, ? examples/s]
inference-nlp-azure-1  | Generating validation split:  96%|█████████▌| 391/408 [00:00<00:00, 3869.76 examples/s]
inference-nlp-azure-1  |                                                                                        
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Generating test split:   0%|          | 0/1725 [00:00<?, ? examples/s]
inference-nlp-azure-1  |                                                                       
inference-nlp-azure-1  | Dataset glue downloaded and prepared to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/2 [00:00<?, ?it/s]
inference-nlp-azure-1  | 100%|██████████| 2/2 [00:00<00:00, 708.80it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/4 [00:00<?, ?ba/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading tokenizer_config.json:   0%|          | 0.00/28.0 [00:00<?, ?B/s][A
inference-nlp-azure-1  | Downloading tokenizer_config.json: 100%|██████████| 28.0/28.0 [00:00<00:00, 23.8kB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading config.json:   0%|          | 0.00/570 [00:00<?, ?B/s][A
inference-nlp-azure-1  | Downloading config.json: 100%|██████████| 570/570 [00:00<00:00, 457kB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading vocab.txt:   0%|          | 0.00/226k [00:00<?, ?B/s][A
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading vocab.txt:  12%|█▏        | 28.0k/226k [00:00<00:01, 195kB/s][A
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading vocab.txt:  69%|██████▉   | 157k/226k [00:00<00:00, 605kB/s] [A
inference-nlp-azure-1  | Downloading vocab.txt: 100%|██████████| 226k/226k [00:00<00:00, 775kB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading tokenizer.json:   0%|          | 0.00/455k [00:00<?, ?B/s][A
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading tokenizer.json:   9%|▉         | 40.0k/455k [00:00<00:01, 277kB/s][A
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading tokenizer.json:  24%|██▎       | 108k/455k [00:00<00:00, 391kB/s] [A
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading tokenizer.json:  91%|█████████ | 412k/455k [00:00<00:00, 1.17MB/s][A
inference-nlp-azure-1  | Downloading tokenizer.json: 100%|██████████| 455k/455k [00:00<00:00, 1.04MB/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  |  25%|██▌       | 1/4 [00:05<00:15,  5.31s/ba]
inference-nlp-azure-1  |  50%|█████     | 2/4 [00:08<00:07,  3.99s/ba]
inference-nlp-azure-1  |  75%|███████▌  | 3/4 [00:11<00:03,  3.54s/ba]
inference-nlp-azure-1  | 100%|██████████| 4/4 [00:14<00:00,  3.31s/ba]
inference-nlp-azure-1  | 100%|██████████| 4/4 [00:14<00:00,  3.58s/ba]
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/2 [00:00<?, ?ba/s]
inference-nlp-azure-1  |  50%|█████     | 1/2 [00:03<00:03,  3.01s/ba]
inference-nlp-azure-1  | 100%|██████████| 2/2 [00:05<00:00,  2.98s/ba]
inference-nlp-azure-1  | 100%|██████████| 2/2 [00:05<00:00,  2.98s/ba]
inference-nlp-azure-1  | 2022-10-18 21:39:50 [INFO] Created a worker pool for first use
inference-nlp-azure-1  | 2022-10-18 21:39:50 [WARNING] Reusing dataset glue (/root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/2 [00:00<?, ?it/s]
inference-nlp-azure-1  | 100%|██████████| 2/2 [00:00<00:00, 662.66it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/4 [00:00<?, ?ba/s]loading configuration file https://huggingface.co/bert-base-uncased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/3c61d016573b14f7f008c02c4e51a366c67ab274726fe2910691e2a761acf43e.37395cee442ab11005bcd270f3c34464dc1704b715b5d7d52b1a461abe3b9e4e
inference-nlp-azure-1  | Model config BertConfig {
inference-nlp-azure-1  |   "_name_or_path": "bert-base-uncased",
inference-nlp-azure-1  |   "architectures": [
inference-nlp-azure-1  |     "BertForMaskedLM"
inference-nlp-azure-1  |   ],
inference-nlp-azure-1  |   "attention_probs_dropout_prob": 0.1,
inference-nlp-azure-1  |   "classifier_dropout": null,
inference-nlp-azure-1  |   "gradient_checkpointing": false,
inference-nlp-azure-1  |   "hidden_act": "gelu",
inference-nlp-azure-1  |   "hidden_dropout_prob": 0.1,
inference-nlp-azure-1  |   "hidden_size": 768,
inference-nlp-azure-1  |   "initializer_range": 0.02,
inference-nlp-azure-1  |   "intermediate_size": 3072,
inference-nlp-azure-1  |   "layer_norm_eps": 1e-12,
inference-nlp-azure-1  |   "max_position_embeddings": 512,
inference-nlp-azure-1  |   "model_type": "bert",
inference-nlp-azure-1  |   "num_attention_heads": 12,
inference-nlp-azure-1  |   "num_hidden_layers": 12,
inference-nlp-azure-1  |   "pad_token_id": 0,
inference-nlp-azure-1  |   "position_embedding_type": "absolute",
inference-nlp-azure-1  |   "transformers_version": "4.21.1",
inference-nlp-azure-1  |   "type_vocab_size": 2,
inference-nlp-azure-1  |   "use_cache": true,
inference-nlp-azure-1  |   "vocab_size": 30522
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 100%|██████████| 4/4 [00:11<00:00,  2.93s/ba]
inference-nlp-azure-1  | 100%|██████████| 4/4 [00:11<00:00,  2.93s/ba]
inference-nlp-azure-1  | 2022-10-18 21:40:03 [INFO] Pass query framework capability elapsed time: 554.0 ms
inference-nlp-azure-1  | 2022-10-18 21:40:03 [INFO] Get FP32 model baseline.
inference-nlp-azure-1  | The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1. If sentence2, idx, sentence1 are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/intel_extension_for_pytorch/frontend.py:261: UserWarning: Conv BatchNorm folding failed during the optimize process.
inference-nlp-azure-1  |   warnings.warn("Conv BatchNorm folding failed during the optimize process.")
inference-nlp-azure-1  | ***** Running Evaluation *****
inference-nlp-azure-1  |   Num examples = 1725
inference-nlp-azure-1  |   Batch size = 8
inference-nlp-azure-1  | 
inference-nlp-azure-1  |   0%|          | 0/216 [00:00<?, ?it/s]
inference-nlp-azure-1  |   1%|          | 2/216 [00:00<01:35,  2.25it/s]
inference-nlp-azure-1  |   1%|▏         | 3/216 [00:01<02:16,  1.56it/s]
inference-nlp-azure-1  |   2%|▏         | 4/216 [00:02<02:30,  1.41it/s]
inference-nlp-azure-1  |   2%|▏         | 5/216 [00:03<02:39,  1.32it/s]
inference-nlp-azure-1  |   3%|▎         | 6/216 [00:04<02:44,  1.28it/s]
inference-nlp-azure-1  |   3%|▎         | 7/216 [00:05<02:45,  1.26it/s]
inference-nlp-azure-1  |   4%|▎         | 8/216 [00:05<02:45,  1.25it/s]
inference-nlp-azure-1  |   4%|▍         | 9/216 [00:06<02:44,  1.26it/s]
inference-nlp-azure-1  |   5%|▍         | 10/216 [00:07<02:44,  1.25it/s]
inference-nlp-azure-1  |   5%|▌         | 11/216 [00:08<02:44,  1.25it/s]
inference-nlp-azure-1  |   6%|▌         | 12/216 [00:09<02:47,  1.22it/s]
inference-nlp-azure-1  |   6%|▌         | 13/216 [00:10<02:46,  1.22it/s]
inference-nlp-azure-1  |   6%|▋         | 14/216 [00:10<02:45,  1.22it/s]
inference-nlp-azure-1  |   7%|▋         | 15/216 [00:11<02:43,  1.23it/s]
inference-nlp-azure-1  |   7%|▋         | 16/216 [00:12<02:43,  1.22it/s]
inference-nlp-azure-1  |   8%|▊         | 17/216 [00:13<02:42,  1.23it/s]
inference-nlp-azure-1  |   8%|▊         | 18/216 [00:14<02:40,  1.23it/s]
inference-nlp-azure-1  |   9%|▉         | 19/216 [00:14<02:41,  1.22it/s]
inference-nlp-azure-1  |   9%|▉         | 20/216 [00:15<02:39,  1.23it/s]
inference-nlp-azure-1  |  10%|▉         | 21/216 [00:16<02:37,  1.24it/s]
inference-nlp-azure-1  |  10%|█         | 22/216 [00:17<02:37,  1.23it/s]
inference-nlp-azure-1  |  11%|█         | 23/216 [00:18<02:37,  1.23it/s]
inference-nlp-azure-1  |  11%|█         | 24/216 [00:18<02:35,  1.23it/s]
inference-nlp-azure-1  |  12%|█▏        | 25/216 [00:19<02:35,  1.23it/s]
inference-nlp-azure-1  |  12%|█▏        | 26/216 [00:20<02:34,  1.23it/s]
inference-nlp-azure-1  |  12%|█▎        | 27/216 [00:21<02:34,  1.22it/s]
inference-nlp-azure-1  |  13%|█▎        | 28/216 [00:22<02:32,  1.23it/s]
inference-nlp-azure-1  |  13%|█▎        | 29/216 [00:23<02:31,  1.23it/s]
inference-nlp-azure-1  |  14%|█▍        | 30/216 [00:23<02:32,  1.22it/s]
inference-nlp-azure-1  |  14%|█▍        | 31/216 [00:24<02:37,  1.17it/s]
inference-nlp-azure-1  |  15%|█▍        | 32/216 [00:25<02:33,  1.20it/s]
inference-nlp-azure-1  |  15%|█▌        | 33/216 [00:26<02:30,  1.21it/s]
inference-nlp-azure-1  |  16%|█▌        | 34/216 [00:27<02:27,  1.23it/s]
inference-nlp-azure-1  |  16%|█▌        | 35/216 [00:27<02:26,  1.24it/s]
inference-nlp-azure-1  |  17%|█▋        | 36/216 [00:28<02:25,  1.24it/s]
inference-nlp-azure-1  |  17%|█▋        | 37/216 [00:29<02:24,  1.24it/s]
inference-nlp-azure-1  |  18%|█▊        | 38/216 [00:30<02:27,  1.21it/s]
inference-nlp-azure-1  |  18%|█▊        | 39/216 [00:31<02:24,  1.23it/s]
inference-nlp-azure-1  |  19%|█▊        | 40/216 [00:32<02:23,  1.23it/s]
inference-nlp-azure-1  |  19%|█▉        | 41/216 [00:32<02:22,  1.23it/s]
inference-nlp-azure-1  |  19%|█▉        | 42/216 [00:33<02:21,  1.23it/s]
inference-nlp-azure-1  |  20%|█▉        | 43/216 [00:34<02:19,  1.24it/s]
inference-nlp-azure-1  |  20%|██        | 44/216 [00:35<02:18,  1.24it/s]
inference-nlp-azure-1  |  21%|██        | 45/216 [00:36<02:17,  1.24it/s]
inference-nlp-azure-1  |  21%|██▏       | 46/216 [00:36<02:18,  1.23it/s]
inference-nlp-azure-1  |  22%|██▏       | 47/216 [00:37<02:20,  1.21it/s]
inference-nlp-azure-1  |  22%|██▏       | 48/216 [00:38<02:19,  1.21it/s]
inference-nlp-azure-1  |  23%|██▎       | 49/216 [00:39<02:19,  1.19it/s]
inference-nlp-azure-1  |  23%|██▎       | 50/216 [00:40<02:16,  1.21it/s]
inference-nlp-azure-1  |  24%|██▎       | 51/216 [00:41<02:16,  1.21it/s]
inference-nlp-azure-1  |  24%|██▍       | 52/216 [00:41<02:14,  1.22it/s]
inference-nlp-azure-1  |  25%|██▍       | 53/216 [00:42<02:12,  1.23it/s]
inference-nlp-azure-1  |  25%|██▌       | 54/216 [00:43<02:11,  1.23it/s]
inference-nlp-azure-1  |  25%|██▌       | 55/216 [00:44<02:10,  1.24it/s]
inference-nlp-azure-1  |  26%|██▌       | 56/216 [00:45<02:08,  1.24it/s]
inference-nlp-azure-1  |  26%|██▋       | 57/216 [00:45<02:09,  1.23it/s]
inference-nlp-azure-1  |  27%|██▋       | 58/216 [00:46<02:08,  1.23it/s]
inference-nlp-azure-1  |  27%|██▋       | 59/216 [00:47<02:06,  1.24it/s]
inference-nlp-azure-1  |  28%|██▊       | 60/216 [00:48<02:07,  1.22it/s]
inference-nlp-azure-1  |  28%|██▊       | 61/216 [00:49<02:06,  1.23it/s]
inference-nlp-azure-1  |  29%|██▊       | 62/216 [00:49<02:04,  1.24it/s]
inference-nlp-azure-1  |  29%|██▉       | 63/216 [00:50<02:04,  1.23it/s]
inference-nlp-azure-1  |  30%|██▉       | 64/216 [00:51<02:03,  1.23it/s]
inference-nlp-azure-1  |  30%|███       | 65/216 [00:52<02:02,  1.23it/s]
inference-nlp-azure-1  |  31%|███       | 66/216 [00:53<02:00,  1.24it/s]
inference-nlp-azure-1  |  31%|███       | 67/216 [00:53<01:59,  1.25it/s]
inference-nlp-azure-1  |  31%|███▏      | 68/216 [00:54<02:02,  1.21it/s]
inference-nlp-azure-1  |  32%|███▏      | 69/216 [00:55<02:00,  1.22it/s]
inference-nlp-azure-1  |  32%|███▏      | 70/216 [00:56<01:59,  1.23it/s]
inference-nlp-azure-1  |  33%|███▎      | 71/216 [00:57<01:57,  1.24it/s]
inference-nlp-azure-1  |  33%|███▎      | 72/216 [00:58<01:56,  1.24it/s]
inference-nlp-azure-1  |  34%|███▍      | 73/216 [00:58<01:55,  1.23it/s]
inference-nlp-azure-1  |  34%|███▍      | 74/216 [00:59<01:54,  1.24it/s]
inference-nlp-azure-1  |  35%|███▍      | 75/216 [01:00<01:54,  1.24it/s]
inference-nlp-azure-1  |  35%|███▌      | 76/216 [01:01<01:54,  1.23it/s]
inference-nlp-azure-1  |  36%|███▌      | 77/216 [01:02<01:53,  1.22it/s]
inference-nlp-azure-1  |  36%|███▌      | 78/216 [01:02<01:52,  1.22it/s]
inference-nlp-azure-1  |  37%|███▋      | 79/216 [01:03<01:51,  1.23it/s]
inference-nlp-azure-1  |  37%|███▋      | 80/216 [01:04<01:50,  1.23it/s]
inference-nlp-azure-1  |  38%|███▊      | 81/216 [01:05<01:48,  1.24it/s]
inference-nlp-azure-1  |  38%|███▊      | 82/216 [01:06<01:49,  1.22it/s]
inference-nlp-azure-1  |  38%|███▊      | 83/216 [01:07<01:50,  1.20it/s]
inference-nlp-azure-1  |  39%|███▉      | 84/216 [01:07<01:48,  1.22it/s]
inference-nlp-azure-1  |  39%|███▉      | 85/216 [01:08<01:47,  1.22it/s]
inference-nlp-azure-1  |  40%|███▉      | 86/216 [01:09<01:51,  1.16it/s]
inference-nlp-azure-1  |  40%|████      | 87/216 [01:10<01:49,  1.18it/s]
inference-nlp-azure-1  |  41%|████      | 88/216 [01:11<01:46,  1.20it/s]
inference-nlp-azure-1  |  41%|████      | 89/216 [01:12<01:44,  1.21it/s]
inference-nlp-azure-1  |  42%|████▏     | 90/216 [01:12<01:43,  1.21it/s]
inference-nlp-azure-1  |  42%|████▏     | 91/216 [01:13<01:41,  1.23it/s]
inference-nlp-azure-1  |  43%|████▎     | 92/216 [01:14<01:41,  1.22it/s]
inference-nlp-azure-1  |  43%|████▎     | 93/216 [01:15<01:42,  1.20it/s]
inference-nlp-azure-1  |  44%|████▎     | 94/216 [01:16<01:40,  1.21it/s]
inference-nlp-azure-1  |  44%|████▍     | 95/216 [01:17<01:39,  1.22it/s]
inference-nlp-azure-1  |  44%|████▍     | 96/216 [01:17<01:38,  1.22it/s]
inference-nlp-azure-1  |  45%|████▍     | 97/216 [01:18<01:36,  1.23it/s]
inference-nlp-azure-1  |  45%|████▌     | 98/216 [01:19<01:35,  1.24it/s]
inference-nlp-azure-1  |  46%|████▌     | 99/216 [01:20<01:33,  1.25it/s]
inference-nlp-azure-1  |  46%|████▋     | 100/216 [01:21<01:32,  1.25it/s]
inference-nlp-azure-1  |  47%|████▋     | 101/216 [01:21<01:32,  1.24it/s]
inference-nlp-azure-1  |  47%|████▋     | 102/216 [01:22<01:32,  1.23it/s]
inference-nlp-azure-1  |  48%|████▊     | 103/216 [01:23<01:31,  1.23it/s]
inference-nlp-azure-1  |  48%|████▊     | 104/216 [01:24<01:35,  1.17it/s]
inference-nlp-azure-1  |  49%|████▊     | 105/216 [01:25<01:33,  1.19it/s]
inference-nlp-azure-1  |  49%|████▉     | 106/216 [01:26<01:31,  1.20it/s]
inference-nlp-azure-1  |  50%|████▉     | 107/216 [01:26<01:29,  1.22it/s]
inference-nlp-azure-1  |  50%|█████     | 108/216 [01:27<01:28,  1.21it/s]
inference-nlp-azure-1  |  50%|█████     | 109/216 [01:28<01:27,  1.23it/s]
inference-nlp-azure-1  |  51%|█████     | 110/216 [01:29<01:26,  1.23it/s]
inference-nlp-azure-1  |  51%|█████▏    | 111/216 [01:30<01:25,  1.23it/s]
inference-nlp-azure-1  |  52%|█████▏    | 112/216 [01:30<01:23,  1.24it/s]
inference-nlp-azure-1  |  52%|█████▏    | 113/216 [01:31<01:21,  1.26it/s]
inference-nlp-azure-1  |  53%|█████▎    | 114/216 [01:32<01:21,  1.25it/s]
inference-nlp-azure-1  |  53%|█████▎    | 115/216 [01:33<01:20,  1.26it/s]
inference-nlp-azure-1  |  54%|█████▎    | 116/216 [01:34<01:20,  1.25it/s]
inference-nlp-azure-1  |  54%|█████▍    | 117/216 [01:34<01:19,  1.25it/s]
inference-nlp-azure-1  |  55%|█████▍    | 118/216 [01:35<01:18,  1.25it/s]
inference-nlp-azure-1  |  55%|█████▌    | 119/216 [01:36<01:18,  1.23it/s]
inference-nlp-azure-1  |  56%|█████▌    | 120/216 [01:37<01:18,  1.23it/s]
inference-nlp-azure-1  |  56%|█████▌    | 121/216 [01:38<01:18,  1.21it/s]
inference-nlp-azure-1  |  56%|█████▋    | 122/216 [01:38<01:16,  1.22it/s]
inference-nlp-azure-1  |  57%|█████▋    | 123/216 [01:39<01:18,  1.19it/s]
inference-nlp-azure-1  |  57%|█████▋    | 124/216 [01:40<01:16,  1.21it/s]
inference-nlp-azure-1  |  58%|█████▊    | 125/216 [01:41<01:15,  1.21it/s]
inference-nlp-azure-1  |  58%|█████▊    | 126/216 [01:42<01:15,  1.20it/s]
inference-nlp-azure-1  |  59%|█████▉    | 127/216 [01:43<01:14,  1.19it/s]
inference-nlp-azure-1  |  59%|█████▉    | 128/216 [01:44<01:13,  1.19it/s]
inference-nlp-azure-1  |  60%|█████▉    | 129/216 [01:44<01:12,  1.21it/s]
inference-nlp-azure-1  |  60%|██████    | 130/216 [01:45<01:11,  1.20it/s]
inference-nlp-azure-1  |  61%|██████    | 131/216 [01:46<01:11,  1.19it/s]
inference-nlp-azure-1  |  61%|██████    | 132/216 [01:47<01:09,  1.22it/s]
inference-nlp-azure-1  |  62%|██████▏   | 133/216 [01:48<01:08,  1.21it/s]
inference-nlp-azure-1  |  62%|██████▏   | 134/216 [01:48<01:07,  1.22it/s]
inference-nlp-azure-1  |  62%|██████▎   | 135/216 [01:49<01:05,  1.24it/s]
inference-nlp-azure-1  |  63%|██████▎   | 136/216 [01:50<01:04,  1.24it/s]
inference-nlp-azure-1  |  63%|██████▎   | 137/216 [01:51<01:03,  1.25it/s]
inference-nlp-azure-1  |  64%|██████▍   | 138/216 [01:52<01:02,  1.25it/s]
inference-nlp-azure-1  |  64%|██████▍   | 139/216 [01:52<01:01,  1.25it/s]
inference-nlp-azure-1  |  65%|██████▍   | 140/216 [01:53<01:00,  1.26it/s]
inference-nlp-azure-1  |  65%|██████▌   | 141/216 [01:54<01:03,  1.19it/s]
inference-nlp-azure-1  |  66%|██████▌   | 142/216 [01:55<01:01,  1.20it/s]
inference-nlp-azure-1  |  66%|██████▌   | 143/216 [01:56<01:00,  1.20it/s]
inference-nlp-azure-1  |  67%|██████▋   | 144/216 [01:57<00:59,  1.21it/s]
inference-nlp-azure-1  |  67%|██████▋   | 145/216 [01:57<00:58,  1.21it/s]
inference-nlp-azure-1  |  68%|██████▊   | 146/216 [01:58<00:57,  1.21it/s]
inference-nlp-azure-1  |  68%|██████▊   | 147/216 [01:59<00:56,  1.22it/s]
inference-nlp-azure-1  |  69%|██████▊   | 148/216 [02:00<00:55,  1.22it/s]
inference-nlp-azure-1  |  69%|██████▉   | 149/216 [02:01<00:54,  1.22it/s]
inference-nlp-azure-1  |  69%|██████▉   | 150/216 [02:02<00:53,  1.23it/s]
inference-nlp-azure-1  |  70%|██████▉   | 151/216 [02:02<00:52,  1.23it/s]
inference-nlp-azure-1  |  70%|███████   | 152/216 [02:03<00:52,  1.23it/s]
inference-nlp-azure-1  |  71%|███████   | 153/216 [02:04<00:50,  1.24it/s]
inference-nlp-azure-1  |  71%|███████▏  | 154/216 [02:05<00:50,  1.24it/s]
inference-nlp-azure-1  |  72%|███████▏  | 155/216 [02:06<00:49,  1.22it/s]
inference-nlp-azure-1  |  72%|███████▏  | 156/216 [02:06<00:48,  1.23it/s]
inference-nlp-azure-1  |  73%|███████▎  | 157/216 [02:07<00:48,  1.22it/s]
inference-nlp-azure-1  |  73%|███████▎  | 158/216 [02:08<00:47,  1.22it/s]
inference-nlp-azure-1  |  74%|███████▎  | 159/216 [02:09<00:47,  1.20it/s]
inference-nlp-azure-1  |  74%|███████▍  | 160/216 [02:10<00:46,  1.20it/s]
inference-nlp-azure-1  |  75%|███████▍  | 161/216 [02:11<00:45,  1.22it/s]
inference-nlp-azure-1  |  75%|███████▌  | 162/216 [02:11<00:43,  1.23it/s]
inference-nlp-azure-1  |  75%|███████▌  | 163/216 [02:12<00:43,  1.22it/s]
inference-nlp-azure-1  |  76%|███████▌  | 164/216 [02:13<00:42,  1.21it/s]
inference-nlp-azure-1  |  76%|███████▋  | 165/216 [02:14<00:41,  1.22it/s]
inference-nlp-azure-1  |  77%|███████▋  | 166/216 [02:15<00:40,  1.22it/s]
inference-nlp-azure-1  |  77%|███████▋  | 167/216 [02:15<00:40,  1.22it/s]
inference-nlp-azure-1  |  78%|███████▊  | 168/216 [02:16<00:39,  1.21it/s]
inference-nlp-azure-1  |  78%|███████▊  | 169/216 [02:17<00:38,  1.21it/s]
inference-nlp-azure-1  |  79%|███████▊  | 170/216 [02:18<00:37,  1.22it/s]
inference-nlp-azure-1  |  79%|███████▉  | 171/216 [02:19<00:37,  1.20it/s]
inference-nlp-azure-1  |  80%|███████▉  | 172/216 [02:20<00:36,  1.21it/s]
inference-nlp-azure-1  |  80%|████████  | 173/216 [02:20<00:35,  1.21it/s]
inference-nlp-azure-1  |  81%|████████  | 174/216 [02:21<00:34,  1.23it/s]
inference-nlp-azure-1  |  81%|████████  | 175/216 [02:22<00:33,  1.24it/s]
inference-nlp-azure-1  |  81%|████████▏ | 176/216 [02:23<00:32,  1.24it/s]
inference-nlp-azure-1  |  82%|████████▏ | 177/216 [02:24<00:32,  1.19it/s]
inference-nlp-azure-1  |  82%|████████▏ | 178/216 [02:25<00:31,  1.20it/s]
inference-nlp-azure-1  |  83%|████████▎ | 179/216 [02:25<00:30,  1.22it/s]
inference-nlp-azure-1  |  83%|████████▎ | 180/216 [02:26<00:29,  1.24it/s]
inference-nlp-azure-1  |  84%|████████▍ | 181/216 [02:27<00:28,  1.24it/s]
inference-nlp-azure-1  |  84%|████████▍ | 182/216 [02:28<00:27,  1.24it/s]
inference-nlp-azure-1  |  85%|████████▍ | 183/216 [02:29<00:26,  1.24it/s]
inference-nlp-azure-1  |  85%|████████▌ | 184/216 [02:29<00:25,  1.24it/s]
inference-nlp-azure-1  |  86%|████████▌ | 185/216 [02:30<00:24,  1.25it/s]
inference-nlp-azure-1  |  86%|████████▌ | 186/216 [02:31<00:24,  1.24it/s]
inference-nlp-azure-1  |  87%|████████▋ | 187/216 [02:32<00:23,  1.25it/s]
inference-nlp-azure-1  |  87%|████████▋ | 188/216 [02:32<00:22,  1.25it/s]
inference-nlp-azure-1  |  88%|████████▊ | 189/216 [02:33<00:21,  1.24it/s]
inference-nlp-azure-1  |  88%|████████▊ | 190/216 [02:34<00:21,  1.24it/s]
inference-nlp-azure-1  |  88%|████████▊ | 191/216 [02:35<00:20,  1.23it/s]
inference-nlp-azure-1  |  89%|████████▉ | 192/216 [02:36<00:19,  1.24it/s]
inference-nlp-azure-1  |  89%|████████▉ | 193/216 [02:37<00:18,  1.23it/s]
inference-nlp-azure-1  |  90%|████████▉ | 194/216 [02:37<00:17,  1.24it/s]
inference-nlp-azure-1  |  90%|█████████ | 195/216 [02:38<00:16,  1.24it/s]
inference-nlp-azure-1  |  91%|█████████ | 196/216 [02:39<00:16,  1.21it/s]
inference-nlp-azure-1  |  91%|█████████ | 197/216 [02:40<00:15,  1.21it/s]
inference-nlp-azure-1  |  92%|█████████▏| 198/216 [02:41<00:14,  1.20it/s]
inference-nlp-azure-1  |  92%|█████████▏| 199/216 [02:42<00:13,  1.22it/s]
inference-nlp-azure-1  |  93%|█████████▎| 200/216 [02:42<00:13,  1.22it/s]
inference-nlp-azure-1  |  93%|█████████▎| 201/216 [02:43<00:12,  1.23it/s]
inference-nlp-azure-1  |  94%|█████████▎| 202/216 [02:44<00:11,  1.23it/s]
inference-nlp-azure-1  |  94%|█████████▍| 203/216 [02:45<00:10,  1.21it/s]
inference-nlp-azure-1  |  94%|█████████▍| 204/216 [02:46<00:09,  1.20it/s]
inference-nlp-azure-1  |  95%|█████████▍| 205/216 [02:46<00:09,  1.21it/s]
inference-nlp-azure-1  |  95%|█████████▌| 206/216 [02:47<00:08,  1.22it/s]
inference-nlp-azure-1  |  96%|█████████▌| 207/216 [02:48<00:07,  1.23it/s]
inference-nlp-azure-1  |  96%|█████████▋| 208/216 [02:49<00:06,  1.24it/s]
inference-nlp-azure-1  |  97%|█████████▋| 209/216 [02:50<00:05,  1.24it/s]
inference-nlp-azure-1  |  97%|█████████▋| 210/216 [02:50<00:04,  1.24it/s]
inference-nlp-azure-1  |  98%|█████████▊| 211/216 [02:51<00:03,  1.25it/s]
inference-nlp-azure-1  |  98%|█████████▊| 212/216 [02:52<00:03,  1.25it/s]
inference-nlp-azure-1  |  99%|█████████▊| 213/216 [02:53<00:02,  1.25it/s]
inference-nlp-azure-1  |  99%|█████████▉| 214/216 [02:54<00:01,  1.23it/s]
inference-nlp-azure-1  | 100%|█████████▉| 215/216 [02:55<00:00,  1.21it/s]
inference-nlp-azure-1  | 100%|██████████| 216/216 [02:55<00:00,  1.36it/s]
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Downloading builder script:   0%|          | 0.00/1.65k [00:00<?, ?B/s][A
inference-nlp-azure-1  | Downloading builder script: 4.21kB [00:00, 3.90MB/s]                   
inference-nlp-azure-1  | 
inference-nlp-azure-1  | 100%|██████████| 216/216 [02:55<00:00,  1.23it/s]
inference-nlp-azure-1  | 2022-10-18 21:43:00 [INFO] Save tuning history to /mnt/azureml/cr/j/e4712a572fab403692800d480981321b/exe/wd/nc_workspace/2022-10-18_21-39-23/./history.snapshot.
inference-nlp-azure-1  | 2022-10-18 21:43:00 [INFO] FP32 baseline is: [Accuracy: 0.8394, Duration (seconds): 177.0837]
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/torch/ao/quantization/qconfig.py:92: UserWarning: QConfigDynamic is going to be deprecated in PyTorch 1.12, please use QConfig instead
inference-nlp-azure-1  |   warnings.warn("QConfigDynamic is going to be deprecated in PyTorch 1.12, please use QConfig instead")
inference-nlp-azure-1  | 2022-10-18 21:43:00 [INFO] Fx trace of the entire model failed, We will conduct auto quantization
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/torch/ao/quantization/observer.py:176: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
inference-nlp-azure-1  |   warnings.warn(
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/torch/nn/quantized/_reference/modules/utils.py:25: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
inference-nlp-azure-1  |   torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/torch/nn/quantized/_reference/modules/utils.py:28: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
inference-nlp-azure-1  |   torch.tensor(weight_qparams["zero_point"], dtype=zero_point_dtype, device=device))
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |*********Mixed Precision Statistics********|
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] +---------------------+-------+------+------+
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |       Op Type       | Total | INT8 | FP32 |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] +---------------------+-------+------+------+
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |      Embedding      |   3   |  3   |  0   |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |      LayerNorm      |   25  |  0   |  25  |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] | quantize_per_tensor |   74  |  74  |  0   |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |        Linear       |   74  |  74  |  0   |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |      dequantize     |   74  |  74  |  0   |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |     input_tensor    |   24  |  24  |  0   |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] |       Dropout       |   24  |  0   |  24  |
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] +---------------------+-------+------+------+
inference-nlp-azure-1  | 2022-10-18 21:43:30 [INFO] Pass quantize model elapsed time: 30514.29 ms
inference-nlp-azure-1  | The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1. If sentence2, idx, sentence1 are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
inference-nlp-azure-1  | /opt/miniconda/lib/python3.8/site-packages/intel_extension_for_pytorch/frontend.py:261: UserWarning: Conv BatchNorm folding failed during the optimize process.
inference-nlp-azure-1  |   warnings.warn("Conv BatchNorm folding failed during the optimize process.")
```
...
```
inference-nlp-azure-1  | tokenizer config file saved in ./outputs/tokenizer_config.json
inference-nlp-azure-1  | Special tokens file saved in ./outputs/special_tokens_map.json
inference-nlp-azure-1  | Configuration saved in ./outputs/config.json
inference-nlp-azure-1  | Convertion complete!
inference-nlp-azure-1  | Cleaning up all outstanding Run operations, waiting 300.0 seconds
inference-nlp-azure-1  | 1 items cleaning up...
inference-nlp-azure-1  | Cleanup took 0.050061702728271484 seconds
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Execution Summary
inference-nlp-azure-1  | =================
inference-nlp-azure-1  | RunId: INC_PTQ_1666128985_788b95f3
inference-nlp-azure-1  | Web View: https://ml.azure.com/runs/INC_PTQ_1666128985_788b95f3?wsid=/subscriptions/0a5dbdd4-ee35-483f-b248-93e05a52cd9f/resourcegroups/intel_azureml_resource/workspaces/cloud_t7_i9&tid=46c98d88-e344-4ed4-8496-4ed7712e255d
inference-nlp-azure-1  | 
inference-nlp-azure-1  | Registering model inc_ptq_bert_model_mrpc
inference-nlp-azure-1  | Found existing cluster, use it.
inference-nlp-azure-1  | Service hf-aks-1
inference-nlp-azure-1  | Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog or local deployment: https://aka.ms/debugimage#debug-locally to debug if deployment takes longer than 10 minutes.
inference-nlp-azure-1  | Running
inference-nlp-azure-1  | 2022-10-19 02:44:14+00:00 Creating Container Registry if not exists.
inference-nlp-azure-1  | 2022-10-19 02:44:14+00:00 Registering the environment.
inference-nlp-azure-1  | 2022-10-19 02:44:15+00:00 Use the existing image.
inference-nlp-azure-1  | 2022-10-19 02:44:17+00:00 Creating resources in AKS.
inference-nlp-azure-1  | 2022-10-19 02:44:18+00:00 Submitting deployment to compute.
inference-nlp-azure-1  | 2022-10-19 02:44:18+00:00 Checking the status of deployment hf-aks-1..
inference-nlp-azure-1  | 2022-10-19 02:45:01+00:00 Checking the status of inference endpoint hf-aks-1.
inference-nlp-azure-1  | Succeeded
inference-nlp-azure-1  | AKS service creation operation finished, operation "Succeeded"
inference-nlp-azure-1  | Healthy
inference-nlp-azure-1  | {'result': '0', 'sentence1': 'Shares of Genentech, a much larger company with several products on the market, rose more than 2 percent.', 'sentence2': 'Shares of Xoma fell 16 percent in early trade, while shares of Genentech, a much larger company with several products on the market, were up 2 percent.', 'logits': 'tensor([[ 2.3388, -2.3361]], grad_fn=<AddmmBackward0>)', 'probability': 'tensor([0.9908, 0.0092], grad_fn=<SoftmaxBackward0>)', 'input_data': "{'input_ids': tensor([[  101,  6661,  1997,  4962, 10111,  2818,  1010,  1037,  2172,  3469,\n          2194,  2007,  2195,  3688,  2006,  1996,  3006,  1010,  3123,  2062,\n          2084,  1016,  3867,  1012,   102,  6661,  1997,  1060,  9626,  3062,\n          2385,  3867,  1999,  2220,  3119,  1010,  2096,  6661,  1997,  4962,\n         10111,  2818,  1010,  1037,  2172,  3469,  2194,  2007,  2195,  3688,\n          2006,  1996,  3006,  1010,  2020,  2039,  1016,  3867,  1012,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}", 'model_path': '/var/azureml-app/azureml-models/inc_ptq_bert_model_mrpc/2/outputs'}
inference-nlp-azure-1  | Classification result: 0
inference-nlp-azure-1 exited with code 0
```