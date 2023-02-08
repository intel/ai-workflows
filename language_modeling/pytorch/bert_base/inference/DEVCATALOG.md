# Intel® NLP workflow for Azure* ML - Inference
Learn how to use Intel's XPU hardware and Intel optimized software to perform inference on the Azure Machine Learning Platform with PyTorch\*, Intel® Extension for PyTorch\*, Hugging Face, and Intel® Neural Compressor.

Check out more workflow examples and reference implementations in the
[Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
This workflow demonstrates how to use Intel’s XPU hardware (e.g.: CPU - Ice Lake or above) and related optimized software to perform inference on the Azure Machine Learning Platform (Azure ML). The main software packages used here are Intel® Extension for PyTorch\*, PyTorch\*, Hugging Face, Azure Machine Learning Platform, and Intel® Neural Compressor. 

Instructions are provided to perform the following:

1. Specify Azure ML information
2. Build a custom docker image for inference
3. Deploy a PyTorch model using Azure ML, with options to change the instance type and number of nodes

For more detailed information, please visit the [Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) GitHub repository.

## Recommended Hardware 
We recommend you use the following hardware for this reference implementation. 
| **Name**   | **Description**               |
| ---------- | ----------------------------- |
| CPU        | Intel CPU - Ice Lake or above |
| Usable RAM | 16 GB                         |
| Disk Size | 256 GB |

## How it Works
This workflow uses the Azure ML infrastructure to deploy a fine-tuned BERT base model. While the following diagram shows the architecture for both training and inference, this specific workflow is focused on the inference portion. You must run the [Intel® NLP workflow for Azure ML - Training](https://github.com/intel/ai-workflows/blob/main/language_modeling/pytorch/bert_base/training/DEVCATALOG.md) workflow first or provide your own trained model.

### Architecture 

AzureML:

![azureml_architecture](https://user-images.githubusercontent.com/43555799/205149722-e37dcec5-5ef2-4440-92f2-9dc243b9e556.jpg)

### Model Spec
The uncased BERT base model is used to demonstrate this workflow.

```python
bert-base-uncased-config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 128,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.21.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

### Dataset 
[Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/research/publication/automatically-constructing-a-corpus-of-sentential-paraphrases/)  is used as the dataset. 

| **Type**                 | **Format** | **Rows** 
| :---                     | :---       | :---     
| Training Dataset         | HuggingFace Dataset  | 3668
| Testing  Dataset         | HuggingFace Dataset  | 1725

## Get Started

#### Download the workflow repository
Clone [Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) repository.
```
git clone https://github.com/intel/Intel-NLP-workflow-for-Azure-ML.git
cd Intel-NLP-workflow-for-Azure-ML
git checkout v1.0.1
```

#### Download the Datasets
The dataset will be downloaded the first time the workflow runs.

## Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.

### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

Because the Docker image is run on a cloud service, you will need Azure credentials to perform training and inference related operations:
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)

### Set Up Docker Image
Pull the provided docker image.
```
docker pull intel/ai-workflows:nlp-azure-inference
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
Below setup and how-to-run sessions are for users who want to use the provided docker image to run the entire pipeline. 
For interactive set up, please go to [Interactive Docker](#interactive-docker).

#### Setup 
Download the `config.json` file from your Azure ML Studio Workspace.

#### How to run 
Run the workflow using the ``docker run`` command, as shown:
```
export AZURE_CONFIG_FILE=<path to config file downloaded from Azure ML Studio Workspace>

docker run \
  --volume ${PWD}/notebooks:/root/notebooks \
  --volume ${PWD}/src:/root/src \
  --volume ${PWD}/${AZURE_CONFIG_FILE}:/root/notebooks/config.json \
  --workdir /root/notebooks \
  --privileged --init -it \
  intel/ai-workflows:nlp-azure-inference \
  sh -c "jupyter nbconvert --to python 1.0-intel-azureml-inference.ipynb && python3 1.0-intel-azureml-inference.py"
```

### Interactive Docker
Follow these setup and how-to-run steps to use an interactive environment.  
For using a Docker pipeline, see the [Run Using Docker](#run-using-docker) section.
#### Setup 

Build the Docker image to prepare the environment used for running the Jupyter notebooks.
```
cd scripts
sh build_main_image.sh
```

Use the Docker image built by ``build_main_image.sh`` to run the Jupyter notebooks. Execute the following command:
```bash
sh start_script.sh
```
After starting the container, execute the following command in the interactive shell.
```bash
cd notebooks
jupyter notebook --allow-root
```
Start the notebook with "inference" in the filename.

## Run Using Bare Metal
This workflow requires Docker and currently cannot be run using bare metal.  

## Expected Output
```
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
```
...
```
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

## Summary and Next Steps
In this workflow, you loaded a Docker image and deployed a PyTorch BERT base model on the Azure Machine Learning Platform using Intel® Xeon® Scalable Processors. The [Intel® NLP workflow for Azure ML - Training](https://github.com/intel/ai-workflows/blob/main/language_modeling/pytorch/bert_base/training/DEVCATALOG.md)  contains a workflow for distributed training on the Azure Machine Learning Platform, which you may use to learn how to train your own model. 

## Learn More
For more information about Intel® NLP workflow for Azure* ML or to read about other relevant workflow
examples, see these guides and software resources:

- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
Workflow will be monitored for issues. Issues or problem spots, and if possible, workarounds will be listed here.

## Support 
[Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) tracks both bugs and enhancement requests using GitHub. We welcome input, however, before filing a request, please make sure you do the following: Search the GitHub issue database.
