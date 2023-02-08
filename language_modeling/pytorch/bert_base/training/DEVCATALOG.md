# Intel® NLP workflow for Azure* ML - Training
Learn how to use Intel's XPU hardware and Intel optimized software to perform distributed training on the Azure Machine Learning Platform with PyTorch\*, Intel® Extension for PyTorch\*, Hugging Face, and Intel® Neural Compressor.

Check out more workflow examples and reference implementations in the
[Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
This workflow demonstrates how to use Intel’s XPU hardware (e.g.: CPU - Ice Lake or above) and related optimized software to perform distributed training on the Azure Machine Learning Platform (Azure ML). The main software packages used here are Intel® Extension for PyTorch\*, PyTorch\*, Hugging Face, Azure Machine Learning Platform, and Intel® Neural Compressor.

 Instructions are provided to perform the following:

1. Specify Azure ML information
2. Build a custom docker image for training
3. Train a PyTorch model using Azure ML, with options to change the instance type and number of nodes

For more detailed information, please visit the [Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) GitHub repository.

## Recommended Hardware 
We recommend you use the following hardware for this reference implementation. 
| **Name**   | **Description**               |
| ---------- | ----------------------------- |
| CPU        | Intel CPU - Ice Lake or above |
| Usable RAM | 16 GB                         |
| Disk Size | 256 GB |

## How it Works
This workflow uses the Azure ML infrastructure to fine-tune a pretrained BERT base model. While the following diagram shows the architecture for both training and inference, this specific workflow is focused on the training portion.  See the [Intel® NLP workflow for Azure ML - Inference](https://github.com/intel/ai-workflows/blob/main/language_modeling/pytorch/bert_base/inference/DEVCATALOG.md) workflow that uses this trained model.

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
The dataset will be downloaded the first time the training runs.


## Run Using Docker
*Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.*

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
docker pull intel/ai-workflows:nlp-azure-training
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
Run the workflow using the ``docker run`` command, as shown:  (example)
```
export AZURE_CONFIG_FILE=<path to config file downloaded from Azure ML Studio Workspace>

docker run \
  --volume ${PWD}/notebooks:/root/notebooks \
  --volume ${PWD}/src:/root/src \
  --volume ${PWD}/${AZURE_CONFIG_FILE}:/root/config.json \
  --workdir /root/notebooks \
  --privileged --init -it \
  intel/ai-workflows:nlp-azure-training \
  sh -c "jupyter nbconvert --to python 1.0-intel-azureml-training.ipynb && python3 1.0-intel-azureml-training.py"
```
### Interactive Docker
Below setup and how-to-run sessions are for users who want to use an interactive environment.  
For docker pipeline, please go to [docker session](#docker).
#### Setup 

Build the docker image to prepare the environment for running the Jupyter notebooks.
```
cd scripts
sh build_main_image.sh
```

Use the Docker image built by ``build_main_image.sh`` to run the Jupyter notebook. Execute the following command:
```bash
sh start_script.sh
```
After starting the container, execute the following command in the interactive shell.
```bash
cd notebooks
jupyter notebook --allow-root
```
Start the notebook with "training" in the filename.

## Run Using Bare Metal
This workflow requires Docker and currently cannot be run using bare metal.  

## Expected Output

```
training-nlp-azure-1  | 
training-nlp-azure-1  |  88%|████████▊ | 95/108 [00:40<00:05,  2.36it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  89%|████████▉ | 96/108 [00:40<00:05,  2.35it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  90%|████████▉ | 97/108 [00:41<00:04,  2.35it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  91%|█████████ | 98/108 [00:41<00:04,  2.33it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  92%|█████████▏| 99/108 [00:42<00:03,  2.32it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  93%|█████████▎| 100/108 [00:42<00:03,  2.30it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  94%|█████████▎| 101/108 [00:42<00:03,  2.29it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  94%|█████████▍| 102/108 [00:43<00:02,  2.23it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  95%|█████████▌| 103/108 [00:43<00:02,  2.27it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  96%|█████████▋| 104/108 [00:44<00:01,  2.29it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  97%|█████████▋| 105/108 [00:44<00:01,  2.27it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  98%|█████████▊| 106/108 [00:45<00:00,  2.28it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |  99%|█████████▉| 107/108 [00:45<00:00,  2.28it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  | 100%|██████████| 108/108 [00:45<00:00,  2.32it/s][A
training-nlp-azure-1  |                                                  
training-nlp-azure-1  | 
training-nlp-azure-1  |                                                  
training-nlp-azure-1  | [A{'eval_loss': 0.60855633020401, 'eval_accuracy': 0.8573913043478261, 'eval_runtime': 46.8883, 'eval_samples_per_second': 36.79, 'eval_steps_per_second': 2.303, 'epoch': 3.0}
training-nlp-azure-1  | 
training-nlp-azure-1  | 100%|██████████| 690/690 [31:31<00:00,  2.27s/it]
training-nlp-azure-1  | 
training-nlp-azure-1  | 100%|██████████| 108/108 [00:46<00:00,  2.32it/s][A
training-nlp-azure-1  | 
training-nlp-azure-1  |                                                  [A
training-nlp-azure-1  | 
training-nlp-azure-1  | Training completed. Do not forget to share your model on huggingface.co/models =)
training-nlp-azure-1  | 
training-nlp-azure-1  | 
training-nlp-azure-1  | 
training-nlp-azure-1  |                                                  
training-nlp-azure-1  | {'train_runtime': 1891.9246, 'train_samples_per_second': 5.816, 'train_steps_per_second': 0.365, 'train_loss': 0.31064462523529496, 'epoch': 3.0}
training-nlp-azure-1  | 
training-nlp-azure-1  | 100%|██████████| 690/690 [31:31<00:00,  2.27s/it]
training-nlp-azure-1  | 100%|██████████| 690/690 [31:31<00:00,  2.74s/it]
training-nlp-azure-1  | Saving model checkpoint to ./outputs/trained_model
training-nlp-azure-1  | Configuration saved in ./outputs/trained_model/config.json
training-nlp-azure-1  | Model weights saved in ./outputs/trained_model/pytorch_model.bin
training-nlp-azure-1  | Time for training: 1922.514419555664s
training-nlp-azure-1  | Cleaning up all outstanding Run operations, waiting 300.0 seconds
training-nlp-azure-1  | 1 items cleaning up...
training-nlp-azure-1  | Cleanup took 5.616384744644165 seconds
training-nlp-azure-1  | 
training-nlp-azure-1  | Execution Summary
training-nlp-azure-1  | =================
training-nlp-azure-1  | RunId: IntelIPEX_HuggingFace_DDP_1666115383_6ff5fb64
training-nlp-azure-1  | Web View: https://ml.azure.com/runs/IntelIPEX_HuggingFace_DDP_1666115383_6ff5fb64?wsid=/subscriptions/0a5dbdd4-ee35-483f-b248-93e05a52cd9f/resourcegroups/intel_azureml_resource/workspaces/cloud_t7_i9&tid=46c98d88-e344-4ed4-8496-4ed7712e255d
training-nlp-azure-1  | 
training-nlp-azure-1  | Length of output paths is not the same as the length of pathsor output_paths contains duplicates. Using paths as output_paths.
training-nlp-azure-1 exited with code 0
```
## Summary and Next Steps
In this workflow, you loaded a docker image and performed distributed training on a PyTorch BERT base model on the Azure Machine Learning Platform using Intel® Xeon® Scalable Processors. See the [Intel® NLP workflow for Azure ML - Inference](https://github.com/intel/ai-workflows/blob/main/language_modeling/pytorch/bert_base/inference/DEVCATALOG.md) workflow that uses this trained model. 

## Learn More
For more information about Intel® NLP workflow for Azure* ML or to read about other relevant workflow
examples, see these guides and software resources:

- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
Issues, problem spots, and their workarounds if possible, will be listed here.

## Support 
[Intel® NLP workflow for Azure* ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML) tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML/issues). Search there before submitting a new issue.
