# **PyTorch BERT Base TRAINING - NLP Azure**

## **Description**

This pipeline provides instructions on how to train the BERT base model on infrastructure provided by Azure Machine Learning with make and docker compose.

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
FINAL_IMAGE_NAME ?= nlp-azure

nlp-azure:
	AZURE_CONFIG_FILE=${AZURE_CONFIG_FILE}  \
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME}  \
	docker compose up nlp-azure --build
		
clean:
	docker compose down
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
    command: sh -c "jupyter nbconvert --to python 1.0-intel-azureml-training.ipynb && python3 1.0-intel-azureml-training.py"
    environment:
      - http_proxy=${http_proxy}
      - https_proxy=${https_proxy}
      - no_proxy=${no_proxy}
    image: ${FINAL_IMAGE_NAME}:training-ubuntu-20.04
    network_mode: "host"
    privileged: true
    volumes:
      - ./azureml/notebooks:/root/notebooks
      - ./azureml/src:/root/src
      - /${AZURE_CONFIG_FILE}:/root/config.json
    working_dir: /root/notebooks
```

# **Azure Machine Learning**

End-to-End AI workflow using the Azure ML Cloud Infrastructure for Training the BERT base model. More Information [here](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML/tree/v1.0). The pipeline runs the `1.0-intel-azureml-training.ipynb` of the [Azure ML](https://github.com/intel/Intel-NLP-workflow-for-Azure-ML/blob/d96750561d88dc04ff20fc900eed44abbda94f0a/notebooks/1.0-intel-azureml-training.ipynb) project. 

## **Quick Start**

* Make sure that the enviroment setup pre-requisites are satisfied per the document [here](../../../../README.md).

* Pull and configure the dependent repo submodule ```git submodule update --init --recursive ```.

* Install [Pipeline Repository Dependencies](../../../../README.md).
 
* Use the quickstart [link](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to setup your Azure ML resources. 

  *  If required, create virtual networks and NAT gateway by following this [link](https://learn.microsoft.com/en-us/azure/virtual-network/nat-gateway/quickstart-create-nat-gateway-portal).
  

* Download the `config.json` file from your Azure ML Studio Workspace. 

* The FP32 model files generated as a result of the training, get stored at the `/root/notebooks/fp32_model_output` folder. 

* Other Variables:

Variable Name | Default | Notes |
:----------------:|:------------------: | :--------------------------------------:|
AZURE_CONFIG_FILE | `$$(pwd)/config.json` | Azure Workspace Configuration file |                       
FINAL_IMAGE_NAME  | nlp-azure | Final Docker Image Name | 

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
#3 DONE 0.4s

#4 [1/3] FROM docker.io/library/ubuntu:20.04@sha256:9c2004872a3a9fcec8cc757ad65c042de1dad4da27de4c70739a6e36402213e3
#4 DONE 0.0s

#5 [2/3] RUN apt-get update &&     apt-get install --no-install-recommends curl=7.68.0-1ubuntu2.13 -y &&     apt-get install --no-install-recommends python3-pip=20.0.2-5ubuntu1.6 -y &&     rm -r /var/lib/apt/lists/*
#5 CACHED

#6 [3/3] RUN pip install --no-cache-dir azureml-sdk==1.45.0 && pip install --no-cache-dir notebook==6.4.12
#6 CACHED

#7 exporting to image
#7 exporting layers done
#7 writing image sha256:5aaf5cd9266ef08eab26d1f355562431618c753c7ad40b88fd9afed8c5aab927 done
#7 naming to docker.io/library/nlp-azure:training-ubuntu-20.04 done
#7 DONE 0.0s
Attaching to training-nlp-azure-1
training-nlp-azure-1  | [NbConvertApp] Converting notebook 1.0-intel-azureml-training.ipynb to python
training-nlp-azure-1  | [NbConvertApp] Writing 8409 bytes to 1.0-intel-azureml-training.py
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint hyperdrive = azureml.train.hyperdrive:HyperDriveRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint automl = azureml.train.automl.run:AutoMLRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.PipelineRun = azureml.pipeline.core.run:PipelineRun._from_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.ReusedStepRun = azureml.pipeline.core.run:StepRun._from_reused_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.StepRun = azureml.pipeline.core.run:StepRun._from_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.scriptrun = azureml.core.script_run:ScriptRun._from_run_dto with exception (cryptography 37.0.4 (/usr/local/lib/python3.8/dist-packages), Requirement.parse('cryptography<39,>=38.0.0'), {'PyOpenSSL', 'pyopenssl'}).
training-nlp-azure-1  | To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code AWUEQBGKV to authenticate.
training-nlp-azure-1  | Building a non-registered environment is not supported. Registering environment.
training-nlp-azure-1  | Performing interactive authentication. Please follow the instructions on the terminal.
training-nlp-azure-1  | Interactive authentication successfully completed.
training-nlp-azure-1  | Loaded existing workspace configuration
training-nlp-azure-1  | Image Build Status: Queued
training-nlp-azure-1  | 
training-nlp-azure-1  | 2022/10/18 17:39:32 Downloading source code...
training-nlp-azure-1  | 2022/10/18 17:39:33 Finished downloading source code
training-nlp-azure-1  | 2022/10/18 17:39:34 Creating Docker network: acb_default_network, driver: 'bridge'
training-nlp-azure-1  | 2022/10/18 17:39:34 Successfully set up Docker network: acb_default_network
training-nlp-azure-1  | 2022/10/18 17:39:34 Setting up Docker configuration...
training-nlp-azure-1  | 2022/10/18 17:39:34 Successfully set up Docker configuration
training-nlp-azure-1  | 2022/10/18 17:39:34 Logging in to registry: b5e8b4b470eb44ca8a9f3fdfa30decb5.azurecr.io
training-nlp-azure-1  | 2022/10/18 17:39:35 Successfully logged into b5e8b4b470eb44ca8a9f3fdfa30decb5.azurecr.io
training-nlp-azure-1  | 2022/10/18 17:39:35 Executing step ID: acb_step_0. Timeout(sec): 5400, Working directory: '', Network: 'acb_default_network'
training-nlp-azure-1  | 2022/10/18 17:39:35 Scanning for dependencies...
training-nlp-azure-1  | 2022/10/18 17:39:36 Successfully scanned dependencies
training-nlp-azure-1  | 2022/10/18 17:39:36 Launching container with name: acb_step_0
training-nlp-azure-1  | Sending build context to Docker daemon  65.54kB
training-nlp-azure-1  | 
training-nlp-azure-1  | Step 1/31 : FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
training-nlp-azure-1  | latest: Pulling from azureml/openmpi4.1.0-ubuntu20.04
training-nlp-azure-1  | d7bfe07ed847: Already exists
training-nlp-azure-1  | de838697cf3d: Pulling fs layer
training-nlp-azure-1  | 0ff300a13686: Pulling fs layer
training-nlp-azure-1  | c300aafe4d93: Pulling fs layer
training-nlp-azure-1  | 6addf04a5cf7: Pulling fs layer
training-nlp-azure-1  | 43deaa364170: Pulling fs layer
training-nlp-azure-1  | c52b6c247a75: Pulling fs layer
training-nlp-azure-1  | 46b6997a5deb: Pulling fs layer
training-nlp-azure-1  | 68511f52e3a9: Pulling fs layer
training-nlp-azure-1  | 3c468a5502b1: Pulling fs layer
training-nlp-azure-1  | 6addf04a5cf7: Waiting
training-nlp-azure-1  | 43deaa364170: Waiting
training-nlp-azure-1  | c52b6c247a75: Waiting
training-nlp-azure-1  | 46b6997a5deb: Waiting
training-nlp-azure-1  | 68511f52e3a9: Waiting
training-nlp-azure-1  | 3c468a5502b1: Waiting
training-nlp-azure-1  | 0ff300a13686: Verifying Checksum
training-nlp-azure-1  | 0ff300a13686: Download complete
training-nlp-azure-1  | 6addf04a5cf7: Verifying Checksum
training-nlp-azure-1  | 6addf04a5cf7: Download complete
training-nlp-azure-1  | c300aafe4d93: Verifying Checksum
training-nlp-azure-1  | c300aafe4d93: Download complete
training-nlp-azure-1  | c52b6c247a75: Verifying Checksum
training-nlp-azure-1  | c52b6c247a75: Download complete
training-nlp-azure-1  | 43deaa364170: Verifying Checksum
training-nlp-azure-1  | 43deaa364170: Download complete
training-nlp-azure-1  | 46b6997a5deb: Verifying Checksum
training-nlp-azure-1  | 46b6997a5deb: Download complete
training-nlp-azure-1  | 68511f52e3a9: Verifying Checksum
training-nlp-azure-1  | 68511f52e3a9: Download complete
training-nlp-azure-1  | 3c468a5502b1: Verifying Checksum
training-nlp-azure-1  | 3c468a5502b1: Download complete
training-nlp-azure-1  | de838697cf3d: Verifying Checksum
training-nlp-azure-1  | de838697cf3d: Download complete
training-nlp-azure-1  | de838697cf3d: Pull complete
training-nlp-azure-1  | Image Build Status: Running
training-nlp-azure-1  | 
training-nlp-azure-1  | 0ff300a13686: Pull complete
training-nlp-azure-1  | c300aafe4d93: Pull complete
training-nlp-azure-1  | 6addf04a5cf7: Pull complete
training-nlp-azure-1  | 43deaa364170: Pull complete
training-nlp-azure-1  | c52b6c247a75: Pull complete
training-nlp-azure-1  | 46b6997a5deb: Pull complete
training-nlp-azure-1  | 68511f52e3a9: Pull complete
training-nlp-azure-1  | 3c468a5502b1: Pull complete
training-nlp-azure-1  | Digest: sha256:7bf1fc7a8163da0ac249f77a6607f21a76b08f1a76e61006d131ac441be3b278
training-nlp-azure-1  | Status: Downloaded newer image for mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
training-nlp-azure-1  |  ---> 19955122af99
training-nlp-azure-1  | Step 2/31 : RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
training-nlp-azure-1  |  ---> Running in bca587fb9a33
training-nlp-azure-1  | Get:1 https://packages.microsoft.com/ubuntu/20.04/prod focal InRelease [10.5 kB]
training-nlp-azure-1  | Get:2 https://packages.microsoft.com/ubuntu/20.04/prod focal/main amd64 Packages [218 kB]
training-nlp-azure-1  | Hit:3 http://archive.ubuntu.com/ubuntu focal InRelease
training-nlp-azure-1  | Get:4 http://security.ubuntu.com/ubuntu focal-security InRelease [114 kB]
training-nlp-azure-1  | Get:5 http://archive.ubuntu.com/ubuntu focal-updates InRelease [114 kB]
training-nlp-azure-1  | Get:6 http://archive.ubuntu.com/ubuntu focal-backports InRelease [108 kB]
training-nlp-azure-1  | Get:7 http://archive.ubuntu.com/ubuntu focal-updates/restricted amd64 Packages [1710 kB]
training-nlp-azure-1  | Get:8 http://security.ubuntu.com/ubuntu focal-security/restricted amd64 Packages [1595 kB]
training-nlp-azure-1  | Get:9 http://archive.ubuntu.com/ubuntu focal-updates/universe amd64 Packages [1219 kB]
training-nlp-azure-1  | Get:10 http://archive.ubuntu.com/ubuntu focal-updates/main amd64 Packages [2689 kB]
training-nlp-azure-1  | Get:11 http://security.ubuntu.com/ubuntu focal-security/universe amd64 Packages [921 kB]
training-nlp-azure-1  | Get:12 http://security.ubuntu.com/ubuntu focal-security/main amd64 Packages [2223 kB]
training-nlp-azure-1  | Fetched 10.9 MB in 2s (4893 kB/s)
training-nlp-azure-1  | Reading package lists...
training-nlp-azure-1  | Reading package lists...
training-nlp-azure-1  | Building dependency tree...
training-nlp-azure-1  | Reading state information...
training-nlp-azure-1  | tzdata is already the newest version (2022c-0ubuntu0.20.04.0).
training-nlp-azure-1  | tzdata set to manually installed.
training-nlp-azure-1  | 0 upgraded, 0 newly installed, 0 to remove and 16 not upgraded.
training-nlp-azure-1  | Removing intermediate container bca587fb9a33
training-nlp-azure-1  |  ---> c6c355e4011e
training-nlp-azure-1  | Step 3/31 : RUN apt-get update &&     apt-get install wget -y &&     apt-get install python3-pip -y &&     apt-get install libgl1 -y &&    apt-get install python3-opencv -y &&    apt-get install git -y &&    apt-get install build-essential -y &&    apt-get install libtool -y &&    apt-get install autoconf -y &&    apt-get install unzip -y &&    apt-get install libssl-dev -y
training-nlp-azure-1  |  ---> Running in 70850b2df207
training-nlp-azure-1  | Hit:1 https://packages.microsoft.com/ubuntu/20.04/prod focal InRelease
training-nlp-azure-1  | Hit:2 http://security.ubuntu.com/ubuntu focal-security InRelease
training-nlp-azure-1  | Hit:3 http://archive.ubuntu.com/ubuntu focal InRelease
training-nlp-azure-1  | Hit:4 http://archive.ubuntu.com/ubuntu focal-updates InRelease
training-nlp-azure-1  | Hit:5 http://archive.ubuntu.com/ubuntu focal-backports InRelease
training-nlp-azure-1  | Reading package lists...
training-nlp-azure-1  | Reading package lists...
training-nlp-azure-1  | Building dependency tree...
training-nlp-azure-1  | Reading state information...
training-nlp-azure-1  | wget is already the newest version (1.20.3-1ubuntu2).
training-nlp-azure-1  | 0 upgraded, 0 newly installed, 0 to remove and 16 not upgraded.
training-nlp-azure-1  | Reading package lists...
training-nlp-azure-1  | Building dependency tree...
training-nlp-azure-1  | Reading state information...
training-nlp-azure-1  | The following additional packages will be installed:
training-nlp-azure-1  |   libexpat1-dev libpython3-dev libpython3.8 libpython3.8-dev python-pip-whl
training-nlp-azure-1  |   python3-dev python3-distutils python3-lib2to3 python3-pkg-resources
training-nlp-azure-1  |   python3-setuptools python3-wheel python3.8-dev zlib1g zlib1g-dev
```
...
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

