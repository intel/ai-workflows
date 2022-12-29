# AI Workflows Infrastructure for Intel® Architecture

## Description
On this page you will find details and instructions on how to set up an environment that supports Intel's AI Pipelines container build and test infrastructure.

## Dependency Requirements
Only Linux systems are currently supported. Please make sure the following are installed in your package manager of choice:
- `make`
- `docker.io`

A full installation of [docker engine](https://docs.docker.com/engine/install/) with docker CLI is required. The recommended docker engine version is `19.03.0+`.

- `docker-compose`

The Docker Compose CLI can be [installed](https://docs.docker.com/compose/install/compose-plugin/#installing-compose-on-linux-systems) both manually and via package manager.

```
$ DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
$ mkdir -p $DOCKER_CONFIG/cli-plugins
$ curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
$ chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

$ docker compose version
Docker Compose version v2.7.0
```

## Build and Run Workflows
Each pipeline will contain specific requirements and instructions for how to provide its specific dependencies and what customization options are possible. Generally, pipelines are run with the following format:

```git submodule update --init --recursive```

This will pull the dependent repo containing the scripts to run the end2end pipeline's inference and/or training.

```<KEY>=<VALUE> ... <KEY>=<VALUE> make <PIPELINE_NAME>```

Where `KEY` and `VALUE` pairs are environment variables that can be used to customize both the pipeline's script options and the resulting container. For more information about the valid `KEY` and `VALUE` pairs, see the README.md file in the folder for each workflow container:

|AI Workflow|Framework/Tool|Mode|
|-|-|-|
|Chronos Time Series Forecasting|Chronos and PyTorch*|[Training](./big-data/chronos/DEVCATALOG.md)
|Document-Level Sentiment Analysis|PyTorch*|[Training](./language_modeling/pytorch/bert_large/training/)|
|Friesian Recommendation System|Spark with TensorFlow|[Training](./big-data/friesian/training/) \| [Inference](./big-data/friesian/DEVCATALOG.md)|
|Habana® Gaudi® Processor Training and Inference using OpenVINO™ Toolkit for U-Net 2D Model|OpenVINO™|[Training and Inference](https://github.com/intel/cv-training-and-inference-openvino/tree/v1.0.0/gaudi-segmentation-unet-ptq)|
|Privacy Preservation|Spark with TensorFlow and PyTorch*|[Training and Inference](./big-data/ppml/DEVCATALOG.md)|
|NLP workflow for AWS Sagemaker|TensorFlow and Jupyter|[Inference](./classification/tensorflow/bert_base/inference/)|
|NLP workflow for Azure ML|PyTorch* and Jupyter|[Training](./language_modeling/pytorch/bert_base/training/) \| [Inference](./language_modeling/pytorch/bert_base/inference/)|
|Protein Structure Prediction|PyTorch*|[Inference](./protein-folding/pytorch/alphafold2/inference/)
|Quantization Aware Training and Inference|OpenVINO™|[Quantization Aware Training(QAT)](https://github.com/intel/nlp-training-and-inference-openvino/tree/v1.0/question-answering-bert-qat)|
|Ray Recommendation System|Ray with PyTorch*|[Training](./big-data/aiok-ray/training/) \| [Inference](./big-data/aiok-ray/inference)|
|RecSys Challenge Analytics With Python|Hadoop and Spark|[Training](./analytics/classical-ml/recsys/training/)|
|Video Streamer|TensorFlow|[Inference](./analytics/tensorflow/ssd_resnet34/inference/)|
|Vision Based Transfer Learning|TensorFlow|[Training](./transfer_learning/tensorflow/resnet50/training/) \| [Inference](./transfer_learning/tensorflow/resnet50/inference/)|
|Wafer Insights|SKLearn|[Inference](./analytics/classical-ml/synthetic/inference/)|


### Cleanup
Each pipeline can remove all resources allocated by executing `make clean`.
