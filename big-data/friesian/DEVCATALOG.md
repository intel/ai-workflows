# Building Large-Scale End-to-End Recommendation Systems with BigDL Friesian

## Overview

[BigDL Friesian](https://bigdl.readthedocs.io/en/latest/doc/Friesian/index.html) is an application framework for building optimized large-scale recommender solutions optimized on Intel Xeon. This workflow demonstrates how to use Friesian to easily build an end-to-end [Wide & Deep Learning](https://arxiv.org/abs/1606.07792) recommennder system on a real-world large dataset provided by Twitter.

## How it Works

- Friesian provides various built-in distributed feature engineering operations and the distributed training of popular recommendation algorithms based on [BigDL Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html) and Spark.
- Friesian provides a complete, highly available and scalable pipeline for online serving (including recall and ranking) as well as nearline updates based on gRPC services.


The overall architecture of Friesian is shown in the following diagram:

<img src="https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/images/architecture.png" width="100%" />


## Get Started

### Dataset Preparation
You can download Twitter Recsys Challenge 2021 dataset from [here](https://recsys-twitter.com/data/show-downloads#). Or you can run the script [`generate_dummy_data.py`]([./generate_dummy_data.py](https://github.com/intel-analytics/BigDL/blob/main/apps/wide-deep-recommendation/generate_dummy_data.py)) to generate a dummy dataset.

To run on a Kubernetes cluster, you may need to put the downloaded data to a shared volume. Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html#load-data-from-network-file-systems-nfs) for more details.

### Docker

- Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html#pull-docker-image) for the docker image for BigDL on K8s.
- Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html#create-a-k8s-client-container) to create a client container for the Kubernetes cluster.

### Environment Preparation
Please follow the steps [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html#prepare-environment) to prepare the Python environment on the client container.

### How to run

- Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html#run-jobs-on-k8s) to run the distributed feature engineering and training workload on a Kubernetes cluster. The scripts are [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd/recsys2021).
- Please refer to [here](https://github.com/intel-analytics/BigDL/tree/main/scala/friesian) to run the online serving workload.

## Recommended Hardware
The hardware below is recommended for use with this reference implementation.

- Intel® 4th Gen Xeon® Scalable Performance processors

## Learn More

- Please check the notebooks [here](https://github.com/intel-analytics/BigDL/tree/main/apps/wide-deep-recommendation) for more detailed descriptions for distributed feature engineering and training.
- Please check [here](https://bigdl.readthedocs.io/en/latest/doc/Friesian/examples.html) for more reference use cases.
- Please check [here](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Friesian/index.html) for more detailed API documentations.

## Known Issues
NA

## Troubleshooting
NA

## Support Forum
Please submit issues [here](https://github.com/intel-analytics/BigDL/issues) and we will track and respond to them daily.
