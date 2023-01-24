# Time Series Forecasting using BigDL-Chronos - Model development and optimization

## Overview

BigDL-Chronos (Chronos in short) is an application framework for building a fast, accurate and scalable time series analysis application, where forecasting is the most popular task. For more detailed information about the framework, go to [Chronos Document](https://bigdl.readthedocs.io/en/latest/doc/Chronos/index.html).

## How it works

- Chronos provides an easy but complete API `TSDataset` for data loading, preprocessing, feature engineering and sampling, where users could fetch their raw tabular data from local files, distributed file system or TS database like Prometheus and transform them to data that can be fed into DL/ML models. It contains 2 backends, pandas for single node users and spark (with the support of `bigdl-orca`) for cluster users.

- Chronos provides `Forecaster` API for 9+ different ML/DL models' training, tunning, evaluation, model optimization, and exporting for deployment.

- 9+ different ML/DL model include traditional models cover traditional models like ARIMA, Prophet to SOTA Deep learning models such as Autoformer, TCN, NBeats, and Seq2Seq.

- During the process, IPEX, intel-tensorflow, onnxruntime, openvino, and intel neural compressor are used as accelerator and low precision tool to accelerate the model during training and inferencing. `bigdl-orca` is used for cluster level distributed training and tunning.

<img width="890" alt="Chronos-workflow" src="https://user-images.githubusercontent.com/43555799/207939545-9e7e1ae4-d46c-4e65-b4a2-1d59ae8b42cf.png">

## Get Started

### Through PyPI

`bigdl-chronos` is released on PyPI as a python library and could be installed on nearly any platform user prefer. Here we provide a typical installation method for single node user who prefer a pytorch backend, for more detailed information, users may refer to our [installaion page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html).

```bash
# conda environment is highly recommeneded
conda create -n bigdl-chronos-pytorch-env python=3.7 setuptools=58.0.4
conda activate bigdl-chronos-pytorch-env
pip install bigdl-chronos[pytorch]
```

#### On colab

Users could also install `bigdl-chronos` easily on google colab, where they only need to install the library through `pip`.

### Through docker

Some users may prefer a docker installation do seperate the environment. Below setup and how-to-run sessions are for users who want to use the provided docker image.

**Pull Docker Image**

```docker pull intel/ai-workflows:time-series-forecaster```

Clone the BigDL repository to the current working directory and checkout the specific tag

```
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
git checkout ai-workflow
```

To run the pipeline, follow the instructions below outside of docker instance.

```
docker run -it --rm -v ${PWD}:/workspace \
   -w /workspace/python/chronos/colab-notebook --init --net=host \
   intel/ai-workflows:time-series-forecaster \
   sh -c "jupyter nbconvert --to python chronos_nyc_taxi_tsdataset_forecaster.ipynb && \
   sed '26,40d' chronos_nyc_taxi_tsdataset_forecaster.py > chronos_taxi_forecaster.py && \
   python chronos_taxi_forecaster.py"
```
**Output**

```
#1 [internal] load build definition from Dockerfile.chronos
#1 transferring dockerfile: 55B done
#1 DONE 0.0s

#2 [internal] load .dockerignore
#2 transferring context: 2B done
#2 DONE 0.0s

#3 [internal] load metadata for docker.io/library/ubuntu:20.04
#3 DONE 0.0s

#4 [1/5] FROM docker.io/library/ubuntu:20.04
#4 DONE 0.0s

#5 [2/5] RUN apt-get update --fix-missing &&     apt-get install -y apt-utils vim curl nano wget unzip git &&     apt-get install -y gcc g++ make &&     apt-get install -y libsm6 libxext6 libxrender-dev &&     apt-get install -y openjdk-8-jre &&     rm /bin/sh &&     ln -sv /bin/bash /bin/sh &&     echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su &&     chgrp root /etc/passwd && chmod ug+rw /etc/passwd &&     wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh &&     chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh &&     ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /usr/local &&     rm Miniconda3-py37_4.12.0-Linux-x86_64.sh
#5 CACHED

#6 [4/5] RUN echo "source activate chronos" > ~/.bashrc
#6 CACHED

#7 [3/5] RUN conda create -y -n chronos python=3.7 setuptools=58.0.4 && source activate chronos &&     pip install --no-cache-dir --pre --upgrade bigdl-chronos[pytorch,automl] matplotlib notebook==6.4.12 &&     pip uninstall -y torchtext
#7 CACHED

#8 [5/5] RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc
#8 CACHED

#9 exporting to image
#9 exporting layers done
#9 writing image sha256:329995e99da4001c6d57e243085145acfce61f5bddabd9459aa598b846eae331 done
#9 naming to docker.io/library/time-series-chronos:training-ubuntu-20.04 done
#9 DONE 0.0s
Attaching to training-time-series-chronos-1
training-time-series-chronos-1  | [NbConvertApp] Converting notebook chronos_nyc_taxi_tsdataset_chronos.ipynb to python
training-time-series-chronos-1  | [NbConvertApp] Writing 10692 bytes to chronos_nyc_taxi_tsdataset_chronos.py
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | /usr/local/envs/chronos/lib/python3.7/site-packages/bigdl/chronos/forecaster/utils.py:157: UserWarning: 'batch_size' cannot be divided with no remainder by 'self.num_processes'. We got 'batch_size' = 32 and 'self.num_processes' = 7
training-time-series-chronos-1  |   format(batch_size, num_processes))
training-time-series-chronos-1  | GPU available: False, used: False
training-time-series-chronos-1  | TPU available: False, using: 0 TPU cores
training-time-series-chronos-1  | IPU available: False, using: 0 IPUs
training-time-series-chronos-1  | HPU available: False, using: 0 HPUs
training-time-series-chronos-1  | Global seed set to 1
training-time-series-chronos-1  | Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/7
```
...

```
 98%|█████████▊| 287/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0155]
Epoch 2:  98%|█████████▊| 288/294 [00:09<00:00, 31.55it/s, loss=0.0157]
Epoch 2:  98%|█████████▊| 289/294 [00:09<00:00, 31.56it/s, loss=0.0157]
Epoch 2:  98%|█████████▊| 289/294 [00:09<00:00, 31.56it/s, loss=0.0162]
Epoch 2:  99%|█████████▊| 290/294 [00:09<00:00, 31.57it/s, loss=0.0162]
Epoch 2:  99%|█████████▊| 290/294 [00:09<00:00, 31.57it/s, loss=0.0165]
Epoch 2:  99%|█████████▉| 291/294 [00:09<00:00, 31.58it/s, loss=0.0165]
Epoch 2:  99%|█████████▉| 291/294 [00:09<00:00, 31.58it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2:  99%|█████████▉| 292/294 [00:09<00:00, 31.59it/s, loss=0.0164]
Epoch 2: 100%|█████████▉| 293/294 [00:09<00:00, 31.58it/s, loss=0.0164]
Epoch 2: 100%|█████████▉| 293/294 [00:09<00:00, 31.58it/s, loss=0.0175]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.0175]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.017] 
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.64it/s, loss=0.017]
Epoch 2: 100%|██████████| 294/294 [00:09<00:00, 31.60it/s, loss=0.017]
training-time-series-forecaster-1  | Global seed set to 1
training-time-series-forecaster-1  | GPU available: False, used: False
training-time-series-forecaster-1  | TPU available: False, using: 0 TPU cores
training-time-series-forecaster-1  | IPU available: False, using: 0 IPUs
training-time-series-forecaster-1  | HPU available: False, using: 0 HPUs
training-time-series-forecaster-1 exited with code 0
```

## Use-cases

### How to guides

How-to guides are bite-sized, executable examples where users could check when meeting with some specific topic during the usage. For friendly experience please visit our [how to guide page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/index.html).

### Tutorials/examples

Here is a use-cases list of `bigdl-chronos`, for more friendly experience please visit our [tutorial page](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/index.html).

| Use case                                                                                                                                                                                                                      | Format   | Model        | Framework            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ | -------------------- |
| [Predict Number of Taxi Passengers with Chronos Forecaster](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-tsdataset-forecaster-quickstart.html)                                                       | Notebook | TCN          | PyTorch              |
| [Tune a Forecasting Task Automatically](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-autotsest-quickstart.html)                                                                                      | Notebook | TCN          | PyTorch, Ray         |
| [Tune a Customized Time Series Forecasting Model with AutoTSEstimator](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.ipynb)              | Notebook | Customized   | PyTorch, Ray         |
| [Auto Tune the Prediction of Network Traffic at the Transit Link of WIDE](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.ipynb)                | Notebook | LSTM         | PyTorch, Ray         |
| [Multivariate Forecasting of Network Traffic at the Transit Link of WIDE](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.ipynb)                 | Notebook | LSTM         | PyTorch              |
| [Multistep Forecasting of Network Traffic at the Transit Link of WIDE](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.ipynb) | Notebook | TCN          | PyTorch              |
| [Stock Price Prediction with LSTMForecaster](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction.ipynb)                                                                           | Notebook | LSTM         | PyTorch              |
| [Stock Price Prediction with ProphetForecaster and AutoProphet](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/use-case/fsi/stock_prediction_prophet.ipynb)                                                | Notebook | Prophet      | Prophet, Ray         |
| [Tune a Time Series Forecasting Model with multi-objective hyperparameter optimization](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb) | Notebook | TCN          | Prophet, Optuna      |
| [Auto tuning prophet on nyc taxi dataset](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/auto_model)                                                                                               | Scripts  | Prophet      | Prophet, Ray         |
| [Use Chronos forecasters in a distributed fashion](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/distributed)                                                                                     | Scripts  | Seq2Seq, TCN | PyTorch, Ray         |
| [Use ONNXRuntime to speed-up forecasters' inferecing](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/onnx)                                                                                         | Scripts  | Seq2Seq      | OnnxRuntime, PyTorch |
| [Quantize Chronos forecasters method to speed-up inference](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/quantization)                                                                           | Scripts  | TCN          | INC, PyTorch         |
| [High dimension time series forecasting with Chronos TCMFForecaster](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/tcmf)                                                                          | Scripts  | TCMF         | Ray, Spark           |
| [Penalize underestimation with LinexLoss](https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/loss/penalize_underestimation.ipynb)                                                                      | Notebook | TCN          | PyTorch              |
| [Serve Chronos forecaster and predict through TorchServe](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/example/serving)                                                                                  | Scripts  | TCN          | TorchServe, PyTorch  |
| [Help pytorch-forecasting improve the training speed of DeepAR model](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/DeepAR)                                                  | Scripts  | DeepAR       | Pytorch-forecasting  |
| [Help pytorch-forecasting improve the training speed of TFT model](https://github.com/intel-analytics/BigDL/tree/main/python/chronos/use-case/pytorch-forecasting/TFT)                                                        | Scripts  | TFT          | Pytorch-forecasting  |

## Recommended Platform

### CPU

- Intel® Xeon® Scalable Performance processors

- Intel® Core® processors

### OS

- Ubuntu 16.04/18.04/20.04/22.04

- Windows (experimentally supported)

- Mac with Intel Chip (experimentally supported)

### Python

- Python 3.7

- Python 3.8 (experimentally supported)

## Learn More

[Chronos Document](https://bigdl.readthedocs.io/en/latest/doc/Chronos/index.html)

## Trouble shooting

[Chronos tips and know issues](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos_known_issue.html)

## Support Forum

We welcome any questions, bug report or feature request to:

- [BigDL issue page](https://github.com/intel-analytics/BigDL/issues)

- [BigDL google user group](https://groups.google.com/g/bigdl-user-group)
