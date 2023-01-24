# Wafer Insights - Inference

## Overview
Wafer Insights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in the Intel fab. For detailed information about the workflow, go to [Wafer Insights](https://github.com/intel/wafer-insights-with-classical-ml) GitHub repository.

## How it Works
Wafer Insights is an interactive data-visualization web application based on Dash and Plotly. It includes 2 major components: a data loader, which generates synthetic fab data for visualization, and a dash app that provides an interface for users to analyze the data and gain insight. Dash is written on top of Plotly.js and React.js, providing an ideal framework for building and deploying data apps with customized user interfaces. The  `src/dashboard` folder contains the code for the dash app and the `src/loaders` folder contains the code for the data loader.

## Get Started

### **Prerequisites**

#### Dependencies
The following libraries are required before you get started:
1. Git
2. Anaconda/Miniconda
3. Docker
4. Python3

#### Download the repo
Clone [Wafer Insights](https://github.com/intel/wafer-insights-with-classical-ml) repository.
```
git clone https://github.com/intel/wafer-insights-with-classical-ml
cd wafer-insights-with-classical-ml
git checkout v1.0.0
```
#### Download the Dataset
Actual measurement data from the Intel fab cannot be shared with the public. Therefore, we provide a synthetic data loader to generate synthetic data using the `make_regression` function from the sklearn library, which has the following format:
| **Type**         | **Format** | **Rows** | **Columns** |
| ---------------- | ---------- | -------- | ----------- |
| Feature Dataset  | Parquet    | 25000    | 2000        |
| Response Dataset | Parquet    | 25000    | 1            |

Refer to [How to Run](#how-to-run) to construct the dataset
### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image.
For bare metal environment, please go to [Bare Metal](#bare-metal).
#### Setup 

##### Pull Docker Image
```
docker pull intel/ai-workflows:wafer-insights
```

##### Set Up Synthetic Data
```
docker run -a stdout \
  -v $(pwd):/workspace \
  --workdir /workspace/src/loaders/synthetic_loader \
  --privileged --init --rm -it \
  intel/ai-workflows:wafer-insights \
  conda run --no-capture-output -n WI python loader.py
```

#### How to Run 

(Optional) Export related proxy into docker environment.
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the pipeline, follow the below instructions outside of the docker instance. 
```
export OUTPUT_DIR=/output
```

```
docker run -a stdout $DOCKER_RUN_ENVS \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PYTHONPATH=$PYTHONPATH:$PWD \
  --volume ${OUTPUT_DIR}:/output \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  -p 8050:8050 \
  --privileged --init --rm -it \
  intel/ai-workflows:wafer-insights \
  conda run --no-capture-output -n WI python src/dashboard/app.py
```

#### Output
```
$ make wafer-insight
WARN[0000] The "PYTHONPATH" variable is not set. Defaulting to a blank string. 
[+] Building 0.1s (9/9) FINISHED                                                                                 
 => [internal] load build definition from Dockerfile.wafer-insights                                         0.0s
 => => transferring dockerfile: 47B                                                                         0.0s
 => [internal] load .dockerignore                                                                           0.0s
 => => transferring context: 2B                                                                             0.0s
 => [internal] load metadata for docker.io/library/ubuntu:20.04                                             0.0s
 => [1/5] FROM docker.io/library/ubuntu:20.04                                                               0.0s
 => CACHED [2/5] RUN apt-get update && apt-get install --no-install-recommends --fix-missing -y     ca-cer  0.0s
 => CACHED [3/5] RUN mkdir -p /workspace                                                                    0.0s
 => CACHED [4/5] RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O  0.0s
 => CACHED [5/5] RUN conda create -yn WI python=3.9 &&     source activate WI &&     conda install -y scik  0.0s
 => exporting to image                                                                                      0.0s
 => => exporting layers                                                                                     0.0s
 => => writing image sha256:dfa7411736694db4d3c8d0032f424fc88f0af98fabd163a659a90d0cc2dfe587                0.0s
 => => naming to docker.io/library/wafer-insights:inference-ubuntu-20.04                                    0.0s
WARN[0000] Found orphan containers ([inference-wafer-analytics-1]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
[+] Running 1/1
 ⠿ Container inference-wafer-insight-1  Recreated                                                           0.1s
Attaching to inference-wafer-insight-1
inference-wafer-insight-1  | [[-5.39543860e-04  2.39971569e-03 -3.42210731e-04 ... -2.35041980e-03
inference-wafer-insight-1  |   -1.81397056e-04 -2.09303234e-03]
inference-wafer-insight-1  |  [-1.00075542e-04 -5.41824409e-04 -2.38435358e-04 ...  3.39901582e-03
inference-wafer-insight-1  |    3.35075678e-04  2.04678475e-03]
inference-wafer-insight-1  |  [-5.14076633e-04 -2.28770984e-03  3.52836617e-04 ... -3.59841471e-03
inference-wafer-insight-1  |   -2.57484490e-03  5.23169035e-04]
inference-wafer-insight-1  |  ...
inference-wafer-insight-1  |  [-3.13805323e-03 -3.16870576e-03  1.28447995e-03 ... -8.94258047e-05
inference-wafer-insight-1  |    8.13668371e-04 -5.02239567e-04]
inference-wafer-insight-1  |  [-7.28863425e-04  2.32030465e-03  1.57134892e-03 ...  2.64884040e-04
inference-wafer-insight-1  |   -2.12739801e-03 -1.98500740e-04]
inference-wafer-insight-1  |  [-1.79534321e-03  6.97006847e-04  4.70415219e-04 ... -4.21349858e-04
inference-wafer-insight-1  |    2.88895727e-03  4.20368128e-04]]
inference-wafer-insight-1  |    fcol`feature_0  fcol`feature_1  ...  fcol`feature_1999              TEST_END_DATE
inference-wafer-insight-1  | 0       -0.000540        0.002400  ...          -0.002093 2022-06-24 17:57:44.060832
inference-wafer-insight-1  | 1       -0.000100       -0.000542  ...           0.002047 2022-06-24 18:02:55.100832
inference-wafer-insight-1  | 2       -0.000514       -0.002288  ...           0.000523 2022-06-24 18:08:06.140832
inference-wafer-insight-1  | 3       -0.000020       -0.003073  ...           0.001036 2022-06-24 18:13:17.180832
inference-wafer-insight-1  | 4       -0.001280        0.001955  ...          -0.000343 2022-06-24 18:18:28.220832
inference-wafer-insight-1  | 
inference-wafer-insight-1  | [5 rows x 2001 columns]
inference-wafer-insight-1  | started_stacking
inference-wafer-insight-1  |            LOT7  WAFER3 PROCESS  ...    MEDIAN DEVREVSTEP TESTNAME`STRUCTURE_NAME
inference-wafer-insight-1  | 0  DG0000000001       0    1234  ... -0.000540      DPMLD          fcol`feature_0
inference-wafer-insight-1  | 1  DG0000000001       1    1234  ... -0.000100      DPMLD          fcol`feature_0
inference-wafer-insight-1  | 2  DG0000000001       2    1234  ... -0.000514      DPMLD          fcol`feature_0
inference-wafer-insight-1  | 3  DG0000000001       3    1234  ... -0.000020      DPMLD          fcol`feature_0
inference-wafer-insight-1  | 4  DG0000000001       4    1234  ... -0.001280      DPMLD          fcol`feature_0
inference-wafer-insight-1  | 
inference-wafer-insight-1  | [5 rows x 10 columns]
inference-wafer-insight-1  | 
inference-wafer-insight-1  | Dash is running on http://0.0.0.0:8050/
inference-wafer-insight-1  | 
inference-wafer-insight-1  |  * Serving Flask app 'app'
inference-wafer-insight-1  |  * Debug mode: on
```

### **Bare Metal** 
Below setup and how-to-run sessions are for users who want to use the bare metal environment.
For docker environment, please go to [Docker](#docker).
#### Setup 
First, set up the environment with conda using:
```
conda create -n WI 
conda activate WI
pip install dash scikit-learn pandas pyarrow colorlover
```
#### How to Run 
To generate synthetic data for testing from the root directory:
```
cd src/loaders/synthetic_loader
python loader.py
```
To run the dashboard:
```
export PYTHONPATH=$PYTHONPATH:$PWD
python src/dashboard/app.py
```
The default dashboard URL is: http://0.0.0.0:8050/

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation.   
| **Name**  | Description                                          |
| --------- | ---------------------------------------------------- |
| CPU       | Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz (96 vCPUs) |
| Free RAM  | 367 GiB/376 GiB                                      |
| Disk Size | 2 TB                                                 | 

**Note:  The code was developed and tested on a machine with this configuration. However, it may be sufficient to use a machine that is much less powerful than the recommended configuration.**

## Useful Resources
[Intel AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)<br>
[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)<br>

## Support
[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)<br>
