# **Video Streamer**

Run a video streamer pipeline that mimics real-time video analysis. Take in real-time data, send it to an endpoint for single-shot object detection, and store the resulting metadata for further review.

Check out more workflow examples and reference implementations in the [Dev Catalog](https://developer.intel.com/aireferenceimplementations)

## Overview

This video streamer pipeline shows you how you can:
* Improve performance for a real time video analytics flow on Intel platforms using Intel optimized software 
* Containerize the reference workflow for easy production level deployment  
* Improve video pre- and post-pocessing on both Intel CPU and GPU
* Improve AI inference speed on both an Intel CPU and GPU

For more details, visit the [Video Streamer](https://github.com/intel/video-streamer)) GitHub repository.

## Hardware Requirements
Bare metal development system and Docker* image running locally have the same system requirements. Specify those reqirements, such as

| Recommended Hardware	| Precision |
| ---- | ---- |
| * Intel® 4th Gen Xeon® Scalable Performance processors |	INT8 |
| * Intel® 4th Gen Xeon® Scalable Performance processors |	BF16 |

## How it Works

* A Gstreamer Pipeline based multimedia framework.
  Gstreamer elements are chained, to create a pipeline where Gstreamer handles the flow of metadata associated with the media.

* Uses TensorFlow for Inference. Inference is implemented as a plugin of Gstreamer.

* OpenCV Image preprocessing (normalization, resize) and drawing Bounding box, labelling
* VDMS stores uploaded metadata to database
* The workflow uses BF16/INT8 precision in SPR, which speeds up the inference time using Intel® AMX, without noticeable loss in accuracy when compared to FP32 precision (using Intel® AVX-512).

Video Streamer Data Flow
![video-pipeline](https://user-images.githubusercontent.com/43555799/205149596-f5054457-ef29-46ba-82e2-a979828d2754.png)

## Get Started

### Download the Workflow Repository
Create a working directory for the workflow and clone the  
[Video Streamer](https://github.com/intel/video-streamer) repository into your working
directory.

```
git clone https://github.com/intel/video-streamer
cd video-streamer
git checkout v1.0.0
```
### Download the Datasets and Models
```
Download a 13MB video file from Intel® IoT DevKit github
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4 -O classroom.mp4
export VIDEO=$(basename $(pwd)/classroom.mp4)
mkdir models
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb -P models
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb -P models
```
### Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.

#### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.


#### Set Up Docker Image
```
docker pull vuiseng9/intellabs-vdms:demo-191220
docker pull intel/ai-workflows:video-streamer
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

#### Run Docker Image

To run the pipeline, follow these instructions outside of the Docker instance.

1) Initiate the VDMS inference endpoint.

   ```
   numactl --physcpubind=0 docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
   ```

2) Initiate the Video-Streamer service.

   ```
   #Create output directory to store results of inference
   export OUTPUT_DIR=/output
   #Run the quick start script using the docker image
   docker run -a stdout $DOCKER_RUN_ENVS \
     --env VIDEO_FILE=/workspace/video-streamer/${VIDEO}  \
     --rm -it --privileged --net=host -p 55555:55555  \
     --volume $(pwd):/workspace/video-streamer  \
     --volume ${OUTPUT_DIR}:${OUTPUT_DIR}  \
     -w /workspace/video-streamer  \
     intel/ai-workflows:video-streamer  \
     /bin/bash ./benchmark.sh && cp -r ../*.txt ${OUTPUT_DIR}
   ```



### Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running a provided Docker image with Docker, see the [Docker
instructions](#run-using-docker).

#### Set Up System Software
Our examples use the ``conda`` package and enviroment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

#### Set Up Workflow
Go to the directory where you cloned the repo and follow these commands to install required software.

```
cd video-streamer
```
##### 1. Video and AI Setup
1) Edit `install.sh` for `mesa-libGL` install  
   In `install.sh`, default command `sudo yum install -y mesa-libGL` is for CentOS. For Ubuntu, change as follows
   ```
   #sudo yum install -y mesa-libGL
   sudo apt install libgl1-mesa-glx
   ```

2) Run the following install script

   create conda environment `vdms-test`
   ```
   conda create -n vdms-test python=3.8
   ```
   activate `vdms-test` then run install
   ```
   conda activate vdms-test
   ./install.sh
   ```

By default, this will install intel-tensorflow-avx512.  If it is necessary to run the workflow using a specific TensorFlow, please update it in `requirements.txt`

##### 2. VDMS Database Setup
* VDMS instance for database uses docker.  
Pull Docker Images
```
docker pull vuiseng9/intellabs-vdms:demo-191220
```

##### 3. Configuration

1) Edit config/pipeline-settings for pipeline setting

   Modify the parameter `gst_plugin_dir` and `video_path` to fit your Gstreamer plugin directory and input video path.

   ```
   mv classroom.mp4 dataset/classroom.mp4
   ```

   For example, we have
   -  `classroom.mp4` in `dataset` folder and
   -  gstreamer installed in `/home/test_usr/miniconda3/envs/vdms-test`. So settings should be:
   ```
   video_path=dataset/classroom.mp4
   gst_plugin_dir=/home/test_usr/miniconda3/envs/vdms-test/lib/gstreamer-1.0
   ```
2) Edit `config/settings.yaml` for inference setting
   - Customize to choose `data_type` from `FP32`, `AMPBF16` and `INT8` for inference. `INT8` is recommended for better performance.

3) CPU Optimization settings are found in two files:

   1) `config/pipeline-settings`
      - `cores_per_pipeline` controls the number of CPU cores to run in the whole pipeline.

   2)  `config/settings.yaml`
   ```
     inter_op_parallelism : "2"  #the number of threads used by independent non-blocking operations in TensorFlow.
     intra_op_parallelism : "4"  #execution of an individual operation can be parallelized on a pool of threads in TensorFlow.
   ```

#### Run Workflow

1) Initiate the VDMS inference endpoint.

   ```
   numactl --physcpubind=0 --membind=1 docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
   ```

2) Start video AI workflow

   ```
   ./run.sh 1
   ```

   `run.sh` is configured to accept a single input parameter which defines how many separate instances of the gstreamer pipelines to run. Each OpenMP thread from a given   instance is pinned to a physical CPU core. For example, when running four pipelines with OMP_NUM_THREADS=4 by configure `config/pipeline-settings`:

   ```
   cores_per_pipeline = 4
   ```

   |*Pipeline*|*Cores*|*Memory*|
   | ---- | ---- | ---- |
   |1| 0-3| Local |
   |2| 4-7| Local |
   |3| 8-11| Local |
   |4|12-15| Local |

   It is very important that the pipelines do not overlap numa domains or any other hardware non-uniformity. These values must be updated for each core architecture to get optimum performance.

   To launch the workload using a single instance, use the following command:
   ```
   ./run.sh 1
   ```

   To launch 14 instances with 4 cores per instance on a dual socket Xeon 8280, just run
   ```
   ./run.sh 14
   ```


### Expected Output

...

```
cpu-video-streamer-1  |  
cpu-video-streamer-1  |         _     _              __ _                                      
cpu-video-streamer-1  |  /\   /(_) __| | ___  ___   / _\ |_ _ __ ___  __ _ _ __ ___   ___ _ __ 
cpu-video-streamer-1  |  \ \ / / |/ _` |/ _ \/ _ \  \ \| __| '__/ _ \/ _` | '_ ` _ \ / _ \ '__|
cpu-video-streamer-1  |   \ V /| | (_| |  __/ (_) | _\ \ |_| | |  __/ (_| | | | | | |  __/ |   
cpu-video-streamer-1  |    \_/ |_|\__,_|\___|\___/  \__/\__|_|  \___|\__,_|_| |_| |_|\___|_|  
cpu-video-streamer-1  | 
cpu-video-streamer-1  |  Intel optimized video streaming pipeline based on GSteamer and Tensorflow
cpu-video-streamer-1  | 
cpu-video-streamer-1  |  /root/conda/envs/vdms-test/lib/gstreamer-1.0:/workspace/vdms-streamer/gst-plugin
cpu-video-streamer-1  | 
cpu-video-streamer-1  | (gst-plugin-scanner:109): GStreamer-WARNING **: 17:44:22.414: Failed to load plugin '/root/conda/envs/vdms-test/lib/gstreamer-1.0/libgstmpg123.so': libmpg123.so.0: cannot open shared object file: No such file or directory
cpu-video-streamer-1  | 
cpu-video-streamer-1  | (gst-plugin-scanner:109): GStreamer-WARNING **: 17:44:22.420: Failed to load plugin '/root/conda/envs/vdms-test/lib/gstreamer-1.0/libgstximagesrc.so': libXdamage.so.1: cannot open shared object file: No such file or directory
cpu-video-streamer-1  | 
cpu-video-streamer-1  | (gst-plugin-scanner:109): GStreamer-WARNING **: 17:44:22.436: Failed to load plugin '/root/conda/envs/vdms-test/lib/gstreamer-1.0/libgstmpg123.so': libmpg123.so.0: cannot open shared object file: No such file or directory
cpu-video-streamer-1  | 
cpu-video-streamer-1  | (gst-plugin-scanner:109): GStreamer-WARNING **: 17:44:22.437: Failed to load plugin '/root/conda/envs/vdms-test/lib/gstreamer-1.0/libgstximagesrc.so': libXdamage.so.1: cannot open shared object file: No such file or directory
cpu-video-streamer-1  | + set +x
cpu-video-streamer-1  | + numactl --physcpubind=0-3 --localalloc gst-launch-1.0 filesrc location=/workspace/video-streamer/classroom.mp4 '!' decodebin '!' videoconvert '!' video/x-raw,format=RGB '!' videoconvert '!' queue '!' gst_detection_tf conf=config/settings.yaml '!' fakesink
cpu-video-streamer-1  | 
cpu-video-streamer-1  | (gst-launch-1.0:174): GStreamer-CRITICAL **: 17:44:24.178: The created element should be floating, this is probably caused by faulty bindings
cpu-video-streamer-1  | INFO:gst_detection_tf:Loading model: models/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
cpu-video-streamer-1  | 2022-07-15 17:44:24.914324: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512_VNNI
cpu-video-streamer-1  | To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
cpu-video-streamer-1  | INFO:gst_detection_tf:Parameters: {'device': 'CPU', 'preproc_fw': 'cv2', 'data_type': 'FP32', 'onednn': True, 'amx': True, 'inter_op_parallelism': '1', 'intra_op_parallelism': '4', 'database': 'VDMS', 'bounding_box': True, 'face_threshold': 0.7, 'label_file': 'dataset/coco.label', 'ssd_resnet34_fp32_model': 'models/ssd_resnet34_fp32_bs1_pretrained_model.pb', 'ssd_resnet34_bf16_model': 'models/ssd_resnet34_bf16_bs1_pretrained_model.pb', 'ssd_resnet34_int8_model': 'models/ssd_resnet34_int8_bs1_pretrained_model.pb', 'ssd_resnet34_fp16_model': 'models/ssd_resnet34_fp16_bs1_pretrained_model.pb', 'ssd_resnet34_fp32_1200x1200_model': 'models/ssd_resnet34_fp32_1200x1200_pretrained_model.pb', 'ssd_resnet34_int8_1200x1200_model': 'models/ssd_resnet34_int8_1200x1200_pretrained_model.pb', 'ssd_resnet34_bf16_1200x1200_model': 'models/ssd_resnet34_bf16_1200x1200_pretrained_model.pb', 'ssd_resnet34_fp16_gpu_model': 'models/gpu/resnet34_tf.22.1.pb', 'ssd_resnet34_fp32_gpu_model': 'models/gpu/resnet34_tf.22.1.pb'}
cpu-video-streamer-1  | Setting pipeline to PAUSED ...
cpu-video-streamer-1  | Pipeline is PREROLLING ...
cpu-video-streamer-1  | Redistribute latency...
cpu-video-streamer-1  | Redistribute latency...
cpu-video-streamer-1  | 2022-07-15 17:44:25.041257: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
cpu-video-streamer-1  | Pipeline is PREROLLED ...
cpu-video-streamer-1  | Setting pipeline to PLAYING ...
cpu-video-streamer-1  | New clock: GstSystemClock
```
...
```
Got EOS from element "pipeline0".
Execution ended after 1:18:36.476950000
Setting pipeline to NULL ...
INFO:gst_detection_tf:Pipeline completes: {'total': 4719.140006303787, 'tf': 4657.993880748749, 'cv': 46.518736600875854, 'np': 1.4765057563781738, 'py': 0.05692410469055176, 'vdms': 0.7926304340362549, 'tf/load_model': 0.4810307025909424, 'np/gst_buf_to_ndarray': 0.5167484283447266, 'cv/normalize': 30.608948469161987, 'cv/resize': 14.067937850952148, 'np/format_image': 0.36579012870788574, 'e2e/preprocess': 56.772953033447266, 'tf/inference': 4657.512850046158, 'np/process_inference_result': 0.5939671993255615, 'py/build_db_data': 0.05692410469055176, 'cv/bound_box': 1.8418502807617188, 'e2e/postprocess': 3.49407958984375, 'vdms/save2db': 0.7926304340362549, 'frames': 8344}
Finished all pipelines
```
## Summary and Next Steps
Run your video analysis workloads on the recommended hardware and use the the provided docker images to get performance improvement from Intel Optimized software.

## Learn More
For more information about <workflow> or to read about other relevant workflow
examples, see these guides and software resources:
[Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

## Support
Video Streamer tracks both bugs and enhancement requests using [Github issues](https://github.com/intel/video-streamer/issues). Search these [GitHub issues](https://github.com/intel/video-streamer/issues) before filing a request or bug to see if it has already been reported.
