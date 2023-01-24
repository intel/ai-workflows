# **Video Streamer**
## Overview

The video streamer pipeline is designed to mimic real-time video analytics. Real-time data is provided to an inference endpoint that executes single-shot object detection. The metadata created during inference is then uploaded to a database for curation.

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
### **Prerequisites**
#### Download the repo
Clone [Video Streamer](https://github.com/intel/video-streamer) repository.
```
git clone https://github.com/intel/video-streamer
cd video-streamer
git checkout v1.0.0
```
#### Download the Video File and Models
```
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4 -O classroom.mp4
export VIDEO=$(basename $(pwd)/classroom.mp4)
mkdir models
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb -P models
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb -P models
```
### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).
#### Setup 

##### Pull Docker Images
```
docker pull vuiseng9/intellabs-vdms:demo-191220
docker pull intel/ai-workflows:video-streamer
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

* Initiate the VDMS inference endpoint.

```
numactl --physcpubind=0 docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
```

* Initiate the Video-Streamer service.

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

#### Output
```
#1 [internal] load build definition from Dockerfile.video-streamer
#1 transferring dockerfile: 47B done
#1 DONE 0.0s

#2 [internal] load .dockerignore
#2 transferring context: 2B done
#2 DONE 0.0s

#3 [internal] load metadata for docker.io/library/centos:8
#3 DONE 0.4s

#4 [1/9] FROM docker.io/library/centos:8@sha256:a27fd8080b517143cbbbab9dfb7c8571c40d67d534bbdee55bd6c473f432b177
#4 DONE 0.0s

#5 [internal] load build context
#5 transferring context: 6.25kB done
#5 DONE 0.0s

#6 [2/9] RUN sed -i.bak '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* &&     sed -i.bak 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* &&     yum distro-sync -y &&     yum --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos -y &&     yum distro-sync -y &&     yum clean all
#6 CACHED

#7 [3/9] RUN yum update -y && yum install -y     python38     python38-pip     which     wget numactl mesa-libGL net-tools &&     yum clean all
#7 CACHED

#8 [4/9] RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh &&     chmod +x miniconda.sh &&     ./miniconda.sh -b -p ~/conda &&     rm ./miniconda.sh &&     ~/conda/bin/conda create -yn vdms-test python=3.8 &&     export PATH=~/conda/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin &&     source activate vdms-test &&     python3 -m pip --no-cache-dir install --upgrade pip     opencv-python==4.5.5.64     protobuf==3.20.1     pyyaml     setuptools     vdms     wheel
#8 CACHED

#9 [5/9] RUN ln -sf $(which python3) /usr/local/bin/python &&     ln -sf $(which python3) /usr/local/bin/python3 &&     ln -sf $(which python3) /usr/bin/python
#9 CACHED

#9 [6/9] RUN source activate vdms-test &&     pip install tensorflow-cpu &&     conda install -y -c conda-forge gst-libav==1.18.4 gst-plugins-good=1.18.4 gst-plugins-bad=1.18.4 gst-plugins-ugly=1.18.4 gst-python=1.18.4 pygobject=3.40.1 &&     conda clean --all
#9 0.993 Collecting tensorflow-cpu
#9 1.087   Downloading tensorflow_cpu-2.10.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (214.4 MB)
#9 3.630      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.4/214.4 MB 10.7 MB/s eta 0:00:00
#9 4.771 Collecting grpcio<2.0,>=1.24.3
#9 4.787   Downloading grpcio-1.48.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
#9 4.844      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 86.2 MB/s eta 0:00:00
#9 4.857 Requirement already satisfied: numpy>=1.20 in /root/conda/envs/vdms-test/lib/python3.8/site-packages (from tensorflow-cpu) (1.23.3)
#9 4.877 Collecting absl-py>=1.0.0
#9 4.886   Downloading absl_py-1.2.0-py3-none-any.whl (123 kB)
#9 4.891      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 123.4/123.4 kB 41.0 MB/s eta 0:00:00
#9 4.923 Collecting keras<2.11,>=2.10.0
#9 4.936   Downloading keras-2.10.0-py2.py3-none-any.whl (1.7 MB)
#9 4.956      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 94.3 MB/s eta 0:00:00
#9 4.983 Collecting keras-preprocessing>=1.1.1
#9 4.992   Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
#9 4.996      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.6/42.6 kB 14.2 MB/s eta 0:00:00
#9 5.030 Collecting tensorboard<2.11,>=2.10
#9 5.038   Downloading tensorboard-2.10.0-py3-none-any.whl (5.9 MB)
#9 5.127      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.9/5.9 MB 68.6 MB/s eta 0:00:00
#9 5.165 Collecting opt-einsum>=2.3.2
#9 5.185   Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
#9 5.191      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 12.7 MB/s eta 0:00:00
#9 5.193 Requirement already satisfied: setuptools in /root/conda/envs/vdms-test/lib/python3.8/site-packages (from tensorflow-cpu) (65.3.0)
```
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

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 
Please go to the directory where you cloned the repo, follow commands to install required software.

```
cd video-streamer
```
##### 1. Video and AI Setup
* 1. Edit `install.sh` for `mesa-libGL` install  
In `install.sh`, default command `sudo yum install -y mesa-libGL` is for CentOS. For Ubuntu, change as follows
```
#sudo yum install -y mesa-libGL
sudo apt install libgl1-mesa-glx
```

* 2. Run the following install script

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

### Configuration

* 1. Edit config/pipeline-settings for pipeline setting

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
* 2. Edit `config/settings.yaml` for inference setting
   - Customize to choose `data_type` from `FP32`, `AMPBF16` and `INT8` for inference. `INT8` is recommended for better performance.

* 3. CPU Optimization settings are found in two files:

* 3.1) `config/pipeline-settings`
   - `cores_per_pipeline` controls the number of CPU cores to run in the whole pipeline.

* 3.2)  `config/settings.yaml`
```
  inter_op_parallelism : "2"  #the number of threads used by independent non-blocking operations in TensorFlow.
  intra_op_parallelism : "4"  #execution of an individual operation can be parallelized on a pool of threads in TensorFlow.
```

### How to Run

* 1. Initiate the VDMS inference endpoint.

```
numactl --physcpubind=0 --membind=1 docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
```

* 2. Start video AI workflow

```
./run.sh 1
```

`run.sh` is configured to accept a single input parameter which defines how many separate instances of the gstreamer pipelines to run. Each OpenMP thread from a given instance is pinned to a physical CPU core. For example, when running four pipelines with OMP_NUM_THREADS=4
by configure `config/pipeline-settings`:

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

# Recommended Hardware
The hardware below is recommended for use with this reference implementation.

| Recommended Hardware	| Precision |
| ---- | ---- |
| * Intel® 4th Gen Xeon® Scalable Performance processors |	INT8 |
| * Intel® 4th Gen Xeon® Scalable Performance processors |	BF16 |

# Useful Resources
[Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

# Support
Video Streamer tracks both bugs and enhancement requests using Github. We welcome input, however, before filing a request, please make sure you do the following: Search the Github issue database.
