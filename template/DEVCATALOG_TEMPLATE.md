# Workflow Title

Start with a short (~300 characters) teaser description of what this workflow
does, useful as the summary in the dev catalog. Use "active" language. For
example,

Learn to use Intel's XPU hardware and Intel optimized software for
distributed training and inference on the Azure Machine Learning Platform
with PyTorch\*, Hugging Face, and Intel® Neural Compressor.

or

Run a video streamer pipeline that mimics real-time video analysis. Take in
real-time data, send it to an endpoint for single-shot object detection, and
store the resulting metadata for further review.

Provide a link back to the dev catalog, for example:

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
Provide high-level information that would help a developer understand why this
workflow might be relevant to them, the **benefits** of the features or
functions showcased (don't just list features), and what they'll learn by trying
this workflow. A bullet list could work well here.

Here are some general authoring guidelines throughout the document:
- Define or explain all acronyms on first use.  If new abbreviations or acronyms
  are used within the diagram's text, explain them in the accompanying
  explanation.
- Make sure diagrams and explanations are consistent (e.g., they use the same
  terminology and flow).
- Text in diagrams must be readable (contrast and size). Provide a larger image
  that's scaled to fit the page so it can be clicked on to see the larger
  version or break an image into separate readable images.
- Use proper Intel legal product names and acknowledgment of third-party
  trademarks with an asterisk (on first use): Intel® Extension for PyTorch\*
  and not IPEX.
- Add comments in example script command/code blocks when it can better explain
  or clarify what's being done.
- Try to eliminate using ``<value>`` placeholders in the template when you write
  the workflow instructions and code examples.  If a value is used more than
  once, it's also a good idea to set an environment variable to the needed value
  and use that throughout the instructions.

End the overview with a link to the workflow's main GitHub repo, for example:

For more details, visit the [Cloud Training and Cloud Inference on Amazon
Sagemaker/Elastic Kubernetes
Service](https://github.com/intel/NLP-Workflow-with-AWS) GitHub repository.

## Hardware Requirements
There are workflow-specific hardware and software setup requirements depending on
how the workflow is run. Bare metal development system and Docker\* image running
locally have the same system requirements. Specify those reqirements, such as

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors|BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |

If Docker runs on a cloud service, specify cloud service requirements.

## How it Works
Explain how the workflow does what it does, including its inputs, processing, and outputs.
Provide a simple architecture or data flow diagram, and additional diagram(s)
with more details if useful.

Mention tuning opportunities and how the developer can interact with or alter
the workflow.

## Get Started

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Main
Repository](<Link to Main GitHub Repository>) repository into your working
directory.

```
# For example...
mkdir ~/work && cd ~/work
git clone https://github.com/intel/workflow-repo.git
cd <workflow repo name>
git checkout <branch>
```

### Download the Datasets
Describe what datasets are used for input to this workflow, and how to download
them if they're not included in the workflow repo or in the Docker image. If the
datasets are particularly large, indicate storage space needed for download and
extraction.

```
cd <datasets parent folder>
mkdir <recommended dataset folder name>
<download datasets using wget, curl, rsync, etc. to dataset folder>
cd ..
```

What information can we document if a developer wants to provide their own
dataset as input? Remind them to put their data into the ``datasets`` directory
we created.

---

## Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
instructions.

If possible, provide an estimate of time to set up and run the workflow
using Docker on the recommended hardware.

### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

If the Docker image is run on a cloud service, mention they may also need
credentials to perform training and inference related operations (such as these
for Azure):
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)

### Set Up Docker Image
Pull the provided docker image.
```
docker pull <Docker image name>
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
Run the workflow using the ``docker run`` command, as shown:  (example)
```
export DATASET_DIR=<path to dataset>
export OUTPUT_DIR=/output
docker run -a stdout $DOCKER_RUN_ENVS \
  --env DATASET=${DATASET} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:/output \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm --pull always \
  intel/ai-workflows:<workflow image name> \
  ./run.sh
```

---

## Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running a provided Docker image with Docker, see the [Docker
instructions](#run-using-docker).

If possible, provide an estimate of time to set up and run
the workflow on bare metal (with recommended HW).

### Set Up System Software
Our examples use the ``conda`` package and enviroment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Mention that other required software is installed by a provided installation script
or if not, provide instructions for installing required software packages and
libraries, along with expected versions of each.

### Set Up Workflow
Run these commands to set up the workflow's conda environment and install required software:
```
cd <working directory>
conda create -n dlsa python=3.8 --yes
conda activate dlsa
sh install.sh
```

### Run Workflow
Use these commands to run the workflow:
```
<bash shell commands>
```

## Expected Output
Explain what a successful execution looks like and where you'll find artifacts
created by analysis or inference from the run (if any).

## Summary and Next Steps
Explain what they've successfully done and what they could try next with this
workflow.  For example, are there different tuning knobs they could try that
would show different results?

## Learn More
For more information about <workflow> or to read about other relevant workflow
examples, see these guides and software resources:

- Put ref links and descriptions here, for example
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- links to other similar or related items from the dev catalog

## Troubleshooting
Document known issues or problem spots, and if possible, workarounds.

## Support
If you have questions or issues about this workflow, contact the [Support Team](support_forum_link).
If there is no support forum, and we want developers to use GitHub issues to submit bugs and enhancement
requests, put a link to that GitHub repo's issues, something like this:

The End-to-end Document Level Sentiment Analysis team tracks both bugs and
enhancement requests using [GitHub
issues](https://github.com/intel/document-level-sentiment-analysis/issues).
Before submitting a suggestion or bug report, search the [DLSA GitHub
issues](https://github.com/intel/document-level-sentiment-analysis/issues) to
see if your issue has already been reported.
