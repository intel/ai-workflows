# Title

## Overview
Short description and provide link to [Main Repository](<Link to Main GitHub Repository>)

## How it Works
Put diagrams here and explain how it works and benefits. (Image links should be absolute URLs. 
For image sizing, use 100% width rather than absolute image size.)

## Get Started

### **Prerequisites**
#### Download the repo
Clone [Main Repository](<Link to Main GitHub Repository>) repository into your working directory.
```
git clone -b <ref> <e2e_repo.git> .
```
#### Download the datasets
`<Direction on Downloading Dataset>`

### **Docker**
Below setup and how-to-run sessions are for users who want to use provided docker image.  
For bare metal environment, please go to [bare metal session](#bare-metal).
#### Setup 

##### Pull Docker Image
```
docker pull <Docker image name>
```

#### How to run 

(Optional) Export related proxy into docker environment.
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \ 
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \ 
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \ 
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \ 
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \ 
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the pipeline, follow below instructions outside of docker instance. 
```
docker run \
  -a stdout $DOCKER_RUN_ENVS \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  --privileged --init -it \
  <Docker image name> \ 
  <Bash shell commands>
```

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment.  
For docker environment, please go to [docker session](#docker).
#### Setup 
```
<Setup commands>
```
#### How to run 


Follow below instructions in the bash shell environment.  
```
<Bash shell commands>
```

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation.   
`<Put recommended hardware here>`

## Useful Resources
`<Put ref links here if any>`

## Support
`<Put support forum here if any>`
