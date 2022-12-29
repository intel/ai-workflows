DATASET_DIR ?= ./data
FINAL_IMAGE_NAME ?= recommendation-ray
OUTPUT_DIR ?= /output
RUN_MODE ?= kaggle
DOCKER_NETWORK_NAME = ray-training

recommendation-ray:
	if [ ! -d "AIOK_Ray/dlrm_all/dlrm/dlrm" ]; then \
		CWD=${PWD}; \
		cd AIOK_Ray/; \
		sh dlrm_all/dlrm/patch_dlrm.sh; \
		cd ${CWD}; \
	fi
	@wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
	-P AIOK_Ray/Dockerfile-ubuntu18.04/ \
	-O AIOK_Ray/Dockerfile-ubuntu18.04/miniconda.sh
	if [ ! "$(shell docker network ls | grep ${DOCKER_NETWORK_NAME})" ]; then \
		docker network create --driver=bridge ${DOCKER_NETWORK_NAME}; \
	fi
	@DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 RUN_MODE=${RUN_MODE} \
 	 docker compose up recommendation-ray --build

clean: 
	docker network rm ${DOCKER_NETWORK_NAME}
	OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET_DIR} docker compose down
	sudo rm -rf ${OUTPUT_DIR}
