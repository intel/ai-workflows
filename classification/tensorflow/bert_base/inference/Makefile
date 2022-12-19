AWS_CSV_FILE ?= credentials.csv
AWS_DATA=$$(pwd)/aws_data
FINAL_IMAGE_NAME ?= nlp-sagemaker
OUTPUT_DIR ?= /output
ROLE ?= role
S3_MODEL_URI ?= link

export AWS_PROFILE := $(shell cat ${AWS_CSV_FILE} | awk -F',' 'NR==2{print $$1}')
export REGION ?= us-west-2

nlp-sagemaker:
	./aws_sagemaker/scripts/setup.sh aws_sagemaker/
	mkdir -p ${AWS_DATA} && cp -r ${HOME}/.aws ${AWS_DATA}/.aws/
	@AWS_PROFILE=${AWS_PROFILE} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 docker compose up --build nlp-sagemaker		
clean:
	if [ -d ${AWS_DATA} ]; then \
		rm -rf ${AWS_DATA}; \
	fi; \
	if [ -d aws/ ]; then \
		rm -rf aws/; \
	fi; \
	if [ -d aws-cli/ ]; then \
		rm -rf aws-cli/; \
	fi; \
	if [ -f awscliv2.zip ]; then \
		rm -f awscliv2.zip; \
	fi
	docker compose down
