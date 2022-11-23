AZURE_CONFIG_FILE ?= $$(pwd)/config.json
FINAL_IMAGE_NAME ?= nlp-azure
FP32_TRAINED_MODEL ?= $$(pwd)/../training/azureml/notebooks/fp32_model_output

nlp-azure:
	mkdir -p ./azureml/notebooks/fp32_model_output && cp -r ${FP32_TRAINED_MODEL} ./azureml/notebooks/
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	AZURE_CONFIG_FILE=${AZURE_CONFIG_FILE} \
	docker compose up nlp-azure --build

clean:
	docker compose down
	rm -rf ./azureml/notebooks/fp32_model_output
