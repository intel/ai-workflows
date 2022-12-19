.PHONY: protein-prediction
DATASET_DIR ?= /dataset
EXPERIMENT_NAME ?= testing
FINAL_IMAGE_NAME ?= protein-structure-prediction
MODEL ?= model_1
OUTPUT_DIR ?= /output

protein-prediction:
	mkdir -p '${OUTPUT_DIR}/weights/extracted' '${OUTPUT_DIR}/logs' '${OUTPUT_DIR}/samples' '${OUTPUT_DIR}/experiments/${EXPERIMENT_NAME}'
	curl -o ${OUTPUT_DIR}/samples/sample.fa https://rest.uniprot.org/uniprotkb/Q6UWK7.fasta
	@EXPERIMENT_NAME=${EXPERIMENT_NAME} \
	 DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL=${MODEL} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
 	docker compose up protein-prediction-inference --build

clean: 
	@DATASET_DIR=${DATASET_DIR} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	docker compose down
