DATASET_DIR ?= /dataset
FINAL_IMAGE_NAME ?= friesian-training
MODEL_OUTPUT ?= /model_output

friesian-training:
	wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
	tar -xvzf dac_sample.tar.gz
	mkdir -p ${DATASET_DIR}/data-csv
	mv dac_sample.txt ${DATASET_DIR}/data-csv/day_0.csv
	rm dac_sample.tar.gz
	@DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL_OUTPUT=${MODEL_OUTPUT} \
 	docker compose up friesian-training --build

clean: 
	@DATASET_DIR=${DATASET_DIR} \
	 OUTPUT_DIR=${MODEL_OUTPUT} \
	docker compose down
