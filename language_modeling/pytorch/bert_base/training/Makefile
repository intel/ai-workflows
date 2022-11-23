AZURE_CONFIG_FILE ?= $$(pwd)/config.json
FINAL_IMAGE_NAME ?= nlp-azure

nlp-azure:
	AZURE_CONFIG_FILE=${AZURE_CONFIG_FILE} \
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	docker compose up nlp-azure --build
		
clean:
	docker compose down
