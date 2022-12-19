FINAL_IMAGE_NAME ?= chronos

chronos:
	FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	docker compose up chronos --build

clean:
	docker compose down
