BATCH_SIZE ?= 32
DATASET_DIR ?= /workspace/data
FINAL_IMAGE_NAME ?= vision-transfer-learning
NAMESPACE ?= argo
NUM_EPOCHS ?= 100
OUTPUT_DIR ?= /output
PLATFORM ?= None
PRECISION ?= FP32
SCRIPT ?= colorectal

vision-transfer-learning:
	@BATCH_SIZE=${BATCH_SIZE} \
	 DATASET_DIR=${DATASET_DIR} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 NUM_EPOCHS=${NUM_EPOCHS} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 PLATFORM=${PLATFORM} \
	 PRECISION=${PRECISION} \
	 SCRIPT=${SCRIPT} \
	 docker compose up vision-transfer-learning --build

argo-single-node: 
	helm install \
	--namespace ${NAMESPACE} \
	--set proxy=${http_proxy} \
	--set workflow.batch_size=${BATCH_SIZE} \
	--set workflow.num_epochs=${NUM_EPOCHS} \
	--set workflow.platform=${PLATFORM} \
	--set workflow.precision=${PRECISION} \
	--set workflow.dataset_dir=${DATASET_DIR} \
	--set workflow.script=${SCRIPT} \
	${FINAL_IMAGE_NAME} ./chart
	argo submit --from wftmpl/${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}

workflow-log:
	argo logs @latest -f -c output-log

clean: 
	docker compose down

helm-clean: 
	kubectl delete wftmpl ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
	helm uninstall ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
