DATASET ?= sst2
DATASET_DIR ?= /data
FINAL_IMAGE_NAME ?= document-level-sentiment-analysis
MODEL ?= bert-large-uncased
NAMESPACE ?= argo
NUM_NODES ?= 2
OUTPUT_DIR ?= /output
PROCESS_PER_NODE ?= 2

hugging-face-dlsa:
	mkdir ./dlsa/profiling-transformers/datasets && cp -r ${DATASET_DIR} ./dlsa/profiling-transformers/datasets
	@DATASET=${DATASET} \
	 FINAL_IMAGE_NAME=${FINAL_IMAGE_NAME} \
	 MODEL=${MODEL} \
	 NUM_NODES=${NUM_NODES} \
	 OUTPUT_DIR=${OUTPUT_DIR} \
	 PROCESS_PER_NODE=${PROCESS_PER_NODE} \
 	 docker compose up hugging-face-dlsa --build
	rm -rf ./dlsa/profiling-transformers/datasets

argo-single-node:
	helm install \
	--namespace ${NAMESPACE} \
	--set proxy=${http_proxy} \
	--set workflow.dataset=${DATASET} \
	--set workflow.num_nodes=${NUM_NODES} \
	--set workflow.model=${MODEL} \
	--set workflow.process_per_node=${PROCESS_PER_NODE} \
	${FINAL_IMAGE_NAME} ./chart
	argo submit --from wftmpl/${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}

workflow-log:
	argo logs @latest -f

clean: 
	rm -rf ./dlsa/profiling-transformers/datasets
	docker compose down

helm-clean: 
	kubectl delete wftmpl ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
	helm uninstall ${FINAL_IMAGE_NAME} --namespace=${NAMESPACE}
