# **BigDL PPML on SGX**

## Overview

[PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) (Privacy Preserving Machine Learning) in [BigDL 2.0](https://github.com/intel-analytics/BigDL) provides a Trusted Cluster Environment for secure Big Data & AI applications, even in an untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning, homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

## How it Works

PPML ensures security for all dimensions of the data lifecycle: data at rest, data in transit, and data in use. Data being transferred on a network is `in transit`, data in storage is `at rest`, and data being processed is `in use`.

![Data Lifecycle](https://user-images.githubusercontent.com/61072813/177720405-60297d62-d186-4633-8b5f-ff4876cc96d6.png)

PPML protects compute and memory by SGX Enclaves, storage (e.g., data and model) by encryption, network communication by remote attestation and Transport Layer Security (TLS), and optional Federated Learning support.

![BigDL PPML](https://user-images.githubusercontent.com/61072813/177922914-f670111c-e174-40d2-b95a-aafe92485024.png)

With BigDL PPML, you can run trusted Big Data & AI applications
- **Trusted Spark SQL & Dataframe**: with trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) in a secure and trusted fashion.
- **Trusted ML (Machine Learning)**: with trusted Big Data analytics and ML/DL support, users can run distributed machine learning (such as MLlib, XGBoost etc.) in a secure and trusted fashion.
- **Trusted DL (Deep Learning)**: with trusted Big Data analytics and ML/DL support, users can run distributed deep learning (such as BigDL, Orca, Nano, DLlib etc.) in a secure and trusted fashion.
- **Trusted FL (Federated Learning)**: with PSI (Private Set Intersection), Secured Aggregation and trusted federated learning support, users can build united models across different parties without compromising privacy, even if these parties have different datasets or features.

## Get Started

### BigDL PPML End-to-End Workflow
![image](https://user-images.githubusercontent.com/61072813/178393982-929548b9-1c4e-4809-a628-10fafad69628.png)
In this section, we take SimpleQuery as an example to go through the entire BigDL PPML end-to-end workflow. SimpleQuery is a simple example to query developers between the ages of 20 and 40 from people.csv. 

#### Step 0. Preparation your environment

Prepare your environment first, including K8s cluster setup, K8s-SGX plugin setup, key/password preparation, key management service (KMS) and attestation service (AS) setup, BigDL PPML client container preparation. **Please follow the detailed steps in** [Prepare Environment](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md). 

Next, you are going to build a base image, and a custom image on top of it to avoid leaving secrets e.g. enclave key in images/containers. After that, you need to register the `MRENCLAVE` in your customer image to Attestation Service Before running your application, and PPML will verify the runtime MREnclave automatically at the backend. The below chart illustrated the whole workflow:
![PPML Workflow with MREnclave](https://user-images.githubusercontent.com/60865256/197942436-7e40d40a-3759-49b4-aab1-826f09760ab1.png)

Start your application with the following guide step by step:

#### Step 1. Prepare your PPML image for the production environment

To build a secure PPML image for a production environment, BigDL prepared a public base image that does not contain any secrets. You can customize your image on top of this base image.

1. Prepare BigDL Base Image

    Users can pull the base image from dockerhub or build it by themselves. 

    Pull the base image
    ```bash
    docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-base:2.2.0-SNAPSHOT
    ```
    
2. Build Custom Image

    When the base image is ready, you need to generate your enclave key which will be used when building a custom image. Keep the enclave key safe for future remote attestations.

    Running the following command to generate the enclave key `enclave-key.pem`, which is used to launch and sign SGX Enclave. 

    ```bash
    cd bigdl-gramine
    openssl genrsa -3 -out enclave-key.pem 3072
    ```

    When the enclave key `enclave-key.pem` is generated, you are ready to build your custom image by running the following command: 

    ```bash
    # under bigdl-gramine dir
    # modify custom parameters in build-custom-image.sh
    ./build-custom-image.sh
    cd ..
    ```

    **Warning:** If you want to skip DCAP (Data Center Attestation Primitives) attestation in runtime containers, you can set `ENABLE_DCAP_ATTESTATION` to *false* in `build-custom-image.sh`, and this will generate a none-attestation image. **But never do this unsafe operation in production!**

    The sensitive enclave key will not be saved in the built image. Two values `mr_enclave` and `mr_signer` are recorded while the Enclave is built, you can find `mr_enclave` and `mr_signer` values in the console log, which are hash values and used to register your MREnclave in the following attestation step.

    ````bash
    [INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
    mr_enclave       : c7a8a42af......
    mr_signer        : 6f0627955......
    ````

    Note: you can also customize the image according to your own needs, e.g. install third-parity python libraries or jars.
    
    Then, start a client container:

    ```
    export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
    echo The k8s master is $K8S_MASTER .
    export DATA_PATH=/YOUR_DIR/data
    export KEYS_PATH=/YOUR_DIR/keys
    export SECURE_PASSWORD_PATH=/YOUR_DIR/password
    export KUBECONFIG_PATH=/YOUR_DIR/kubeconfig
    export LOCAL_IP=$LOCAL_IP
    export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT # or the custom image built by yourself

    sudo docker run -itd \
        --privileged \
        --net=host \
        --name=bigdl-ppml-client-k8s \
        --cpuset-cpus="0-4" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
        -v $KUBECONFIG_PATH:/root/.kube/config \
        -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
        -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
        -e LOCAL_IP=$LOCAL_IP \
        $DOCKER_IMAGE bash
    ```

#### Step 2. Encrypt and Upload Data

Encrypt the input data of your Big Data & AI applications (here we use SimpleQuery) and then upload encrypted data to the Network File System (NFS) server. More details in [Encrypt Your Data](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

1. Generate the input data `people.csv` for SimpleQuery application
you can use [generate_people_csv.py](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.

2. Encrypt `people.csv`
    ```
    docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $input_file_path"
    ```

#### Step 3. Build Big Data & AI applications

To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml). The code of SimpleQuery is in [here](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQuerySparkExample.scala), it is already built into bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar, and the jar is put into PPML image.

#### Step 4. SGX Attestation 

   Enter the client container:
   ```
   sudo docker exec -it bigdl-ppml-client-k8s bash
   ```
   
1. Disable attestation

    If you do not need the attestation, you can disable the attestation service. You should configure spark-driver-template.yaml and spark-executor-template.yaml to set `ATTESTATION` value to `false`. By default, the attestation service is disabled. 
    ``` yaml
    apiVersion: v1
    kind: Pod
    spec:
      ...
        env:
          - name: ATTESTATION
            value: false
      ...
    ```

2. Enable attestation

    The bi-attestation guarantees that the MREnclave in runtime containers is a secure one made by you. Its workflow is as below:
    ![image](https://user-images.githubusercontent.com/60865256/198168194-d62322f8-60a3-43d3-84b3-a76b57a58470.png)
    
    To enable attestation, you should have a running Attestation Service in your environment. 

    **2.1. Deploy EHSM KMS & AS**

      KMS (Key Management Service) and AS (Attestation Service) make sure applications of the customer run in the SGX MREnclave signed above by customer-self, rather than a fake one fake by an attacker.

      BigDL PPML uses EHSM as a reference KMS & AS, you can follow the guide [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#deploy-bigdl-ehsm-kms-on-kubernetes-with-helm-charts) to deploy EHSM in your environment.

    **2.2. Enroll in EHSM**

    Execute the following command to enroll yourself in EHSM, The `<kms_ip>` is your configured-ip of EHSM service in the deployment section:

    ```bash
    curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"
    ......
    {"code":200,"message":"successful","result":{"apikey":"E8QKpBB******","appid":"8d5dd3b*******"}}
    ```

    You will get a `appid` and `apikey` pair. Please save it for later use.

    **2.3. Attest EHSM Server (optional)**

    You can attest EHSM server and verify the service is trusted before running workloads to avoid sending your secrets to a fake service.

    To attest EHSM server, start a BigDL container using the custom image built before. **Note**: this is the other container different from the client.

    ```bash
    export KEYS_PATH=YOUR_LOCAL_SPARK_SSL_KEYS_FOLDER_PATH
    export LOCAL_IP=YOUR_LOCAL_IP
    export CUSTOM_IMAGE=YOUR_CUSTOM_IMAGE_BUILT_BEFORE
    export PCCS_URL=YOUR_PCCS_URL # format like https://1.2.3.4:xxxx, obtained from KMS services or a self-deployed one

    sudo docker run -itd \
        --privileged \
        --net=host \
        --cpuset-cpus="0-5" \
        --oom-kill-disable \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        --name=gramine-verify-worker \
        -e LOCAL_IP=$LOCAL_IP \
        -e PCCS_URL=$PCCS_URL \
        $CUSTOM_IMAGE bash
    ```

    Enter the docker container:

    ```bash
    sudo docker exec -it gramine-verify-worker bash
    ```

    Set the variables in `verify-attestation-service.sh` before running it:

      ```
      `ATTESTATION_URL`: URL of attestation service. Should match the format `<ip_address>:<port>`.

      `APP_ID`, `API_KEY`: The appID and apiKey pair generated by your attestation service.

      `ATTESTATION_TYPE`: Type of attestation service. Currently support `EHSMAttestationService`.

      `CHALLENGE`: Challenge to get quote of attestation service which will be verified by local SGX SDK. Should be a BASE64 string. It can be a casual BASE64 string, for example, it can be generated by the command `echo anystring|base64`.
      ```

    In the container, execute `verify-attestation-service.sh` to verify the attestation service quote.

      ```bash
      bash verify-attestation-service.sh
      ```

    **2.4. Register your MREnclave to EHSM**

    Register the MREnclave with metadata of your MREnclave (appid, apikey, mr_enclave, mr_signer) obtained in the above steps to EHSM through running a python script:

    ```bash
    # At /ppml/trusted-big-data-ml inside the container now
    python register-mrenclave.py --appid <your_appid> \
                                --apikey <your_apikey> \
                                --url https://<kms_ip>:9000 \
                                --mr_enclave <your_mrenclave_hash_value> \
                                --mr_signer <your_mrensigner_hash_value>
    ```
    You will receive a response containing a `policyID` and save it which will be used to attest runtime MREnclave when running distributed Kubernetes application.

    **2.5. Enable Attestation in configuration**

    First, upload `appid`, `apikey` and `policyID` obtained before to Kubernetes as secrets:
    
    ```bash
    kubectl create secret generic kms-secret \
                      --from-literal=app_id=YOUR_KMS_APP_ID \
                      --from-literal=api_key=YOUR_KMS_API_KEY \
                      --from-literal=policy_id=YOUR_POLICY_ID
    ```
    
    Configure `spark-driver-template.yaml` and `spark-executor-template.yaml` to enable Attestation as follows:
    ``` yaml
    apiVersion: v1
    kind: Pod
    spec:
      containers:
      - name: spark-driver
        securityContext:
          privileged: true
        env:
          - name: ATTESTATION
            value: true
          - name: PCCS_URL
            value: your_pccs_url  -----> <set_the_value_to_your_pccs_url>
          - name: ATTESTATION_URL
            value: your_attestation_url
          - name: APP_ID
            valueFrom:
              secretKeyRef:
                name: kms-secret
                key: app_id
          - name: API_KEY
            valueFrom:
              secretKeyRef:
                name: kms-secret
                key: app_key
          - name: ATTESTATION_POLICYID
            valueFrom:
              secretKeyRef:
                name: policy-id-secret
                key: policy_id
    ...
    ```
    You should get `Attestation Success!` in logs after you [submit a PPML job](#step-4-submit-job) if the quote generated with `user_report` is verified successfully by Attestation Service, or you will get `Attestation Fail! Application killed!` and the job will be stopped.

#### Step 5. Submit Job

When the Big Data & AI application and its input data is prepared, you are ready to submit BigDL PPML jobs. You need to choose the deploy mode and the way to submit job first.

* **There are 4 modes to submit job**:

    1. **local mode**: run jobs locally without connecting to a cluster. It is exactly the same as using spark-submit to run your application: `$SPARK_HOME/bin/spark-submit --class "SimpleApp" --master local[4] target.jar`, driver and executors are not protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703141-63209559-05e1-4c4d-b096-6b862a9bed8a.png" width='250px' />
        </p>


    2. **local SGX mode**: run jobs locally with SGX guarded. As the picture shows, the client JVM is running in a SGX Enclave so that driver and executors can be protected.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703165-2afc280d-6a3d-431d-9856-dd5b3659214a.png" width='250px' />
        </p>


    3. **client SGX mode**: run jobs in k8s client mode with SGX guarded. As we know, in K8s client mode, the driver is deployed locally as an external client to the cluster. With **client SGX mode**, the executors running in K8S cluster are protected by SGX, the driver running in client is also protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703216-70588315-7479-4b6c-9133-095104efc07d.png" width='500px' />
        </p>


    4. **cluster SGX mode**: run jobs in k8s cluster mode with SGX guarded. As we know, in K8s cluster mode, the driver is deployed on the k8s worker nodes like executors. With **cluster SGX mode**, the driver and executors running in K8S cluster are protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703234-e45b8fe5-9c61-4d17-93ef-6b0c961a2f95.png" width='500px' />
        </p>


* **There are two options to submit PPML jobs**:
    * use [PPML CLI](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli) to submit jobs manually
    * use [helm chart](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli#helm-chart) to submit jobs automatically

Here we use **k8s client mode** and **PPML CLI** to run SimpleQuery. Check other modes, please see [PPML CLI Usage Examples](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli#usage-examples). Alternatively, you can also use Helm to submit jobs automatically, see the details in [Helm Chart Usage](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli#helm-chart).

  1. enter the ppml container
      ```
      docker exec -it bigdl-ppml-client-k8s bash
      ```
  2. run simplequery on k8s client mode
      ```
      #!/bin/bash
      export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
      bash bigdl-ppml-submit.sh \
              --master $RUNTIME_SPARK_MASTER \
              --deploy-mode client \
              --sgx-enabled true \
              --sgx-driver-jvm-memory 12g \
              --sgx-executor-jvm-memory 12g \
              --driver-memory 32g \
              --driver-cores 8 \
              --executor-memory 32g \
              --executor-cores 8 \
              --num-executors 2 \
              --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
              --name simplequery \
              --verbose \
              --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
              --jars local:///ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar \
              local:///ppml/trusted-big-data-ml/work/data/simplequery/spark-encrypt-io-0.3.0-SNAPSHOT.jar \
              --inputPath /ppml/trusted-big-data-ml/work/data/simplequery/people_encrypted \
              --outputPath /ppml/trusted-big-data-ml/work/data/simplequery/people_encrypted_output \
              --inputPartitionNum 8 \
              --outputPartitionNum 8 \
              --inputEncryptModeValue AES/CBC/PKCS5Padding \
              --outputEncryptModeValue AES/CBC/PKCS5Padding \
              --primaryKeyPath /ppml/trusted-big-data-ml/work/data/simplequery/keys/primaryKey \
              --dataKeyPath /ppml/trusted-big-data-ml/work/data/simplequery/keys/dataKey \
              --kmsType EHSMKeyManagementService
              --kmsServerIP your_ehsm_kms_server_ip \
              --kmsServerPort your_ehsm_kms_server_port \
              --ehsmAPPID your_ehsm_kms_appid \
              --ehsmAPIKEY your_ehsm_kms_apikey
      ```


  3. check runtime status: exit the container or open a new terminal

      To check the logs of the Spark driver, run
      ```
      sudo kubectl logs $( sudo kubectl get pod | grep "simplequery.*-driver" -m 1 | cut -d " " -f1 )
      ```
      To check the logs of a Spark executor, run
      ```
      sudo kubectl logs $( sudo kubectl get pod | grep "simplequery-.*-exec" -m 1 | cut -d " " -f1 )
      ```
  
  4. If you setup [PPML Monitoring](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#optional-k8s-monitioring-setup), you can check PPML Dashboard to monitor the status at http://kubernetes_master_url:3000

#### Step 6. Monitor Job by History Server

You can monitor spark events using a history server. The history server provides an interface to watch and log spark performance and metrics.
     
First, create a shared directory that can be accessed by both the client and the other worker containers in your cluster. For example, you can create an empty directory under the mounted NFS path or HDFS. The spark drivers and executors will write their event logs to this destination, and the history server will read logs here as well.
     
Second, enter your client container and edit `$SPARK_HOME/conf/spark-defaults.conf`, where the histroy server reads the configurations:
```
spark.eventLog.enabled           true
spark.eventLog.dir               <your_shared_dir_path> ---> e.g. file://<your_nfs_dir_path> or hdfs://<your_hdfs_dir_path>
spark.history.fs.logDirectory    <your_shared_dir_path> ---> similiar to spark.eventLog.dir
```
     
Third, run the below command and the history server will start to watch automatically:
```
$SPARK_HOME/sbin/start-history-server.sh
```
     
Next, when you run spark jobs, enable writing driver and executor event logs in java/spark-submit commands by setting spark conf like below:
```
...
--conf spark.eventLog.enabled=true \
--conf spark.eventLog.dir=<your_shared_dir_path> \
...
```
     
Starting spark jobs, you can find event log files at `<your_shared_dir_path>` like:
```
$ ls
local-1666143241860 spark-application-1666144573580
     
$ cat spark-application-1666144573580
......
{"Event":"SparkListenerJobEnd","Job ID":0,"Completion Time":1666144848006,"Job Result":{"Result":"JobSucceeded"}}
{"Event":"SparkListenerApplicationEnd","Timestamp":1666144848021}
```
     
You can use these logs to analyze spark jobs. Moreover, you are also allowed to surf from a web UI provided by the history server by accessing `http://localhost:18080`:
![history server UI](https://user-images.githubusercontent.com/60865256/196840282-6584f36e-5e72-4144-921e-4536d3391f05.png)    


#### Step 7. Decrypt Results

When the job is done, you can decrypt and read the results of the job. More details in [Decrypt Job Result](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

  ```
  docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $apikey $input_path"
  ```

https://user-images.githubusercontent.com/61072813/184758643-821026c3-40e0-4d4c-bcd3-8a516c55fc01.mp4

## Recommended Hardware

The hardware below is recommended for use with this reference implementation.

Intel® 3th Gen Xeon® Scalable Performance processors or later

## Learn More

[BigDL PPML](https://github.com/intel-analytics/BigDL/tree/main/ppml)
[Tutorials](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/examples.html)

## Troubleshooting

1. Is SGX supported on CentOS 6/7?
No. Please upgrade your OS if possible.

2. Do we need Internet connection for SGX node?
No. We can use PCCS for registration and certificate download. Only PCCS need Internet connection.

3. Does PCCS require SGX?
No. PCCS can be installed on any server with Internet connection.

4. Can we turn off SGX attestation?
Of course. But, turning off attestation will break the integrity provided by SGX. Attestation is turned off to simplify installation for a quick start.

5. Does we need to rewrite my applications?
No. In most cases, you don't have to rewrite your applications

## Support Forum

- [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)
- [User Group](https://groups.google.com/forum/#!forum/bigdl-user-group)
- [Github Issues](https://github.com/intel-analytics/BigDL/issues)
---