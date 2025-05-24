<h2 style="text-align:center;"> MLOps Capstone Project: Email Spam Detector</h2>

<img title="Email Spam Detector Architecture Overview" src="src\image\mlops overview.png">

<h3>Project Description</h3>

<p>
The capstone project demonstrates the data and modelling pipelines built based on the main aspects of MLOps, such as modelling experimentation and tracking, model registry, workflow orchestration, model deployment and monitoring.

The problem statement of 
</p>

<h3>Problem Statement and Objective</h3>

<p>
Many spam emails still managed to get into my email inbox daily. The chore of having to manually report each one of them as spam is repetitive and time wasting. The goal is to create a spam detector to automate this process by treating the problem as a classification.
</p>

<h3>Steps Overview</h3>
<p>The order of the pipeline is as follows:</p>
<ol>
  <li>Data Collection</li>
  <li>Model Experimentation and Tracking, and Orchestration</li>
  <li>Model Deployment</li>
  <li>Monitoring and Orchestration</li>
</ol>

<h2>Pipeline Steps Execution</h2>
<h3>Data Collection</h3>
<img title="Data Collection" src="src\image\data collect.png">
<p>
Instead of building the API to fetch emails from my email client, I decided to simulate it using <a href="https://huggingface.co/datasets/Deysi/spam-detection-dataset" target="_blank">Deysi/spam-detection-dataset</a>. The dataset consists of 8,180 train samples and 2,730 test samples. A small subset from  both training and test datasets are used to train and test the model respectively. The test dataset subset also act as the reference data for data drift monitoring. Whenever unseen samples are needed, data is randomly fetched from the training dataset.
</p>

<h3>Model Experimentation and Tracking</h3>
<img title="Model Experimentation and Tracking" src="src\image\model exp.png">
<p>
I first started experimenting the solution on a Jupyter Notebook, <i>notebook/basic_modelling.ipynb</i>. Then refactored the notebook codes into <i>model_training/training.py</i>. Amazon EC2, Amazon RDS and Amazon S3 are setup to host the MLFlow tracking server, to store MLFlow metadata and artifacts respectively.

Once an Amazon EC2 instance is running with MLFlow installations done, run this command with all the variables substituted:
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://postgres:<password>@<aws_rds_hostname>:5432/mlflow_db --default-artifact-root s3://<s3_bucket_name>
```
The MLFlow UI will be available at `http://<ec2_public_address>:5000`
<img title="MLFlow UI" src="src\image\mlflow ui.png">

Once the MLFlow tracking server is ready, the training code can be ran. The overview of <i>model_training/training.py</i> is as such:
<ol>
  <li>Initialize MLFlow tracking URI and experiment name</li>
  <li>Get the training and test datasets</li>
  <li>Data preprocessing</li>
  <li>Model training and hyperparameter runing</li>
  <li>Model registry staging</li>
</ol>

To ease the modelling process, it is deployed to Prefect Cloud.
For Prefect Version 2.11.0, these are the steps to create a deployment:
<ol>
  <li>Make sure terminal directory is at root.</li>
  <li>Run <i>prefect init</i> and follow the UI instructions, I chose git</li>
  <li>Run <i>prefect deploy</i>  and follow the UI instructions</li>
  <li>Run <i>prefect worker start --pool 'mlops-capstone'</i> to start worker</li>
</ol>

Once the deployment is successful, a flow run can be executed from `Spam Detector Capstone/mlops-capstone-spam-detector`. <img title="Prefect Training Flow" src="src\image\training prefect.png">

At the end of the run, a model will be staged in MLFlow 
Model Registry.
<img title="MLFlow registry" src="src\image\mlflow registry.png">
</p>

<h3>Model Deployment</h3>
<img title="Model Deployment" src="src\image\deployment.png">
<p>
In this section, I created an inference script, <i>inference.py</i> with the following steps:
<ol>
  <li>Get unseen data, simulated by getting randomly from the training dataset</li>
  <li>Preprocess unseen data</li>
  <li>Initialize MLFlow tracking URI</li>
  <li>Load production model from MLFlow Model Registry</li>
  <li>Predict on the unseen data</li>
  <li>Write results to parquet, to simulate passing the results to email client API</li>
</ol>

The inference script is containerized using a DockerFile with can be hosted on Amazon ECS.

Steps to build and run the container:
<ol>
  <li>Make sure terminal directory is at root.</li>
  <li>Run <i>docker build -t spam-detection-inference:v1 -f .\deployment\Dockerfile .</i></li>
  <li>Run <i>docker run -it -e AWS_ACCESS_KEY_ID=  &lt;XXX&gt; -e AWS_SECRET_ACCESS_KEY= &lt;XXX&gt; spam-detection-inference:v1  &lt;MLFLOW_TRACKING_URL&gt;</i> with all the variables substituted. 
</li>
</ol>

Once the run the successful, you should see a <i>spam_detection.parquet</i> file in the S3 bucket with the current year and date as part of the prefix key.
<img title="Spam Prediction in S3" src="src\image\s3 prediction.png">
</p>

<h3>Monitoring</h3>
<img title="Monitoring" src="src\image\monitoring.png">
<p>
The monitoring section leverages the <a href='https://www.evidentlyai.com/' target="_blank">Evidently AI</a> library. I have mainly modified  <i>evidently_metrics_calculation.py</i> from the course to fit my use case.
The steps of the script are:
<ol>
  <li>Prepare PostgreSQL database and table</li>
  <li>Initialize MLFlow tracking URI</li>
  <li>Load production model from MLFlow Model Registry</li>
  <li>Get unseen data, simulated by getting randomly from the training dataset</li>
  <li>Preprocess unseen data</li>
  <li>Predict on the unseen data</li>
  <li>Get reference data from MLFlow production model run</li>
  <li>Calculate drift metrics at 5 random time intervals</li>
</ol>

From the root directory, run
```
docker compose -f .\monitoring\docker-compose.yml up --build -d
```
to prepare PostgreSQL, Adminer and Grafana. Likewise this can be deployed to Amazon ECS.

Since the text embeddings play a large role in the model's performance, three embedding drift metrics are used, namely with the methods: classifier model, maximum mean discrepancy and cosine distance. This is based on a good <a href="https://www.evidentlyai.com/blog/embedding-drift-detection"  target="_blank">blog writeup</a> by Evidently AI. A drift is considered to be occured when at least two drifts from the mentioned three methods are detected. When a drift is detected, a flow from the model training deployment `Spam Detector Capstone/mlops-capstone-spam-detector` is automatically ran to retrain the model.

The monitoring script has also been deployed to Prefect Cloud with the same steps as described earlier and scheduled to run daily. A flow run can be executed from `Batch Monitoring/batch-drift-monitoring`.
<img title="Prefect Monitoring Flow" src="src\image\monitor prefect.png">

Once everything has run successfully, Adminer can be logged in from `http://localhost:8081/`
<img title="Adminer Settings" src="src\image\adminer settings.png">
and you should see some data in the `embedding_drift_metrics` table.
<img title="Adminer Data" src="src\image\adminer data.png">

The Grafana dashboard can be accessed from `http://localhost:3001/`.
<img title="Grafana Dashboard" src="src\image\grafana viz.png">

</p>

<h3>Code best practices</h3>
<p>
The following have been developed:
<ol>
  <li>Unit test</li>
  <li>Integration test</li>
  <li>auto code formatter using Black</li>
  <li>Makefile</li>
</ol>
</p>

## Dagster Deployment and Execution

This section outlines how to build and run the Dagster user code, which includes the training and monitoring jobs.

### 1. Prerequisites

*   **Docker:** Ensure Docker is installed and running on your system.
*   **Pipenv:** The project uses Pipenv for dependency management. Make sure `Pipfile` and `Pipfile.lock` are up-to-date. You can typically update the lock file with `pipenv lock`.
*   **External Services:** The Dagster jobs interact with external services:
    *   **PostgreSQL:** Used for storing monitoring metrics by the `batch_monitoring_job` and potentially by MLflow if configured similarly. The `prep_db_op` within the monitoring job expects to connect to a PostgreSQL instance to create the `monitor` database and `embedding_drift_metrics` table. The `postgres_db` resource in `defs.py` also needs connection details for this `monitor` database.
    *   **MLflow:** Used for experiment tracking, model registry, and artifact storage. Both `detector_training_job` and `batch_monitoring_job` require access to an MLflow tracking server.
    *   Ensure these services are accessible from where you run the Docker container.

### 2. Building the Dagster User Code Docker Image

The `Dockerfile.dagster_user_code` is provided to containerize the Dagster user code. This image will contain all your job definitions, ops, and necessary Python dependencies.

To build the image, run the following command from the project root:
```bash
docker build -t dagster_user_code:latest -f Dockerfile.dagster_user_code .
```

### 3. Running Dagster Locally using Docker (Development Mode)

You can run the Dagster webserver (Dagit) locally using the Docker image built in the previous step. This is useful for development, testing, and manually launching job runs.

```bash
# Example:
docker run -d -p 3000:3000 \
  -e POSTGRES_HOST=<your_postgres_host_or_ip> \
  -e POSTGRES_PORT=5430 \
  -e POSTGRES_USER=<your_postgres_user> \
  -e POSTGRES_PASSWORD=<your_postgres_password> \
  # POSTGRES_DB is typically 'monitor' as configured in defs.py for the resource,
  # and used by prep_db_op. No need to set POSTGRES_DB env var here if using defaults.
  -e MLFLOW_TRACKING_URI=<your_mlflow_tracking_uri> \
  # Add AWS credentials if your MLflow setup uses S3 for artifacts and jobs need direct S3 access
  # -e AWS_ACCESS_KEY_ID=<your_aws_access_key_id> \
  # -e AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key> \
  # -e AWS_DEFAULT_REGION=<your_aws_region> \
  # To persist Dagster's run history and schedules outside the container, mount DAGSTER_HOME:
  # -v /path/on/host/to/dagster_home:/opt/dagster/dagster_home \
  # -e DAGSTER_HOME=/opt/dagster/dagster_home \
  --name dagster_dev_container \
  dagster_user_code:latest
```

**Explanation of Environment Variables:**

*   `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: These are used by ops like `prep_db_op` (monitoring) and the `postgres_db` resource defined in `defs.py`. They should point to your PostgreSQL instance where the `monitor` database (for metrics) is or will be created.
*   `MLFLOW_TRACKING_URI`: Points to your MLflow tracking server. Used by both training and monitoring jobs.
*   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`: Required if your MLflow artifact store is S3 and your jobs/ops interact with S3 directly (e.g., `boto3` calls not mediated by MLflow client's S3 handling). The MLflow client itself usually handles S3 access transparently if its backend store and artifact root are S3 URIs and the environment it runs in (the container) has appropriate credentials or IAM roles.
*   `DAGSTER_HOME` (Optional): If you want to persist Dagster's run history, schedule data, and other instance-specific information outside the Docker container, you can set `DAGSTER_HOME` to a path within the container and mount a volume to that path. For example, set `DAGSTER_HOME=/opt/dagster/dagster_home` and add `-v /local/path/to/dagster_home:/opt/dagster/dagster_home` to the `docker run` command.

Once the container is running, you can access the Dagster webserver at `http://localhost:3000`.

### 4. Overview of Production Deployment

For a production environment, you would typically deploy Dagster with a more robust setup. This often involves:

*   **Dagster Cloud:** A fully managed service provided by Elementl.
*   **Self-Hosted Deployment:**
    *   **User Code Deployment:** The Docker image (`dagster_user_code:latest`) you built is a key component. It represents your "user code" and is loaded by Dagster services.
    *   **Dagster Webserver:** Runs the Dagit UI and GraphQL API. Can be run as a separate container or process.
    *   **Dagster Daemon:** Manages schedules, sensors, and run queuing. Runs as a separate container or process.
    *   **Dagster Agent(s):** (For Kubernetes, ECS, Docker, etc.) Launch runs in the execution environment. The agent fetches user code (often your Docker image) to execute jobs.
    *   **Run Launcher:** Configured to launch runs, e.g., using `DockerRunLauncher` to run jobs in new containers from your user code image, or `K8sRunLauncher` for Kubernetes.
    *   **Storage:** PostgreSQL is typically used for instance storage (run history, schedules). Artifact storage (e.g., for op outputs) might be S3, GCS, or a network filesystem.

This project's `Dockerfile.dagster_user_code` provides the foundation for the user code deployment part of such a setup. The default CMD `dagster dev ...` is suitable for development. For production, you would typically have the image run `dagster api grpc -h 0.0.0.0 -p <grpc_port> -f defs.py` when deployed as a user code server that agents or the webserver can query via gRPC.
