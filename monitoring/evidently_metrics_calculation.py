from datetime import datetime, timedelta
import time
import pandas as pd
import psycopg2

import mlflow
from mlflow import MlflowClient

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from prefect import task, flow
from prefect.deployments import run_deployment


from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric, Comment
from evidently.metrics.data_drift.embedding_drift_methods import distance, mmd

'''
Steps at root directory:
1. docker compose -f .\monitoring\docker-compose.yml up --build
2. python .\monitoring\evidently_metrics_calculation.py 
'''

SEND_TIMEOUT = 10

CREATE_TABLE_STATEMENT = """
drop table if exists embedding_drift_metrics;
create table embedding_drift_metrics(
	timestamp timestamp,
	model_drift float,
	mmd_drift float,
	cosine_dist_drift float,
    total_drift_detected bool
)
"""


@task(name="MLFlow Init")
def init_mlflow(mlflow_tracking_uri):
    '''
    Initialise MLFlow for experiment tracking.

    Args:
        mlflow_tracking_uri: the uri of the mlflow server
        mlflow_experiment_name: the name of the experiment

    Returns:
        client: the mlflow client
        optuna_mlflow_callback: the optuna mlflow callback


    '''
    client = MlflowClient(mlflow_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return client


@task(name="Load sentence transformer")
def load_preprocessor(device='cpu'):
    '''
    Loads the sentence transformer model.

    Args:
        device: the device to use for the model

    Returns:
        sentence_model: the sentence transformer model
    '''
    return SentenceTransformer('all-mpnet-base-v2', device=device)


@task(task_run_name="Embedding text")
def embed_text(df, sentence_model):
    '''
    Embeds dataset texts.

    Args:
        df: the dataframe containing the text
        sentence_model: the sentence transformer model

    Returns:
        embeddings: the embeddings of the text
    '''
    embeddings = sentence_model.encode(
        df['text'].values, show_progress_bar=False, batch_size=32
    )

    return embeddings


@task(task_run_name="Get unseen text")
def get_unseen_data():
    '''
    Simulate fetching 300 random unseen texts

    Returns:
        unseen_df: the dataframe containing the unseen texts
    '''
    spam_detection_dataset = load_dataset("Deysi/spam-detection-dataset")
    spam_detection_dataset.set_format(type='pandas')

    # Assuming unseen data is from the train dataset
    unseen_df = spam_detection_dataset['train'][:].sample(300, random_state=10)
    return unseen_df


@task(task_run_name="Get production model run id")
def get_mlflow_prod_model_run_id(mlflow_client):
    '''
    Get the production model run id.

    Args:
        mlflow_client: the mlflow client

    Returns:
        production_model_run_id: the production model run id
    '''

    # Get all registered models
    registered_models = mlflow_client.search_registered_models(
        filter_string=f"name='spam-detector'"
    )

    production_model_run_id = [
        [
            model_version.run_id
            for model_version in reg_model.latest_versions
            if model_version.current_stage == 'Production'
        ]
        for reg_model in registered_models
    ][0][0]

    return production_model_run_id


@task(task_run_name="Get production model reference data")
def get_reference_data(production_model_run_id, mlflow_client):
    '''
    Get the reference data used in validating the production model.

    Args:
        production_model_run_id: the production model run id
        mlflow_client: the mlflow client

    Returns:
        reference_df: the dataframe containing the reference data
    '''
    run_info = mlflow_client.get_run(production_model_run_id)
    s3_artifact_uri = run_info.info.artifact_uri

    reference_df = pd.read_csv(s3_artifact_uri + '/dataset/train_embeddings_df.csv')
    return reference_df


@task(task_run_name="Load production model")
def load_model(production_model_run_id, mlflow_client):
    '''
    Load production model from MLFlow Model Registry

    Args:
        mlflow_client: the mlflow client

    Returns:
        production_model: the production model
    '''
    production_model_url = f'runs:/{production_model_run_id}/models'

    production_model = mlflow.pyfunc.load_model(production_model_url)

    return production_model


@task(name="prepare database", retries=2, retry_delay_seconds=5)
def prep_db():
    """
    Create the database and tables.
    """
    conn = psycopg2.connect("host=localhost port=5430 user=postgres password=example")
    conn.autocommit = True
    cursor = conn.cursor()
    query = cursor.execute("SELECT 1 FROM pg_database WHERE datname='monitor'")
    res = cursor.fetchall()
    if len(res) == 0:
        cursor.execute("create database monitor;")
        cursor.close()
        conn.close()

    with psycopg2.connect(
        "host=localhost port=5430 dbname=monitor user=postgres password=example"
    ) as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_STATEMENT)
        cursor.close()


@task(name="calculate drift metrics", retries=2, retry_delay_seconds=5)
def calculate_metrics_postgresql(reference_df, prediction_df):
    '''
    Calculates the drift metrics based on the prediction and reference data.
    The drift scores are written to the database.

    Args:
        reference_df: the dataframe containing the reference data
        prediction_df: the dataframe containing the prediction data

    Returns:
        None
    '''
    column_mapping = ColumnMapping(
        embeddings={
            'text_embeddings': reference_df.drop(['prediction'], axis=1).columns
        },
        prediction='prediction',
        target=None,
    )

    report = Report(
        metrics=[
            Comment(
                'Drift Methods: Model, Maximum Mean Discrepancy (MMD), Cosine Distance'
            ),
            EmbeddingsDriftMetric('text_embeddings'),
            EmbeddingsDriftMetric(
                'text_embeddings',
                drift_method=mmd(
                    threshold=0.015,
                    bootstrap=None,
                    quantile_probability=0.95,
                    pca_components=None,
                ),
            ),
            EmbeddingsDriftMetric(
                'text_embeddings',
                drift_method=distance(
                    dist='cosine',
                    threshold=0.2,
                    pca_components=None,
                    bootstrap=None,
                    quantile_probability=0.05,
                ),
            ),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=prediction_df,
        column_mapping=column_mapping,
    )
    # report.save_html('metric_report.html')

    result = report.as_dict()

    model_drift = result['metrics'][1]['result']['drift_score']
    mmd_drift = result['metrics'][2]['result']['drift_score']
    cosine_dist_drift = result['metrics'][3]['result']['drift_score']

    model_drift_detected = result['metrics'][1]['result']['drift_detected']
    mmd_drift_detected = result['metrics'][2]['result']['drift_detected']
    cosine_dist_drift_detected = result['metrics'][3]['result']['drift_detected']
    # Drift is detected if at least 2 drifts are True
    total_drift_detected = (
        sum([model_drift_detected, mmd_drift_detected, cosine_dist_drift_detected]) >= 2
    )

    # If there's a drift, start retraining job
    if total_drift_detected:
        response = run_deployment(
            name='Spam Detector Capstone/mlops-capstone-spam-detector'
        )

    with psycopg2.connect(
        "host=localhost port=5430 dbname=monitor user=postgres password=example"
    ) as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(
            "insert into embedding_drift_metrics(timestamp, model_drift, mmd_drift, cosine_dist_drift, total_drift_detected) values (%s, %s, %s, %s, %s)",
            (
                datetime.now(),
                model_drift,
                mmd_drift,
                cosine_dist_drift,
                total_drift_detected,
            ),
        )


@flow(name="Batch Monitoring")
def batch_monitoring(mlflow_tracking_uri: str = 'http://18.142.46.37:5000/') -> None:
    prep_db()

    mlflow_client = init_mlflow(mlflow_tracking_uri)
    production_model_run_id = get_mlflow_prod_model_run_id(mlflow_client)
    prod_model = load_model(production_model_run_id, mlflow_client)

    unseen_df = get_unseen_data()
    sentence_model = load_preprocessor('cpu')
    unseen_embeddings = embed_text(unseen_df, sentence_model)
    predictions = prod_model.predict(unseen_embeddings)
    prediction_df = pd.DataFrame(unseen_embeddings)
    prediction_df['prediction'] = predictions
    prediction_df.columns = [str(header) for header in prediction_df.columns]

    reference_df = get_reference_data(production_model_run_id, mlflow_client)
    reference_df.drop('label', axis=1, inplace=True)
    reference_df['prediction'] = prod_model.predict(reference_df)
    reference_df.columns = [str(header) for header in reference_df.columns]

    last_send = datetime.now() - timedelta(seconds=10)
    for i in range(0, 5):
        calculate_metrics_postgresql(reference_df, prediction_df)

        new_send = datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + timedelta(seconds=20)


if __name__ == '__main__':
    batch_monitoring()
