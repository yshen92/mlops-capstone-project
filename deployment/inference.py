import sys
import pandas as pd
from datetime import datetime

import mlflow
from mlflow import MlflowClient

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def load_preprocessor(device='cpu'):
    '''
    Loads the sentence transformer model.

    Args:
        device: the device to use for the model

    Returns:
        sentence_model: the sentence transformer model
    '''
    return SentenceTransformer('all-mpnet-base-v2', device=device)


def embed_text(text, sentence_model):
    '''
    Embeds dataset texts.

    Args:
        df: the dataframe containing the text
        sentence_model: the sentence transformer model

    Returns:
        embeddings: the embeddings of the text
    '''
    embeddings = sentence_model.encode(text, show_progress_bar=False, batch_size=32)

    return embeddings


def preprocess_data(df):
    '''
    Preprocesses by embedding text.

    Args:
        df: the dataframe containing the text

    Returns:
        embeddings: the embeddings of the text
    '''
    # to use CPU only for inference for simplicity
    sentence_model = load_preprocessor('cpu')
    return embed_text(df['text'].values, sentence_model)


def load_model(mlflow_client):
    '''
    Load production model from MLFlow Model Registry

    Args:
        mlflow_client: the mlflow client

    Returns:
        production_model: the production model
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

    production_model_url = f'runs:/{production_model_run_id}/models'

    production_model = mlflow.pyfunc.load_model(production_model_url)

    return production_model


def fetch_data():
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


def get_current_year_and_month():
    now = datetime.now()
    return now.year, now.month


def spam_detection(mlflow_tracking_uri=None):
    unseen_df = fetch_data()

    unseen_embeddings = preprocess_data(unseen_df)

    if mlflow_tracking_uri == None:
        mlflow_tracking_uri = sys.argv[1]
    mlflow_client = MlflowClient(mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    prod_model = load_model(mlflow_client)

    predictions = prod_model.predict(unseen_embeddings)

    prediction_df = pd.DataFrame(predictions, columns=['prediction'])
    year, month = get_current_year_and_month()
    prediction_df['text_id'] = f'{year:04d}-{month:02d}_' + unseen_df.index.astype(str)

    output_file = f's3://mlops-capstone-prediction/year={year:04d}/month={month:02d}/spam_detection.parquet'
    prediction_df.to_parquet(
        output_file, engine='pyarrow', compression=None, index=False
    )


if __name__ == '__main__':
    spam_detection()
