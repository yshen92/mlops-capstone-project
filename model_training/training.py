import os
import pandas as pd
from datetime import date

from torch import cuda
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

import optuna
from optuna.integration.mlflow import MLflowCallback

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

## TODO: Move to README
# Steps
# 1. Start EC2, clear off db and s3 artifacts if needed
# 2. Run mlflow server on EC2
# 3. Run training.py

# Prefect steps:
# 1. prefect init
# 2. prefect deploy

@task(name="MLFlow Init")
def init_mlflow(mlflow_tracking_uri, mlflow_experiment_name):
    client = MlflowClient(mlflow_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    optuna_mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_tracking_uri,
        metric_name='accuracy',
        create_experiment=False,
    )

    return client, optuna_mlflow_callback

@task(name="Get spam training and testing datasets")
def get_data():
    '''
    Loads the dataset from the Deysi/spam-detection-dataset.
    The dataset contains the following columns:
        - text: the text to be identified
        - label: the label of the text (not spam, spam)

    Returns:
        train_df: a dataframe containing the training data
        test_df: a dataframe containing the testing data
    
    '''
    spam_detection_dataset = load_dataset("Deysi/spam-detection-dataset")
    spam_detection_dataset.set_format(type='pandas')

    train_df = spam_detection_dataset['train'][:].sample(1500, random_state=10)
    test_df = spam_detection_dataset['test'][:].sample(500, random_state=10)

    return train_df, test_df

@task(name="Load sentence transformer")
def load_preprocessor(device='cpu'):
    return SentenceTransformer('model_training\sentence-transformers_all-mpnet-base-v2', device=device)

@task(task_run_name="Embedding {embed_type} data")
def embed_text(df, sentence_model, embed_type='train'):
    embeddings = sentence_model.encode(df['text'].values, show_progress_bar=False, batch_size=32)

    return embeddings

@task(log_prints=True, name="Model hyperparameter tuning")
def hyperparameter_tuning(train_embeddings_df, test_embeddings_df, optuna_mlflow_callback):
    def objective(trial):
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 5, 100, log=True)
        clf = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
        clf.fit(train_embeddings_df.drop('label', axis=1), train_embeddings_df['label'])

        predictions = clf.predict(test_embeddings_df.drop('label', axis=1))
        accuracy = accuracy_score(test_embeddings_df['label'], predictions)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[optuna_mlflow_callback])

    return study

@task(name="Get best experiment run")
def find_best_run(study, mlflow_experiment_name):
    spam_detection_experiment=dict(mlflow.get_experiment_by_name(mlflow_experiment_name))
    experiment_id=spam_detection_experiment['experiment_id']

    # Get based on the best trial value with the lowest n_estimators
    best_run = mlflow.search_runs( 
        experiment_ids=experiment_id,
        filter_string=f'metrics.accuracy = 1',
        run_view_type= ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=['parameters.rf_n_estimators ASC']
    )

    return best_run

@task(name="Train the best model")
def train_best_model(best_run, train_embeddings_df, test_embeddings_df):
    with mlflow.start_run():
        best_max_depth = int(best_run['params.rf_max_depth'].values[0])
        best_n_estimators = int(best_run['params.rf_n_estimators'].values[0])
        mlflow.log_params({'rf_max_depth': best_max_depth, 'rf_n_estimators': best_n_estimators})

        best_clf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth)
        best_clf.fit(train_embeddings_df.drop('label', axis=1), train_embeddings_df['label'])

        best_predictions = best_clf.predict(test_embeddings_df.drop('label', axis=1))
        accuracy = accuracy_score(test_embeddings_df['label'], best_predictions)
        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(best_clf, artifact_path="models", registered_model_name='spam-detector')

        # Log datasets as MLFlow artifacts
        if not os.path.exists('temp/'):
            os.makedirs('temp/')
        train_embeddings_df.to_csv('temp/train_embeddings_df.csv', index=False)
        test_embeddings_df.to_csv('temp/test_embeddings_df.csv', index=False)
        mlflow.log_artifact('temp/train_embeddings_df.csv', 'dataset')
        mlflow.log_artifact('temp/test_embeddings_df.csv', 'dataset')

        # Prefect artifact to report trained model accuracy
        markdown_accuracy_report = f"""# Accuracy Report

        ## Summary

        Spam Detection

        ## Random Forest Classifier Accuracy

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {accuracy:.2f} |
        """

        create_markdown_artifact(
            key="spam-detector-accuracy-model-report", markdown=markdown_accuracy_report
        )

    return model_info

@task(log_prints=True, name="Productionize the model")
def stage_model(client, model_info):
    trained_model_run_id = model_info.run_id

    # Get all registered models
    registered_models = client.search_registered_models(
        filter_string=f"name='spam-detector'"
    )

    if len(registered_models) == 1:
        if registered_models[0].latest_versions[0].run_id == trained_model_run_id:
            if registered_models[0].latest_versions[0].current_stage == 'None':
                model_version = registered_models[0].latest_versions[0].version
                client.transition_model_version_stage(
                    name='spam-detector',
                    version=model_version,
                    stage='Production',
                    archive_existing_versions=False
                )
                print(
                    f'Productionized version {model_version} of spam-detector model.'
                )
            else:
                # a sole registered model with a stage doesn't need to be transitioned
                pass
        else:
            # the sole registered model should be the same as the registered trained model
            pass
    else:
        production_model_run_id = [
            [model_version.run_id for model_version in reg_model.latest_versions if model_version.current_stage=='Production'] 
            for reg_model in registered_models
        ]

        trained_model_version = [
                [model_version for model_version in reg_model.latest_versions if model_version.run_id==trained_model_run_id] 
                for reg_model in registered_models
        ]

        # There should only be 1 production model at a time
        production_model_run = client.get_run(production_model_run_id[0])
        production_model_accuracy = production_model_run.data.metrics['accuracy']

        trained_model_run = client.get_run(trained_model_run_id)
        trained_model_accuracy = trained_model_run.data.metrics['accuracy']

        # Newly trained model challenges production model to replace production model if accuracy is better
        # Else trained model is archived
        if trained_model_accuracy > production_model_accuracy:
            new_stage = "Production"
            client.transition_model_version_stage(
                name='spam-detector',
                version=trained_model_version.version,
                stage=new_stage,
                archive_existing_versions=True,
            )
            print( f'Productionized version {trained_model_version.version} of spam-detector model.')
        else:
            new_stage = "Archived"
            client.transition_model_version_stage(
                name='spam-detector',
                version=trained_model_version.version,
                stage=new_stage,
                archive_existing_versions=False,
            )
            print( f'Archived version {trained_model_version.version} of spam-detector model.')

@flow(name="Spam Detector Capstone")
def detector_training_main(
    mlflow_tracking_uri: str = "http://18.142.178.133:5000/",
    mlflow_experiment_name: str = "spam-detection-experiment",
) -> None:
    
    mlflow_client, optuna_mlflow_callback = init_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    train_df, test_df = get_data()

    device = 'cuda' if cuda.is_available() else 'cpu'
    sentence_model = load_preprocessor(device)

    train_embeddings = embed_text(train_df, sentence_model, embed_type='training')
    test_embeddings = embed_text(test_df, sentence_model, embed_type='testing')

    # Prepapre embeddings dataset for hyperparameter tuning and model training
    train_embeddings_df = pd.DataFrame(train_embeddings)
    train_embeddings_df['label'] = train_df['label'].values
    test_embeddings_df = pd.DataFrame(test_embeddings)
    test_embeddings_df['label'] = test_df['label'].values

    study = hyperparameter_tuning(train_embeddings_df, test_embeddings_df, optuna_mlflow_callback)

    best_run = find_best_run(study, mlflow_experiment_name)
    model_info = train_best_model(best_run, train_embeddings_df, test_embeddings_df)

    stage_model(mlflow_client, model_info)

if __name__ == '__main__':
    detector_training_main() 


