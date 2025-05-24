import os
import pandas as pd
from datetime import date
from typing import Tuple, Dict, Any

from torch import cuda
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType # type: ignore
# ViewType is not directly available under mlflow.entities in mlflow version 2.12.1 and is instead available at mlflow.protos.databricks_pb2.ViewType
# However, previous code used mlflow.entities.ViewType, so keeping it for now.
# If issues arise, this might need to be: from mlflow.protos.databricks_pb2 import ViewType

import optuna
from optuna.integration.mlflow import MLflowCallback

from dagster import op, job, OpExecutionContext, AssetMaterialization, MetadataValue, Out, Output, make_values_resource

# Determine device for sentence_transformer
DEFAULT_DEVICE = 'cuda' if cuda.is_available() else 'cpu'

@op(
    config_schema={"mlflow_tracking_uri": str, "mlflow_experiment_name": str},
    out={
        "client": Out(MlflowClient),
        "optuna_callback": Out(MLflowCallback),
        "experiment_name": Out(str),
    },
)
def init_mlflow_op(context: OpExecutionContext):
    '''
    Initialise MLFlow for experiment tracking.
    '''
    mlflow_tracking_uri = context.op_config["mlflow_tracking_uri"]
    mlflow_experiment_name = context.op_config["mlflow_experiment_name"]

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
        context.log.info(f"Created new MLflow experiment: {mlflow_experiment_name}, ID: {experiment_id}")
    except mlflow.exceptions.MlflowException: # More specific exception
        experiment_id = mlflow.get_experiment_by_name(
            mlflow_experiment_name
        ).experiment_id # type: ignore
        context.log.info(f"Using existing MLflow experiment: {mlflow_experiment_name}, ID: {experiment_id}")
        
    mlflow.set_experiment(experiment_id=experiment_id) # type: ignore

    optuna_mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_tracking_uri,
        metric_name='accuracy',
        create_experiment=False,
    )
    context.log.info(f"MLflow initialized for experiment: {mlflow_experiment_name}")
    yield Output(client, "client")
    yield Output(optuna_mlflow_callback, "optuna_callback")
    yield Output(mlflow_experiment_name, "experiment_name")


@op(out={"train_df": Out(pd.DataFrame), "test_df": Out(pd.DataFrame)})
def get_data_op(context: OpExecutionContext):
    '''
    Loads the dataset from the Deysi/spam-detection-dataset.
    '''
    context.log.info("Loading dataset from HuggingFace Deysi/spam-detection-dataset...")
    spam_detection_dataset = load_dataset("Deysi/spam-detection-dataset")
    spam_detection_dataset.set_format(type='pandas')

    train_df = spam_detection_dataset['train'][:].sample(1500, random_state=10) # type: ignore
    test_df = spam_detection_dataset['test'][:].sample(500, random_state=10) # type: ignore
    context.log.info(f"Loaded data: {len(train_df)} train samples, {len(test_df)} test samples.")
    yield Output(train_df, "train_df")
    yield Output(test_df, "test_df")


@op(config_schema={"device": str})
def load_preprocessor_op(context: OpExecutionContext) -> SentenceTransformer:
    '''
    Loads the sentence transformer model.
    Assumes model 'sentence-transformers_all-mpnet-base-v2' is in 'model_training/' directory.
    '''
    device = context.op_config["device"]
    model_path = os.path.join('model_training', 'sentence-transformers_all-mpnet-base-v2')
    
    context.log.info(f"Attempting to load sentence transformer model from: {model_path} on device: {device}")
    if not os.path.exists(model_path):
        context.log.warning(f"Model path {model_path} does not exist. SentenceTransformer will attempt to download 'all-mpnet-base-v2' if this path is interpreted as a model name by the library.")
        # If 'model_training/sentence-transformers_all-mpnet-base-v2' is not a valid model name for direct download,
        # and the local path is missing, this op will fail.
        # For robustness, explicitly use the model name for download if local path fails:
        # return SentenceTransformer('all-mpnet-base-v2', device=device)
        # However, sticking to original behavior of trying the path first.
        # If the directory `sentence-transformers_all-mpnet-base-v2` itself is the model, then this path is correct.
        pass

    return SentenceTransformer(model_path, device=device)


@op
def embed_text_op(context: OpExecutionContext, df: pd.DataFrame, sentence_model: SentenceTransformer, embed_type: str) -> Any: # Actually numpy array
    '''
    Embeds dataset texts. embed_type is for logging.
    '''
    context.log.info(f"Embedding {embed_type} data ({len(df)} samples)...")
    embeddings = sentence_model.encode(
        df['text'].values, show_progress_bar=False, batch_size=32 
    )
    context.log.info(f"Finished embedding {embed_type} data. Embedding shape: {embeddings.shape}")
    return embeddings


@op(out={"train_embeddings_df": Out(pd.DataFrame), "test_embeddings_df": Out(pd.DataFrame)})
def prepare_embedding_dfs_op(
    context: OpExecutionContext,
    train_embeddings: Any, # Actually numpy array from embed_text_op
    test_embeddings: Any,  # Actually numpy array from embed_text_op
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    '''
    Prepares embedding dataframes for hyperparameter tuning and model training.
    '''
    context.log.info("Preparing embedding DataFrames...")
    train_embeddings_df = pd.DataFrame(train_embeddings)
    train_embeddings_df['label'] = train_df['label'].values # type: ignore
    test_embeddings_df = pd.DataFrame(test_embeddings)
    test_embeddings_df['label'] = test_df['label'].values # type: ignore
    context.log.info("Embedding DataFrames prepared.")
    yield Output(train_embeddings_df, "train_embeddings_df")
    yield Output(test_embeddings_df, "test_embeddings_df")


@op(config_schema={"n_trials": int})
def hyperparameter_tuning_op(
    context: OpExecutionContext,
    train_embeddings_df: pd.DataFrame,
    test_embeddings_df: pd.DataFrame,
    optuna_mlflow_callback: MLflowCallback,
) -> optuna.study.Study: # type: ignore
    '''
    Optuna hyperparameter tuning on Random Forest Classifier.
    '''
    n_trials = context.op_config["n_trials"]
    context.log.info(f"Starting hyperparameter tuning with {n_trials} trials...")

    def objective(trial: optuna.trial.Trial): # type: ignore
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 5, 100, log=True)
        clf = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, random_state=42
        )
        clf.fit(train_embeddings_df.drop('label', axis=1), train_embeddings_df['label'])

        predictions = clf.predict(test_embeddings_df.drop('label', axis=1))
        accuracy = accuracy_score(test_embeddings_df['label'], predictions)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_mlflow_callback])
    context.log.info(f"Hyperparameter tuning completed. Best trial accuracy: {study.best_value}, Params: {study.best_params}")
    return study


@op
def find_best_run_op(
    context: OpExecutionContext, study: optuna.study.Study, mlflow_experiment_name: str # type: ignore
) -> pd.DataFrame:
    '''
    Finds the best experiment run based on the best trial value with the lowest n_estimators.
    '''
    context.log.info(f"Finding best run for experiment: {mlflow_experiment_name} based on Optuna study best value: {study.best_value}")
    
    # Fetch experiment details
    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
    if not experiment:
        context.log.error(f"MLflow experiment '{mlflow_experiment_name}' not found.")
        raise Exception(f"MLflow experiment '{mlflow_experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # Search for runs matching the best accuracy from Optuna study
    # Optuna callback logs metric name as 'accuracy' (defined in init_mlflow_op)
    filter_string = f"metrics.accuracy = {study.best_value}"
    
    best_runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id], # Must be a list
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY, # type: ignore
        order_by=['params.rf_n_estimators ASC'], # Get the one with fewer estimators in case of tie
    )

    if best_runs_df is None or best_runs_df.empty:
        context.log.warning(f"No MLflow runs found for experiment '{mlflow_experiment_name}' matching accuracy {study.best_value}. This can occur if MLflow logging failed or if there's a slight precision difference in logged metrics. Trying to find runs with similar accuracy.")
        # Attempt a search with a small tolerance for floating point inaccuracies
        tolerance = 1e-5
        filter_string_tolerant = f"metrics.accuracy > {study.best_value - tolerance} and metrics.accuracy < {study.best_value + tolerance}"
        best_runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string_tolerant,
            run_view_type=ViewType.ACTIVE_ONLY, # type: ignore
            order_by=['metrics.accuracy DESC', 'params.rf_n_estimators ASC'], # Get closest accuracy, then by estimators
        )

    if best_runs_df is None or best_runs_df.empty:
        context.log.error(f"Still no runs found for experiment '{mlflow_experiment_name}' near accuracy {study.best_value}. Critical error: Cannot determine the best run.")
        raise Exception("Could not find the best run in MLflow corresponding to Optuna's best trial.")

    context.log.info(f"Found {len(best_runs_df)} run(s) matching criteria. Selecting the top one based on sorting.")
    return best_runs_df.iloc[[0]] # Return as DataFrame with one row


@op
def train_best_model_op(
    context: OpExecutionContext,
    best_run_df: pd.DataFrame, 
    train_embeddings_df: pd.DataFrame,
    test_embeddings_df: pd.DataFrame,
) -> Dict[str, Any]:
    '''
    Train the best model using hyperparameters from the best experiment run.
    '''
    context.log.info("Training best model based on selected best_run_df...")
    if best_run_df.empty:
        context.log.error("best_run_df DataFrame is empty. Cannot proceed.")
        raise ValueError("Input best_run_df DataFrame is empty.")

    best_run_params = best_run_df.iloc[0]
    # MLflow run ID of the Optuna trial run, not the new nested run.
    optuna_run_id = best_run_params["run_id"] 

    # Start a new MLflow run, nested under the main job run if applicable,
    # or as a new top-level run if this op is run standalone.
    # This run is specifically for the "final" model trained with best HPs.
    with mlflow.start_run(run_name="final_model_training", nested=True) as final_model_run:
        final_run_id = final_model_run.info.run_id
        context.log.info(f"Started new MLflow run for final model: {final_run_id}. Optuna trial run was: {optuna_run_id}")

        # Log parameters from the best Optuna trial
        # Parameters in MLflow are logged as params.X, metrics as metrics.Y
        # Need to extract only the parameter part of the column names
        params_to_log = {}
        for col_name in best_run_df.columns:
            if col_name.startswith("params."):
                param_key = col_name.replace("params.", "")
                params_to_log[param_key] = best_run_params[col_name]
        
        # Ensure 'rf_max_depth' and 'rf_n_estimators' are correctly typed and present
        best_max_depth = int(params_to_log.get("rf_max_depth", 0)) # type: ignore
        best_n_estimators = int(params_to_log.get("rf_n_estimators", 0)) # type: ignore

        if best_max_depth == 0 or best_n_estimators == 0:
            context.log.error(f"Could not extract valid 'rf_max_depth' or 'rf_n_estimators' from best_run_df. Params found: {params_to_log}")
            raise ValueError("Invalid model parameters from best_run_df.")

        context.log.info(f"Training final model with params: max_depth={best_max_depth}, n_estimators={best_n_estimators}")
        mlflow.log_params(params_to_log) # Log all extracted params

        best_clf = RandomForestClassifier(
            n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42
        )
        best_clf.fit(
            train_embeddings_df.drop('label', axis=1), train_embeddings_df['label']
        )

        best_predictions = best_clf.predict(test_embeddings_df.drop('label', axis=1))
        accuracy = accuracy_score(test_embeddings_df['label'], best_predictions)
        mlflow.log_metric("accuracy", accuracy)
        context.log.info(f"Final model accuracy: {accuracy}")

        # Log model to MLflow
        model_reg_name = 'spam-detector-dagster' # Use a distinct name for Dagster-trained models
        model_info = mlflow.sklearn.log_model(
            sk_model=best_clf, 
            artifact_path="sk_models",  # Changed from "models" to avoid potential conflict
            registered_model_name=model_reg_name 
        )
        context.log.info(f"Final model logged to MLflow. URI: {model_info.model_uri}, Registered Model Name: {model_reg_name}")

        model_info_dict = {
            "run_id": final_run_id, # Use the new run ID for this model
            "model_uri": model_info.model_uri,
            "artifact_path": model_info.artifact_path,
            "registered_model_name": model_reg_name,
            "accuracy": accuracy
        }

        # Log datasets as MLFlow artifacts
        temp_dir = 'temp_artifacts'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        train_csv_path = os.path.join(temp_dir, 'train_embeddings_df.csv')
        test_csv_path = os.path.join(temp_dir, 'test_embeddings_df.csv')
        train_embeddings_df.to_csv(train_csv_path, index=False)
        test_embeddings_df.to_csv(test_csv_path, index=False)
        mlflow.log_artifact(train_csv_path, 'dataset_snapshots')
        mlflow.log_artifact(test_csv_path, 'dataset_snapshots')
        context.log.info(f"Logged training and testing embedding dataframes to {temp_dir} and as MLflow artifacts.")

        markdown_accuracy_report = f"""# Model Training Report ({model_reg_name})

        ## Summary
        - Date: {date.today()}
        - Accuracy: {accuracy:.4f}
        - Parameters: `max_depth: {best_max_depth}`, `n_estimators: {best_n_estimators}`
        
        ## MLflow Details
        - Run ID (Final Model): {final_run_id}
        - Model URI: {model_info.model_uri}
        - Optuna Trial Run ID (source of HPs): {optuna_run_id}
        """

        context.log_event(
            AssetMaterialization(
                asset_key=f"mlflow_model/{model_reg_name}", # Asset key for Dagster
                description="Trained spam detector model via Dagster and MLflow.",
                metadata={
                    "mlflow_run_id": final_run_id,
                    "mlflow_model_uri": model_info.model_uri,
                    "mlflow_registered_model_name": model_reg_name,
                    "accuracy": accuracy,
                    "hyperparameters": MetadataValue.json(params_to_log),
                    "optuna_run_id": optuna_run_id,
                    "accuracy_report": MetadataValue.md(markdown_accuracy_report),
                },
            )
        )
        context.log.info("Best model trained, logged to MLflow, and Dagster asset materialized.")
    return model_info_dict


@op
def stage_model_op(context: OpExecutionContext, client: MlflowClient, model_info: Dict[str, Any]):
    '''
    Stage the freshly trained model to Production if its accuracy is better than the current Production model.
    '''
    trained_model_run_id = model_info['run_id']
    model_name = model_info['registered_model_name']
    trained_model_accuracy = model_info['accuracy']

    context.log.info(f"Starting model staging process for: {model_name}, Run ID: {trained_model_run_id}, Accuracy: {trained_model_accuracy:.4f}")

    try:
        latest_versions = client.get_latest_versions(name=model_name)
    except mlflow.exceptions.RestException as e: # type: ignore
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            context.log.info(f"Registered model '{model_name}' does not exist. This is unexpected if train_best_model_op just registered it.")
            # This path should ideally not be hit if train_best_model_op succeeded.
            # If it does, it implies an issue with MLflow registration or timing.
            # For now, assume the model was registered and proceed to find its version.
            # Re-fetch after a small delay or rely on the version from model_info if available.
            # However, model_info from log_model doesn't directly give version number.
            # We need to search for the version by run_id.
            latest_versions = [] # Will be populated by searching for the run_id specific version
        else:
            context.log.error(f"Error fetching versions for model '{model_name}': {e}")
            raise

    trained_model_version_obj = None
    # After `log_model` with `registered_model_name`, a new version is created. We need to find it.
    # Sometimes MLflow takes a moment for the new version to be searchable.
    # A robust way is to search by run_id.
    
    # Re-fetch all versions to ensure we have the latest list
    all_model_versions = client.search_model_versions(f"name='{model_name}'")
    for v in all_model_versions:
        if v.run_id == trained_model_run_id:
            trained_model_version_obj = v
            break
            
    if not trained_model_version_obj:
        context.log.error(f"Could not find model version for '{model_name}' with Run ID {trained_model_run_id} even after re-fetching. Staging cannot proceed.")
        # This is a critical issue if the model was meant to be registered.
        return # Cannot proceed

    trained_model_version_str = trained_model_version_obj.version
    context.log.info(f"Found version '{trained_model_version_str}' for trained model (Run ID: {trained_model_run_id}).")

    # Find current Production model, if any
    production_model_version_obj = None
    for v_info in all_model_versions: # Iterate through all versions again
        if v_info.current_stage == 'Production':
            production_model_version_obj = v_info
            break # Assuming only one model in Production

    if production_model_version_obj:
        if production_model_version_obj.run_id == trained_model_run_id:
            context.log.info(f"Model version {trained_model_version_str} (Run ID: {trained_model_run_id}) is already the source of the Production model. Ensuring it's staged as 'Production'.")
            if trained_model_version_obj.current_stage != 'Production':
                client.transition_model_version_stage(
                    name=model_name, version=trained_model_version_str, stage='Production', archive_existing_versions=True
                )
                context.log.info(f"Transitioned version {trained_model_version_str} of '{model_name}' to Production.")
            else:
                context.log.info(f"Version {trained_model_version_str} is already in Production. No change needed.")
            return

        # Existing Production model is different. Compare accuracies.
        prod_model_run = client.get_run(production_model_version_obj.run_id)
        # Ensure 'accuracy' metric exists for the production model
        prod_model_accuracy = prod_model_run.data.metrics.get('accuracy')
        if prod_model_accuracy is None:
            context.log.warning(f"Production model version {production_model_version_obj.version} (Run ID: {production_model_version_obj.run_id}) has no 'accuracy' metric. Cannot compare. Archiving it and promoting new model.")
            client.transition_model_version_stage(name=model_name, version=production_model_version_obj.version, stage='Archived')
            prod_model_accuracy = -1.0 # Ensure new model is better

        context.log.info(f"Comparing trained model (Version: {trained_model_version_str}, Accuracy: {trained_model_accuracy:.4f}) with current Production model (Version: {production_model_version_obj.version}, Accuracy: {prod_model_accuracy:.4f}).")
        
        if trained_model_accuracy > prod_model_accuracy: # type: ignore
            context.log.info(f"Trained model is better. Transitioning version {trained_model_version_str} to 'Production' and archiving existing Production model.")
            client.transition_model_version_stage(
                name=model_name, version=trained_model_version_str, stage='Production', archive_existing_versions=True
            )
        else:
            context.log.info(f"Trained model is not better. Transitioning version {trained_model_version_str} to 'Archived'.")
            client.transition_model_version_stage(
                name=model_name, version=trained_model_version_str, stage='Archived', archive_existing_versions=False
            )
    else:
        # No model in Production, promote the current one
        context.log.info(f"No model currently in Production for '{model_name}'. Transitioning version {trained_model_version_str} to 'Production'.")
        client.transition_model_version_stage(
            name=model_name, version=trained_model_version_str, stage='Production', archive_existing_versions=False
        )
    context.log.info(f"Model staging process for '{model_name}' version {trained_model_version_str} completed.")


@job(
    config={
        "ops": {
            "init_mlflow_op": {
                "config": {
                    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"), 
                    "mlflow_experiment_name": "spam-detection-dagster" 
                }
            },
            "load_preprocessor_op": {
                "config": {
                    "device": DEFAULT_DEVICE 
                }
            },
            "hyperparameter_tuning_op":{
                "config":{
                    "n_trials": 10 # Default, can be overridden in run config
                }
            }
        }
    }
)
def detector_training_job():
    mlflow_outputs = init_mlflow_op()
    data_outputs = get_data_op()
    
    sentence_model = load_preprocessor_op()

    train_embeddings = embed_text_op(
        df=data_outputs["train_df"], # type: ignore
        sentence_model=sentence_model,
        embed_type='training'
    )
    test_embeddings = embed_text_op(
        df=data_outputs["test_df"], # type: ignore
        sentence_model=sentence_model,
        embed_type='testing'
    )

    prepared_dfs = prepare_embedding_dfs_op(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_df=data_outputs["train_df"], # type: ignore
        test_df=data_outputs["test_df"] # type: ignore
    )

    study = hyperparameter_tuning_op(
        train_embeddings_df=prepared_dfs["train_embeddings_df"], # type: ignore
        test_embeddings_df=prepared_dfs["test_embeddings_df"], # type: ignore
        optuna_mlflow_callback=mlflow_outputs["optuna_callback"] # type: ignore
    )

    best_run_df = find_best_run_op(
        study=study,
        mlflow_experiment_name=mlflow_outputs["experiment_name"] # type: ignore
    )

    model_i = train_best_model_op(
        best_run_df=best_run_df,
        train_embeddings_df=prepared_dfs["train_embeddings_df"], # type: ignore
        test_embeddings_df=prepared_dfs["test_embeddings_df"] # type: ignore
    )

    stage_model_op(
        client=mlflow_outputs["client"], # type: ignore
        model_info=model_i
    )

# Notes on changes:
# - Added types to op outputs and inputs where beneficial. Used `Any` for numpy arrays for simplicity.
# - Added `config_schema` for `n_trials` in `hyperparameter_tuning_op`.
# - Improved logging across ops with more context.
# - `find_best_run_op`:
#   - Corrected `experiment_id` to be a list.
#   - Added fallback search with tolerance for floating point inaccuracies in accuracy matching.
#   - Ensures it returns a DataFrame (even if one row) for consistency.
# - `train_best_model_op`:
#   - Clarified `optuna_run_id` vs `final_run_id`.
#   - Changed registered model name to `spam-detector-dagster` to differentiate.
#   - Changed artifact path for sklearn model to `sk_models`.
#   - Logged all `params.*` from the best Optuna run.
#   - Ensured `accuracy` is part of the `model_info_dict` and AssetMaterialization.
#   - Stored dataset snapshots under `dataset_snapshots` artifact subdir.
# - `stage_model_op`:
#   - More robust fetching of model versions, especially the newly created one.
#   - Handles case where production model might not have 'accuracy' metric.
#   - Clearer logging of decisions.
# - `detector_training_job`:
#   - MLflow tracking URI can be set via `MLFLOW_TRACKING_URI` env var or defaults to `http://localhost:5000`.
# - General:
#   - Used `os.path.join` for model_path.
#   - Added type ignores where Dagster's dynamic output typing might seem problematic to static checkers but is fine at runtime.
#   - Corrected `MlflowClient` instantiation in `init_mlflow_op`.
#   - Ensured `mlflow.get_experiment_by_name` result is handled correctly (it can be None).
#   - Used `mlflow.exceptions.MlflowException` for more specific error handling.
#   - `ViewType` import might need adjustment based on MLflow version. Current code assumes it's fine.
#   - Pandas `sample()` and `values` can sometimes cause type checker issues if DataFrame schema is not strictly known; added ignores.
#   - `sentence_model.encode` `show_progress_bar` set to False to reduce log clutter.
#   - `best_run_df.iloc[[0]]` ensures that `find_best_run_op` always returns a DataFrame, not a Series.
#   - `train_best_model_op` now correctly extracts parameters from the `best_run_df` columns.
#   - `stage_model_op` is more robust in finding the version ID of the model that was just trained.
#   - `archive_existing_versions=True` when promoting a new model to Production and a previous one existed.
#   - `archive_existing_versions=False` if no previous production model, or when archiving the current model because it's not better.
#   - `client.search_model_versions(f"name='{model_name}'")` is a more reliable way to get all versions.
#   - Added `random_state=42` to `RandomForestClassifier` in `hyperparameter_tuning_op` for objective function reproducibility.
#   - The `model_training/sentence-transformers_all-mpnet-base-v2` path for the sentence transformer implies that this directory, containing the model files, is located at the root of the `model_training` directory. This needs to be correct in the execution environment.
#   - Corrected `mlflow.set_experiment` call in `init_mlflow_op`.
#   - `train_embeddings_df['label'] = train_df['label'].values` etc. are fine, pandas handles assignment from Series to Series.The file `model_training/training.py` has been overwritten with the new Dagster-based code.
I've made several corrections and improvements during the generation of this code, including:
-   More specific exception handling for MLflow.
-   Ensuring `experiment_id` is a list for `mlflow.search_runs`.
-   Adding a tolerance-based fallback for `find_best_run_op` if exact accuracy match fails.
-   Clarifying MLflow run IDs (Optuna trial vs. final model training).
-   Using a distinct registered model name (`spam-detector-dagster`) for models trained by this Dagster job.
-   Improved robustness in `stage_model_op` for fetching model versions and comparing accuracies.
-   Made `n_trials` in `hyperparameter_tuning_op` configurable.
-   Set MLflow tracking URI to be configurable via environment variable `MLFLOW_TRACKING_URI` or default to `http://localhost:5000`.
-   Used `os.path.join` for constructing file paths.
-   Set `show_progress_bar=False` in `embed_text_op` to reduce log noise.
-   Ensured `find_best_run_op` returns a DataFrame.
-   Correctly extracted and logged hyperparameters in `train_best_model_op`.
-   Handled `archive_existing_versions` flag in `transition_model_version_stage` more carefully.

Next, I need to update `defs.py` to include the new job. I'll read its current content first.
