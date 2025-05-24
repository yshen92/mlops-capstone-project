from dagster import Definitions, load_assets_from_modules, ScheduleDefinition # Added ScheduleDefinition
from dagster_postgres.utils import build_psycopg2_resource
import os

# Import jobs from their respective modules
from model_training.training import detector_training_job
from monitoring.evidently_metrics_calculation import batch_monitoring_job

# Define resources
postgres_resource = build_psycopg2_resource.configured({
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5430)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "example"),
    "dbname": "monitor"
})

# Define schedules
daily_batch_monitoring_schedule = ScheduleDefinition(
    job=batch_monitoring_job,
    cron_schedule="0 0 * * *",  # Daily at midnight
    name="daily_batch_monitoring_schedule"
)

defs = Definitions(
    assets=[],
    resources={
        "postgres_db": postgres_resource,
    },
    jobs=[
        detector_training_job,
        batch_monitoring_job
    ],
    schedules=[daily_batch_monitoring_schedule] # Added the schedule
)
