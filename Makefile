# make all MLFLOW_TRACKING_URL=<value>

unit_&_integration_tests:
	bash tests/run.sh $(MLFLOW_TRACKING_URL)

quality_checks:
	black .