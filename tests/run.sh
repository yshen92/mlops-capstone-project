#!/bin/bash

# Check if a URL argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./tests/run.sh MLFLOW_TRACKING_URL"
    exit 1
fi

# Set the URL variable
MLFLOW_TRACKING_URL="$1"

# Activate the pipenv virtual environment
pipenv shell

# Run command to test text embeddings length
python -c "from tests import unit_tests; unit_tests.test_text_embeddings_length()"

# Run command to test text embeddings values
python -c "from tests import unit_tests; unit_tests.test_text_embeddings()"

# Run command for integration test
python -c "from tests import integration_test; integration_test.TestIntegration('$MLFLOW_TRACKING_URL')"


