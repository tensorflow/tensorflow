#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROD_REPOSITORY="https://upload.pypi.org/legacy/"
TEST_REPOSITORY="https://test.pypi.org/legacy/"

twine upload \
    --username "$PYPI_USERNAME" \
    --password "$PYPI_PASSWORD" \
    --repository-url "$PROD_REPOSITORY" \
    "$DIR/../python/dist/"*

