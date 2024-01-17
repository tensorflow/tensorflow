#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# script to run pip-compile for keras, tensorboard, estimator deps.
# if there is a change in requirements.in then all lock files will be updated
# accordingly.

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euo pipefail -o history

# Check for required arguments
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_tensorflow_wheel> <python_version>"
  exit 1
fi

TENSORFLOW_WHEEL_PATH="$1"
PYTHON_VERSION="$2"

# All commands run relative to this directory
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ../requirements_updater || exit 1

# Create the requirements_wheel_test.in file
echo "tensorflow @ file://localhost/$TENSORFLOW_WHEEL_PATH" > requirements_wheel_test.in

# Create the requirements_lock file
REQUIREMENTS_LOCK_FILE="requirements_lock_${PYTHON_VERSION}.txt"
touch "$REQUIREMENTS_LOCK_FILE"

### Update the requirements_lock file
bazel run --experimental_convenience_symlinks=ignore --repo_env=REQUIREMENTS_FILE_NAME=requirements_wheel_test.in //:requirements_${PYTHON_VERSION}.update

# Move the updated file to the appropriate directory
mv "$REQUIREMENTS_LOCK_FILE" ../wheel_test/

echo "All tasks completed successfully."
