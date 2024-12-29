#!/bin/bash
# Copyright 2024 The OpenXLA Authors.
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

set -x

source ./config.sh

# Create tmp and cache directories if they don't exist
mkdir -p "$TMP_DIR"
mkdir -p "$CACHE_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  # Create a virtual environment
  python3 -m venv "$VENV_DIR"
  # Activate the virtual environment
  source "$VENV_DIR"/bin/activate
  # Install Gemma2 Flax dependencies
  pip install -r requirements.txt
else
  # Activate the virtual environment
  source "$VENV_DIR"/bin/activate
fi


TAR_FILE="${CACHE_DIR}/gemma-2-flax-2b-it.tar"
# Download and extract Gemma2 Flax model files
if [[ ! -d "$MODEL_DIR" ]]; then
  # Copy the tar file to the tmp directory
  wget -P "$CACHE_DIR" https://storage.googleapis.com/xla-benchmarking-temp/gemma-2-flax-2b-it.tar
  # Change to cache directory and extract the tar file
  cd "$CACHE_DIR"
  tar -xf "$TAR_FILE"
fi
