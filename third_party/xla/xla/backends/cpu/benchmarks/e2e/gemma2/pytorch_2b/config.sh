#!/bin/bash
# Copyright 2025 The OpenXLA Authors.
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

# Temporary directory for the virtual environment
export TMP_DIR="${HOME}/tmp"

# Cache directory for the Gemma2 Flax model
export CACHE_DIR="${HOME}/.cache"

# Path to virtual enviornment
export VENV_DIR="${TMP_DIR}/gemma-2-pyTorch"

# Path to the Gemma2 Flax model files
export MODEL_DIR="${CACHE_DIR}/gemma-2-pyTorch-2b-it"

