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

# Load variables from config.sh
source ./config.sh

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found. Please run setup.sh first."
else
  # Activate the virtual environment
  source "$VENV_DIR"/bin/activate

  # Run the benchmark
  python benchmark.py
fi