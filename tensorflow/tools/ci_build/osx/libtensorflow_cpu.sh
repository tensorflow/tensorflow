#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# Script to produce binary release of libtensorflow (C API, Java jars etc.).

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# See comments at the top of this file for details.
source "${SCRIPT_DIR}/../builds/libtensorflow.sh"

# Configure script
export PYTHON_BIN_PATH="/usr/bin/python"
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_CUDA=0
export TF_NEED_OPENCL=0
export TF_NEED_MKL=0
export COMPUTECPP_PATH="/usr/local"

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
build_libtensorflow_tarball "-cpu-darwin-$(uname -m)"
