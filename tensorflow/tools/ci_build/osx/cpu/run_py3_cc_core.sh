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
#
# ==============================================================================

set -e
set -x

N_JOBS=$(sysctl -n hw.ncpu)
N_JOBS=$((N_JOBS+1))

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Run configure.
export TF_NEED_CUDA=0
export PYTHON_BIN_PATH=$(which python3)
yes "" | $PYTHON_BIN_PATH configure.py
which bazel
bazel test --test_tag_filters=-no_oss,-gpu,-tpu,-benchmark-test,-nomac,-no_mac,-mac_excluded \
    --announce_rc \
    --test_timeout 300,450,1200,3600 \
    --test_size_filters=small,medium \
    --jobs=${N_JOBS} --build_tests_only --test_output=errors -k -- \
    //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/...
