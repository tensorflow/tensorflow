#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_BUILD__JOBS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`

export TF_NEED_ROCM=0

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      -k \
      --test_tag_filters=-no_oss,-oss_excluded,-oss_serial,-gpu,-multi_gpu,-tpu,-no_rocm,-benchmark-test,-v1only \
      --jobs=${N_BUILD_JOBS} \
      --local_test_jobs=${N_BUILD_JOBS} \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_size_filters=small,medium,large \
      -- \
      //tensorflow/... \
      -//tensorflow/core/tpu/... \
      -//tensorflow/lite/... \
