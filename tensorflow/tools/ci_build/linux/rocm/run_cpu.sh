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
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION

export TF_NEED_ROCM=0

if [ -f /usertools/cpu.bazelrc ]; then
        # Use the bazelrc files in /usertools if available
	if [ ! -d /tf ];then
           # The bazelrc files in /usertools expect /tf to exist
           mkdir /tf
        fi
        bazel \
          --bazelrc=/usertools/cpu.bazelrc \
          test \
          --config=sigbuild_local_cache \
          --config=pycpp \
          --test_tag_filters=-no_cuda_on_cpu_tap,-no-gpu,-optimize.mlir.test,-requires-gpu-nvidia,-tpu,-v1only,-oss_serial,-no_windows,-no_oss \
          --action_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
          --local_test_jobs=${N_BUILD_JOBS} \
         --test_timeout 920,2400,7200,9600 \
          --jobs=${N_BUILD_JOBS}
else
         yes "" | $PYTHON_BIN_PATH configure.py


        # Run bazel test command. Double test timeouts to avoid flakes.
        # xla/mlir_hlo/tests/Dialect/gml_st tests disabled in 09/08/22 sync
        bazel test \
              -k \
              --test_tag_filters=-no_oss,-oss_excluded,-oss_serial,-gpu,-multi_gpu,-tpu,-cuda-only,-benchmark-test,-v1only \
              --test_lang_filters=cc,py \
	            --jobs=30 \
              --local_ram_resources=60000 \
              --local_cpu_resources=15 \
              --local_test_jobs=${N_BUILD_JOBS} \
              --test_timeout 920,2400,7200,9600 \
              --build_tests_only \
              --test_output=errors \
              --test_sharding_strategy=disabled \
              --test_size_filters=small,medium \
              --test_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
              -- \
              //tensorflow/... \
              -//tensorflow/compiler/tf2tensorrt/... \
              -//tensorflow/core/tpu/... \
              -//tensorflow/lite/... \
              -//tensorflow/tools/toolchains/...
fi 
