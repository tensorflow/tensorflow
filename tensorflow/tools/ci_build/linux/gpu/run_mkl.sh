#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

#default config
DEFAULT_CONFIG="--config=cuda"
OMPTHREADS="--action_env=OMP_NUM_THREADS=$N_JOBS"
KMP_BLOCKTIME="--test_env=KMP_BLOCKTIME=0"

#install packages needed

pip install tensorflow-estimator tensorboard
pip install --upgrade  tf-estimator-nightly
pip install keras-nightly 

pip list

# Run configure.
export TF_NEED_CUDA=1
export TF_CUDA_COMPUTE_CAPABILITIES=6.0
export PYTHON_BIN_PATH=`which python`
yes "" | TF_NEED_CUDA=1 TF_CUDA_COMPUTE_CAPABILITIES=6.0 $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
# Setting KMP_BLOCKTIME to 0 lets OpenMP threads to sleep right after parallel execution
# in an MKL primitive. This reduces the effects of an oversubscription of OpenMP threads
# caused by executing multiple tests concurrently.

ENABLE_ONEDNN=""
CONFIG=${DEFAULT_CONFIG}
if [[ $# -ge 1 ]]; then
  if [[ "$1" == "eigencuda" ]]; then
     echo "uses all default values for eigen"
     ENABLE_ONEDNN=""
  else
     ENABLE_ONEDNN="--action_env=TF_ENABLE_ONEDNN_OPTS=1"
  fi
fi

#using default targets from tensorflow project
source "./tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh"
if [[ -z "$DEFAULT_BAZEL_TARGETS" ]]; then
   DEFAULT_BAZEL_TARGETS="//tensorflow/...  -//tensorflow/compiler/...  -//tensorflow/lite/..."
else
   DEFAULT_BAZEL_TARGETS="${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/..."
fi
echo "DEFAULT_BAZEL_TARGETS: $DEFAULT_BAZEL_TARGETS "

echo ""
echo "Bazel will test with CONFIG=${CONFIG}, ENABLE_ONEDNN=${ENABLE_ONEDNN}"
echo ""
#Bazel test command with two option eigencuda or dnllcuda

bazel test \
    --test_tag_filters=gpu,-no_gpu,-benchmark-test,-no_gpu_presubmit,-no_cuda11,-v1only,-no_oss,-oss_excluded,-oss_serial \
    --build_tag_filters=gpu,-no_gpu,-benchark-test,-no_oss,-oss_excluded,-oss_serial,-no_gpu_presubmit,-no_cuda11,-v1only \
    --test_lang_filters=cc,py \
    -c opt -k \
    --test_timeout 300,450,1200,3600 \
    --build_tests_only \
    --local_test_jobs=1 \
    --cache_test_results \
    ${CONFIG} \
    ${KMP_BLOCKTIME} \
    ${OMPTHREADS} \
    ${ENABLE_ONEDNN} \
    --test_output=errors \
    -- ${DEFAULT_BAZEL_TARGETS}
