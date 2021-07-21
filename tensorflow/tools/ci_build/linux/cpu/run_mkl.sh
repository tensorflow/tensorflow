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

# This script accepts only one parameter: either the word "eigen", or an
# integer value greater than 0 that is passed to the bazel test command
# via the OMP_NUM_THREADS action environment variable. If an integer is
# passed, the script assumes it is running in DNNL mode; the
# OMP_NUM_THREADS variable is irrelevant in eigen mode.

set -e
set -x

#using default targets from tensorflow project
source "./tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh"
if [[ -z "$DEFAULT_BAZEL_TARGETS" ]]; then
   DEFAULT_BAZEL_TARGETS="//tensorflow/...  -//tensorflow/compiler/...  -//tensorflow/lite/..."
else
   DEFAULT_BAZEL_TARGETS="${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/..."
fi
echo "DEFAULT_BAZEL_TARGETS: $DEFAULT_BAZEL_TARGETS "

DEFAULT_OMP_NUM_THREADS="10"
DEFAULT_CONFIG="--config=mkl"

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Run configure.
export TF_NEED_CUDA=0
export PYTHON_BIN_PATH=`which python3`
yes "" | $PYTHON_BIN_PATH configure.py

# Get parameters from command-line rather than from env.
# Setting OMP_THREADS for low performing benchmarks.
#   Default value(=core count) degrades performance of some benchmark cases.
#   Optimal thread count is case specific.
RE_DIGITS_ONLY="^[0-9]+$"
MIN_OMP_THREADS=1
CONFIG=""
OMPTHREADS=""
BLOCK_FORMAT=""
ENABLE_ONEDNN=""
KMP_BLOCKTIME=""
if [[ $# -ge 1 ]]; then
  if [[ "$1" == "eigen" ]]; then
     echo "uses all default values for eigen"
  elif [[ "$1" == "threadpool" ]]; then
    ENABLE_ONEDNN="--action_env=TF_ENABLE_ONEDNN_OPTS=1"
  elif [[ "$1" =~ ${RE_DIGITS_ONLY} && $1 -ge ${MIN_OMP_THREADS} ]]; then
    CONFIG="${DEFAULT_CONFIG}"
    OMPTHREADS="--action_env=OMP_NUM_THREADS=${1}"
    BLOCK_FORMAT="--action_env=TF_ENABLE_MKL_NATIVE_FORMAT=0"
    KMP_BLOCKTIME="--test_env=KMP_BLOCKTIME=0"
  else
    echo "${1} isn't a valid configuration or"
    echo "number of OM_NUM_THREADS. Exiting..."
    exit 1
  fi
else  # No parameters were passed in so set default values.
  CONFIG="${DEFAULT_CONFIG}"
  OMPTHREADS="--action_env=OMP_NUM_THREADS=${DEFAULT_OMP_NUM_THREADS}"
  BLOCK_FORMAT="--action_env=TF_ENABLE_MKL_NATIVE_FORMAT=0"
  KMP_BLOCKTIME="--test_env=KMP_BLOCKTIME=0"
fi

echo ""
echo "Bazel will test with CONFIG=${CONFIG}, BLOCK_FORMAT=${BLOCK_FORMAT}, ENABLE_ONEDNN=${ENABLE_ONEDNN}, KMP_BLOCKTIME=${KMP_BLOCKTIME}  and OMPTHREADS=${OMPTHREADS}"
echo ""

# Run bazel test command. Double test timeouts to avoid flakes.
# Setting KMP_BLOCKTIME to 0 lets OpenMP threads to sleep right after parallel
# execution in an MKL primitive. This reduces the effects of an oversubscription
# of OpenMP threads caused by executing multiple tests concurrently.
bazel test \
    --test_tag_filters=-no_oss,-no_oss_py2,-oss_serial,-gpu,-tpu,-benchmark-test,-v1only \
    --test_lang_filters=cc,py \
    -k \
    --jobs=${N_JOBS} \
    --test_timeout 300,450,1200,3600 \
    --build_tests_only \
    ${CONFIG} \
    ${KMP_BLOCKTIME} \
    ${OMPTHREADS} \
    ${ENABLE_ONEDNN} \
    ${BLOCK_FORMAT} \
    --config=opt \
    --test_output=errors \
    -- ${DEFAULT_BAZEL_TARGETS}
