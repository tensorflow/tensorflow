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
MODE=${MODE:-"mkl"}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-""}

echo ""
echo "MODE:${MODE}"
echo "OMP_NUM_THREADS:${OMP_NUM_THREADS}"
echo ""


N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Run configure.
export TF_NEED_CUDA=0
export PYTHON_BIN_PATH=`which python3`
yes "" | $PYTHON_BIN_PATH configure.py
if [[ "$MODE" == "eigen" ]]; then
    CONFIG=""
    OMPTHREADS=""
else
    CONFIG="--config=mkl"
# Setting OMP_THREADS for low performing benchmarks.
#   Default value(=core count) degrades performance of some benchmark cases. 
#   Optimal thread count is case specific. 
#   An argument can be passed to script, the value of which is used if given.
#   Otherwise OMP_NUM_THREADS is set to 10
    if [[ -z $OMP_NUM_THREADS ]]; then
        OMPTHREADS="--action_env=OMP_NUM_THREADS=10"
    else 
        OMPTHREADS="--action_env=OMP_NUM_THREADS=$OMP_NUM_THREADS"
    fi
fi

# Run bazel test command. Double test timeouts to avoid flakes.
# Setting KMP_BLOCKTIME to 0 lets OpenMP threads to sleep right after parallel execution
# in an MKL primitive. This reduces the effects of an oversubscription of OpenMP threads
# caused by executing multiple tests concurrently.
bazel test --test_tag_filters=-no_oss,-no_oss_py2,-oss_serial,-gpu,-tpu,-benchmark-test --test_lang_filters=cc,py -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 --build_tests_only \
    ${CONFIG} --test_env=KMP_BLOCKTIME=0 ${OMPTHREADS} --config=opt --test_output=errors -- \
    //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/... -//tensorflow/lite/...
