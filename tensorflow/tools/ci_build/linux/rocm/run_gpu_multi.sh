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
N_TEST_JOBS=1 # run tests serially

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
else
    if [[ -z "${ROCM_PATH}" ]]; then
        ROCM_INSTALL_DIR=/opt/rocm/
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

# Run configure.
export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION

export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

if [ ! -d /tf ];then
    # The bazelrc files expect /tf to exist
    mkdir /tf
fi

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
        --local_test_jobs=${N_TEST_JOBS} \
        --jobs=30 \
        --local_ram_resources=60000 \
        --local_cpu_resources=15 \
        --verbose_failures \
        --config=rocm \
        --config=nonpip_multi_gpu \
        --config=sigbuild_local_cache \
        --action_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
        --repo_env="ROCM_PATH=$ROCM_PATH" \


#  Started failing with 210906 sync
# FAILED : //tensorflow/core/kernels:collective_nccl_test_2gpu \

# no_rocm : //tensorflow/python/keras/distribute:keras_dnn_correctness_test_2gpu \
# no_rocm : //tensorflow/python/keras/distribute:keras_embedding_model_correctness_test_2gpu \
      
# TIMEOUT : //tensorflow/python/distribute:values_test_2gpu \
# TIMEOUT : //tensorflow/python/keras/distribute:keras_image_model_correctness_test_2gpu \
# TIMEOUT : //tensorflow/python/keras/distribute:keras_rnn_model_correctness_test_2gpu \
# TIMEOUT : //tensorflow/python/keras/distribute:saved_model_mixed_api_test_2gpu \
# TIMEOUT : //tensorflow/python/keras/distribute:saved_model_save_load_test_2gpu \

# Started timing-out with ROCm 4.1
# TIMEOUT : //tensorflow/python/keras/distribute:keras_premade_models_test_2gpu \

# Became FLAKY with  ROCm 4.1
# FLAKY : //tensorflow/python/distribute:strategy_common_test_2gpu \
# FLAKY : //tensorflow/python/distribute:strategy_common_test_xla_2gpu \
# FLAKY : //tensorflow/python/distribute:strategy_gather_test_2gpu \
# FLAKY : //tensorflow/python/distribute:strategy_gather_test_xla_2gpu \
# FLAKY : //tensorflow/python/keras/distribute:custom_training_loop_metrics_test_2gpu \
# FLAKY : //tensorflow/python/keras/distribute:custom_training_loop_models_test_2gpu \

# FAILED : //tensorflow/python/keras/distribute:distribute_strategy_test_2gpu \

# FAILED ON CI Node only : //tensorflow/python/distribute:collective_all_reduce_strategy_test_2gpu \
# See run : http://ml-ci.amd.com:21096/job/tensorflow/job/github-prs-rocmfork-develop-upstream/job/rocm-latest-ubuntu-gpu-multi/216/console

# FAILED ON CI Node only : //tensorflow/python/keras/distribute:keras_save_load_test_2gpu \
# Starting with ROCm 4.1, see run : http://ml-ci.amd.com:21096/job/tensorflow/job/github-prs-rocmfork-develop-upstream/job/rocm-latest-ubuntu-gpu-multi/241/console

# FAILED  //tensorflow/python/keras/distribute:minimize_loss_test_2gpu \
# potential breaking commit : https://github.com/tensorflow/tensorflow/commit/74e39c8fa60079862597c9db506cd15b2443a5a2

# NO MORE MULTI_GPU : //tensorflow/python/keras/distribute:checkpointing_test_2gpu \
# multi_gpu tag was commented out in this commit : https://github.com/tensorflow/tensorflow/commit/b87d02a3f8d8b55045bf4250dd72e746357a3eba
