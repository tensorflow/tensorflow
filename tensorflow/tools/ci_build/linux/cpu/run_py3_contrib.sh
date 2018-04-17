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

# Run configure.
export TF_NEED_CUDA=0
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=`which python3`
yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --test_tag_filters=-no_oss,-oss_serial,-gpu,-benchmark-test -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 --config=opt \
    --test_output=errors -- \
    //tensorflow/contrib/... \
    -//tensorflow/contrib/lite/... \
    //tensorflow/contrib/lite:context_test \
    //tensorflow/contrib/lite:framework \
    //tensorflow/contrib/lite:interpreter_test \
    //tensorflow/contrib/lite:model_test \
    //tensorflow/contrib/lite/toco:toco \
    //tensorflow/contrib/lite:simple_memory_arena_test \
    //tensorflow/contrib/lite:string_util_test \
    //tensorflow/contrib/lite/kernels:activations_test \
    //tensorflow/contrib/lite/kernels:add_test \
    //tensorflow/contrib/lite/kernels:basic_rnn_test \
    //tensorflow/contrib/lite/kernels:concatenation_test \
    //tensorflow/contrib/lite/kernels:conv_test \
    //tensorflow/contrib/lite/kernels:depthwise_conv_test \
    //tensorflow/contrib/lite/kernels:embedding_lookup_test \
    //tensorflow/contrib/lite/kernels:embedding_lookup_sparse_test \
    //tensorflow/contrib/lite/kernels:fully_connected_test \
    //tensorflow/contrib/lite/testing:generated_examples_zip_test \
    //tensorflow/contrib/lite/kernels:hashtable_lookup_test \
    //tensorflow/contrib/lite/kernels:local_response_norm_test \
    //tensorflow/contrib/lite/kernels:lsh_projection_test \
    //tensorflow/contrib/lite/kernels:lstm_test \
    //tensorflow/contrib/lite/kernels:l2norm_test \
    //tensorflow/contrib/lite/kernels:mul_test \
    //tensorflow/contrib/lite/kernels:pooling_test \
    //tensorflow/contrib/lite/kernels:reshape_test \
    //tensorflow/contrib/lite/kernels:resize_bilinear_test \
    //tensorflow/contrib/lite/kernels:skip_gram_test \
    //tensorflow/contrib/lite/kernels:softmax_test \
    //tensorflow/contrib/lite/kernels:space_to_depth_test \
    //tensorflow/contrib/lite/kernels:svdf_test
