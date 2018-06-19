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
export PYTHON_BIN_PATH=`which python3`
export CC_OPT_FLAGS='-mavx'

export TF_NEED_ROCM=1

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --config=rocm --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-benchmark-test -k \
    --test_lang_filters=py --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --build_tests_only --test_output=errors --local_test_jobs=4 --config=opt \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute -- \
    //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/... \
    -//tensorflow/python/eager:backprop_test \
    -//tensorflow/python/eager:function_test \
    -//tensorflow/python/estimator:boosted_trees_test   \
    -//tensorflow/python/feature_column:feature_column_test \
    -//tensorflow/python/keras:activations_test \
    -//tensorflow/python/keras:core_test \
    -//tensorflow/python/keras:gru_test \
    -//tensorflow/python/keras:local_test \
    -//tensorflow/python/keras:lstm_test \
    -//tensorflow/python/keras:model_subclassing_test \
    -//tensorflow/python/keras:normalization_test \
    -//tensorflow/python/keras:pooling_test \
    -//tensorflow/python/keras:sequential_test \
    -//tensorflow/python/keras:simplernn_test \
    -//tensorflow/python/keras:training_eager_test \
    -//tensorflow/python/keras:wrappers_test \
    -//tensorflow/python/kernel_tests:atrous_conv2d_test \
    -//tensorflow/python/kernel_tests:batch_matmul_op_test \
    -//tensorflow/python/kernel_tests:bias_op_test \
    -//tensorflow/python/kernel_tests:bincount_op_test \
    -//tensorflow/python/kernel_tests:cholesky_op_test \
    -//tensorflow/python/kernel_tests:concat_op_test \
    -//tensorflow/python/kernel_tests:control_flow_ops_py_test \
    -//tensorflow/python/kernel_tests:conv_ops_3d_test \
    -//tensorflow/python/kernel_tests:conv_ops_test \
    -//tensorflow/python/kernel_tests:conv1d_test \
    -//tensorflow/python/kernel_tests:conv2d_backprop_filter_grad_test \
    -//tensorflow/python/kernel_tests:conv2d_transpose_test \
    -//tensorflow/python/kernel_tests:cwise_ops_test  \
    -//tensorflow/python/kernel_tests:dct_ops_test \
    -//tensorflow/python/kernel_tests:depthwise_conv_op_test \
    -//tensorflow/python/kernel_tests:fft_ops_test \
    -//tensorflow/python/kernel_tests:functional_ops_test \
    -//tensorflow/python/kernel_tests:init_ops_test \
    -//tensorflow/python/kernel_tests:linalg_grad_test \
    -//tensorflow/python/kernel_tests:linalg_ops_test \
    -//tensorflow/python/kernel_tests:losses_test \
    -//tensorflow/python/kernel_tests:lrn_op_test \
    -//tensorflow/python/kernel_tests:matmul_op_test \
    -//tensorflow/python/kernel_tests:matrix_inverse_op_test \
    -//tensorflow/python/kernel_tests:matrix_solve_ls_op_test \
    -//tensorflow/python/kernel_tests:matrix_triangular_solve_op_test \
    -//tensorflow/python/kernel_tests:metrics_test \
    -//tensorflow/python/kernel_tests:neon_depthwise_conv_op_test \
    -//tensorflow/python/kernel_tests:pool_test \
    -//tensorflow/python/kernel_tests:pooling_ops_3d_test \
    -//tensorflow/python/kernel_tests:pooling_ops_test \
    -//tensorflow/python/kernel_tests:qr_op_test \
    -//tensorflow/python/kernel_tests:reduction_ops_test   \
    -//tensorflow/python/kernel_tests:scan_ops_test \
    -//tensorflow/python/kernel_tests:scatter_nd_ops_test \
    -//tensorflow/python/kernel_tests:scatter_ops_test \
    -//tensorflow/python/kernel_tests:segment_reduction_ops_test \
    -//tensorflow/python/kernel_tests:self_adjoint_eig_op_test \
    -//tensorflow/python/kernel_tests:split_op_test \
    -//tensorflow/python/kernel_tests:svd_op_test \
    -//tensorflow/python/kernel_tests:tensor_array_ops_test \
    -//tensorflow/python/kernel_tests:tensordot_op_test \
    -//tensorflow/python/kernel_tests:variable_scope_test \
    -//tensorflow/python/profiler/internal:run_metadata_test \
    -//tensorflow/python/profiler:profile_context_test \
    -//tensorflow/python/profiler:profiler_test \
    -//tensorflow/python:adam_test \
    -//tensorflow/python:cluster_test \
    -//tensorflow/python:cost_analyzer_test \
    -//tensorflow/python:function_test \
    -//tensorflow/python:gradient_checker_test \
    -//tensorflow/python:gradients_test \
    -//tensorflow/python:histogram_ops_test \
    -//tensorflow/python:image_grad_test \
    -//tensorflow/python:image_ops_test \
    -//tensorflow/python:layers_normalization_test \
    -//tensorflow/python:layout_optimizer_test \
    -//tensorflow/python:learning_rate_decay_test \
    -//tensorflow/python:math_ops_test \
    -//tensorflow/python:memory_optimizer_test \
    -//tensorflow/python:nn_fused_batchnorm_test \
    -//tensorflow/python:timeline_test \
    -//tensorflow/python:virtual_gpu_test \
    -//tensorflow/python:function_def_to_graph_test \
    -//tensorflow/python:gradient_descent_test \
    -//tensorflow/python:momentum_test \
    -//tensorflow/python/keras:models_test \
    -//tensorflow/python/keras:training_test \
    -//tensorflow/python/keras:cudnn_recurrent_test \
    -//tensorflow/python/kernel_tests/distributions:beta_test \
    -//tensorflow/python/kernel_tests/distributions:dirichlet_test \
    -//tensorflow/python/kernel_tests/distributions:student_t_test \
    -//tensorflow/python/estimator:baseline_test \
    -//tensorflow/python/estimator:dnn_test \
    -//tensorflow/python/estimator:estimator_test \
    -//tensorflow/python/estimator:linear_test \
    -//tensorflow/python/estimator:dnn_linear_combined_test

# Note: temp. disabling 93 unit tests in order to esablish a CI baseline (2018/06/13)
