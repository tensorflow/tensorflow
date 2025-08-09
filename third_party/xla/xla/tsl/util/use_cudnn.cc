/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tsl/util/use_cudnn.h"

#include <cstdint>

#include "xla/tsl/util/env_var.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/stringpiece.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tsl {

#define ADD_BOOL_CUDNN_FLAG(func_name, flag_name, default_value) \
  bool func_name() {                                             \
    bool value = default_value;                                  \
    absl::Status status =                                        \
        ReadBoolFromEnvVar(#flag_name, default_value, &value);   \
    if (!status.ok()) {                                          \
      LOG(ERROR) << status;                                      \
    }                                                            \
    return value;                                                \
  }

// Whether to enable Cudnn runtime compiled kernels which are able to support
// more general fusion patterns but might increase the warmup time.
// TODO(kaixih@nvidia): we can make it default when Cudnn further improves the
// runtime compilation overhead.
bool CudnnUseRuntimeFusion() {
  static bool result = [] {
    bool value = false;
#if GOOGLE_CUDA
    absl::Status status =
        ReadBoolFromEnvVar("TF_CUDNN_USE_RUNTIME_FUSION", false, &value);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
#endif  // GOOGLE_CUDA
    return value;
  }();
  return result;
}

ADD_BOOL_CUDNN_FLAG(CudnnUseAutotune, TF_CUDNN_USE_AUTOTUNE, true);
// Whether to auto-tuning Cudnn RNN forward and backward pass to pick
// statistically the best cudnnRNNAlgo_t and cudnnMathType_t.
// The flag is disabled when TF_DEBUG_CUDNN_RNN is turned on.
ADD_BOOL_CUDNN_FLAG(CudnnRnnUseAutotune, TF_CUDNN_RNN_USE_AUTOTUNE, true);
ADD_BOOL_CUDNN_FLAG(CudnnDisableConv1x1Optimization,
                    TF_CUDNN_DISABLE_CONV_1X1_OPTIMIZATION, false);

// Whether to run Cudnn RNN forward and backward in debug mode, where users can
// force a specified cudnnRNNAlgo_t and cudnnMathType_t, when used together with
// the following two env vars:
// TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS
// TF_DEBUG_CUDNN_RNN_ALGO
// By default it is disabled and only intended for testing and profiling.
ADD_BOOL_CUDNN_FLAG(DebugCudnnRnn, TF_DEBUG_CUDNN_RNN, false);
// If using TENSOR_OP_MATH in Cudnn RNN for both forward and backward pass. Only
// effective when TF_DEBUG_CUDNN_RNN is true.
// Note none of the persistent RNN algorithm support TENSOR_OP_MATH before
// Cudnn 7.1. See Nvidia Cudnn manual for more details.
ADD_BOOL_CUDNN_FLAG(DebugCudnnRnnUseTensorOps,
                    TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS, false);
#undef ADD_BOOL_CUDNN_FLAG

#define ADD_INT64_CUDNN_FLAG(func_name, flag_name, default_value) \
  int64_t func_name() {                                           \
    int64_t value = default_value;                                \
    absl::Status status =                                         \
        ReadInt64FromEnvVar(#flag_name, default_value, &value);   \
    if (!status.ok()) {                                           \
      LOG(ERROR) << status;                                       \
    }                                                             \
    return value;                                                 \
  }
// Cudnn RNN algorithm to use for both forward and backward pass. Only effective
// when TF_DEBUG_CUDNN_RNN is true. See Nvidia Cudnn manual for allowed
// cudnnRNNAlgo_t.
ADD_INT64_CUDNN_FLAG(DebugCudnnRnnAlgo, TF_DEBUG_CUDNN_RNN_ALGO, -1);
#undef ADD_INT64_CUDNN_FLAG

bool ShouldCudnnGroupedConvolutionBeUsed(const int32_t filter_rows,
                                         const int32_t filter_cols,
                                         const int32_t in_depth,
                                         const int32_t out_depth) {
  return in_depth == out_depth && filter_rows == filter_cols &&
         (filter_rows == 1 || filter_rows == 3 || filter_rows == 5 ||
          filter_rows == 7);
}

}  // namespace tsl
