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

#include "tensorflow/core/util/use_cudnn.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

#define ADD_BOOL_CUDNN_FLAG(func_name, flag_name, default_value)           \
  bool func_name() {                                                       \
    bool value = default_value;                                            \
    Status status = ReadBoolFromEnvVar(#flag_name, default_value, &value); \
    if (!status.ok()) {                                                    \
      LOG(ERROR) << status;                                                \
    }                                                                      \
    return value;                                                          \
  }

ADD_BOOL_CUDNN_FLAG(CanUseCudnn, TF_USE_CUDNN, true);
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

#define ADD_INT64_CUDNN_FLAG(func_name, flag_name, default_value)           \
  int64 func_name() {                                                       \
    int64 value = default_value;                                            \
    Status status = ReadInt64FromEnvVar(#flag_name, default_value, &value); \
    if (!status.ok()) {                                                     \
      LOG(ERROR) << status;                                                 \
    }                                                                       \
    return value;                                                           \
  }
// Cudnn RNN algorithm to use for both forward and backward pass. Only effective
// when TF_DEBUG_CUDNN_RNN is true. See Nvidia Cudnn manual for allowed
// cudnnRNNAlgo_t.
ADD_INT64_CUDNN_FLAG(DebugCudnnRnnAlgo, TF_DEBUG_CUDNN_RNN_ALGO, -1);
#undef ADD_INT64_CUDNN_FLAG

FP16ConvMode CudnnConvComputeMode() {
  string value;
  Status status = ReadStringFromEnvVar("TF_FP16_CONV_MODE", "accurate", &value);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  string lowercase_value = absl::AsciiStrToLower(value);
  if (lowercase_value == "accurate") {
    return FP16ConvMode::kAccurate;
  } else if (lowercase_value == "fast") {
    return FP16ConvMode::kFast;
  } else {
    LOG(ERROR) << "FP16ConvMode only supports two modes, ACCURATE and FAST. "
                  "Got unknown mode: "
               << value;
  }
  return FP16ConvMode::kAccurate;
}

bool IsCudnnSupportedFilterSize(const int32 filter_rows,
                                const int32 filter_cols, const int32 in_depth,
                                const int32 out_depth) {
  return in_depth == out_depth && filter_rows == filter_cols &&
         (filter_rows == 1 || filter_rows == 3 || filter_rows == 5 ||
          filter_rows == 7);
}

}  // namespace tensorflow
