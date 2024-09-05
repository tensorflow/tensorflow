/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_NUMERIC_OPTIONS_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_NUMERIC_OPTIONS_UTILS_H_

#include "xla/stream_executor/numeric_options.h"
#include "xla/tsl/util/determinism.h"
#include "tensorflow/core/util/env_var.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace tensorflow {

inline stream_executor::NumericOptions GetNumericOptions() {
  return stream_executor::NumericOptions{
      /*require_determinism=*/tsl::OpDeterminismRequired(),
      /*allow_tf32=*/tsl::tensor_float_32_execution_enabled()};
}

inline stream_executor::NumericOptions GetNumericOptionsForCuDnn() {
  static bool cudnn_deterministic_env_var = [] {
    bool cudnn_deterministic = false;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_CUDNN_DETERMINISTIC",
                                   /*default_val=*/false,
                                   &cudnn_deterministic));
    return cudnn_deterministic;
  }();
  stream_executor::NumericOptions result = GetNumericOptions();
  result.require_determinism |= cudnn_deterministic_env_var;
  return result;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_NUMERIC_OPTIONS_UTILS_H_
