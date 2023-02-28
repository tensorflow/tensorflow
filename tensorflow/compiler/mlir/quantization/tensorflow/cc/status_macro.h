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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_STATUS_MACRO_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_STATUS_MACRO_H_

#include "tensorflow/tsl/platform/macros.h"

namespace tensorflow {
namespace quantization {

// Similar to TF_RETURN_IF_ERROR but used for `absl::Status`.
#define TF_QUANT_RETURN_IF_ERROR(expr)                   \
  do {                                                   \
    ::absl::Status _status = (expr);                     \
    if (TF_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_STATUS_MACRO_H_
