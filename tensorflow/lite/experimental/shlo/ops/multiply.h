/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_MULTIPLY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_MULTIPLY_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct MultiplyOp {
  struct Attributes {};
};

MultiplyOp Create(MultiplyOp::Attributes);
absl::Status Prepare(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_MULTIPLY_H_
