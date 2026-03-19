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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct IsFiniteOp {
  struct Attributes {};
};

IsFiniteOp Create(const IsFiniteOp::Attributes& attributes);
absl::Status Prepare(IsFiniteOp& op, const Tensor& operand, Tensor& result);
absl::Status Evaluate(IsFiniteOp& op, const Tensor& operand, Tensor& result);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_
