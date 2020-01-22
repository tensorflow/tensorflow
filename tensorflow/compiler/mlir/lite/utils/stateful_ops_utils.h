/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STATEFUL_OPS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STATEFUL_OPS_UTILS_H_

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir

namespace mlir {
namespace TFL {

// Check if the given op has stateful operands and return their stateful
// operand indices.
bool IsStatefulOp(Operation* op, std::vector<int>* stateful_operand_indices);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STATEFUL_OPS_UTILS_H_
