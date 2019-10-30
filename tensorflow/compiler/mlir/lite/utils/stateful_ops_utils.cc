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

#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"

#include <vector>

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

bool IsStatefulOp(Operation* op, std::vector<int>* stateful_operand_indices) {
  if (auto tfl = dyn_cast_or_null<TFL::LSTMOp>(op)) {
    *stateful_operand_indices = tfl.GetStatefulOperands();
    return true;
  }

  if (auto tfl = dyn_cast_or_null<TFL::UnidirectionalSequenceLSTMOp>(op)) {
    *stateful_operand_indices = tfl.GetStatefulOperands();
    return true;
  }

  if (auto tfl = dyn_cast_or_null<TFL::UnidirectionalSequenceRNNOp>(op)) {
    *stateful_operand_indices = tfl.GetStatefulOperands();
    return true;
  }

  if (auto tfl = dyn_cast_or_null<TFL::SVDFOp>(op)) {
    *stateful_operand_indices = tfl.GetStatefulOperands();
    return true;
  }

  return false;
}

}  // namespace TFL
}  // namespace mlir
