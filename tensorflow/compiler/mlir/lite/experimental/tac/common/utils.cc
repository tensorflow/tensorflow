/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"

#include "mlir/IR/OpDefinition.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

bool NotTFLQuantDequantizeOp(Operation* op) {
  if (!op) return false;
  if (llvm::isa<TFL::QuantizeOp, TFL::DequantizeOp>(op)) return false;
  return true;
}

bool IsTerminatorOp(Operation* op) {
  if (!op) return false;
  return op->hasTrait<OpTrait::IsTerminator>();
}

// Try to guess the inference type of the op.
InferenceType GetInferenceType(Operation* op) {
  bool float_type_observed = false;
  bool int8_type_observed = false;
  bool uint8_type_observed = false;
  for (auto& input : op->getOpOperands()) {
    auto input_type = input.get().getType();
    if (IsF32ShapedType(input_type)) {
      float_type_observed = true;
    } else if (IsQI8Type(input_type)) {
      int8_type_observed = true;
    } else if (IsQUI8Type(input_type)) {
      uint8_type_observed = true;
    }
  }

  // We should not observe both uint8 & int8.
  if (int8_type_observed && uint8_type_observed) return UNKNOWN;

  if (float_type_observed) {
    if (int8_type_observed || uint8_type_observed) {
      return HYBRID;
    } else {
      return FLOAT;
    }
  }

  if (int8_type_observed) {
    return QUANTIZED_INT8;
  }

  if (uint8_type_observed) {
    return QUANTIZED_UINT8;
  }

  // Default to float inference.
  return FLOAT;
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
