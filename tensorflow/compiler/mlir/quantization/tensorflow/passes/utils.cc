/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"

#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace quant {

bool HasQuantizedTensors(Operation* op) {
  if (IsOpNotQuantizable(op)) return false;
  for (Type operand_type : op->getOperandTypes()) {
    auto tensor_type = operand_type.dyn_cast<TensorType>();
    if (tensor_type && tensor_type.getElementType().isa<QuantizedType>()) {
      return true;
    }
  }
  for (Type result_type : op->getResultTypes()) {
    auto tensor_type = result_type.dyn_cast<TensorType>();
    if (tensor_type && tensor_type.getElementType().isa<QuantizedType>()) {
      return true;
    }
  }
  return false;
}

Type CloneTypeWithNewElementType(Type old_type, Type element_type) {
  if (!old_type.isa<ShapedType>()) return {};

  return old_type.cast<ShapedType>().clone(element_type);
}
}  // namespace quant
}  // namespace mlir
