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
#include "tensorflow/compiler/mlir/lite/utils/variables_utils.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TFL {
namespace utils {

bool IsSupportedVariableType(Operation* op) {
  ShapedType type;
  if (llvm::isa<TF::ReadVariableOp>(op)) {
    type = op->getResult(0).getType().cast<ShapedType>();
  } else if (llvm::isa<TF::AssignVariableOp>(op)) {
    type = op->getOperand(1).getType().cast<ShapedType>();
  } else if (llvm::isa<TF::VarHandleOp>(op)) {
    type = op->getResult(0)
               .getType()
               .cast<ShapedType>()
               .getElementType()
               .cast<TF::TensorFlowTypeWithSubtype>()
               .GetSubtypes()
               .back();
  }
  return IsSupportedVariableType(type);
}

bool IsSupportedVariableType(ShapedType type) {
  auto element_type = type.getElementType();
  // Check complex types.
  if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>()) {
    auto complex_element_type = complex_type.getElementType();
    if (complex_element_type.isF32() || complex_element_type.isF64())
      return true;
  }
  // Check quantized types.
  if (auto quant_type = element_type.dyn_cast<mlir::quant::QuantizedType>()) {
    // TFLite supports QI16, QI32, QI8, and QUI8
    if ((quant_type.getStorageTypeIntegralWidth() == 16 &&
         quant_type.isSigned()) ||
        quant_type.getStorageTypeIntegralWidth() == 8 ||
        (quant_type.getStorageTypeIntegralWidth() == 32 &&
         quant_type.isSigned()))
      return true;
  }
  return element_type.isF32() || element_type.isF64() ||
         element_type.isInteger(1) || element_type.isInteger(8) ||
         element_type.isInteger(32) || element_type.isSignlessInteger(64);
}

}  // namespace utils
}  // namespace TFL
}  // namespace mlir
