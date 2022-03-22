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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {

// Returns true if 'op' is non const op. Returns false otherwise or if
// 'op' is null.
inline bool IsNonConstOp(Operation* op) {
  if (!op) return false;
  if (llvm::isa<arith::ConstantOp, mlir::func::ConstantOp>(op)) return false;
  if (op->hasTrait<OpTrait::ConstantLike>()) return false;
  if (llvm::isa<TFL::ConstOp, TFL::QConstOp>(op)) return false;
  return true;
}

// Returns true if 'op' is a terminator op, otherwise false.
bool IsTerminatorOp(Operation* op);

// Returns true if 'op' is not TFL Quant / Dequant op. Returns False otherwise
// or if 'op' is null.
bool NotTFLQuantDequantizeOp(Operation* op);

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t) {
  if (auto shaped_type = t.dyn_cast_or_null<ShapedType>()) {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Return true when the given element_type is QI8.
inline bool IsQI8Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         quantized_type.isSigned();
}

// Return true when the given element_type is QUI8.
inline bool IsQUI8Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 8 &&
         !quantized_type.isSigned();
}

// Return true when the given element_type is QI32.
inline bool IsQI32Type(Type t) {
  auto quantized_type = quant::QuantizedType::getQuantizedElementType(t);
  return quantized_type != nullptr &&
         quantized_type.getStorageTypeIntegralWidth() == 32 &&
         quantized_type.isSigned();
}

// Try to guess the inference type of the op.
InferenceType GetInferenceType(Operation* op);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_UTILS_H_
