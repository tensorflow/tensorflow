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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_

#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {

using llvm::ArrayRef;
using mlir::Operation;
using mlir::ShapedType;
using mlir::Value;

// Returns true if each Operand at given indices have the same static shape.
inline bool OpHasSameStaticShapes(Operation* op,
                                  llvm::ArrayRef<int> operand_idxs) {
  if (op->getNumOperands() == 0 || operand_idxs.empty()) return true;
  const int first_opr_idx = operand_idxs[0];
  ArrayRef<int64_t> shape =
      op->getOperand(first_opr_idx).getType().dyn_cast<ShapedType>().getShape();
  for (int opr_idx : operand_idxs) {
    Value operand = op->getOperand(opr_idx);
    auto shaped_type = operand.getType().dyn_cast<ShapedType>();
    if (!shaped_type || !shaped_type.hasStaticShape()) {
      return false;
    }
    if (shape != shaped_type.getShape()) {
      return false;
    }
  }
  return true;
}

// Returns true if each Operand has the same static shape.
inline bool OpHasSameStaticShapes(Operation* op) {
  llvm::OwningArrayRef<int> operand_idxs(op->getNumOperands());
  std::iota(operand_idxs.begin(), operand_idxs.end(), 0);
  return OpHasSameStaticShapes(op, operand_idxs);
}
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
