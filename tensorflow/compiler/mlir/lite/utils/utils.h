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

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {

using llvm::ArrayRef;
using mlir::Operation;
using mlir::ShapedType;
using mlir::Value;

// Returns true if all tensor value in `values` has static shape and same shape.
inline bool OpHasSameStaticShapes(Operation* op) {
  auto values = op->getOperands();
  int operand_num = 0;
  ArrayRef<int64_t> shape;
  for (Value value : values) {
    auto shaped_type = value.getType().dyn_cast<ShapedType>();
    if (!shaped_type || !shaped_type.hasStaticShape()) {
      return false;
    }
    if (operand_num == 0) {
      shape = shaped_type.getShape();
    } else {
      if (shape != shaped_type.getShape()) {
        return false;
      }
    }
    ++operand_num;
  }
  return true;
}
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
