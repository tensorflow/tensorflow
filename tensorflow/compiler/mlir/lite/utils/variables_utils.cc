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

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace utils {

bool IsSupportedVariableType(Operation* op) {
  ShapedType type;
  if (llvm::isa<TF::ReadVariableOp>(op)) {
    type = op->getResult(0).getType().cast<ShapedType>();
  } else if (llvm::isa<TF::AssignVariableOp>(op)) {
    type = op->getOperand(1).getType().cast<ShapedType>();
  }
  return type.getElementType().isF32();
}

}  // namespace utils
}  // namespace TFL
}  // namespace mlir
