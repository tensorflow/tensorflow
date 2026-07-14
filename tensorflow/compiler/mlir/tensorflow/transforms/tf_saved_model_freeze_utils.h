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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace tf_saved_model {
// Container to hold all update actions on ops.
// Key: Operation to update.
// Value: optional list of argument indices to delete from this op.
// Note that we use MapVector because we want to iterate on the same order
// of insertion.
LogicalResult EraseObsoleteResourceUses(
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>
        arguments_to_erase);

// Traces usage of 'var_handle_op' or 'resources' and replaces it's usage with
// constant value 'value'. All op operands updates are captured in
// 'arguments_to_erase'.
LogicalResult ReplaceVarWithConstant(
    mlir::Value::use_range uses, ElementsAttr value,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase);
}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_
