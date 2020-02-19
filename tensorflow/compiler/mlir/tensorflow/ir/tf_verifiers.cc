/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_verifiers.h"

#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

LogicalResult VerifyLayoutSensitiveInterface(Operation* op) {
  auto layout_sensitive_interface = cast<LayoutSensitiveInterface>(op);

  if (!llvm::all_of(
          layout_sensitive_interface.GetLayoutDependentArgs(),
          [&](int64_t index) { return index < op->getNumOperands(); })) {
    return op->emitOpError("layout dependent argument index is out of bound");
  }

  if (!llvm::all_of(
          layout_sensitive_interface.GetLayoutDependentResults(),
          [&](int64_t index) { return index < op->getNumResults(); })) {
    return op->emitOpError("layout dependent result index is out of bound");
  }

  return success();
}

}  // namespace TF
}  // namespace mlir
