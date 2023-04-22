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

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"

namespace mlir {
namespace TF {

namespace {

template <typename Interface>
LogicalResult VerifyLayoutDependentArgsAndResults(Operation* op,
                                                  Interface interface) {
  auto valid_operand = [&](int64_t idx) { return idx < op->getNumOperands(); };
  if (!llvm::all_of(interface.GetLayoutDependentArgs(), valid_operand)) {
    return op->emitOpError("layout dependent argument index is out of bound");
  }

  auto valid_result = [&](int64_t idx) { return idx < op->getNumResults(); };
  if (!llvm::all_of(interface.GetLayoutDependentResults(), valid_result)) {
    return op->emitOpError("layout dependent result index is out of bound");
  }

  return success();
}

}  // namespace

LogicalResult VerifyLayoutSensitiveInterface(Operation* op) {
  auto layout_sensitive_interface = cast<LayoutSensitiveInterface>(op);
  return VerifyLayoutDependentArgsAndResults(op, layout_sensitive_interface);
}

LogicalResult VerifyFoldOperandsTransposeInterface(Operation* op) {
  auto fold_operands_transpose = cast<FoldOperandsTransposeInterface>(op);
  return VerifyLayoutDependentArgsAndResults(op, fold_operands_transpose);
}

}  // namespace TF
}  // namespace mlir
