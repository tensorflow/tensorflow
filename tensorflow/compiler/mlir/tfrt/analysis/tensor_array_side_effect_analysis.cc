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
#include "tensorflow/compiler/mlir/tfrt/analysis/tensor_array_side_effect_analysis.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace tfrt_compiler {

bool IsTensorArrayOp(mlir::Operation* op) {
  return llvm::isa<mlir::TF::TensorArrayV3Op, mlir::TF::TensorArrayScatterV3Op,
                   mlir::TF::TensorArrayGatherV3Op,
                   mlir::TF::TensorArrayReadV3Op,
                   mlir::TF::TensorArrayWriteV3Op>(op);
}

static bool FunctionContainsOnlyNoSideEffectOpOrTensorArrayOp(
    mlir::FuncOp func_op) {
  for (mlir::Operation& op : func_op.front()) {
    if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op) &&
        !IsTensorArrayOp(&op))
      return false;
  }

  return true;
}

TensorArraySideEffectAnalysis::TensorArraySideEffectAnalysis(
    mlir::ModuleOp module) {
  for (auto func_op : module.getOps<mlir::FuncOp>()) {
    if (FunctionContainsOnlyNoSideEffectOpOrTensorArrayOp(func_op)) {
      set_.insert(func_op);
    }
  }
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
