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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/util.h"

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.h.inc"

namespace tensorflow {
namespace mlrt_compiler {

bool UseFallback(mlir::Operation *op) {
  if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) return false;

  // TODO(b/173017701): have a centralized place to hold the information
  // whether a TF op should be lowered to FallbackExecute op.
  // LINT.IfChange(fallback_allow_list)
  return !llvm::isa<mlir::TF::_TfrtSetResourceOp, mlir::TF::_TfrtGetResourceOp,
                    mlir::TF::BatchFunctionOp, mlir::TF::CaseOp,
                    mlir::TF::StatefulPartitionedCallOp,
                    mlir::TF::PartitionedCallOp, mlir::TF::LegacyCallOp,
                    mlir::TF::IfOp, mlir::TF::WhileOp,
                    mlir::TF::TPUCompileMlirAndExecuteOp>(op);
  // LINT.ThenChange(tf_to_mlrt.cc:fallback_allow_list)
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
