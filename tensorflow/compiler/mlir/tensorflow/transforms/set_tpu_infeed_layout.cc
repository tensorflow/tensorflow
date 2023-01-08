/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/set_tpu_infeed_layout.h"

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/adjust_layout.h"

namespace mlir {

bool SetTPUInfeedLayout(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  auto res = mlir_module->walk([&](mlir::TF::InfeedDequeueTupleOp op) {
    mlir::OpBuilder builder(op.getContext());
    std::vector<mlir::Type> result_types;

    for (mlir::Type t : op.getResultTypes()) {
      auto ty = t.cast<mlir::TensorType>();
      if (!ty.hasStaticShape()) return mlir::WalkResult::interrupt();
      result_types.push_back(t);
    }

    auto layout = mlir::mhlo::GetTPUInfeedLayout(
        mlir::TupleType::get(builder.getContext(), result_types), builder);
    if (failed(layout)) return mlir::WalkResult::interrupt();
    // Do not append a UnitAttr for the "token" operand here to avoid
    // compilation failure when exporting the "layouts" attribute to a graph
    // node. Instead, add the UnitAttr during LegalizeTF pass.
    op->setAttr("layouts", layout.value());

    return mlir::WalkResult::advance();
  });
  return !res.wasInterrupted();
}

}  // namespace mlir
