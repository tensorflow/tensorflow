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

#include "tensorflow/core/ir/tf_op_registry.h"

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"

namespace mlir {
namespace tfg {
TensorFlowOpRegistryInterface::TensorFlowOpRegistryInterface(Dialect *dialect)
    : TensorFlowOpRegistryInterface(dialect, tensorflow::OpRegistry::Global()) {
}

// Returns true if the op is stateful.
static bool IsStatefulImpl(const tensorflow::OpRegistry *registry,
                           StringRef op_name) {
  const tensorflow::OpRegistrationData *op_reg_data =
      registry->LookUp(op_name.str());
  // If an op definition was not found, conservatively assume stateful.
  if (!op_reg_data) return true;
  return op_reg_data->op_def.is_stateful();
}

bool TensorFlowOpRegistryInterface::isStateful(Operation *op) const {
  // Handle TFG internal ops.
  if (op->hasTrait<OpTrait::IntrinsicOperation>()) return false;
  if (auto func = dyn_cast<GraphFuncOp>(op)) return func.getIsStateful();
  // Handle TFG region ops.
  // TODO(jeffniu): Region ops should be marked with a trait.
  StringRef op_name = op->getName().stripDialect();
  if (op->getNumRegions() && op_name.ends_with("Region"))
    op_name = op_name.drop_back(/*len("Region")=*/6);
  return IsStatefulImpl(registry_, op_name);
}
}  // namespace tfg
}  // namespace mlir
