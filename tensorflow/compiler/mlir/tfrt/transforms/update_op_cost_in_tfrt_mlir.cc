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
#include "tensorflow/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"

namespace tensorflow {
namespace tfrt_compiler {

constexpr char kCostAttrName[] = "_tfrt_cost";
constexpr char kOpKeyAttrName[] = "op_key";

void UpdateOpCostInTfrtMlir(mlir::ModuleOp op,
                            const tfrt_stub::CostRecorder& cost_recorder) {
  mlir::Builder builder(op);
  op.walk([&](mlir::Operation* op) {
    // TODO(b/259602527): Add unit test for the precedence.
    // Registered cost function has higher priority than online cost analysis.
    if (HasCostFunctionRegistered(op->getName().getStringRef())) return;
    // Only update ops with existing cost attr.
    const auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(kCostAttrName);
    if (!cost_attr) return;
    // Only fallback ops have `op_key`s.
    const auto op_key_attr =
        op->getAttrOfType<mlir::IntegerAttr>(kOpKeyAttrName);
    if (!op_key_attr) return;
    // Set the cost attr with a new value.
    const int64_t op_key = op_key_attr.getInt();
    op->setAttr(kCostAttrName, builder.getI64IntegerAttr(
                                   cost_recorder.GetCostNanosecond(op_key)));
  });
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
