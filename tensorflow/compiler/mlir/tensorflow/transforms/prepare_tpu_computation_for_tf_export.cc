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

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {
namespace {

class PrepareTpuComputationForTfExportPass
    : public PassWrapper<PrepareTpuComputationForTfExportPass, FunctionPass> {
  void runOnFunction() override;
};

void PrepareTpuComputationForTfExportPass::runOnFunction() {
  auto func = getFunction();
  OpBuilder builder(func.getBody());
  for (int i = 0; i < func.getNumArguments(); ++i) {
    constexpr char kShardingAttr[] = "mhlo.sharding";
    if (auto sharding =
            func.getArgAttrOfType<mlir::StringAttr>(i, kShardingAttr)) {
      if (!sharding.getValue().empty()) {
        BlockArgument arg = func.getArgument(i);
        // TODO(hinsu): Instead of setting both 'sharding' and '_XlaSharding'
        // attributes, only set the 'sharding' attribute. Both attributes are
        // currently required as the XlaSharding xla op kernel doesn't use the
        // 'sharding' attribute.
        auto updated_arg = builder.create<TF::XlaShardingOp>(
            func.getLoc(), arg.getType(), arg, sharding, sharding);
        func.getArgument(i).replaceAllUsesExcept(
            updated_arg, llvm::SmallPtrSet<Operation*, 1>({updated_arg}));
      }

      func.removeArgAttr(i, builder.getIdentifier(kShardingAttr));
    }

    // TODO(prakalps, hinsu): Utilize aliasing output attribute instead of
    // dropping it. This only affects performance and is not required for
    // correctness.
    constexpr char kAliasingAttr[] = "tf.aliasing_output";
    func.removeArgAttr(i, builder.getIdentifier(kAliasingAttr));
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreatePrepareTpuComputationForTfExportPass() {
  return std::make_unique<PrepareTpuComputationForTfExportPass>();
}

}  // namespace TF
}  // namespace mlir
