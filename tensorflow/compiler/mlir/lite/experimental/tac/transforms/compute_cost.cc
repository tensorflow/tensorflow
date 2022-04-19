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

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/cost.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// We will caculate the total compute cost for each Func Op.
//
// The compute cost is simply an add-up of the costs of all the operations
// within the FuncOp. (Excluding const ops since they're just "data".)
// We will ignore quant/dequant/requant costs within the Func Op as well,
// intuition:
//
// The assumpution is that quant/dequant/requant will only happen at the begin
// and the end of the FuncOp (basically the "boundaries" of the subgraph).
// So we can imagine if multiple "same-inference-typed" graph are presented at
// the same time, the quant/dequant ops pair can be squashed:
//
//         dequant         ------------
//            |
//          ops...             FuncOp1
//            |
//         quant           -------------
//           |         <--- can be squashed
//         dequant         -------------
//            |
//        ops...               FuncOp2
//           |
//         quant          ---------------
//
// But it's true quant & dequant ops can happen "within" the FuncOp as well,
// normally as "quantization params" adjust. We should check more careful to
// include those as those ops wouldn't be "squashed".

class ComputeCostPass
    : public mlir::PassWrapper<ComputeCostPass, mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComputeCostPass)

 private:
  llvm::StringRef getArgument() const final { return "tfl-compute-cost"; }
  llvm::StringRef getDescription() const final {
    return "Compute the total cost for each available subgraph.";
  }
  void runOnOperation() override;
};

void ComputeCostPass::runOnOperation() {
  auto module = getOperation();

  for (auto func : module.getOps<func::FuncOp>()) {
    // We only care about those functions annotated with "tac.interface_name".
    auto interface_name = GetInterFaceName(func);
    if (!interface_name.hasValue()) continue;

    auto target = GetTargetAnnotation(func);
    if (!target.hasValue()) {
      func.emitError("we cannot get hardware info for this function.");
      signalPassFailure();
    }

    float total_cost = GetCostForFunc(&func, target.getValue());
    OpBuilder builder(func);
    UpdateCost(func, total_cost, &builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateComputeCostPass() {
  return std::make_unique<ComputeCostPass>();
}

static PassRegistration<ComputeCostPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
