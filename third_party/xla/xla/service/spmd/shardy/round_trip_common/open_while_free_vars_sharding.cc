/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/round_trip_common/open_while_free_vars_sharding.h"

#include <memory>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::StringRef;
using ::mlir::func::FuncOp;
using ::mlir::sdy::TensorShardingAttr;

class OpenWhileFreeVarsShardingPass
    : public mlir::PassWrapper<OpenWhileFreeVarsShardingPass,
                               mlir::OperationPass<FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenWhileFreeVarsShardingPass)

  void runOnOperation() final {
    FuncOp funcOp = getOperation();
    mlir::IRRewriter rewriter(funcOp);

    funcOp.walk([&](mlir::stablehlo::WhileOp op) {
      llvm::SetVector<mlir::Value> freeVars;
      mlir::getUsedValuesDefinedAbove(op->getRegions(), freeVars);
      rewriter.setInsertionPoint(op);
      for (mlir::Value freeVar : freeVars) {
        TensorShardingAttr sharding = mlir::sdy::getSharding(freeVar);
        if (!sharding || sharding.getRank() == 0) {
          continue;
        }
        auto fullyOpenSharding = TensorShardingAttr::getFullyOpenLike(sharding);
        if (fullyOpenSharding == sharding) {
          // The sharding of the `freeVar` is already fully open, no need to add
          // a sharding constraint.
          continue;
        }
        auto shardingConstraint =
            rewriter.create<mlir::sdy::ShardingConstraintOp>(
                freeVar.getLoc(), freeVar, fullyOpenSharding);
        // Only replace uses in the regions of the while op.
        rewriter.replaceUsesWithIf(
            freeVar, shardingConstraint, [op](mlir::OpOperand& use) {
              return op->isProperAncestor(use.getOwner());
            });
      }
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-open-while-free-vars-sharding";
  }

  StringRef getDescription() const override {
    return "Adds a fully open sharding constraint to free variables of while "
           "op that already have a sharding.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createOpenWhileFreeVarsShardingPass() {
  return std::make_unique<OpenWhileFreeVarsShardingPass>();
}

void registerOpenWhileFreeVarsShardingPass() {
  mlir::registerPass(createOpenWhileFreeVarsShardingPass);
}

}  // namespace sdy
}  // namespace xla
