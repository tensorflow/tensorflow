/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project

#include <functional>  // NOLINT
#include <utility>     // NOLINT

#include "mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"

namespace mlir {
namespace bisect {
namespace {

constexpr int64_t kMaxWhileIterations = 1;

// Rewrites a while loop to execute its body a fixed number of times. The
// condition is executed, but its result is ignored.
// For ease of implementation, this generates scf.execute_region ops. These are
// subsequently canonicalized away.
llvm::SmallVector<OwningOpRef<ModuleOp>> InlineScfWhile(BisectState&,
                                                        scf::WhileOp whileOp) {
  llvm::SmallVector<OwningOpRef<ModuleOp>> result;
  for (int64_t num_executions = 0; num_executions <= kMaxWhileIterations;
       ++num_executions) {
    using ::mlir::scf::ExecuteRegionOp;

    auto [module, op] = CloneModuleFor(whileOp);
    OpBuilder b(op);
    llvm::SmallVector<scf::ExecuteRegionOp> regions;

    auto wrap_region_in_execute = [&, loc = op.getLoc()](mlir::Region& region) {
      regions
          .emplace_back(b.create<ExecuteRegionOp>(
              loc,
              region.getBlocks().front().getTerminator()->getOperandTypes(),
              mlir::ValueRange{}))
          .getRegion()
          .takeBody(region);
    };

    wrap_region_in_execute(op.getBefore());
    // Replace the condition terminator with a yield terminator.
    {
      auto& before_block = regions[0].getRegion().getBlocks().front();
      OpBuilder before_builder(before_block.getTerminator());
      IRRewriter before_rewriter(before_builder);
      before_rewriter.replaceOpWithNewOp<scf::YieldOp>(
          before_block.getTerminator(),
          before_block.getTerminator()->getOperands());
    }

    // Clone the execute region ops the requested number of times.
    if (num_executions > 0) {
      wrap_region_in_execute(op.getAfter());
      for (int64_t i = 0; i < num_executions - 1; ++i) {
        b.insert(regions.emplace_back(regions[0].clone()));
        b.insert(regions.emplace_back(regions[1].clone()));
      }
      b.insert(regions.emplace_back(regions[0].clone()));
    }

    // Rewire region arguments and erase them.
    for (int64_t i = 0; i < regions.size(); ++i) {
      auto args = i == 0 ? ValueRange{op.getOperands()}
                         : ValueRange{regions[i - 1].getResults()};
      bool is_after_region = (i & 1) == 1;
      auto& region = regions[i].getRegion();
      for (int64_t arg = static_cast<int64_t>(region.getNumArguments()) - 1;
           arg >= 0; --arg) {
        region.getArgument(arg).replaceAllUsesWith(
            args[is_after_region ? arg + 1 : arg]);
        region.eraseArgument(arg);
      }
    }
    op->replaceAllUsesWith(regions.back().getResults().drop_front(1));
    op->erase();
    result.push_back(std::move(module));
  }
  return result;
}

SmallVector<OwningOpRef<ModuleOp>> ReduceScfForallBounds(
    BisectState&, scf::ForallOp forall_op) {
  SmallVector<OwningOpRef<ModuleOp>> result;
  SmallVector<OpFoldResult> newUpperBound{forall_op.getMixedUpperBound()};
  OpBuilder b(forall_op);
  for (auto& ub : newUpperBound) {
    auto constantOr = mlir::getConstantIntValue(ub);
    if (!constantOr.has_value()) continue;

    ub = b.getIndexAttr(*constantOr - 1);
  }
  auto [module, op] = CloneModuleFor(forall_op);
  b.setInsertionPoint(op);
  SmallVector<Value> dynamicUpperBound;
  SmallVector<int64_t> staticUpperBound;
  dispatchIndexOpFoldResults(newUpperBound, dynamicUpperBound,
                             staticUpperBound);
  op.getDynamicUpperBoundMutable().assign(dynamicUpperBound);
  op.setStaticUpperBound(staticUpperBound);

  result.push_back(std::move(module));
  return result;
}

REGISTER_MLIR_REDUCE_STRATEGY(ReduceScfForallBounds);
REGISTER_MLIR_REDUCE_STRATEGY(InlineScfWhile);

}  // namespace
}  // namespace bisect
}  // namespace mlir
