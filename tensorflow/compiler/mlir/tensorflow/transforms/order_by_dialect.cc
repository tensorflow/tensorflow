/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/topological_sort.h"

namespace mlir {
namespace TF {
namespace {

std::vector<Operation*> groupOperationsByDialect(Block& block);

#define GEN_PASS_DEF_ORDERBYDIALECTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Reorder operations so that consecutive ops stay in the same dialect, as far
// as possible. This is to optimize the op order for the group-by-dialect pass,
// which factors consecutive same-dialect ops into functions.
class OrderByDialectPass
    : public impl::OrderByDialectPassBase<OrderByDialectPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrderByDialectPass)
  void runOnOperation() override;
};

int DialectOrdering(Operation* predecessor, Operation* op) {
  return predecessor && predecessor->getName().getDialectNamespace() ==
                            op->getName().getDialectNamespace();
}

void OrderByDialectPass::runOnOperation() {
  ModuleOp module = getOperation();
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    std::vector<std::pair<Operation*, Operation*>> side_effect_data;
    const detail::SideEffectAnalysisInfo* info = nullptr;
    auto extra_dependencies =
        [&](Operation* op,
            bool incoming) -> llvm::SmallVector<Operation*, 4> const& {
      return incoming ? info->DirectControlPredecessors(op)
                      : info->DirectControlSuccessors(op);
    };
    // Some tests have recursive calls and other shenanigans, so allow
    // them to skip side effect analysis.
    if (!func->hasAttr("ignore_side_effects_for_testing")) {
      info =
          &getAnalysis<mlir::TF::SideEffectAnalysis>().GetAnalysisForFunc(func);
    }
    func->walk([&](Operation* function) {
      for (Region& region : function->getRegions()) {
        for (Block& block : region.getBlocks()) {
          if (block.empty()) continue;
          auto ops = SortBlockTopologically(
              block, DialectOrdering,
              info ? extra_dependencies : no_extra_dependencies);
          // Replace the block with the reordered block.
          for (Operation* op : ops) {
            op->remove();
            block.push_back(op);
          }
        }
      }
    });
  }
}

}  // namespace

std::unique_ptr<Pass> CreateOrderByDialectPass() {
  return std::make_unique<OrderByDialectPass>();
}

void RegisterOrderByDialectPass() { registerPass(CreateOrderByDialectPass); }

}  // namespace TF
}  // namespace mlir
