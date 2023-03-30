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

#include <memory>
#include <string>
#include <unordered_map>

#include "gml_st/transforms/passes.h"
#include "gml_st/utils/tensor_utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_COLLECTSTATSPASS
#include "gml_st/transforms/passes.h.inc"

using NameToOpMap =
    std::unordered_map<std::string, llvm::SmallVector<Operation *, 4>>;

struct CollectStatsPass : public impl::CollectStatsPassBase<CollectStatsPass> {
  using CollectStatsPassBase<CollectStatsPass>::CollectStatsPassBase;

  explicit CollectStatsPass(int64_t level) { detailLevel = level; }

  void runOnOperation() override {
    if (detailLevel <= 0) return;
    func::FuncOp func = getOperation();

    func.walk([&](Operation *op) {
      if (!isa<TilingInterface, tensor::CollapseShapeOp, tensor::EmptyOp,
               tensor::ExpandShapeOp, tensor::PackOp, tensor::PadOp,
               tensor::UnPackOp>(op))
        return WalkResult::advance();

      std::string key = op->getName().getStringRef().str();
      if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        key += isDegenerateReshapeOp(collapseShapeOp) ? " (degenerate)"
                                                      : " (non-degenerate)";
      }
      map[key].push_back(op);
      return WalkResult::advance();
    });

    printStats();
  }

 private:
  void printStats() {
    llvm::outs() << "*** Tileable ops stats (detail level " << detailLevel
                 << ") ***\n";
    for (auto it : map) {
      auto name = it.first;
      auto ops = it.second;
      llvm::outs() << ops.size() << "x " << name << "\n";
      // If we want the op name only, stop here.
      if (detailLevel == 1) continue;
      for (size_t i = 0; i < ops.size(); ++i) {
        auto *op = ops[i];
        llvm::outs().indent(2) << i + 1 << ". ";
        op->print(llvm::outs());
        llvm::outs() << '\n';
        // If we want the full op string only, stop here.
        if (detailLevel == 2) continue;
        // Otherwise print info about the producers and consumers of the op.
        llvm::outs().indent(4) << "Producers:\n";
        for (auto operand : op->getOperands()) {
          if (auto loopLikeProducer =
                  operand.getDefiningOp<LoopLikeOpInterface>()) {
            llvm::outs().indent(6)
                << loopLikeProducer->getName().getStringRef() << '\n';
          } else {
            operand.print(llvm::outs().indent(6));
            llvm::outs() << '\n';
          }
        }
        llvm::outs().indent(4) << "Consumers:\n";
        for (auto user : op->getUsers()) {
          user->print(llvm::outs().indent(6));
          llvm::outs() << '\n';
        }
      }
      llvm::outs() << '\n';
    }
  }

  int64_t detailLevel;
  NameToOpMap map;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCollectStatsPass() {
  return std::make_unique<CollectStatsPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createCollectStatsPass(
    int64_t level) {
  return std::make_unique<CollectStatsPass>(level);
}

}  // namespace gml_st
}  // namespace mlir
