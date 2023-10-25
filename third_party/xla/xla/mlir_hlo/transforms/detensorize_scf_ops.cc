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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_DETENSORIZESCFOPSPASS
#include "transforms/passes.h.inc"

namespace {

bool isUnitTensor(Value value) {
  if (auto tensorTy = value.getType().dyn_cast<RankedTensorType>()) {
    return tensorTy.getRank() == 0;
  }
  return false;
}

template <typename T>
struct RegionOpPattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter& rewriter) const override {
    // If none of the operands or results are unit tensors, exit early.
    if (!llvm::any_of(op->getOperands(), isUnitTensor) &&
        !llvm::any_of(op->getResults(), isUnitTensor))
      return failure();

    auto* result = rewriter.clone(*op.getOperation());

    auto unitTensors = [](auto&& range) {
      return llvm::make_filter_range(llvm::enumerate(range), [](auto it) {
        return isUnitTensor(it.value());
      });
    };

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(result);
    for (auto [index, operand] : unitTensors(result->getOperands())) {
      result->setOperand(index, b.create<tensor::ExtractOp>(operand));
    }

    // Fix any block arguments in the op. We're detensorizing all arguments that
    // are unit tensors, so it's safe to do this in all blocks as well (assuming
    // no ops have arguments appearing out of thin air). Inside blocks, we still
    // use tensors. This pass expects linalg-detensorize to run next.
    for (auto& region : result->getRegions()) {
      for (auto& block : region.getBlocks()) {
        for (auto [index, arg] : unitTensors(block.getArguments())) {
          b.setInsertionPointToStart(&block);
          // Change the argument type to a scalar, but repack it into a tensor.
          arg.setType(
              arg.getType().template cast<RankedTensorType>().getElementType());
          auto converted = b.create<tensor::FromElementsOp>(
              RankedTensorType::get({}, arg.getType()), arg);
          arg.replaceAllUsesExcept(converted, converted.getOperation());
        }

        // In the terminator, we have to unpack unit tensors from the block.
        for (auto [index, operand] :
             unitTensors(block.getTerminator()->getOperands())) {
          b.setInsertionPoint(block.getTerminator());
          block.getTerminator()->setOperand(
              index, b.create<tensor::ExtractOp>(operand));
        }
      }
    }

    b.setInsertionPointAfter(result);
    llvm::SmallVector<Value> results = result->getResults();
    for (auto [index, opResult] : unitTensors(results)) {
      // Fix the result type in the SCF op (it's actually a scalar now).
      auto oldType = opResult.getType().template cast<RankedTensorType>();
      opResult.setType(oldType.getElementType());

      // Convert the scalar back to a tensor in the output.
      results[index] = b.create<tensor::FromElementsOp>(oldType, opResult);
    }
    rewriter.replaceOp(op.getOperation(), results);
    return success();
  }
};

struct DetensorizeScfOpsPass
    : public impl::DetensorizeScfOpsPassBase<DetensorizeScfOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<RegionOpPattern<scf::WhileOp>, RegionOpPattern<scf::ForOp>,
                 RegionOpPattern<scf::IfOp>>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::createDetensorizeScfOpsPass() {
  return std::make_unique<mlir::DetensorizeScfOpsPass>();
}
