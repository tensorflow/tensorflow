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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

struct ShapeReificationPattern : public OpRewritePattern<shape::ShapeOfOp> {
  explicit ShapeReificationPattern(MLIRContext *ctx)
      : OpRewritePattern<shape::ShapeOfOp>(ctx) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
    auto origin = op.getArg().getDefiningOp<InferShapedTypeOpInterface>();
    if (!origin) return failure();
    SmallVector<Value, 1> reifications;
    if (failed(origin.reifyReturnTypeShapes(rewriter, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
    Value shape = reifications[op.getArg().cast<OpResult>().getResultNumber()];

    // Insert cast, if needed.
    if (shape.getType() != op.getType()) {
      shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(), shape);
    }

    rewriter.replaceOp(op, shape);
    return success();
  }
};

struct ShapeReificationThroughAssumingOpsPattern
    : public OpRewritePattern<shape::AssumingOp> {
  using OpRewritePattern<shape::AssumingOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::AssumingOp aop,
                                PatternRewriter &rewriter) const override {
    // Analyze in which results' values and shapes we are interested.
    size_t numResults = aop->getNumResults();
    SmallVector<SmallVector<shape::ShapeOfOp>> shapeUsersPerResult;
    shapeUsersPerResult.reserve(numResults);
    SmallVector<bool> hasNonShapeUsersPerResult;
    hasNonShapeUsersPerResult.reserve(numResults);
    for (Value result : aop.getResults()) {
      auto &shapeUsers = shapeUsersPerResult.emplace_back();
      auto &hasNonShapeUsers = hasNonShapeUsersPerResult.emplace_back(false);
      for (Operation *user : result.getUsers()) {
        if (auto sop = llvm::dyn_cast<shape::ShapeOfOp>(user)) {
          shapeUsers.push_back(sop);
        } else {
          hasNonShapeUsers = true;
        }
      }
    }

    // Fail, if there is nothing to make progress on.
    if (llvm::all_of(shapeUsersPerResult, [](auto it) { return it.empty(); }) &&
        llvm::all_of(hasNonShapeUsersPerResult, [](auto it) { return it; })) {
      return failure();
    }

    // Create a new assuming op.
    auto newAop = rewriter.create<shape::AssumingOp>(
        aop.getLoc(), aop.getWitness(), [&](OpBuilder &b, Location loc) {
          // From the old assuming op, move all ops over to this new one, except
          // the yield terminator.
          Block *aopBody = aop.getBody();
          auto yop =
              llvm::cast<shape::AssumingYieldOp>(aopBody->getTerminator());
          Block *newAopBody = b.getInsertionBlock();
          auto &dstOps = newAopBody->getOperations();
          auto &srcOps = aopBody->getOperations();
          dstOps.splice(dstOps.begin(), srcOps, srcOps.begin(),
                        yop->getIterator());

          // Collect all the values that have non-shape uses to yield them from
          // the body. Also, create the needed `shape_of` ops at the end of the
          // body and yield these results.
          b.setInsertionPointToEnd(newAopBody);
          SmallVector<Value> results;
          SmallVector<Value> shapeResults;
          for (const auto &it : llvm::enumerate(yop.getOperands())) {
            if (hasNonShapeUsersPerResult[it.index()]) {
              results.push_back(it.value());
            }
            if (!shapeUsersPerResult[it.index()].empty()) {
              shapeResults.push_back(
                  b.create<shape::ShapeOfOp>(loc, it.value()));
            }
          }
          results.append(shapeResults);
          return results;
        });

    // Find the replacement values for the old assuming op.
    size_t i = 0;
    auto newAopResults = newAop.getResults();
    auto replacement = llvm::to_vector<8>(llvm::map_range(
        hasNonShapeUsersPerResult, [&](bool hasNonShapeUses) -> Value {
          return hasNonShapeUses ? newAopResults[i++] : nullptr;
        }));

    // Replace all the shape uses with the shape values from the new assuming
    // region.
    for (const auto &shapeUsers : shapeUsersPerResult) {
      if (shapeUsers.empty()) continue;
      for (shape::ShapeOfOp sop : shapeUsers) {
        rewriter.replaceOp(sop, newAopResults[i]);
      }
      i++;
    }
    assert(i == newAopResults.size() &&
           "expect to use all results of the new assuming op");

    // Finally, replace the old assuming op.
    rewriter.replaceOp(aop, replacement);
    return success();
  }
};

struct ShapeReificationPass
    : public ShapeReificationPassBase<ShapeReificationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }

  void runOnOperation() override {
    // Collect patterns.
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateShapeReificationPatterns(ctx, &patterns);

    // Apply patterns from the bottom up. This ensures to need no more than one
    // iteration.
    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = false;
    func::FuncOp f = getOperation();
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns), cfg))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateShapeReificationPatterns(MLIRContext *ctx,
                                      RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      ShapeReificationPattern,
      ShapeReificationThroughAssumingOpsPattern>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateShapeReificationPass() {
  return std::make_unique<ShapeReificationPass>();
}

}  // namespace mhlo
}  // namespace mlir
