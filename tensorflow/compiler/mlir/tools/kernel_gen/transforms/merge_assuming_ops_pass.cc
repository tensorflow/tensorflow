/* Copyright 2021 The OpenXLA Authors.

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

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/Base.h"  // from @stablehlo
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace kernel_gen {

#define GEN_PASS_DEF_MERGEASSUMINGOPSPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

namespace {

struct ShapeReificationPattern : public OpRewritePattern<shape::ShapeOfOp> {
  explicit ShapeReificationPattern(MLIRContext *context)
      : OpRewritePattern<shape::ShapeOfOp>(context) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
    // Only reify shape computation if operand allows for it.
    auto shapeOrigin = op.getArg().getDefiningOp<InferShapedTypeOpInterface>();
    if (!shapeOrigin) return failure();

    llvm::SmallVector<Value, 1> reifications;
    if (failed(shapeOrigin.reifyReturnTypeShapes(
            rewriter, shapeOrigin->getOperands(), reifications)))
      return failure();
    assert(reifications.size() == 1);
    Value reifiedShape = reifications.front();

    // Insert cast if needed.
    if (reifiedShape.getType() != op.getType()) {
      reifiedShape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                     reifiedShape);
    }

    rewriter.replaceOp(op, reifiedShape);
    return success();
  }
};

template <typename OpTy>
struct InlineBroadcastedShapeOperandsPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Find all the shape operands, direct and indirect.
    SmallVector<Value, 8> inlinedOperands;
    for (Value direct : op->getOperands()) {
      if (auto bcastOp = direct.getDefiningOp<shape::BroadcastOp>()) {
        for (Value indirect : bcastOp->getOperands())
          inlinedOperands.push_back(indirect);
      } else {
        inlinedOperands.push_back(direct);
      }
    }

    // Only rewrite if it makes a difference.
    if (inlinedOperands.size() == op.getNumOperands()) return failure();

    // Inline shape operands.
    rewriter.replaceOpWithNewOp<OpTy>(op, op->getResultTypes(), inlinedOperands,
                                      op->getAttrs());
    return success();
  }
};

LogicalResult moveUpIntoAssumingOpMatchAndRewrite(Operation *op,
                                                  PatternRewriter &rewriter) {
  // Only implemented for single-result ops.
  if (op->getNumResults() != 1) return failure();

  // Find a preceding `assuming` op.
  auto *theBlock = op->getBlock();
  Operation *prev = op->getPrevNode();
  while (prev != nullptr && !llvm::isa<shape::AssumingOp>(prev))
    prev = prev->getPrevNode();
  auto assumingOp = llvm::dyn_cast_or_null<shape::AssumingOp>(prev);
  if (!assumingOp) return failure();
  assert(assumingOp->getBlock() == theBlock && op->getBlock() == theBlock &&
         "expect assuming op and root op to be in the same block");

  // Make sure that all operands will be available after moving.
  auto isAvailable = [&](Value v) {
    Operation *def = v.getDefiningOp();
    return def == nullptr || def->getBlock() != theBlock ||
           !assumingOp->isBeforeInBlock(def);
  };
  if (!llvm::all_of(op->getOperands(), isAvailable)) return failure();

  Block *body = assumingOp.getBody();
  auto yieldOp = llvm::cast<shape::AssumingYieldOp>(body->getTerminator());

  // Find the operands to use if the op was within the assuming region. We
  // will later use their copies, as we copy the assuming op and its body.
  SmallVector<Value, 8> newOperandsUnmapped =
      llvm::to_vector<8>(llvm::map_range(op->getOperands(), [&](Value v) {
        for (const auto &result : llvm::enumerate(assumingOp->getResults())) {
          if (result.value() == v) return yieldOp->getOperand(result.index());
        }
        return v;
      }));

  // Insert the rewritten assuming op right before the old one.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(assumingOp);
  auto newAssumingOp = rewriter.create<shape::AssumingOp>(
      assumingOp.getLoc(), assumingOp.getWitness(),
      [&](OpBuilder &b, Location) {
        // Copy body.
        IRMapping mapping;
        for (auto &nested : body->without_terminator())
          b.clone(nested, mapping);

        // Copy op into the new body and use the mapped operands.
        for (auto it : llvm::zip(op->getOperands(), newOperandsUnmapped)) {
          Value oldOperand, newOperandUnmapped;
          std::tie(oldOperand, newOperandUnmapped) = it;
          mapping.map(oldOperand, mapping.lookupOrDefault(newOperandUnmapped));
        }
        Operation *newOp = b.clone(*op, mapping);

        // Yield the previous results and also the new ones.
        auto mappedResults = llvm::to_vector<8>(llvm::map_range(
            yieldOp.getOperands(),
            [&](Value v) { return mapping.lookupOrDefault(v); }));
        mappedResults.append(newOp->getResults().begin(),
                             newOp->getResults().end());
        return mappedResults;
      });

  // Replace the assuming op and the root op with the corresponding result
  // values.
  ValueRange newAssumingOpResults = newAssumingOp->getResults();
  rewriter.replaceOp(assumingOp, newAssumingOpResults.drop_back());
  rewriter.replaceOp(op, newAssumingOpResults.back());
  return success();
}

/// Move operation into a preceding assuming op. This allows to process
/// operations that depend on the assuming op's results. It will eventually
/// allow to make assuming regions' constraints independent from each other.
template <typename OpTy>
struct MoveUpIntoAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return moveUpIntoAssumingOpMatchAndRewrite(op.getOperation(), rewriter);
  }
};

// Move elementwise operations into a preceding assuming op. This will
// eventually allow for more fusion opportunities.
struct MoveElementwiseOpsUpIntoAssumingOpPattern : public RewritePattern {
  explicit MoveElementwiseOpsUpIntoAssumingOpPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Apply to all elementwise and broadcasting elementwise operations with no
    // side effects.
    if (!op->hasTrait<mlir::OpTrait::Elementwise>() &&
        !op->hasTrait<hlo::OpTrait::BroadcastingElementwise>()) {
      return failure();
    }
    if (!isMemoryEffectFree(op)) return failure();

    return moveUpIntoAssumingOpMatchAndRewrite(op, rewriter);
  }
};

// Move operation into an assuming region if all uses are within its body.
LogicalResult moveDownIntoAssumingOpMatchAndRewrite(Operation *op,
                                                    PatternRewriter &rewriter) {
  auto users = op->getUsers();
  auto it = users.begin();
  auto end = users.end();
  if (it == end) return failure();

  // Find candidate assuming op.
  auto assumingOp = (it++)->getParentOfType<shape::AssumingOp>();
  if (!assumingOp || assumingOp->isProperAncestor(op)) return failure();

  // Make sure all uses are within the unique assuming op's body.
  while (it != end) {
    auto hopefullySameAssumingOp = (it++)->getParentOfType<shape::AssumingOp>();
    if (!hopefullySameAssumingOp || hopefullySameAssumingOp != assumingOp) {
      return failure();
    }
  }

  // Move op into the assuming region.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(assumingOp.getBody());
  Operation *newOp = rewriter.clone(*op);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

// Move elementwise operations into succeeding assuming regions. This will
// eventually allow for more fusion opportunities.
struct MoveElementwiseOpsDownIntoAssumingOpPattern : public RewritePattern {
  explicit MoveElementwiseOpsDownIntoAssumingOpPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Apply to all elementwise and broadcasting elementwise operations with no
    // side effects.
    if (!op->hasTrait<mlir::OpTrait::Elementwise>() &&
        !op->hasTrait<hlo::OpTrait::BroadcastingElementwise>()) {
      return failure();
    }
    if (!isMemoryEffectFree(op)) return failure();

    return moveDownIntoAssumingOpMatchAndRewrite(op, rewriter);
  }
};

/// Move operation out of assuming op. This is only valid for
/// constraint-independent ops, like `cstr_broadcastable` and `shape_of`. It
/// will eventually allow to make assuming regions' constraints independent from
/// each other.
template <typename OpTy>
struct MoveUpOutOfAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Must be inside of an assuming op.
    auto assumingOp = op->template getParentOfType<shape::AssumingOp>();
    if (!assumingOp) return failure();

    // Operands must not be defined within the assuming op.
    Block *body = assumingOp.getBody();
    auto isAvailable = [&](Value v) {
      Operation *def = v.getDefiningOp();
      return def == nullptr || def->getBlock() != body;
    };
    if (!llvm::all_of(op->getOperands(), isAvailable)) return failure();

    // Move op before the assuming region.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(assumingOp);
    Operation *newOp = rewriter.clone(*op);
    rewriter.replaceOp(op, newOp->getResults());

    // If the assuming region yields none of the new op's results, these values
    // are exclusively used in the assuming op's body. In these cases there is
    // no need for further rewrites.
    auto isNewOpResult = [newOp](Value v) {
      return llvm::is_contained(newOp->getResults(), v);
    };
    auto yieldOp = cast<shape::AssumingYieldOp>(body->getTerminator());
    if (llvm::none_of(yieldOp.getOperands(), isNewOpResult)) return success();

    // If the assuming region yields any of the new op's results, these values
    // can instead bypass the assuming region. There is no need to yield them
    // explicitly as they are assumed to be independent. The assuming op is
    // rewritten accordingly.
    SmallVector<Value, 2> replacementValues;
    auto newAssumingOp = rewriter.create<shape::AssumingOp>(
        assumingOp.getLoc(), assumingOp.getWitness(),
        [&](OpBuilder &b, Location) {
          // Copy body.
          IRMapping mapping;
          for (Operation &nested : body->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect new yield operands.
          SmallVector<Value, 2> newYieldOperands;
          for (Value result : yieldOp.getOperands()) {
            if (isNewOpResult(result)) {
              replacementValues.push_back(result);
            } else {
              newYieldOperands.push_back(mapping.lookupOrDefault(result));
              replacementValues.push_back(nullptr);
            }
          }
          return newYieldOperands;
        });

    // Use the assuming op's results for the missing replacement values.
    auto src = newAssumingOp.getResults().begin();
    for (auto &dst : replacementValues) {
      if (dst) continue;
      dst = *src++;
    }

    rewriter.replaceOp(assumingOp, replacementValues);
    return success();
  }
};

/// Merge assuming regions if their constraints are independent from each other.
struct MergeAssumingOpsPattern : public OpRewritePattern<shape::AssumingOp> {
  using OpRewritePattern<shape::AssumingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::AssumingOp op,
                                PatternRewriter &rewriter) const override {
    // Merge assuming op with directly preceding one if both witnesses are
    // available.
    auto precedingOp =
        llvm::dyn_cast_or_null<shape::AssumingOp>(op->getPrevNode());
    if (!precedingOp) return failure();
    if (op.getWitness().getDefiningOp() == precedingOp) return failure();

    // Merge witnesses.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(precedingOp);
    Value newWitness = rewriter.create<shape::AssumingAllOp>(
        op.getWitness().getDefiningOp()->getLoc(),
        ValueRange{precedingOp.getWitness(), op.getWitness()});

    // Merge assuming ops.
    Block *body_a = precedingOp.getBody();
    Block *body_b = op.getBody();
    auto newAssumingOp = rewriter.create<shape::AssumingOp>(
        precedingOp.getLoc(), newWitness, [&](OpBuilder &b, Location) {
          // Copy preceding op's body.
          IRMapping mapping;
          for (auto &nested : body_a->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Map result values of preceding assuming op.
          auto yieldOpA =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_a->getTerminator());
          for (auto pair :
               llvm::zip(precedingOp->getResults(), yieldOpA.getOperands())) {
            mapping.map(std::get<0>(pair),
                        mapping.lookupOrDefault(std::get<1>(pair)));
          }

          // Copy op's body.
          for (auto &nested : body_b->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect merged assuming op's results.
          SmallVector<Value, 4> mappedResults;
          auto yieldOpB =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_b->getTerminator());
          for (Value v : yieldOpA.getOperands()) {
            mappedResults.push_back(mapping.lookupOrDefault(v));
          }
          for (Value v : yieldOpB.getOperands()) {
            mappedResults.push_back(mapping.lookupOrDefault(v));
          }
          return mappedResults;
        });

    // Replace the two assuming ops with the new corresponding results.
    ValueRange newResults = newAssumingOp->getResults();
    size_t splitAt = precedingOp->getNumResults();
    rewriter.replaceOp(precedingOp, newResults.take_front(splitAt));
    rewriter.replaceOp(op, newResults.drop_front(splitAt));
    return success();
  }
};

struct EliminateDuplicateCstrBroadcastableOps
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern<shape::CstrBroadcastableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    // Search for previous occurence of the same constraint.
    Operation *it = op->getPrevNode();
    while (it != nullptr) {
      if (auto candidate = llvm::dyn_cast<shape::CstrBroadcastableOp>(it)) {
        if (candidate.getShapes() == op.getShapes()) {
          rewriter.replaceOp(op, candidate.getResult());
          return success();
        }
      }
      it = it->getPrevNode();
    }

    return failure();
  }
};

void populateMergeAssumingOpsPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
  patterns->add<
      EliminateDuplicateCstrBroadcastableOps,
      InlineBroadcastedShapeOperandsPattern<shape::CstrBroadcastableOp>,
      MergeAssumingOpsPattern, MoveElementwiseOpsDownIntoAssumingOpPattern,
      MoveElementwiseOpsUpIntoAssumingOpPattern,
      MoveUpIntoAssumingOpPattern<shape::AssumingAllOp>,
      MoveUpIntoAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveUpIntoAssumingOpPattern<shape::ShapeOfOp>,
      MoveUpOutOfAssumingOpPattern<shape::AssumingAllOp>,
      MoveUpOutOfAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveUpOutOfAssumingOpPattern<shape::ShapeOfOp>, ShapeReificationPattern>(
      context);
  mhlo::DynamicBroadcastInDimOp::getCanonicalizationPatterns(*patterns,
                                                             context);
  mhlo::DynamicReshapeOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingAllOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingOp::getCanonicalizationPatterns(*patterns, context);
  shape::BroadcastOp::getCanonicalizationPatterns(*patterns, context);
  shape::CstrBroadcastableOp::getCanonicalizationPatterns(*patterns, context);
  tensor::CastOp::getCanonicalizationPatterns(*patterns, context);
}

struct MergeAssumingOpsPass
    : public impl::MergeAssumingOpsPassBase<MergeAssumingOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateMergeAssumingOpsPatterns(ctx, &patterns);
    GreedyRewriteConfig config;
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace kernel_gen
}  // namespace mlir
