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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
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
    auto shape_origin = op.arg().getDefiningOp<InferShapedTypeOpInterface>();
    if (!shape_origin) return failure();

    llvm::SmallVector<Value, 1> reifications;
    if (failed(shape_origin.reifyReturnTypeShapes(
            rewriter, shape_origin->getOperands(), reifications)))
      return failure();
    assert(reifications.size() == 1);
    Value reified_shape = reifications.front();

    // Insert cast if needed.
    if (reified_shape.getType() != op.getType()) {
      reified_shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                      reified_shape);
    }

    rewriter.replaceOp(op, reified_shape);
    return success();
  }
};

template <typename OpTy>
struct InlineBroadcastedShapeOperandsPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Find all the shape operands, direct and indirect.
    SmallVector<Value, 8> inlined_operands;
    for (Value direct : op->getOperands()) {
      if (auto bcast_op = direct.getDefiningOp<shape::BroadcastOp>()) {
        for (Value indirect : bcast_op->getOperands())
          inlined_operands.push_back(indirect);
      } else {
        inlined_operands.push_back(direct);
      }
    }

    // Only rewrite if it makes a difference.
    if (inlined_operands.size() == op.getNumOperands()) return failure();

    // Inline shape operands.
    rewriter.replaceOpWithNewOp<OpTy>(op, op->getResultTypes(),
                                      inlined_operands, op->getAttrs());
    return success();
  }
};

LogicalResult MoveIntoAssumingOpMatchAndRewrite(Operation *op,
                                                PatternRewriter &rewriter) {
  // Only move into immediately preceding `assuming` op.
  auto assuming_op =
      llvm::dyn_cast_or_null<shape::AssumingOp>(op->getPrevNode());
  if (!assuming_op) return failure();

  Block *body = assuming_op.getBody();
  auto yield_op = cast<shape::AssumingYieldOp>(body->getTerminator());

  // Find the operands to use if the op was within the assuming region. We
  // will later use their copies, as we copy the assuming op and its body.
  SmallVector<Value, 8> new_operands_unmapped =
      llvm::to_vector<8>(llvm::map_range(op->getOperands(), [&](Value v) {
        for (auto result : llvm::enumerate(assuming_op->getResults())) {
          if (result.value() == v) return yield_op->getOperand(result.index());
        }
        return v;
      }));

  // Insert the rewritten assuming op right before the old one.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(assuming_op);
  auto new_assuming_op = rewriter.create<shape::AssumingOp>(
      assuming_op.getLoc(), assuming_op.witness(), [&](OpBuilder &b, Location) {
        // Copy body.
        BlockAndValueMapping mapping;
        for (auto &nested : body->without_terminator())
          b.clone(nested, mapping);

        // Copy op into the new body and use the mapped operands.
        for (auto it : llvm::zip(op->getOperands(), new_operands_unmapped)) {
          Value old_operand, new_operand_unmapped;
          std::tie(old_operand, new_operand_unmapped) = it;
          mapping.map(old_operand,
                      mapping.lookupOrDefault(new_operand_unmapped));
        }
        Operation *new_op = b.clone(*op, mapping);

        // Yield the previous results and also the new ones.
        auto mapped_results = llvm::to_vector<8>(llvm::map_range(
            yield_op.operands(),
            [&](Value v) { return mapping.lookupOrDefault(v); }));
        mapped_results.append(new_op->getResults().begin(),
                              new_op->getResults().end());
        return mapped_results;
      });

  // Replace the assuming op and the root op with the corresponding result
  // value.
  ValueRange new_assuming_op_results = new_assuming_op->getResults();
  rewriter.replaceOp(assuming_op, new_assuming_op_results.drop_back());
  rewriter.replaceOp(op, new_assuming_op_results.back());
  return success();
}

/// Move operation into a preceding assuming op. This allows to process
/// operations that depend on the assuming op's results. It will eventually
/// allow to make assuming regions' constraints independent from each other.
template <typename OpTy>
struct MoveIntoAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return MoveIntoAssumingOpMatchAndRewrite(op.getOperation(), rewriter);
  }
};

// Move elementwise operations into assuming regions. This will eventually allow
// for more fusion opportunities.
struct MoveElementwiseOpsIntoAssumingOpPattern : public RewritePattern {
  explicit MoveElementwiseOpsIntoAssumingOpPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Apply to all elementwise and broadcasting elementwise operations.
    if (!op->hasTrait<mlir::OpTrait::Elementwise>() &&
        !op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>())
      return failure();

    return MoveIntoAssumingOpMatchAndRewrite(op, rewriter);
  }
};

/// Move operation out of assuming op. This is only valid for
/// constraint-independent ops, like `cstr_broadcastable` and `shape_of`. It
/// will eventually allow to make assuming regions' constraints independent from
/// each other.
template <typename OpTy>
struct MoveOutOfAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Must be inside of an assuming op.
    auto assuming_op = op->template getParentOfType<shape::AssumingOp>();
    if (!assuming_op) return failure();

    // Operands must not be defined within the assuming op.
    Block *body = assuming_op.getBody();
    auto is_available = [&](Value v) {
      Operation *def = v.getDefiningOp();
      return def == nullptr || def->getBlock() != body;
    };
    if (!llvm::all_of(op->getOperands(), is_available)) return failure();

    // Move op before the assuming region.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(assuming_op);
    Operation *new_op = rewriter.clone(*op);
    rewriter.replaceOp(op, new_op->getResults());

    // If the assuming region yields none of the new op's results, these values
    // are exclusively used in the assuming op's body. In these cases there is
    // no need for further rewrites.
    auto is_new_op_result = [&](Value v) {
      return llvm::is_contained(new_op->getResults(), v);
    };
    auto yield_op = cast<shape::AssumingYieldOp>(body->getTerminator());
    if (llvm::none_of(yield_op.operands(), is_new_op_result)) return success();

    // If the assuming region yields any of the new op's results, these values
    // can instead bypass the assuming region. There is no need to yield them
    // explicitly as they are assumed to be independent. The assuming op is
    // rewritten accordingly.
    SmallVector<Value, 2> replacement_values;
    auto new_assuming_op = rewriter.create<shape::AssumingOp>(
        assuming_op.getLoc(), assuming_op.witness(),
        [&](OpBuilder &b, Location) {
          // Copy body.
          BlockAndValueMapping mapping;
          for (Operation &nested : body->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect new yield operands.
          SmallVector<Value, 2> new_yield_operands;
          for (Value result : yield_op.operands()) {
            if (is_new_op_result(result)) {
              replacement_values.push_back(result);
            } else {
              new_yield_operands.push_back(mapping.lookup(result));
              replacement_values.push_back(nullptr);
            }
          }
          return new_yield_operands;
        });

    // Use the assuming op's results for the missing replacement values.
    auto src = new_assuming_op.getResults().begin();
    for (auto &dst : replacement_values) {
      if (dst) continue;
      dst = *src++;
    }

    rewriter.replaceOp(assuming_op, replacement_values);
    return success();
  }
};

/// Merge assuming regions if their constraints are independent from each other.
struct MergeAssumingOpsPattern : public OpRewritePattern<shape::AssumingOp> {
  using OpRewritePattern<shape::AssumingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::AssumingOp op,
                                PatternRewriter &rewriter) const override {
    // Merge assuming op with directly preceding one if both witnesses are
    // availiable.
    auto preceding_op =
        llvm::dyn_cast_or_null<shape::AssumingOp>(op->getPrevNode());
    if (!preceding_op) return failure();
    if (op.witness().getDefiningOp() == preceding_op) return failure();

    // Merge witnesses.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(preceding_op);
    Value new_witness = rewriter.create<shape::AssumingAllOp>(
        op.witness().getDefiningOp()->getLoc(),
        ValueRange{preceding_op.witness(), op.witness()});

    // Merge assuming ops.
    Block *body_a = preceding_op.getBody();
    Block *body_b = op.getBody();
    auto new_assuming_op = rewriter.create<shape::AssumingOp>(
        preceding_op.getLoc(), new_witness, [&](OpBuilder &b, Location) {
          // Copy preceding op's body.
          BlockAndValueMapping mapping;
          for (auto &nested : body_a->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Map result values of preceding assuming op.
          auto yield_op_a =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_a->getTerminator());
          for (auto pair :
               llvm::zip(preceding_op->getResults(), yield_op_a.operands())) {
            mapping.map(std::get<0>(pair),
                        mapping.lookupOrDefault(std::get<1>(pair)));
          }

          // Copy op's body.
          for (auto &nested : body_b->without_terminator()) {
            b.clone(nested, mapping);
          }

          // Collect merged assuming op's results.
          SmallVector<Value, 4> mapped_results;
          auto yield_op_b =
              llvm::dyn_cast<shape::AssumingYieldOp>(body_b->getTerminator());
          for (Value v : yield_op_a.operands()) {
            mapped_results.push_back(mapping.lookupOrDefault(v));
          }
          for (Value v : yield_op_b.operands()) {
            mapped_results.push_back(mapping.lookupOrDefault(v));
          }
          return mapped_results;
        });

    // Replace the two assuming ops with the new corresponding results.
    ValueRange new_results = new_assuming_op->getResults();
    size_t split_at = preceding_op->getNumResults();
    rewriter.replaceOp(preceding_op, new_results.take_front(split_at));
    rewriter.replaceOp(op, new_results.drop_front(split_at));
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
        if (candidate.shapes() == op.shapes()) {
          rewriter.replaceOp(op, candidate.result());
          return success();
        }
      }
      it = it->getPrevNode();
    }

    return failure();
  }
};

struct EarlyBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern<DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast_op,
                                PatternRewriter &rewriter) const override {
    Operation *producer_op = bcast_op.operand().getDefiningOp();
    if (!producer_op ||
        !producer_op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() ||
        !producer_op->hasTrait<mlir::OpTrait::Elementwise>()) {
      return failure();
    }

    // Materialize broadcast on operands.
    SmallVector<Value, 2> bcasted_operands;
    Location loc = bcast_op.getLoc();
    ArrayRef<int64_t> ty_shape = bcast_op.getType().getShape();
    for (Value operand : producer_op->getOperands()) {
      // The broadcast only works on ranked operations.
      auto operand_ty = operand.getType().dyn_cast<RankedTensorType>();
      if (!operand_ty) {
        return bcast_op.emitError()
               << "Can only move up broadcasts over ranked tensor operands.";
      }

      auto bcasted_operand_ty =
          RankedTensorType::get(ty_shape, operand_ty.getElementType());
      bcasted_operands.push_back(rewriter.create<DynamicBroadcastInDimOp>(
          loc, bcasted_operand_ty, operand, bcast_op.output_dimensions(),
          bcast_op.broadcast_dimensions()));
    }

    // Create a copy of the producer op with the new broadcasted operands.
    OperationState new_producer_op_state(
        loc, producer_op->getName().getStringRef(), bcasted_operands,
        bcast_op.getType(), producer_op->getAttrs());
    Operation *new_producer_op =
        rewriter.createOperation(new_producer_op_state);

    // The original result of the broadcast now falls directly out of the new
    // producer op. Use it instead.
    rewriter.replaceOp(bcast_op, new_producer_op->getResults());

    return success();
  }
};

struct BroadcastPropagationPass
    : public BroadcastPropagationPassBase<BroadcastPropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::PopulateBroadcastsPropagationPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateBroadcastsPropagationPatterns(MLIRContext *context,
                                           OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      EliminateDuplicateCstrBroadcastableOps,
      InlineBroadcastedShapeOperandsPattern<shape::CstrBroadcastableOp>,
      MergeAssumingOpsPattern,
      MoveElementwiseOpsIntoAssumingOpPattern,
      MoveIntoAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveIntoAssumingOpPattern<shape::ShapeOfOp>,
      MoveOutOfAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveOutOfAssumingOpPattern<shape::ShapeOfOp>,
      EarlyBroadcastInDimOpPattern,
      ShapeReificationPattern>(context);
  // clang-format on
  mhlo::DynamicBroadcastInDimOp::getCanonicalizationPatterns(*patterns,
                                                             context);
  mhlo::DynamicReshapeOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingAllOp::getCanonicalizationPatterns(*patterns, context);
  shape::AssumingOp::getCanonicalizationPatterns(*patterns, context);
  shape::BroadcastOp::getCanonicalizationPatterns(*patterns, context);
  shape::CstrBroadcastableOp::getCanonicalizationPatterns(*patterns, context);
  tensor::CastOp::getCanonicalizationPatterns(*patterns, context);
}

std::unique_ptr<FunctionPass> createBroadcastPropagationPass() {
  return std::make_unique<BroadcastPropagationPass>();
}

}  // namespace mhlo
}  // namespace mlir
