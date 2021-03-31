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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
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
    if (failed(shape_origin.reifyReturnTypeShapes(rewriter, reifications)))
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

/// Move operation into a preceeding assuming op. This allows to process
/// operations that depend on the assuming op's results. It will eventually
/// allow to make assuming regions' constraints independent from each other.
template <typename OpTy>
struct MoveIntoAssumingOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Only move into immediately preceeding `assuming` op.
    auto assuming_op =
        llvm::dyn_cast_or_null<shape::AssumingOp>(op->getPrevNode());
    if (!assuming_op) return failure();

    Block *body = assuming_op.getBody();
    auto yield_op = cast<shape::AssumingYieldOp>(body->getTerminator());

    // Find the operands to use if the op was within the assuming region. We
    // will later use their copies, as we copy the assuming op and its body.
    SmallVector<Value, 8> new_operands_unmapped;
    for (auto operand : op->getOperands()) {
      new_operands_unmapped.push_back(operand);
      for (auto result : llvm::enumerate(assuming_op->getResults())) {
        if (result.value() == operand)
          new_operands_unmapped.back() = yield_op->getOperand(result.index());
      }
    }

    // Insert the rewritten assuming op right before the old one.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(assuming_op);
    auto new_assuming_op = rewriter.create<shape::AssumingOp>(
        assuming_op.getLoc(), assuming_op.witness(),
        [&](OpBuilder &b, Location loc) {
          // Copy body.
          BlockAndValueMapping mapping;
          for (auto &nested : body->without_terminator())
            b.clone(nested, mapping);

          // Copy op into the new body and use the mapped operands.
          SmallVector<Value, 2> new_operands;
          for (Value v_unmapped : new_operands_unmapped) {
            Value v = mapping.lookupOrDefault(v_unmapped);
            new_operands.push_back(v);
          }
          Value new_op = b.create<OpTy>(loc, op->getResultTypes(), new_operands,
                                        op->getAttrs());

          // Yield the previous results and also the new one.
          SmallVector<Value, 2> mapped_results;
          for (auto result : yield_op.operands())
            mapped_results.push_back(mapping.lookupOrDefault(result));
          mapped_results.push_back(new_op);
          return mapped_results;
        });

    // Replace the assuming op and the root op with the corresponding result
    // value.
    ValueRange new_assuming_op_results = new_assuming_op->getResults();
    rewriter.replaceOp(assuming_op, new_assuming_op_results.drop_back());
    rewriter.replaceOp(op, new_assuming_op_results.back());
    return success();
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

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(assuming_op);
    rewriter.replaceOpWithNewOp<OpTy>(op, op->getResultTypes(),
                                      op->getOperands(), op->getAttrs());
    return success();
  }
};

// TODO(frgossen): Only move up broadcasting operations if there is a consumer.
struct MoveUpBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern<DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast_op,
                                PatternRewriter &rewriter) const override {
    Operation *producer_op = bcast_op.operand().getDefiningOp();
    if (!producer_op ||
        !producer_op->hasTrait<OpTrait::SameOperandsAndResultShape>() ||
        !producer_op->hasTrait<OpTrait::Elementwise>()) {
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

struct MoveUpDynamicBroadcastsForFusionPass
    : public PassWrapper<MoveUpDynamicBroadcastsForFusionPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::PopulateMoveUpDynamicBroadcastsForFusionPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateMoveUpDynamicBroadcastsForFusionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      InlineBroadcastedShapeOperandsPattern<shape::CstrBroadcastableOp>,
      MoveIntoAssumingOpPattern<shape::ShapeOfOp>,
      MoveIntoAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveOutOfAssumingOpPattern<shape::CstrBroadcastableOp>,
      MoveOutOfAssumingOpPattern<shape::ShapeOfOp>,
      MoveUpBroadcastInDimOpPattern,
      ShapeReificationPattern>(context);
  // clang-format on
}

std::unique_ptr<FunctionPass> createMoveUpDynamicBroadcastsForFusionPass() {
  return std::make_unique<MoveUpDynamicBroadcastsForFusionPass>();
}

}  // namespace mhlo
}  // namespace mlir
