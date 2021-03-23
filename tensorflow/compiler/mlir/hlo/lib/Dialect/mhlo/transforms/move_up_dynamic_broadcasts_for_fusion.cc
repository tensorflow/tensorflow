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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

bool IsShapeOfOpMovable(Value arg) {
  return arg.getDefiningOp<InferShapedTypeOpInterface>();
}

struct ShapeOfOpConversion : public OpConversionPattern<shape::ShapeOfOp> {
  explicit ShapeOfOpConversion(MLIRContext *context)
      : OpConversionPattern<shape::ShapeOfOp>(context) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(
      shape::ShapeOfOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    shape::ShapeOfOp::Adaptor transformed(operands);

    // Only reify shape computation if operand allows for it.
    if (!IsShapeOfOpMovable(transformed.arg())) return failure();

    auto shape_origin =
        transformed.arg().getDefiningOp<InferShapedTypeOpInterface>();
    llvm::SmallVector<Value, 1> reified_shapes;
    if (failed(shape_origin.reifyReturnTypeShapes(rewriter, reified_shapes)))
      return failure();

    assert(reified_shapes.size() == 1);
    Value reified_shape = reified_shapes.front();
    if (reified_shape.getType() != op.getType()) {
      reified_shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                      reified_shape);
    }

    rewriter.replaceOp(op, reified_shapes.front());
    return success();
  }
};

// We can only move up broadcasting ops that apply to the result of a
// shape-preserving operation.
bool isDynamicBroadcastInDimOpMovable(Value operand) {
  Operation *producer_op = operand.getDefiningOp();
  return producer_op != nullptr &&
         producer_op->hasTrait<OpTrait::SameOperandsAndResultShape>() &&
         producer_op->hasTrait<OpTrait::Elementwise>();
}

// TODO(frgossen): Only move up broadcasting operations if there is a consumer.
struct MoveUpBroadcastInDimOpConversion
    : public OpConversionPattern<DynamicBroadcastInDimOp> {
  explicit MoveUpBroadcastInDimOpConversion(MLIRContext *context)
      : OpConversionPattern<DynamicBroadcastInDimOp>(context) {}

  LogicalResult matchAndRewrite(
      DynamicBroadcastInDimOp bcast_op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    DynamicBroadcastInDimOp::Adaptor transformed(operands);
    if (!isDynamicBroadcastInDimOpMovable(transformed.operand()))
      return failure();

    // Materialize broadcast on operands.
    SmallVector<Value, 2> bcasted_operands;
    Location loc = bcast_op.getLoc();
    ArrayRef<int64_t> ty_shape = bcast_op.getType().getShape();
    Operation *producer_op = transformed.operand().getDefiningOp();
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
          loc, bcasted_operand_ty, operand, transformed.output_dimensions(),
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
    // Setup target legality.
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    PopulateMoveUpDynamicBroadcastsForFusionLegality(&target);

    // Populate rewrite patterns.
    OwningRewritePatternList patterns;
    mhlo::PopulateMoveUpDynamicBroadcastsForFusionPatterns(&ctx, &patterns);

    // Apply transformation.
    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateMoveUpDynamicBroadcastsForFusionLegality(
    ConversionTarget *target) {
  target->addLegalDialect<MhloDialect, StandardOpsDialect, shape::ShapeDialect,
                          tensor::TensorDialect>();
  target->addDynamicallyLegalOp<shape::ShapeOfOp>(
      [](shape::ShapeOfOp op) { return !IsShapeOfOpMovable(op.arg()); });
  target->addDynamicallyLegalOp<DynamicBroadcastInDimOp>(
      [](DynamicBroadcastInDimOp op) {
        return !isDynamicBroadcastInDimOpMovable(op.operand());
      });
}

void PopulateMoveUpDynamicBroadcastsForFusionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<ShapeOfOpConversion,
                   MoveUpBroadcastInDimOpConversion>(context);
  // clang-format on
}

std::unique_ptr<FunctionPass> createMoveUpDynamicBroadcastsForFusionPass() {
  return std::make_unique<MoveUpDynamicBroadcastsForFusionPass>();
}

}  // namespace mhlo
}  // namespace mlir
