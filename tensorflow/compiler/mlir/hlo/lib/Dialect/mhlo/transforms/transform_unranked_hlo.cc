/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

// TODO(herhut): Generate these out of op definitions.
#define MAP_XLA_OPERATION_CWISE_UNARY(fn, sep)                                 \
  fn(AbsOp) sep fn(CeilOp) sep fn(ClzOp) sep fn(CosOp) sep fn(ExpOp)           \
      sep fn(Expm1Op) sep fn(FloorOp) sep fn(ImagOp) sep fn(IsFiniteOp)        \
          sep fn(LogOp) sep fn(Log1pOp) sep fn(LogisticOp) sep fn(NotOp)       \
              sep fn(NegOp) sep fn(PopulationCountOp) sep fn(RealOp)           \
                  sep fn(RoundOp) sep fn(RsqrtOp) sep fn(SignOp) sep fn(SinOp) \
                      sep fn(SqrtOp) sep fn(TanhOp)

// TODO(herhut): Generate these out of op definitions.
#define MAP_XLA_OPERATION_CWISE_BINARY(fn, sep)                           \
  fn(AddOp) sep fn(Atan2Op) sep fn(ComplexOp) sep fn(DivOp) sep fn(MaxOp) \
      sep fn(MinOp) sep fn(MulOp) sep fn(PowOp) sep fn(RemOp)             \
          sep fn(ShiftLeftOp) sep fn(ShiftRightArithmeticOp)              \
              sep fn(ShiftRightLogicalOp) sep fn(SubOp)

// TODO(herhut): Generate these out of op definitions.
#define MAP_CHLO_OPERATION_CWISE_UNARY(fn, sep)                         \
  fn(AcosOp) sep fn(AtanOp) sep fn(ErfOp) sep fn(ErfcOp) sep fn(SinhOp) \
      sep fn(TanOp)

template <typename OpTy>
inline void AddLegalOpOnRankedTensor(ConversionTarget *target) {
  target->addDynamicallyLegalOp<OpTy>([](OpTy op) {
    return llvm::all_of(op.getOperation()->getOperandTypes(),
                        [&](Type t) { return t.isa<RankedTensorType>(); });
  });
}

/// Element-wise operations on unranked tensors can be applied to the flattened
/// tensor operands with the same effect.  This pattern rewrites every such
/// operation to
///   (i)   flatten the input tensor,
///   (ii)  apply the operation, and
///   (iii) restore the original shape.
template <typename OpTy>
struct ElementwiseOpConversion : public OpRewritePattern<OpTy> {
  explicit ElementwiseOpConversion(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Don't apply conversion unless all operands are unranked.
    if (!llvm::all_of(op.getOperation()->getOperands(), [&](Value operand) {
          return operand.getType().isa<UnrankedTensorType>();
        })) {
      return failure();
    }

    // Get operands' shape.
    auto loc = op.getLoc();
    Type extentTensorTy = shape::getExtentTensorType(rewriter.getContext());
    SmallVector<Value, 3> operandShapes;
    for (Value operand : op.getOperation()->getOperands()) {
      Value shape =
          rewriter.create<shape::ShapeOfOp>(loc, extentTensorTy, operand);
      operandShapes.push_back(shape);
    }
    Value shape =
        operandShapes.size() == 1
            ? operandShapes.front()
            : rewriter.create<shape::AnyOp>(loc, extentTensorTy, operandShapes);

    // Derive flat shape.
    Type indexTy = rewriter.getIndexType();
    Value numElements =
        rewriter.create<shape::NumElementsOp>(loc, indexTy, shape);
    Value flatShape = rewriter.create<TensorFromElementsOp>(loc, numElements);

    // Flatten operands.
    SmallVector<Value, 3> flatOperands;
    for (Value operand : op.getOperation()->getOperands()) {
      Type operandElementTy =
          operand.getType().template cast<ShapedType>().getElementType();
      Type flatTy =
          RankedTensorType::get({ShapedType::kDynamicSize}, operandElementTy);
      Value flat = rewriter.create<mhlo::DynamicReshapeOp>(loc, flatTy, operand,
                                                           flatShape);
      flatOperands.push_back(flat);
    }

    // Apply operation to flattened operands.
    Type resultElementTy =
        op.getType().template cast<ShapedType>().getElementType();
    Type flatResultTy =
        RankedTensorType::get({ShapedType::kDynamicSize}, resultElementTy);
    Value flatResult =
        rewriter.create<OpTy>(loc, flatResultTy, flatOperands, op.getAttrs());

    // Restore original shape.
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, op.getType(),
                                                        flatResult, shape);

    return success();
  }
};

// Converts a broadcasting binary operation with a scalar operand and an
// unranked operand to a ranked broadcasting operation by dynamically reshaping
// the unranked operand to a 1D tensor. This will always be safe because
// broadcasting from a scalar to another shape always works.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertUnrankedScalarDynamicBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    typename ChloOpTy::Adaptor transformed(operands);
    Value lhs = transformed.lhs();
    Value rhs = transformed.rhs();

    auto lhs_ranked_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto lhs_unranked_type = lhs.getType().dyn_cast<UnrankedTensorType>();

    auto rhs_ranked_type = rhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_unranked_type = rhs.getType().dyn_cast<UnrankedTensorType>();

    bool lhs_is_scalar = lhs_ranked_type &&
                         lhs_ranked_type.getShape().empty() &&
                         rhs_unranked_type;
    bool rhs_is_scalar = rhs_ranked_type &&
                         rhs_ranked_type.getShape().empty() &&
                         lhs_unranked_type;

    // Only support the case where exactly one operand is scalar and the other
    // is unranked. Other patterns in chlo-to-hlo legalization will create more
    // efficient lowerings for cases where both ranks are known or will handle
    // the more generic case of both inputs being unranked.
    if (!(lhs_is_scalar ^ rhs_is_scalar)) return failure();

    auto result_type = op.getResult().getType().template dyn_cast<TensorType>();

    // Reshape the non-scalar value into a dynamically sized, rank-1 tensor
    Value shape =
        rewriter.create<shape::ShapeOfOp>(loc, lhs_is_scalar ? rhs : lhs);
    Value num_elements = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value size_tensor =
        rewriter.create<TensorFromElementsOp>(loc, num_elements);
    Value reshaped = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, RankedTensorType::get({-1}, result_type.getElementType()),
        lhs_is_scalar ? rhs : lhs, size_tensor);

    // Create a new ranked Chlo op that will be further lowered by other
    // patterns into Mhlo.
    SmallVector<Value, 2> new_operands{lhs_is_scalar ? lhs : reshaped,
                                       rhs_is_scalar ? rhs : reshaped};
    Value computed =
        rewriter.create<ChloOpTy>(loc, SmallVector<Type, 1>{reshaped.getType()},
                                  new_operands, op.getAttrs());

    // Reshape the result back into an unranked tensor.
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, result_type,
                                                        computed, shape);

    return success();
  }
};

// Handles lowering of the following pattern to patterns that will be further
// matched by other patterns until they result in LHLO:
//   %result = "chlo.op"(%lhs, %rhs) : (<*xTy>, <*xTy>) -> <*xTy>
//
// The sequence of specializations this handles is:
//   - Either operand being scalar
//   - Operands having equal shapes
//   - The resulting value being any of ranks [2,6]
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertUnrankedDynamicBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    typename ChloOpTy::Adaptor transformed(operands);
    Value lhs = transformed.lhs();
    Value rhs = transformed.rhs();
    auto lhs_type = lhs.getType().dyn_cast<UnrankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<UnrankedTensorType>();
    auto result_type = op.getResult().getType().template dyn_cast<TensorType>();

    // Only support unranked operands. If either operand is ranked, another
    // pattern will handle the lowering.
    if (!lhs_type || !rhs_type) return failure();

    // If lhs is scalar
    auto if_op = rewriter.create<scf::IfOp>(
        loc, result_type, IsScalarTensor(rewriter, op, lhs), true);
    OpBuilder if_lhs_scalar_builder =
        if_op.getThenBodyBuilder(rewriter.getListener());
    Value reshaped_lhs = if_lhs_scalar_builder.create<TensorCastOp>(
        loc, RankedTensorType::get({}, lhs_type.getElementType()), lhs);
    Value if_lhs_scalar_result = if_lhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{reshaped_lhs, rhs},
        op.getAttrs());
    if_lhs_scalar_builder.create<scf::YieldOp>(loc, if_lhs_scalar_result);

    // If lhs is NOT scalar
    //
    // See if rhs is scalar
    OpBuilder else_lhs_scalar_builder =
        if_op.getElseBodyBuilder(rewriter.getListener());
    auto if_rhs_scalar_op = else_lhs_scalar_builder.create<scf::IfOp>(
        loc, result_type, IsScalarTensor(else_lhs_scalar_builder, op, rhs),
        true);
    else_lhs_scalar_builder.create<scf::YieldOp>(loc,
                                                 if_rhs_scalar_op.getResult(0));
    OpBuilder if_rhs_scalar_builder =
        if_rhs_scalar_op.getThenBodyBuilder(rewriter.getListener());
    Value reshaped_rhs = if_rhs_scalar_builder.create<TensorCastOp>(
        loc, RankedTensorType::get({}, lhs_type.getElementType()), rhs);
    Value if_rhs_scalar_result = if_rhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{lhs, reshaped_rhs},
        op.getAttrs());
    if_rhs_scalar_builder.create<scf::YieldOp>(loc, if_rhs_scalar_result);

    // If NEITHER shape is scalar
    //
    // See if shapes are equal.
    OpBuilder else_no_scalars_builder =
        if_rhs_scalar_op.getElseBodyBuilder(rewriter.getListener());
    Value shape_of_lhs =
        else_no_scalars_builder.create<shape::ShapeOfOp>(loc, lhs);
    Value shape_of_rhs =
        else_no_scalars_builder.create<shape::ShapeOfOp>(loc, rhs);
    Value equal_shapes = else_no_scalars_builder.create<shape::ShapeEqOp>(
        loc, shape_of_lhs, shape_of_rhs);

    auto if_eq_shapes_op = else_no_scalars_builder.create<scf::IfOp>(
        loc, result_type, equal_shapes, true);
    else_no_scalars_builder.create<scf::YieldOp>(loc,
                                                 if_eq_shapes_op.getResult(0));

    OpBuilder if_eq_shapes_builder =
        if_eq_shapes_op.getThenBodyBuilder(rewriter.getListener());
    Value non_broadcast_op =
        Adaptor::CreateOp(op, result_type, lhs, rhs, if_eq_shapes_builder);
    if_eq_shapes_builder.create<scf::YieldOp>(loc, non_broadcast_op);

    // If shapes are not scalar, nor equal
    //
    // See if values are of a rank that we support.
    OpBuilder if_neq_shapes_builder =
        if_eq_shapes_op.getElseBodyBuilder(rewriter.getListener());
    if_neq_shapes_builder.create<scf::YieldOp>(
        loc, HandleBroadcastAndOp(if_neq_shapes_builder, op, lhs, rhs));

    rewriter.replaceOp(op, {if_op.getResult(0)});
    return success();
  }

 private:
  // Returns the dyanamic result of checking the given value is a scalar
  // tensor.
  Value IsScalarTensor(OpBuilder &rewriter, ChloOpTy op, Value tensor) const {
    auto loc = op.getLoc();

    Value shape_of_tensor = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    Value rank_tensor = rewriter.create<shape::RankOp>(
        loc, rewriter.getIndexType(), shape_of_tensor);
    return rewriter.create<CmpIOp>(loc, rewriter.getI1Type(), CmpIPredicate::eq,
                                   rank_tensor,
                                   rewriter.create<ConstantIndexOp>(loc, 0));
  }

  // Create the if statement and code for a broadcasting op with a result of a
  // given rank.
  scf::IfOp createRankSpecializedBroadcastAndOp(OpBuilder &builder, ChloOpTy op,
                                                Value lhs, Value rhs,
                                                Value actual_rank,
                                                int targeted_rank) const {
    auto loc = op.getLoc();

    // Create the if block to place the current specialized logic in.
    Value greater_rank_is_n = builder.create<CmpIOp>(
        loc, CmpIPredicate::eq, actual_rank,
        builder.create<ConstantIndexOp>(loc, targeted_rank));
    auto if_op =
        builder.create<scf::IfOp>(loc, lhs.getType(), greater_rank_is_n, true);
    OpBuilder if_builder = if_op.getThenBodyBuilder(builder.getListener());

    // Handle shape broadcasting and inferrence.
    Value lhs_shape = if_builder.create<shape::ShapeOfOp>(loc, lhs);
    Value rhs_shape = if_builder.create<shape::ShapeOfOp>(loc, rhs);
    SmallVector<int64_t, 6> ranked_shape(targeted_rank, 1);
    auto unknown_rank_extent_tensor_type = RankedTensorType::get(
        {RankedTensorType::kDynamicSize}, builder.getIndexType());
    auto known_rank_extent_tensor_type =
        RankedTensorType::get({targeted_rank}, builder.getIndexType());
    auto reshaped_type = RankedTensorType::get(
        llvm::SmallVector<int64_t, 6>(targeted_rank,
                                      RankedTensorType::kDynamicSize),
        lhs.getType().template dyn_cast<TensorType>().getElementType());
    Value ranked_shape_val = if_builder.create<shape::ConstShapeOp>(
        loc, known_rank_extent_tensor_type,
        mlir::DenseIntElementsAttr::get(known_rank_extent_tensor_type,
                                        ranked_shape));
    Value extended_lhs = if_builder.create<shape::BroadcastOp>(
        loc, unknown_rank_extent_tensor_type, lhs_shape, ranked_shape_val,
        nullptr);
    Value extended_lhs_casted = if_builder.create<TensorCastOp>(
        loc, known_rank_extent_tensor_type, extended_lhs);
    Value extended_rhs = if_builder.create<shape::BroadcastOp>(
        loc, unknown_rank_extent_tensor_type, rhs_shape, ranked_shape_val,
        nullptr);
    Value extended_rhs_casted = if_builder.create<TensorCastOp>(
        loc, known_rank_extent_tensor_type, extended_rhs);

    // 1. Reshape operands to the given rank (with the same number of elements)
    // 2. Compute the ranked-broadcasted ChloOp (which will assert that the ops
    //    can be broadcasted and do the actual broadcasting)
    // 3. Type erase the output back to unranked
    Value reshaped_lhs = if_builder.create<mhlo::DynamicReshapeOp>(
        loc, reshaped_type, lhs, extended_lhs_casted);
    Value reshaped_rhs = if_builder.create<mhlo::DynamicReshapeOp>(
        loc, reshaped_type, rhs, extended_rhs_casted);
    Value result = if_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{reshaped_type},
        ArrayRef<Value>{reshaped_lhs, reshaped_rhs}, op.getAttrs());
    Value reshaped_result = if_builder.create<TensorCastOp>(
        loc, UnrankedTensorType::get(reshaped_type.getElementType()), result);
    if_builder.create<scf::YieldOp>(loc, reshaped_result);

    // Return the if_op, so the result can be used and the else block can be
    // used for the next rank specialized step.
    return if_op;
  }

  // Iterates over the desired ranks to be specialized and generates the code
  // snippet for each case.
  Value HandleBroadcastAndOp(OpBuilder &rewriter, ChloOpTy op, Value lhs,
                             Value rhs) const {
    constexpr int max_rank_specialization = 7;
    auto loc = op.getLoc();

    // Find the larger rank of the 2 operands.
    auto extent_tensor_type = RankedTensorType::get({ShapedType::kDynamicSize},
                                                    rewriter.getIndexType());
    Value lhs_shape =
        rewriter.create<shape::ShapeOfOp>(loc, extent_tensor_type, lhs);
    Value rhs_shape =
        rewriter.create<shape::ShapeOfOp>(loc, extent_tensor_type, rhs);
    Value lhs_rank =
        rewriter.create<shape::RankOp>(loc, rewriter.getIndexType(), lhs_shape);
    Value rhs_rank =
        rewriter.create<shape::RankOp>(loc, rewriter.getIndexType(), rhs_shape);
    Value greater_rank_lhs =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs_rank, rhs_rank);
    Value greater_rank =
        rewriter.create<SelectOp>(loc, greater_rank_lhs, lhs_rank, rhs_rank);

    // Generate a list of nested if/else statements to handle rank
    // specializations from 1-6.
    scf::IfOp if_op = createRankSpecializedBroadcastAndOp(rewriter, op, lhs,
                                                          rhs, greater_rank, 1);

    // Put each subsequent rank specialization inside the else statement of the
    // previous one.
    OpBuilder else_builder = if_op.getElseBodyBuilder(rewriter.getListener());
    for (int i = 2; i < max_rank_specialization; i++) {
      auto inner_if = createRankSpecializedBroadcastAndOp(else_builder, op, lhs,
                                                          rhs, greater_rank, i);

      else_builder.create<scf::YieldOp>(loc, inner_if.getResult(0));
      else_builder = inner_if.getElseBodyBuilder(rewriter.getListener());
    }

    // Fire an assertion if none of the rank specializations applied (one of the
    // ranks was greater than 6).
    else_builder.create<AssertOp>(
        loc, else_builder.create<ConstantIntOp>(loc, 0, 1),
        "Input for dynamic binary op lowering was of a rank greater than 6");
    else_builder.create<scf::YieldOp>(loc, lhs);

    // Return the result of the outermost if statement.
    return if_op.getResult(0);
  }
};

struct TransformUnrankedHloPass
    : public PassWrapper<TransformUnrankedHloPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    // Setup conversion target.
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<mhlo::MhloDialect, StandardOpsDialect,
                           shape::ShapeDialect, scf::SCFDialect>();
    target.addLegalOp<FuncOp>();
#define ADD_LEGAL_MHLO(op) AddLegalOpOnRankedTensor<mhlo::op>(&target)
#define ADD_LEGAL_CHLO(op) AddLegalOpOnRankedTensor<chlo::op>(&target)
    MAP_XLA_OPERATION_CWISE_UNARY(ADD_LEGAL_MHLO, ;);
    MAP_XLA_OPERATION_CWISE_BINARY(ADD_LEGAL_MHLO, ;);
    MAP_CHLO_OPERATION_CWISE_UNARY(ADD_LEGAL_CHLO, ;);
#undef ADD_LEGAL_MHLO
#undef ADD_LEGAL_CHLO
    AddLegalOpOnRankedTensor<mhlo::CompareOp>(&target);
    AddLegalOpOnRankedTensor<mhlo::SelectOp>(&target);
    target.addDynamicallyLegalDialect<chlo::HloClientDialect>(
        [](Operation *op) {
          return !llvm::any_of(op->getOperandTypes(), [](Type type) {
            return type.isa<UnrankedTensorType>();
          });
        });

    // Populate rewrite patterns.
    OwningRewritePatternList patterns;
    PopulateTransformUnrankedHloPatterns(&ctx, &patterns);

    // Apply transformation.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
#define MAP_UNARY(op) ElementwiseOpConversion<mhlo::op>
#define MAP_BINARY(op) ElementwiseOpConversion<mhlo::op>
#define MAP_CHLO_UNARY(op) ElementwiseOpConversion<chlo::op>
#define COMMA ,
  // clang-format off
  patterns->insert<
      MAP_XLA_OPERATION_CWISE_UNARY(MAP_UNARY, COMMA),
      MAP_XLA_OPERATION_CWISE_BINARY(MAP_BINARY, COMMA),
      MAP_CHLO_OPERATION_CWISE_UNARY(MAP_CHLO_UNARY, COMMA),
      ElementwiseOpConversion<mhlo::CompareOp>,
      ElementwiseOpConversion<mhlo::SelectOp>>(context);
  // clang-format on
#undef MAP_UNARY
#undef MAP_BINARY
#undef MAP_CHLO_UNARY
#undef COMMA
  chlo::PopulateForBroadcastingBinaryOp<
      ConvertUnrankedDynamicBroadcastBinaryOp>(context, patterns);
  chlo::PopulateForBroadcastingBinaryOp<
      ConvertUnrankedScalarDynamicBroadcastBinaryOp>(context, patterns);
}

std::unique_ptr<FunctionPass> createTransformUnrankedHloPass() {
  return std::make_unique<TransformUnrankedHloPass>();
}

}  // namespace mlir
