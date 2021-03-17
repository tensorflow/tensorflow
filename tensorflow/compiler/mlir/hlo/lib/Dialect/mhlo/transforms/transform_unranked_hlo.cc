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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

// TODO(herhut): Generate these out of op definitions.
#define MAP_XLA_OPERATION_CWISE_UNARY(fn, sep)                                \
  fn(AbsOp) sep fn(CeilOp) sep fn(ClzOp) sep fn(ConvertOp) sep fn(CosOp)      \
      sep fn(ExpOp) sep fn(Expm1Op) sep fn(FloorOp) sep fn(ImagOp)            \
          sep fn(IsFiniteOp) sep fn(LogOp) sep fn(Log1pOp) sep fn(LogisticOp) \
              sep fn(NotOp) sep fn(NegOp) sep fn(PopulationCountOp)           \
                  sep fn(RealOp) sep fn(RoundOp) sep fn(RsqrtOp)              \
                      sep fn(SignOp) sep fn(SinOp) sep fn(SqrtOp)             \
                          sep fn(TanhOp)

// TODO(herhut): Generate these out of op definitions.
#define MAP_XLA_OPERATION_CWISE_BINARY(fn, sep)                            \
  fn(AddOp) sep fn(AndOp) sep fn(Atan2Op) sep fn(ComplexOp) sep fn(DivOp)  \
      sep fn(MaxOp) sep fn(MinOp) sep fn(MulOp) sep fn(OrOp) sep fn(PowOp) \
          sep fn(RemOp) sep fn(ShiftLeftOp) sep fn(ShiftRightArithmeticOp) \
              sep fn(ShiftRightLogicalOp) sep fn(SubOp) sep fn(XorOp)

// TODO(herhut): Generate these out of op definitions.
#define MAP_CHLO_OPERATION_CWISE_UNARY(fn, sep)                            \
  fn(AcosOp) sep fn(AcoshOp) sep fn(AsinOp) sep fn(AsinhOp) sep fn(AtanOp) \
      sep fn(AtanhOp) sep fn(ConjOp) sep fn(CoshOp) sep fn(DigammaOp)      \
          sep fn(ErfOp) sep fn(ErfcOp) sep fn(IsInfOp) sep fn(LgammaOp)    \
              sep fn(SinhOp) sep fn(TanOp)

// TODO(herhut): Generate these out of op definitions.
#define MAP_CHLO_OPERATION_CWISE_BINARY(fn, sep) fn(PolygammaOp) sep fn(ZetaOp)

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
    Value flatShape = rewriter.create<tensor::FromElementsOp>(loc, numElements);

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
        rewriter.create<OpTy>(loc, flatResultTy, flatOperands, op->getAttrs());

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

    auto scalar_element_type = lhs_is_scalar ? lhs_ranked_type.getElementType()
                                             : rhs_ranked_type.getElementType();
    auto result_type = op.getResult().getType().template dyn_cast<TensorType>();
    auto result_element_type = result_type.getElementType();

    // Reshape the non-scalar value into a dynamically sized, rank-1 tensor
    Value shape =
        rewriter.create<shape::ShapeOfOp>(loc, lhs_is_scalar ? rhs : lhs);
    Value num_elements = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value size_tensor =
        rewriter.create<tensor::FromElementsOp>(loc, num_elements);
    Value reshaped = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, RankedTensorType::get({-1}, scalar_element_type),
        lhs_is_scalar ? rhs : lhs, size_tensor);

    // Create a new ranked Chlo op that will be further lowered by other
    // patterns into Mhlo.
    SmallVector<Value, 2> new_operands{lhs_is_scalar ? lhs : reshaped,
                                       rhs_is_scalar ? rhs : reshaped};
    Value computed = rewriter.create<ChloOpTy>(
        loc, TypeRange{RankedTensorType::get({-1}, result_element_type)},
        new_operands, op->getAttrs());

    // Reshape the result back into an unranked tensor.
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, result_type,
                                                        computed, shape);

    return success();
  }
};

template <typename ChloOpTy, typename HloOpTy>
struct ConvertUnrankedDynamicBroadcastOpHelper {
  // Returns the dynamic result of checking the given value is effectively a
  // scalar shape (i.e. the number of elements is 1).
  static Value GreaterRankIsN(OpBuilder &builder, Location loc,
                              Value actual_rank, int targeted_rank) {
    return builder.create<CmpIOp>(
        loc, CmpIPredicate::eq, actual_rank,
        builder.create<ConstantIndexOp>(loc, targeted_rank));
  }

  static scf::IfOp createIfOpForRankSpecializedBroadcastAndOp(
      OpBuilder &builder, ChloOpTy op, Value actual_rank, int targeted_rank) {
    // Create the if block to place the current specialized logic in.
    Value greater_rank_is_n =
        GreaterRankIsN(builder, op.getLoc(), actual_rank, targeted_rank);
    return builder.create<scf::IfOp>(op.getLoc(), op.getResult().getType(),
                                     greater_rank_is_n, true);
  }

  static Value createBroadcastToKnownRank(OpBuilder &builder, ChloOpTy op,
                                          Value shape, int targeted_rank) {
    auto loc = op.getLoc();
    SmallVector<int64_t, 6> ranked_shape(targeted_rank, 1);
    auto unknown_rank_extent_tensor_type = RankedTensorType::get(
        {RankedTensorType::kDynamicSize}, builder.getIndexType());
    auto known_rank_extent_tensor_type =
        RankedTensorType::get({targeted_rank}, builder.getIndexType());
    Value ranked_shape_val = builder.create<shape::ConstShapeOp>(
        loc, known_rank_extent_tensor_type,
        mlir::DenseIntElementsAttr::get(known_rank_extent_tensor_type,
                                        ranked_shape));
    Value extended_value = builder.create<shape::BroadcastOp>(
        loc, unknown_rank_extent_tensor_type, shape, ranked_shape_val, nullptr);
    return builder.create<tensor::CastOp>(loc, known_rank_extent_tensor_type,
                                          extended_value);
  }

  // Create the if statement and code for a broadcasting op with a result of a
  // given rank.
  static void createRankSpecializedBroadcastAndOp(OpBuilder &if_builder,
                                                  ChloOpTy op,
                                                  ValueRange operands,
                                                  ValueRange operand_shapes,
                                                  int targeted_rank) {
    auto loc = op.getLoc();
    SmallVector<Value, 2> reshaped_operands;

    auto dynamic_dimensions = llvm::SmallVector<int64_t, 6>(
        targeted_rank, RankedTensorType::kDynamicSize);

    for (auto it : llvm::zip(operands, operand_shapes)) {
      Value operand, shape;
      std::tie(operand, shape) = it;
      // Handle shape broadcasting and inference.
      Value extended_operand_casted =
          createBroadcastToKnownRank(if_builder, op, shape, targeted_rank);

      // 1. Reshape operands to the given rank (with the same number of
      // elements)
      // 2. Compute the ranked-broadcasted ChloOp (which will assert that the
      // ops
      //    can be broadcasted and do the actual broadcasting)
      // 3. Type erase the output back to unranked
      auto reshaped_type = RankedTensorType::get(
          dynamic_dimensions,
          operand.getType().template dyn_cast<TensorType>().getElementType());
      Value reshaped_operand = if_builder.create<mhlo::DynamicReshapeOp>(
          loc, reshaped_type, operand, extended_operand_casted);
      reshaped_operands.push_back(reshaped_operand);
    }
    auto result_element_type = op.getResult()
                                   .getType()
                                   .template dyn_cast<TensorType>()
                                   .getElementType();
    auto result_type =
        RankedTensorType::get(dynamic_dimensions, result_element_type);
    Value result = if_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, reshaped_operands, op->getAttrs());
    Value reshaped_result = if_builder.create<tensor::CastOp>(
        loc, UnrankedTensorType::get(result_element_type), result);
    if_builder.create<scf::YieldOp>(loc, reshaped_result);
  }

  // Iterates over the desired ranks to be specialized and generates the code
  // snippet for each case.
  static Value HandleBroadcastAndOp(OpBuilder &rewriter, ChloOpTy op,
                                    ValueRange operands) {
    auto loc = op.getLoc();

    // Get the minimum broadcast shapes of the operands.
    SmallVector<Value> shapes;
    shapes.reserve(operands.size());
    auto extent_tensor_type = RankedTensorType::get({ShapedType::kDynamicSize},
                                                    rewriter.getIndexType());
    for (Value operand : operands) {
      Value shape =
          rewriter.create<shape::ShapeOfOp>(loc, extent_tensor_type, operand);
      shapes.push_back(shape);
    }
    auto broadcast_shape = rewriter.create<shape::BroadcastOp>(
        loc, extent_tensor_type, shapes, nullptr);
    SmallVector<Type> result_types(shapes.size(), extent_tensor_type);
    auto reduced_shapes =
        rewriter
            .create<chlo::MinimumBroadcastShapesOp>(loc, result_types, shapes)
            .results();
    SmallVector<Value> reshaped_operands;
    reshaped_operands.reserve(operands.size());
    for (auto it : llvm::zip(operands, reduced_shapes)) {
      Value operand;
      Value reduced_shape;
      std::tie(operand, reduced_shape) = it;
      auto reshaped_operand = rewriter.create<mhlo::DynamicReshapeOp>(
          loc, operand.getType(), operand, reduced_shape);
      reshaped_operands.push_back(reshaped_operand);
    }

    // Find the largest rank of the operands.
    Value greater_rank;
    for (Value shape : reduced_shapes) {
      Value rank =
          rewriter.create<shape::RankOp>(loc, rewriter.getIndexType(), shape);
      if (!greater_rank) {
        greater_rank = rank;
      } else {
        Value greater_rank_compare = rewriter.create<CmpIOp>(
            loc, CmpIPredicate::sgt, greater_rank, rank);
        greater_rank = rewriter.create<SelectOp>(loc, greater_rank_compare,
                                                 greater_rank, rank);
      }
    }

    // Generate a list of nested if/else statements to handle rank
    // specializations from 1 to `kMaxRankSpecialization`.
    scf::IfOp if_op = createIfOpForRankSpecializedBroadcastAndOp(
        rewriter, op, greater_rank, 1);
    OpBuilder if_builder = if_op.getThenBodyBuilder(rewriter.getListener());
    createRankSpecializedBroadcastAndOp(if_builder, op, reshaped_operands,
                                        reduced_shapes, 1);

    // Put each subsequent rank specialization inside the else statement of the
    // previous one.
    OpBuilder else_builder = if_op.getElseBodyBuilder(rewriter.getListener());
    constexpr int kMaxRankSpecialization = 5;
    for (int i = 2; i < kMaxRankSpecialization; i++) {
      auto inner_if = createIfOpForRankSpecializedBroadcastAndOp(
          else_builder, op, greater_rank, i);
      if_builder = inner_if.getThenBodyBuilder(rewriter.getListener());
      createRankSpecializedBroadcastAndOp(if_builder, op, reshaped_operands,
                                          reduced_shapes, i);
      else_builder.create<scf::YieldOp>(loc, inner_if.getResult(0));
      else_builder = inner_if.getElseBodyBuilder(rewriter.getListener());
    }
    // Fire an assertion if none of the rank specializations applied (one of
    // the ranks was greater than `kMaxRankSpecialization`).
    else_builder.create<AssertOp>(
        loc,
        GreaterRankIsN(else_builder, op.getLoc(), greater_rank,
                       kMaxRankSpecialization),
        "Input for dynamic binary op lowering was of a rank greater than " +
            std::to_string(kMaxRankSpecialization));
    // Add the rank 5 specialization to the innermost else block.
    createRankSpecializedBroadcastAndOp(else_builder, op, reshaped_operands,
                                        reduced_shapes, kMaxRankSpecialization);

    // Return the reshaped result of the outermost if statement.
    auto result = if_op.getResult(0);
    auto reshaped_result = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, result.getType(), result, broadcast_shape);
    return reshaped_result;
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

    Value shape_of_lhs = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value shape_of_rhs = rewriter.create<shape::ShapeOfOp>(loc, rhs);

    // If lhs has exactly one element
    auto if_op = rewriter.create<scf::IfOp>(
        loc, result_type, IsSingleElementShape(rewriter, op, shape_of_lhs),
        true);
    OpBuilder if_lhs_scalar_builder =
        if_op.getThenBodyBuilder(rewriter.getListener());
    Value reshaped_lhs = if_lhs_scalar_builder.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get({}, lhs_type.getElementType()), lhs);
    Value if_lhs_scalar_result = if_lhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{reshaped_lhs, rhs},
        op->getAttrs());
    Value extended_if_lhs_scalar_result =
        extendToBroadcastShape(if_lhs_scalar_builder, loc, if_lhs_scalar_result,
                               shape_of_lhs, shape_of_rhs);
    if_lhs_scalar_builder.create<scf::YieldOp>(loc,
                                               extended_if_lhs_scalar_result);

    // If lhs does not have exactly one element
    //
    // See if rhs has exactly one element
    OpBuilder else_lhs_scalar_builder =
        if_op.getElseBodyBuilder(rewriter.getListener());
    auto if_rhs_scalar_op = else_lhs_scalar_builder.create<scf::IfOp>(
        loc, result_type,
        IsSingleElementShape(else_lhs_scalar_builder, op, shape_of_rhs), true);
    else_lhs_scalar_builder.create<scf::YieldOp>(loc,
                                                 if_rhs_scalar_op.getResult(0));
    OpBuilder if_rhs_scalar_builder =
        if_rhs_scalar_op.getThenBodyBuilder(rewriter.getListener());
    Value reshaped_rhs = if_rhs_scalar_builder.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get({}, rhs_type.getElementType()), rhs);
    Value if_rhs_scalar_result = if_rhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{lhs, reshaped_rhs},
        op->getAttrs());
    Value extended_if_rhs_scalar_result =
        extendToBroadcastShape(if_rhs_scalar_builder, loc, if_rhs_scalar_result,
                               shape_of_lhs, shape_of_rhs);
    if_rhs_scalar_builder.create<scf::YieldOp>(loc,
                                               extended_if_rhs_scalar_result);

    // If NEITHER shape has exactly one element
    //
    // See if shapes are equal.
    OpBuilder else_no_scalars_builder =
        if_rhs_scalar_op.getElseBodyBuilder(rewriter.getListener());
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

    // If shapes do not have exactly one element, nor are equal
    //
    // See if values are of a rank that we support.
    OpBuilder if_neq_shapes_builder =
        if_eq_shapes_op.getElseBodyBuilder(rewriter.getListener());
    if_neq_shapes_builder.create<scf::YieldOp>(
        loc, ConvertUnrankedDynamicBroadcastOpHelper<
                 ChloOpTy, HloOpTy>::HandleBroadcastAndOp(if_neq_shapes_builder,
                                                          op, {lhs, rhs}));

    rewriter.replaceOp(op, {if_op.getResult(0)});
    return success();
  }

 private:
  // Returns the dynamic result of checking the given value is effectively a
  // scalar shape (i.e. the number of elements is 1).
  Value IsSingleElementShape(OpBuilder &rewriter, ChloOpTy op,
                             Value shape_of_tensor) const {
    auto loc = op.getLoc();

    Value num_elements =
        rewriter.create<shape::NumElementsOp>(loc, shape_of_tensor);
    return rewriter.create<CmpIOp>(loc, rewriter.getI1Type(), CmpIPredicate::eq,
                                   num_elements,
                                   rewriter.create<ConstantIndexOp>(loc, 1));
  }

  Value extendToBroadcastShape(OpBuilder &builder, Location loc, Value value,
                               Value shape_of_lhs, Value shape_of_rhs) const {
    auto unknown_rank_extent_tensor_type = RankedTensorType::get(
        {RankedTensorType::kDynamicSize}, builder.getIndexType());
    Value broadcast_shape =
        builder.create<shape::BroadcastOp>(loc, unknown_rank_extent_tensor_type,
                                           shape_of_lhs, shape_of_rhs, nullptr);
    return builder.create<mhlo::DynamicReshapeOp>(loc, value.getType(), value,
                                                  broadcast_shape);
  }
};

// Rank-specialize chlo.broadcast_select ops.
struct ConvertUnrankedDynamicBroadcastSelectOp
    : public OpConversionPattern<chlo::BroadcastSelectOp> {
  using OpConversionPattern<chlo::BroadcastSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      chlo::BroadcastSelectOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // For now only do the bare minimum and specialize for every rank. There is
    // more potential for optimization here. This also is missing the
    // specialization for rank 0.
    rewriter.replaceOp(
        op, {ConvertUnrankedDynamicBroadcastOpHelper<
                chlo::BroadcastSelectOp,
                mhlo::SelectOp>::HandleBroadcastAndOp(rewriter, op, operands)});
    return success();
  }
};

struct TransformUnrankedHloPass
    : public PassWrapper<TransformUnrankedHloPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect>();
  }

  void runOnFunction() override {
    // Setup conversion target.
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<chlo::HloClientDialect, mhlo::MhloDialect,
                           StandardOpsDialect, shape::ShapeDialect,
                           scf::SCFDialect, tensor::TensorDialect>();
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
    mhlo::PopulateTransformUnrankedHloPatterns(&ctx, &patterns);

    // Apply transformation.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

namespace mhlo {

void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
#define MAP_HLO(op) ElementwiseOpConversion<mhlo::op>
#define MAP_CHLO(op) ElementwiseOpConversion<chlo::op>
#define COMMA ,
  // clang-format off
  patterns->insert<
      MAP_XLA_OPERATION_CWISE_UNARY(MAP_HLO, COMMA),
      MAP_XLA_OPERATION_CWISE_BINARY(MAP_HLO, COMMA),
      MAP_CHLO_OPERATION_CWISE_UNARY(MAP_CHLO, COMMA),
      MAP_CHLO_OPERATION_CWISE_BINARY(MAP_CHLO, COMMA),
      ElementwiseOpConversion<mhlo::CompareOp>,
      ElementwiseOpConversion<mhlo::SelectOp>>(context);
  // clang-format on
#undef MAP_HLO
#undef MAP_CHLO
#undef COMMA
  chlo::PopulateForBroadcastingBinaryOp<
      ConvertUnrankedDynamicBroadcastBinaryOp>(context, patterns);
  chlo::PopulateForBroadcastingBinaryOp<
      ConvertUnrankedScalarDynamicBroadcastBinaryOp>(context, patterns);
  patterns->insert<ConvertUnrankedDynamicBroadcastSelectOp>(context);
}

std::unique_ptr<FunctionPass> createTransformUnrankedHloPass() {
  return std::make_unique<TransformUnrankedHloPass>();
}

}  // namespace mhlo
}  // namespace mlir
