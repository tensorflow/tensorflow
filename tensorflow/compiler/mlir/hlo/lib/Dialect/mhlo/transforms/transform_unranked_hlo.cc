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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
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
namespace mhlo {
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

template <typename OpTy>
inline void AddLegalOpOnRankedTensor(ConversionTarget *target) {
  target->addDynamicallyLegalOp<OpTy>([](OpTy op) {
    return llvm::all_of((op.getOperation())->getOperandTypes(),
                        [&](Type t) { return t.isa<RankedTensorType>(); });
  });
}

/// Unary element-wise operations on unranked tensors can be applied to the
/// flattened tensor with the same effect.
/// This pattern rewrites every such operation to
///   (i)   flatten the input tensor,
///   (ii)  apply the unary operation, and
///   (iii) restore the original shape.
template <typename OpTy>
struct UnaryElementwiseOpConversion : public OpRewritePattern<OpTy> {
  explicit UnaryElementwiseOpConversion(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Don't apply conversion to ops with statically shaped operands.
    Value operand = op.getOperand();
    auto operandTy = operand.getType().dyn_cast<TensorType>();
    if (operandTy.hasRank()) return failure();

    // Generate IR to flatten the operand.
    auto loc = op.getLoc();
    Type extentTensorTy = shape::getExtentTensorType(rewriter.getContext());
    Value shape =
        rewriter.create<shape::ShapeOfOp>(loc, extentTensorTy, operand);
    Type indexTy = rewriter.getIndexType();
    Value numElements =
        rewriter.create<shape::NumElementsOp>(loc, indexTy, shape);
    Value flatShape = rewriter.create<TensorFromElementsOp>(loc, numElements);
    auto flatTensorTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                              operandTy.getElementType());
    Value flatOperand = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, flatTensorTy, operand, flatShape);

    // Generate IR for the actual operation.
    Value flatResult = rewriter.create<OpTy>(loc, flatTensorTy, flatOperand);

    // Generate IR to restore the original shape.
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, operandTy,
                                                        flatResult, shape);

    return success();
  }
};

/// Binary element-wise operation on unranked tensors can be applied to the
/// flattened operand tensors with the same effect.
/// This pattern rewrites every such operation to
///   (i)   flatten the operand tensors,
///   (ii)  apply the binary operation, and
//    (iii) restore the original shape.
template <typename OpTy>
struct BinaryElementwiseOpConversion : public OpRewritePattern<OpTy> {
  explicit BinaryElementwiseOpConversion(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Don't apply conversion unless both operands are unranked.
    if (op.lhs().getType().template isa<RankedTensorType>() ||
        op.rhs().getType().template isa<RankedTensorType>()) {
      return failure();
    }

    // Flatten operands.
    auto loc = op.getLoc();
    Type extentTensorTy = shape::getExtentTensorType(rewriter.getContext());
    Value shapeLhs =
        rewriter.create<shape::ShapeOfOp>(loc, extentTensorTy, op.lhs());
    Value shapeRhs =
        rewriter.create<shape::ShapeOfOp>(loc, extentTensorTy, op.rhs());
    Value shape = rewriter.create<shape::AnyOp>(loc, extentTensorTy,
                                                ValueRange{shapeLhs, shapeRhs});
    Type indexTy = rewriter.getIndexType();
    Value numElements =
        rewriter.create<shape::NumElementsOp>(loc, indexTy, shape);
    Value flatShape = rewriter.create<TensorFromElementsOp>(loc, numElements);
    TensorType lhsTy = op.lhs().getType().template cast<TensorType>();
    Type flatLhsTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                           lhsTy.getElementType());
    Value flatLhs =
        rewriter.create<DynamicReshapeOp>(loc, flatLhsTy, op.lhs(), flatShape);
    TensorType rhsTy = op.rhs().getType().template cast<TensorType>();
    Type flatRhsTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                           rhsTy.getElementType());
    Value flatRhs =
        rewriter.create<DynamicReshapeOp>(loc, flatRhsTy, op.rhs(), flatShape);

    // Apply actual operation to flattened operands.
    Value flatResult = rewriter.create<OpTy>(loc, flatLhs, flatRhs);

    // Restore original shape.
    rewriter.replaceOpWithNewOp<DynamicReshapeOp>(op, op.getType(), flatResult,
                                                  shape);

    return success();
  }
};

struct TransformUnrankedHloPass
    : public PassWrapper<TransformUnrankedHloPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }

  void runOnFunction() override {
    // Setup conversion target.
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<MhloDialect, StandardOpsDialect,
                           shape::ShapeDialect>();
    target.addLegalOp<FuncOp>();
#define ADD_LEGAL(op) AddLegalOpOnRankedTensor<op>(&target)
    MAP_XLA_OPERATION_CWISE_UNARY(ADD_LEGAL, ;);
    MAP_XLA_OPERATION_CWISE_BINARY(ADD_LEGAL, ;);
#undef ADD_LEGAL

    // Populate rewrite patterns.
    OwningRewritePatternList patterns;
    PopulateTransformUnrankedHloPatterns(&ctx, &patterns);

    // Apply transformation.
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      return signalPassFailure();
  }
};

}  // namespace

void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
  // TODO(frgossen): Populate all unary and binary operations.
  // clang-format off
#define MAP_UNARY(op) UnaryElementwiseOpConversion<op>
#define MAP_BINARY(op) BinaryElementwiseOpConversion<op>
#define COMMA ,
  patterns->insert<
      MAP_XLA_OPERATION_CWISE_UNARY(MAP_UNARY, COMMA),
      MAP_XLA_OPERATION_CWISE_BINARY(MAP_BINARY, COMMA)
      >(context);
#undef MAP_UNARY
#undef MAP_BINARY
#undef COMMA
  // clang-format on
}

std::unique_ptr<::mlir::Pass> createTransformUnrankedHloPass() {
  return std::make_unique<TransformUnrankedHloPass>();
}

}  // namespace mhlo
}  // namespace mlir
