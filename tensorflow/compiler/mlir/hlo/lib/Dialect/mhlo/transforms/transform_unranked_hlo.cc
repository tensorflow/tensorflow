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
#define MAP_CHLO_OPERATION_CWISE_UNARY(fn, sep) \
  fn(TanOp) sep fn(AcosOp) sep fn(SinhOp)

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
                           shape::ShapeDialect>();
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
}

std::unique_ptr<FunctionPass> createTransformUnrankedHloPass() {
  return std::make_unique<TransformUnrankedHloPass>();
}

}  // namespace mlir
