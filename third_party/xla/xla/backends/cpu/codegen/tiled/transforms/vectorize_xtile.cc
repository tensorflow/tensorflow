/* Copyright 2026 The OpenXLA Authors.

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
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DEF_VECTORIZEXTILEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

template <typename OpTy>
struct VectorizeElementwiseOp : public mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) return mlir::failure();

    rewriter.replaceOpWithNewOp<OpTy>(op, new_type, adaptor.getOperands(),
                                      op->getAttrs());
    return mlir::success();
  }
};

struct VectorizeReshape
    : public mlir::OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ReshapeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_vector_type = mlir::cast<mlir::VectorType>(
        this->getTypeConverter()->convertType(op.getType()));

    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        op, result_vector_type, adaptor.getOperand());
    return mlir::success();
  }
};

struct VectorizeBroadcastInDim
    : public mlir::OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Value source_vector = adaptor.getOperand();
    auto result_vector_type = mlir::cast<mlir::VectorType>(
        this->getTypeConverter()->convertType(op.getType()));

    llvm::ArrayRef<int64_t> source_shape =
        mlir::cast<mlir::VectorType>(source_vector.getType()).getShape();
    llvm::ArrayRef<int64_t> broadcast_dims = op.getBroadcastDimensions();

    llvm::SmallVector<int64_t> intermediate_shape(result_vector_type.getRank(),
                                                  1);
    for (auto [input_dim, result_dim] : llvm::enumerate(broadcast_dims)) {
      intermediate_shape[result_dim] = source_shape[input_dim];
    }

    mlir::Value intermediate_vector = mlir::vector::ShapeCastOp::create(
        rewriter, op->getLoc(),
        mlir::VectorType::get(intermediate_shape,
                              result_vector_type.getElementType()),
        source_vector);

    mlir::Value broadcast_op = mlir::vector::BroadcastOp::create(
        rewriter, op->getLoc(), result_vector_type, intermediate_vector);

    rewriter.replaceOp(op, broadcast_op);
    return mlir::success();
  }
};

template <typename... Ops>
void populateVectorizePatterns(mlir::TypeConverter& converter,
                               mlir::RewritePatternSet& patterns) {
  patterns.add<VectorizeElementwiseOp<Ops>...>(converter,
                                               patterns.getContext());
}

class VectorizeXTilePass
    : public impl::VectorizeXTilePassBase<VectorizeXTilePass> {
 public:
  using VectorizeXTilePassBase::VectorizeXTilePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::TypeConverter type_converter;
    type_converter.addConversion(
        [](mlir::Type type) -> std::optional<mlir::Type> {
          if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            return mlir::VectorType::get(tensor_type.getShape(),
                                         tensor_type.getElementType());
          }
          return type;
        });

    type_converter.addSourceMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                          result_type, inputs)
              .getResult(0);
        });

    type_converter.addTargetMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                          result_type, inputs)
              .getResult(0);
        });

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::vector::VectorDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<xtile::XTileDialect>();  // Keep other XTile ops
                                                    // legal (like EntryFuncOp)

    target.addIllegalOp<mlir::stablehlo::ReshapeOp,
                        mlir::stablehlo::BroadcastInDimOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect>(
        [&](mlir::Operation* op) { return type_converter.isLegal(op); });
    target.addDynamicallyLegalDialect<mlir::math::MathDialect>(
        [&](mlir::Operation* op) { return type_converter.isLegal(op); });

    mlir::RewritePatternSet patterns(context);
    patterns.add<VectorizeReshape, VectorizeBroadcastInDim>(type_converter,
                                                            context);

    populateVectorizePatterns<
        mlir::arith::AddFOp, mlir::arith::AddIOp, mlir::arith::SubFOp,
        mlir::arith::SubIOp, mlir::arith::MulFOp, mlir::arith::MulIOp,
        mlir::arith::DivFOp, mlir::arith::DivSIOp, mlir::arith::DivUIOp,
        mlir::arith::RemFOp, mlir::arith::RemSIOp, mlir::arith::RemUIOp,
        mlir::arith::MaximumFOp, mlir::arith::MaxSIOp, mlir::arith::MaxUIOp,
        mlir::arith::MinimumFOp, mlir::arith::MinSIOp, mlir::arith::MinUIOp,
        mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp,
        mlir::arith::NegFOp, mlir::arith::SelectOp, mlir::arith::CmpFOp,
        mlir::arith::CmpIOp, mlir::arith::ExtFOp, mlir::arith::TruncFOp,
        mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, mlir::arith::FPToSIOp,
        mlir::arith::FPToUIOp, mlir::arith::SIToFPOp, mlir::arith::UIToFPOp,
        mlir::arith::TruncIOp, mlir::arith::IndexCastOp, mlir::math::AbsIOp,
        mlir::math::AbsFOp, mlir::math::CeilOp, mlir::math::FloorOp,
        mlir::math::RoundEvenOp, mlir::math::AcosOp, mlir::math::AcoshOp,
        mlir::math::AsinOp, mlir::math::AsinhOp, mlir::math::Atan2Op,
        mlir::math::AtanhOp, mlir::math::CosOp, mlir::math::CoshOp,
        mlir::math::ExpOp, mlir::math::ErfOp, mlir::math::ExpM1Op,
        mlir::math::LogOp, mlir::math::Log1pOp, mlir::math::IPowIOp,
        mlir::math::PowFOp, mlir::math::RsqrtOp, mlir::math::SinOp,
        mlir::math::SinhOp, mlir::math::SqrtOp, mlir::math::TanOp,
        mlir::math::TanhOp, mlir::math::CbrtOp, mlir::math::IsFiniteOp>(
        type_converter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
