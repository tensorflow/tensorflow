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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_ELEMENTWISETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct ElementwiseConverter
    : public mlir::OpTraitConversionPattern<mlir::OpTrait::Elementwise> {
 public:
  using OpTraitConversionPattern::OpTraitConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation* op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> new_result_types;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      new_result_types))) {
      return rewriter.notifyMatchFailure(op, "failed to convert type");
    }

    mlir::IRMapping mapping;
    mapping.map(op->getOperands(), operands);
    mlir::Operation* new_op = rewriter.clone(*op, mapping);

    for (auto [results, new_type] :
         llvm::zip(new_op->getResults(), new_result_types)) {
      results.setType(new_type);
    }

    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct ConstantConversionPattern
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto dense_attr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValueAttr());
    if (!dense_attr) {
      return rewriter.notifyMatchFailure(op, "expected a DenseElementsAttr");
    }

    auto tensor_type =
        mlir::dyn_cast<mlir::RankedTensorType>(dense_attr.getType());
    if (!tensor_type || !tensor_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static ranked tensor type");
    }

    auto vector_type = mlir::VectorType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());

    // Reshape the dense attribute to the new vector type.
    auto vector_attr = dense_attr.reshape(vector_type);

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, vector_attr);

    return mlir::success();
  }
};

struct ReshapeConversionPattern
    : public mlir::OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern<mlir::stablehlo::ReshapeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!tensor_type || !tensor_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static ranked tensor type");
    }

    auto vector_type = mlir::VectorType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());

    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        op, vector_type, adaptor.getOperand());

    return mlir::success();
  }
};

struct TensorBitcastConversionPattern
    : public mlir::OpConversionPattern<mlir::tensor::BitcastOp> {
  using OpConversionPattern<mlir::tensor::BitcastOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::BitcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!tensor_type || !tensor_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static ranked tensor type");
    }

    auto vector_type = mlir::VectorType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());

    rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, vector_type,
                                                        adaptor.getSource());

    return mlir::success();
  }
};

struct ElementwiseToVectorPass
    : public impl::ElementwiseToVectorPassBase<ElementwiseToVectorPass> {
  void runOnOperation() override {
    mlir::TypeConverter type_converter;

    // Default conversion: keep type as is.
    type_converter.addConversion([](mlir::Type type) { return type; });

    // Convert static ranked tensors to vectors.
    type_converter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
      if (type.hasStaticShape()) {
        return mlir::VectorType::get(type.getShape(), type.getElementType());
      }
      return type;
    });

    // Source materialization: Vector -> Tensor
    type_converter.addSourceMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr;
          }
          return WriteVectorToTensor(builder, inputs.front());
        });

    // Target materialization: Tensor -> Vector
    type_converter.addTargetMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr;
          }
          return ReadTensorToVector(builder, inputs.front());
        });

    mlir::ConversionTarget target(getContext());

    // Mark elementwise ops as illegal if they use tensors that can be converted
    // to vectors.
    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation* op) -> std::optional<bool> {
          if (op->hasTrait<mlir::OpTrait::Elementwise>()) {
            return type_converter.isLegal(op);
          }
          return std::nullopt;
        });

    // Mark constant ops as illegal if they use tensors that can be converted to
    // vectors.
    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [&](mlir::arith::ConstantOp op) {
          return type_converter.isLegal(op.getOperation());
        });

    target.addDynamicallyLegalOp<mlir::stablehlo::ReshapeOp>(
        [&](mlir::stablehlo::ReshapeOp op) {
          return type_converter.isLegal(op.getOperation());
        });

    target.addDynamicallyLegalOp<mlir::tensor::BitcastOp>(
        [&](mlir::tensor::BitcastOp op) {
          return type_converter.isLegal(op.getOperation());
        });

    target.addLegalOp<mlir::vector::ShapeCastOp>();

    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<ElementwiseConverter, ConstantConversionPattern,
                 ReshapeConversionPattern, TensorBitcastConversionPattern>(
        type_converter, &getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateElementwiseToVectorPass() {
  return std::make_unique<ElementwiseToVectorPass>();
}

}  // namespace xla::cpu
