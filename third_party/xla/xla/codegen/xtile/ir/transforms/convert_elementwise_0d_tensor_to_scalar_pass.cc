/* Copyright 2025 The OpenXLA Authors.

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

#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep

namespace xla::xtile {

#define GEN_PASS_DEF_CONVERTELEMENTWISE0DTENSORTOSCALARPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

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

    if (dense_attr.size() != 1) {
      return rewriter.notifyMatchFailure(op, "expected a single element");
    }

    auto scalar_attr = dense_attr.getValues<mlir::TypedAttr>()[0];
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, scalar_attr);

    return mlir::success();
  }
};

struct ConvertElementwise0DTensorToScalarPass
    : public impl::ConvertElementwise0DTensorToScalarPassBase<
          ConvertElementwise0DTensorToScalarPass> {
  void runOnOperation() override {
    mlir::TypeConverter type_converter;
    type_converter.addConversion([](mlir::Type type) { return type; });

    type_converter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
      if (type.getRank() == 0) {
        return type.getElementType();
      }
      return type;
    });

    type_converter.addSourceMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr;
          }
          return mlir::tensor::FromElementsOp::create(builder, loc, result_type,
                                                      inputs.front());
        });

    type_converter.addTargetMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type result_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1) {
            return nullptr;
          }
          return mlir::tensor::ExtractOp::create(builder, loc, inputs.front());
        });

    mlir::ConversionTarget target(getContext());

    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation* op) -> std::optional<bool> {
          if (op->hasTrait<mlir::OpTrait::Elementwise>()) {
            return type_converter.isLegal(op);
          }
          return std::nullopt;
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [&](mlir::arith::ConstantOp op) {
          return type_converter.isLegal(op.getOperation());
        });

    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<ElementwiseConverter, ConstantConversionPattern>(
        type_converter, &getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

}  // namespace xla::xtile
