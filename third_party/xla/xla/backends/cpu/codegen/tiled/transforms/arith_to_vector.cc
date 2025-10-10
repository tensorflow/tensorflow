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

#include <cassert>
#include <memory>
#include <utility>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_ARITHTOVECTORPASS
#define GEN_PASS_DEF_ARITHTOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

// This converter defines the rules for mapping types from the source (tensors)
// to the target (vectors).
class TensorToVectorTypeConverter : public mlir::TypeConverter {
 public:
  TensorToVectorTypeConverter() {
    // Keep all other types as-is.
    addConversion([](mlir::Type type) { return type; });

    // Convert RankedTensorType to VectorType.
    addConversion([](mlir::RankedTensorType type) -> mlir::Type {
      // We can only convert tensors with a static shape to vectors.
      if (type.hasStaticShape()) {
        return mlir::VectorType::get(type.getShape(), type.getElementType());
      }
      return nullptr;  // Return null if the type cannot be converted.
    });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// A generic pattern to convert an Arith op from tensor-based to vector-based.
// This works for any operation that is a member of the Arith dialect and
// has the same operand/result structure.
template <typename ArithOp>
class ArithOpConversion : public mlir::OpConversionPattern<ArithOp> {
 public:
  using mlir::OpConversionPattern<ArithOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ArithOp op, typename ArithOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> new_result_types;
    mlir::LogicalResult results_ok = this->getTypeConverter()->convertTypes(
        op->getResultTypes(), new_result_types);
    if (results_ok.failed()) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }

    rewriter.replaceOpWithNewOp<ArithOp>(op, new_result_types,
                                         adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

// We need to specify the ConstantOp conversion explicitly as it doesn't follow
// the simple operands & results of the other Arith ops.
template <>
class ArithOpConversion<mlir::arith::ConstantOp>
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
 public:
  using mlir::OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op,
      typename mlir::arith::ConstantOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = getTypeConverter()->convertType(op.getType());
    mlir::ShapedType shaped_type = mlir::dyn_cast<mlir::ShapedType>(new_type);
    if (!shaped_type) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }

    auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, new_type, dense_attr.reshape(shaped_type));
    return mlir::success();
  }
};

template <typename... ArithOps>
void AddArithOpConversionsImpl(mlir::ConversionTarget& target,
                               mlir::RewritePatternSet& patterns,
                               TensorToVectorTypeConverter& typeConverter) {
  target.addDynamicallyLegalOp<ArithOps...>(
      [&](mlir::Operation* op) { return typeConverter.isLegal(op); });
  patterns.add<ArithOpConversion<ArithOps>...>(typeConverter,
                                               patterns.getContext());
}

void AddArithOpConversions(mlir::ConversionTarget& target,
                           mlir::RewritePatternSet& patterns,
                           TensorToVectorTypeConverter& typeConverter) {
  namespace ma = ::mlir::arith;
  AddArithOpConversionsImpl<
      // go/keep-sorted start
      // clang-format off
        ma::AddFOp,
        ma::AddIOp,
        ma::AddUIExtendedOp,
        ma::AndIOp,
        ma::BitcastOp,
        ma::CeilDivSIOp,
        ma::CeilDivUIOp,
        ma::CmpFOp,
        ma::CmpIOp,
        ma::ConstantOp,
        ma::DivFOp,
        ma::DivSIOp,
        ma::DivUIOp,
        ma::ExtFOp,
        ma::ExtSIOp,
        ma::ExtUIOp,
        ma::FPToSIOp,
        ma::FPToUIOp,
        ma::FloorDivSIOp,
        ma::IndexCastOp,
        ma::IndexCastUIOp,
        ma::MaxNumFOp,
        ma::MaxSIOp,
        ma::MaxUIOp,
        ma::MaximumFOp,
        ma::MinNumFOp,
        ma::MinSIOp,
        ma::MinUIOp,
        ma::MinimumFOp,
        ma::MulFOp,
        ma::MulIOp,
        ma::MulSIExtendedOp,
        ma::MulUIExtendedOp,
        ma::NegFOp,
        ma::OrIOp,
        ma::RemFOp,
        ma::RemSIOp,
        ma::RemUIOp,
        ma::SIToFPOp,
        ma::ScalingExtFOp,
        ma::ScalingTruncFOp,
        ma::SelectOp,
        ma::ShLIOp,
        ma::ShRSIOp,
        ma::ShRUIOp,
        ma::SubFOp,
        ma::SubIOp,
        ma::TruncFOp,
        ma::TruncIOp,
        ma::UIToFPOp,
        ma::XOrIOp
      // clang-format on
      // go/keep-sorted end
      >(target, patterns, typeConverter);
}

struct ArithToVectorPass
    : public impl::ArithToVectorPassBase<ArithToVectorPass> {
  void runOnOperation() override {
    auto* context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(*context);
    mlir::RewritePatternSet patterns(context);
    TensorToVectorTypeConverter typeConverter;
    AddArithOpConversions(target, patterns, typeConverter);

    mlir::ConversionConfig config;
    config.buildMaterializations = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateArithToVectorPass() {
  return std::make_unique<ArithToVectorPass>();
}

}  // namespace xla::cpu
