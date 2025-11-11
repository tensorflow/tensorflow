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
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_ELEMENTALTENSORTOVECTORPASS
#define GEN_PASS_DEF_ELEMENTALTENSORTOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// This converter defines the rules for mapping types from the source (tensors)
// to the target (vectors).
class TensorToVectorTypeConverter : public mlir::TypeConverter {
 public:
  TensorToVectorTypeConverter() {
    // Keep all non-tensor types as-is.
    addConversion([](mlir::Type type) { return type; });

    // Convert RankedTensorType to VectorType.
    addConversion([](mlir::RankedTensorType type) -> mlir::Type {
      // We can only convert tensors with a static shape to vectors.
      if (!type.hasStaticShape()) {
        return nullptr;  // Return null if the type cannot be converted.
      }
      return mlir::VectorType::get(type.getShape(), type.getElementType());
    });

    addSourceMaterialization([](mlir::OpBuilder& builder,
                                mlir::Type result_type, mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) {
        return nullptr;
      }

      return WriteVectorToTensor(builder, inputs.front());
    });

    addTargetMaterialization([](mlir::OpBuilder& builder,
                                mlir::Type result_type, mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) {
        return nullptr;
      }

      return ReadTensorToVector(builder, inputs.front());
    });
  }

 private:
  static llvm::SmallVector<mlir::Value> MakeZeroIndices(
      mlir::OpBuilder& builder, mlir::Location loc, int64_t rank) {
    return llvm::SmallVector<mlir::Value>(
        rank, mlir::arith::ConstantIndexOp::create(builder, loc, 0));
  }
};

// A generic pattern to convert an elemental op from tensor-based to
// vector-based.
template <typename ElementalOp>
class ElementalOpConversion : public mlir::OpConversionPattern<ElementalOp> {
 public:
  using mlir::OpConversionPattern<ElementalOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ElementalOp op, typename ElementalOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> new_result_types;
    mlir::LogicalResult results_ok = this->getTypeConverter()->convertTypes(
        op->getResultTypes(), new_result_types);
    if (results_ok.failed()) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }

    rewriter.replaceOpWithNewOp<ElementalOp>(
        op, new_result_types, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

// We need to specify the ConstantOp conversion explicitly as it doesn't follow
// the simple operands & results of the other Arith ops.
template <>
class ElementalOpConversion<mlir::arith::ConstantOp>
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

template <typename... ElementalOps>
void AddAElementalOpConversionsImpl(
    mlir::ConversionTarget& target, mlir::RewritePatternSet& patterns,
    TensorToVectorTypeConverter& typeConverter) {
  target.addDynamicallyLegalOp<ElementalOps...>(
      [&](mlir::Operation* op) { return typeConverter.isLegal(op); });
  patterns.add<ElementalOpConversion<ElementalOps>...>(typeConverter,
                                                       patterns.getContext());
}

void AddArithOpConversions(mlir::ConversionTarget& target,
                           mlir::RewritePatternSet& patterns,
                           TensorToVectorTypeConverter& typeConverter) {
  AddAElementalOpConversionsImpl<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
#undef GET_OP_LIST
      >(target, patterns, typeConverter);
}

void AddMathOpConversions(mlir::ConversionTarget& target,
                          mlir::RewritePatternSet& patterns,
                          TensorToVectorTypeConverter& typeConverter) {
  AddAElementalOpConversionsImpl<
#define GET_OP_LIST
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"
#undef GET_OP_LIST
      >(target, patterns, typeConverter);
}

struct ElementalTensorToVectorPass
    : public impl::ElementalTensorToVectorPassBase<
          ElementalTensorToVectorPass> {
  void runOnOperation() override {
    auto* context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(*context);
    mlir::RewritePatternSet patterns(context);
    TensorToVectorTypeConverter typeConverter;
    AddArithOpConversions(target, patterns, typeConverter);
    AddMathOpConversions(target, patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateElementalTensorToVectorPass() {
  return std::make_unique<ElementalTensorToVectorPass>();
}

}  // namespace xla::cpu
