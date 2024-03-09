/* Copyright 2020 The OpenXLA Authors.

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

// This file implements logic for translating mixed IR to buffer form.
// Currently it supports MHLO and some operations from the Standard dialect.

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CONVERTTOSIGNLESSPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// Generic pattern that rewrites any op by rewriting its operands and result
// types. Regions are also rewritten.
class ConvertToSignless : public ConversionPattern {
 public:
  ConvertToSignless(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto* newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region& before = std::get<0>(regions);
      Region& parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// A pattern that converts the type of the attribute used as an operand for
// arith.constant
class ConvertConstantToSignless
    : public OpConversionPattern<arith::ConstantOp> {
 public:
  ConvertConstantToSignless(TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(
      arith::ConstantOp constantOp, arith::ConstantOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We only care about unsigned integers
    if (!adaptor.getValue().isa<DenseIntElementsAttr>()) return failure();

    auto values = llvm::to_vector(
        adaptor.getValue().cast<DenseIntElementsAttr>().getValues<APInt>());
    Type type = typeConverter->convertType(constantOp.getType());
    auto shapedType = type.dyn_cast<ShapedType>();
    auto newValues = DenseIntElementsAttr::get(
        shapedType, values);

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, newValues);
    return success();
  }
};

struct ConvertToSignlessPass
    : public impl::ConvertToSignlessPassBase<ConvertToSignlessPass> {
 public:
  void runOnOperation() override {
    auto& context = getContext();
    ConversionTarget target(context);

    mhlo::RemoveSignTypeConverter converter;
    target.markUnknownOpDynamicallyLegal([&](auto op) {
      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return converter.isLegal(op.getType()) &&
             converter.isLegal(op.getValue().getType());
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertToSignless, ConvertConstantToSignless>(converter,
                                                               &context);
    // FuncOp is special as it has type encoding via attributes.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSignlessPass() {
  return std::make_unique<ConvertToSignlessPass>();
}

}  // namespace mhlo
}  // namespace mlir
