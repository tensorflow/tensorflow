/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// TOSA has no notion of complex datatypes, it represents a single complex
// input tensor using two floating point input tensors corresponding to the
// "real" and "imag" parts of the complex input.
//
// To maintain the correct mapping of these tensors during the legalization
// from TFL to TOSA, a complex tensor of shape [x, ..., y] is converted to
// a single floating point tensor of shape [x, ..., y, 2] where each resulting
// pair of values can be used to represent a complex value, which ensures a
// 1:1 mapping between TFL and TOSA input/output tensors. In legalization,
// "unrealized_conversion_cast" operations are inserted to express this
// conversion.
//
// This pass removes complex tensors from the graph by rewriting them using
// the above [x, ..., y, 2] floating point format. Consequently, it removes
// any remaining "unrealized_conversion_cast" operations and ensures the
// resulting graph is free of illegal complex tensors.

#include <iterator>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"       // from @llvm-project
#include "mlir/IR/Builders.h"                   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"          // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"               // from @llvm-project
#include "mlir/IR/PatternMatch.h"               // from @llvm-project
#include "mlir/Pass/PassRegistry.h"             // from @llvm-project
#include "mlir/Support/LogicalResult.h"         // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-strip-complex-types"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSASTRIPCOMPLEXTYPESPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class StripComplexTypes
    : public impl::TosaStripComplexTypesPassBase<StripComplexTypes> {
 public:
  explicit StripComplexTypes() {}
  void runOnOperation() override;
};

class ComplexTypeConverter : public TypeConverter {
 public:
  static Type convertTensor(RankedTensorType type) {
    if (auto elementType = type.getElementType().dyn_cast<ComplexType>()) {
      llvm::SmallVector<int64_t> newShape;
      for (auto dim : type.getShape()) {
        newShape.push_back(dim);
      }
      newShape.push_back(2);
      return RankedTensorType::get(newShape, elementType.getElementType());
    }
    return type;
  }

  explicit ComplexTypeConverter() { addConversion(convertTensor); }
};

// Handles the type conversion component of the TypeConversion. This updates
// conversion patterns that used the original complex tensor types to be
// updated to the non-complex variants.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(MLIRContext* context, TypeConverter& converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (isa<func::FuncOp>(op)) {
      return failure();
    }

    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, op->getAttrs(), op->getSuccessors());
    for (Region& r : op->getRegions()) {
      Region* newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation* newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static bool isIllegalType(Type type) {
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return shapedType.getElementType().isa<ComplexType>();
  }
  return false;
}

void StripComplexTypes::runOnOperation() {
  ComplexTypeConverter converter;
  ConversionTarget target(getContext());

  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  // Operations are legal if they don't contain any illegal type.
  target.markUnknownOpDynamicallyLegal([](Operation* op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : funcOp.getFunctionType().getResults()) {
        if (isIllegalType(type)) return false;
      }
    }
    for (Type type : op->getResultTypes()) {
      if (type && isIllegalType(type)) return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (type && isIllegalType(type)) return false;
    }
    return true;
  });

  auto func = getOperation();
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  if (failed(applyFullConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createStripComplexTypesPass() {
  return std::make_unique<StripComplexTypes>();
}
}  // namespace tosa
}  // namespace mlir
