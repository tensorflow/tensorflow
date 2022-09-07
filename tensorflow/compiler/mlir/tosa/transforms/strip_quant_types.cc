/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This pass converts a TFLite uint8 graph to the int8 domain, with adaptors at
// input and output tensors. This is needed because TOSA precision is
// implemented in the int8 domain. This pass does:
// 1. match TFL::QConst with uint8, generate TFL::QConst with int8 with value
// remapped.
// 2. insert tosa.RESCALE uint8 -> int8 if block argument (placeholder of graph)
// is uint8 typed.
// 3. insert tosa.RESCALE int8 -> uint8 if original returned tensor is uint8
// typed.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-convert-tfl-uint8"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSASTRIPQUANTTYPESPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class StripQuantTypes
    : public impl::TosaStripQuantTypesPassBase<StripQuantTypes> {
 public:
  explicit StripQuantTypes() {}
  void runOnOperation() override;
};

class QuantTypeConverter : public TypeConverter {
 public:
  static Type convertType(Type type) {
    if (auto qType = type.dyn_cast<quant::QuantizedType>()) {
      if (qType.isSigned() || qType.getStorageTypeIntegralWidth() != 8) {
        return IntegerType::get(type.getContext(),
                                qType.getStorageTypeIntegralWidth());
      }

      return IntegerType::get(type.getContext(),
                              qType.getStorageTypeIntegralWidth(),
                              IntegerType::SignednessSemantics::Unsigned);
    }
    return type;
  }
  static Type convertTensor(RankedTensorType type) {
    auto newType = RankedTensorType::get(type.getShape(),
                                         convertType(type.getElementType()));
    return newType;
  }
  explicit QuantTypeConverter() {
    addConversion([](Type type) { return convertType(type); });
    addConversion(convertTensor);
  }
};

// Handles the type conversion component of the TypeConversion. This updates
// conversion patterns that used the original Quant types to be updated to
// the non-quant variants.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(MLIRContext* context, TypeConverter& converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<Type, 4> newResults;
    if (isa<func::FuncOp>(op)) {
      return failure();
    }

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
  if (type.isa<quant::QuantizedType>()) return true;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return isIllegalType(shapedType.getElementType());
  }
  return false;
}

void StripQuantTypes::runOnOperation() {
  QuantTypeConverter converter;
  ConversionTarget target(getContext());

  target.addIllegalDialect<quantfork::QuantizationForkDialect>();
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

  auto* ctx = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);

  if (failed(applyFullConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createStripQuantTypesPass() {
  return std::make_unique<StripQuantTypes>();
}
}  // namespace tosa
}  // namespace mlir
