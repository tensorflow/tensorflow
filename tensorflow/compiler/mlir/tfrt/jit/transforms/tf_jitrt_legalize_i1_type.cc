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

#include <algorithm>
#include <optional>
#include <utility>

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {

using llvm::APInt;
using llvm::ArrayRef;
using llvm::dyn_cast;
using llvm::SmallVector;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DenseElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::Type;
using mlir::TypeConverter;
using mlir::Value;
using mlir::func::FuncOp;

#define GEN_PASS_DEF_JITRTLEGALIZEI1TYPES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

static std::optional<Type> PromoteI1ToI8(Type input_type) {
  if (auto integer_type = input_type.dyn_cast<IntegerType>()) {
    if (integer_type.getWidth() == 1)
      return integer_type.scaleElementBitwidth(8);
  }

  return std::nullopt;
}

/// TypeConverter that turns 'i1' tensors into 'i8' tensors.
class I1TypeConverter : public mlir::TypeConverter {
 public:
  using TypeConverter::convertType;

  I1TypeConverter() {
    // Catch-all type conversion.
    addConversion([](Type type) { return type; });

    addConversion([](RankedTensorType tensor_type) -> std::optional<Type> {
      auto maybe_promoted_i8_type = PromoteI1ToI8(tensor_type.getElementType());
      if (!maybe_promoted_i8_type) return tensor_type;
      return RankedTensorType::get(tensor_type.getShape(),
                                   *maybe_promoted_i8_type);
    });
  }
};

static bool isLegalType(const Type type) {
  if (auto tensor_type = type.dyn_cast<RankedTensorType>()) {
    if (auto integer_type =
            tensor_type.getElementType().dyn_cast<IntegerType>()) {
      return integer_type.getWidth() != 1;
    }
  }

  return true;
}

static bool isLegalAttribute(NamedAttribute attr) {
  if (auto int_attr = attr.getValue().dyn_cast<DenseIntElementsAttr>()) {
    // Only RankedTensorType is expected.
    ShapedType shaped_type = int_attr.getType();
    if (!shaped_type.isa<RankedTensorType>()) return true;
    return !shaped_type.getElementType().isInteger(/*width=*/1);
  }

  // TODO(diegocaballero): Add support for TypeAttr if/when we have a use case.

  return true;
}

static NamedAttribute convertAttribute(NamedAttribute attr,
                                       ConversionPatternRewriter &rewriter) {
  if (auto int_attr = attr.getValue().dyn_cast<DenseIntElementsAttr>()) {
    ShapedType shaped_type = int_attr.getType();
    // Only RankedTensorType is expected.
    if (!shaped_type.isa<RankedTensorType>()) return attr;
    if (!shaped_type.getElementType().isInteger(/*width=*/1)) return attr;

    // Convert internal bool attribute representation to 8-bit integer.
    SmallVector<APInt, 4> new_i8_values;
    for (bool bool_val : int_attr.getValues<bool>()) {
      new_i8_values.push_back(
          bool_val ? APInt::getOneBitSet(/*numBits=*/8, /*bitNo=*/0)
                   : APInt::getZero(/*numBits=*/8));
    }

    auto i8_tensor_type =
        RankedTensorType::get(shaped_type.getShape(), rewriter.getI8Type());
    return NamedAttribute(
        attr.getName(), DenseElementsAttr::get(i8_tensor_type, new_i8_values));
  }

  // TODO(diegocaballero): Add support for TypeAttr if/when we have a use case.

  return attr;
}

/// Generic conversion pattern that replaces any operation (except FuncOp) using
/// 'i1' tensors with the same operation using 'i8' tensors.
struct I1ToI8GenericConversionPattern : public ConversionPattern {
  using ConversionPattern::ConversionPattern;

  I1ToI8GenericConversionPattern(I1TypeConverter &type_converter,
                                 MLIRContext *context)
      : ConversionPattern(type_converter, MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> converted_operands,
      ConversionPatternRewriter &rewriter) const override {
    // Convert attributes.
    SmallVector<NamedAttribute, 4> new_attrs;
    for (NamedAttribute attr : op->getAttrs())
      new_attrs.push_back(convertAttribute(attr, rewriter));

    // Convert result types.
    SmallVector<Type, 4> new_result_types;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           new_result_types)))
      return mlir::failure();

    // Create a new op using the converted attributes, operands and result
    // types. If the existing op has regions, we move them to the new op and
    // convert their signature.
    OperationState new_op_state(op->getLoc(), op->getName().getStringRef(),
                                converted_operands, new_result_types, new_attrs,
                                op->getSuccessors());

    for (Region &region : op->getRegions()) {
      Region *new_region = new_op_state.addRegion();
      rewriter.inlineRegionBefore(region, *new_region, new_region->begin());

      TypeConverter::SignatureConversion signature_conv(
          new_region->getNumArguments());
      if (failed(typeConverter->convertSignatureArgs(
              new_region->getArgumentTypes(), signature_conv)))
        return mlir::failure();
      rewriter.applySignatureConversion(new_region, signature_conv);
    }

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
  }
};

static void populateI1TypeConversionPatterns(I1TypeConverter &type_converter,
                                             RewritePatternSet &patterns) {
  patterns.add<I1ToI8GenericConversionPattern>(type_converter,
                                               patterns.getContext());
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(
      patterns, type_converter);
}

struct JitRtLegalizeI1TypesPass
    : public impl::JitRtLegalizeI1TypesBase<JitRtLegalizeI1TypesPass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    I1TypeConverter type_converter;

    ConversionTarget target(context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      // Check legality of attributes.
      auto attrs = op->getAttrs();
      if (std::any_of(attrs.begin(), attrs.end(), [&](NamedAttribute attr) {
            return !isLegalAttribute(attr);
          }))
        return false;

      // Check legality of FuncOp.
      if (FuncOp func_op = dyn_cast<FuncOp>(op)) {
        auto input_types = func_op.getFunctionType().getInputs();
        auto result_types = func_op.getFunctionType().getResults();
        return std::all_of(
                   input_types.begin(), input_types.end(),
                   [&](const Type type) { return isLegalType(type); }) &&
               std::all_of(result_types.begin(), result_types.end(),
                           [&](const Type type) { return isLegalType(type); });
      }

      // Check legality of any other op.
      auto operand_types = op->getOperandTypes();
      auto result_types = op->getResultTypes();
      return std::all_of(operand_types.begin(), operand_types.end(),
                         [](Type type) { return isLegalType(type); }) &&
             std::all_of(result_types.begin(), result_types.end(),
                         [](Type type) { return isLegalType(type); });
    });

    RewritePatternSet patterns(&context);
    populateI1TypeConversionPatterns(type_converter, patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
CreateJitRtLegalizeI1TypesPass() {
  return std::make_unique<JitRtLegalizeI1TypesPass>();
}

}  // namespace tensorflow
