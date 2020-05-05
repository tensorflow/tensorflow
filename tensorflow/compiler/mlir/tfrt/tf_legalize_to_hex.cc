/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements lowering of Tf dialect to TFRT Hex kernels.
//
// Current lowering is a placeholder performing trivial conversion
// for integer constants and additions.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace {

// Pattern rewrite rules for "tf.Const", "tf.Add" and "return" ops.
bool isInt32LikeType(Type t) {
  if (t.isSignlessInteger(32)) return true;
  if (auto ttype = t.dyn_cast<RankedTensorType>()) {
    if (ttype.hasStaticShape() && ttype.getNumElements() == 1 &&
        ttype.getElementType().isSignlessInteger(32))
      return true;
  }
  return false;
}

// Replaces 32-bit integer TF::ConstOp with "hex.constant_int" op.
struct ConstOpConversion : public ConversionPattern {
  explicit ConstOpConversion(MLIRContext *context)
      : ConversionPattern(TF::ConstOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto constOp = cast<TF::ConstOp>(op);
    if (!isInt32LikeType(constOp.getType())) return failure();

    auto valueAttr = constOp.value();
    auto newAttr = Attribute();

    // Convert constant op if it has an integer or dense elements attribute.
    // Other kinds of element attributes are not converted for now.
    if (valueAttr.isa<IntegerAttr>()) {
      newAttr = valueAttr;
    } else if (auto v = valueAttr.dyn_cast<SplatElementsAttr>()) {
      if (v.isSplat()) newAttr = v.getSplatValue();
    }
    if (!newAttr) return failure();

    mlir::OperationState state(constOp.getLoc(), "hex.constant_int");
    state.types.push_back(rewriter.getIntegerType(32));
    state.addAttribute("value", newAttr);
    auto newOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

// Replaces 32-bit integer TF::Add op with "hex.add_int" op.
struct AddOpConversion : public ConversionPattern {
  explicit AddOpConversion(MLIRContext *context)
      : ConversionPattern(TF::AddOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto addOp = cast<TF::AddOp>(op);

    if (!isInt32LikeType(operands[0].getType()) ||
        !isInt32LikeType(operands[1].getType()))
      return failure();

    auto int32Ty = rewriter.getIntegerType(32);
    mlir::OperationState state(addOp.getLoc(), "hex.add_int", operands,
                               {int32Ty}, {});
    auto newOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

// Replaces return op that has no arguments with "hex.return" op.
struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReturnOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getNumOperands() != 0) return failure();

    mlir::OperationState state(srcOp.getLoc(), "hex.return");
    rewriter.createOperation(state);

    rewriter.eraseOp(srcOp);
    return success();
  }
};

// Legalize TF operations to host program dialect.
struct TfLegalizeToHex
    : public PassWrapper<TfLegalizeToHex, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) -> Type {
      // Convert single element tensor type of int32s to int32 type
      if (isInt32LikeType(type)) {
        return IntegerType::get(32, type.getContext());
      }
      return Type();
    });

    OwningRewritePatternList patterns;

    // For now, replace only int32 TF::OpConst, TF::OpAdd and OpReturn with
    // "hex.constant_int", "hex.add_int" and "hex.return", respectively.
    patterns.insert<ConstOpConversion, AddOpConversion, ReturnOpConversion>(
        ctx);

    ConversionTarget target(*ctx);
    const auto legal = ConversionTarget::LegalizationAction::Legal;
    target.setOpAction(OperationName(StringRef("hex.constant_int"), ctx),
                       legal);
    target.setOpAction(OperationName(StringRef("hex.add_int"), ctx), legal);
    target.setOpAction(OperationName(StringRef("hex.return"), ctx), legal);
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();

    auto result =
        applyFullConversion(getOperation(), target, patterns, &converter);
    if (failed(result)) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToHexPass() {
  return std::make_unique<TfLegalizeToHex>();
}

static PassRegistration<TfLegalizeToHex> pass(
    "tf-legalize-to-hex",
    "Convert TF dialect to the TF runtime host program dialect.");
}  // namespace mlir
