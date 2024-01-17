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

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_POPULATESHAPEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

class PopulateShapeForCustomAggregatorOp
    : public OpConversionPattern<TF::CustomAggregatorOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::CustomAggregatorOp op, TF::CustomAggregatorOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input_shape_type = op.getInput().getType().dyn_cast<Type>();
    auto output_shape_type = op.getOutput().getType();

    if (!input_shape_type.isa<RankedTensorType>()) {
      input_shape_type = adaptor.getInput().getType();
    }

    if (input_shape_type.isa<RankedTensorType>() &&
        !output_shape_type.isa<RankedTensorType>() &&
        TF::HasCompatibleElementTypes(input_shape_type, output_shape_type)) {
      auto new_op = rewriter.create<TF::CustomAggregatorOp>(
          op->getLoc(), /*output=*/input_shape_type,
          /*args=*/adaptor.getInput(),
          /*Id=*/op.getId());
      new_op->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, new_op);
      return success();
    }
    return failure();
  }
};

class PopulateShapeForXlaCallModuleOp
    : public OpConversionPattern<TF::XlaCallModuleOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::XlaCallModuleOp op, TF::XlaCallModuleOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      op->emitError("XlaCallModuleOp doesn't have 1 output.");
      return failure();
    }
    // Assume XlaCallModuleOp only has 1 output.
    auto output_shape_type = op->getResultTypes()[0];
    if (!output_shape_type.isa<RankedTensorType>()) {
      auto output_shape_attr = op.getSout()[0].dyn_cast<tf_type::ShapeAttr>();
      if (!output_shape_attr.hasRank()) {
        return failure();
      }
      auto new_output_shape_type = tensorflow::GetTypeFromTFTensorShape(
          output_shape_attr.getShape(),
          getElementTypeOrSelf(op.getResultTypes()[0]));
      auto new_op = rewriter.create<TF::XlaCallModuleOp>(
          op->getLoc(), /*output=*/new_output_shape_type,
          /*args=*/adaptor.getOperands(),
          /*version=*/op.getVersionAttr(),
          /*module=*/op.getModuleAttr(),
          /*Sout=*/op.getSoutAttr());
      new_op->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, new_op);
      return success();
    }
    return failure();
  }
};

class PopulateShapePass
    : public impl::PopulateShapePassBase<PopulateShapePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PopulateShapePass)

  explicit PopulateShapePass() = default;

 private:
  void runOnOperation() override;
};

void PopulateShapePass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<TF::CustomAggregatorOp>([](Operation *op) {
    auto custom_aggregator_op = llvm::dyn_cast<TF::CustomAggregatorOp>(op);
    return custom_aggregator_op.getInput().getType().isa<RankedTensorType>() &&
           custom_aggregator_op.getOutput().getType().isa<RankedTensorType>();
  });
  target.addDynamicallyLegalOp<TF::XlaCallModuleOp>([](Operation *op) {
    if (op->getNumResults() != 1) return true;
    return op->getResultTypes()[0].isa<RankedTensorType>();
  });

  patterns
      .add<PopulateShapeForCustomAggregatorOp, PopulateShapeForXlaCallModuleOp>(
          context);

  if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}
}  // namespace

}  // namespace mlir::quant::stablehlo
