/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies quantization on TFLite dialect.

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_LOWERQUANTANNOTATIONSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

class RewriteFakeQuantCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

 public:
  explicit RewriteFakeQuantCompositeOp(MLIRContext* context)
      : OpRewritePattern<stablehlo::CompositeOp>(context) {
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.fake_quant") {
      return failure();
    }

    SmallVector<double, 4> scales;
    SmallVector<int64_t, 4> zero_points;
    int num_bits;
    bool is_signed;

    if (failed(FillCompositeParams(op, scales, zero_points, num_bits,
                                   is_signed))) {
      return failure();
    }

    ShapedType input_shaped_type = cast<ShapedType>(op.getOperand(0).getType());
    Type input_element_type = input_shaped_type.getElementType();
    Type quantized_element_type;
    if (scales.size() == 1) {
      quantized_element_type = GetPerTensorQuantizedTensorType(
          rewriter, scales[0], zero_points[0],
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          /*narrow_range=*/false, is_signed);
    } else {
      int32_t quantized_dimension;
      if (auto quantized_dimension_attr = llvm::dyn_cast_or_null<IntegerAttr>(
              op.getCompositeAttributes().get("quantization_dimension"))) {
        quantized_dimension =
            quantized_dimension_attr.getValue().getSExtValue();
      } else {
        return failure();
      }
      quantized_element_type = GetPerAxisQuantizedTensorType(
          rewriter, scales, zero_points, quantized_dimension,
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          /*narrow_range=*/false, is_signed);
    }
    RankedTensorType intermediate_type = RankedTensorType::get(
        input_shaped_type.getShape(), quantized_element_type);
    TFL::QuantizeOp tfl_quantize_op = rewriter.create<TFL::QuantizeOp>(
        op.getLoc(), intermediate_type,
        /*input=*/op.getOperand(0),
        /*qtype=*/TypeAttr::get(intermediate_type));

    Type output_type = op.getType(0);
    TFL::DequantizeOp tfl_dequantize_op = rewriter.create<TFL::DequantizeOp>(
        op.getLoc(), output_type, /*input=*/tfl_quantize_op);

    rewriter.replaceAllOpUsesWith(op, tfl_dequantize_op.getOutput());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LowerQuantAnnotationsPass
    : public impl::LowerQuantAnnotationsPassBase<LowerQuantAnnotationsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQuantAnnotationsPass)

  void runOnOperation() override;
};

void LowerQuantAnnotationsPass::runOnOperation() {
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<RewriteFakeQuantCompositeOp>(&ctx);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();
  target.addLegalDialect<quant::QuantDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  // Declare all the MHLO ops as legal except for the quantization composites we
  // want to lower.
  target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
      [](Operation* op) {
        auto mhlo_op = dyn_cast_or_null<stablehlo::CompositeOp>(op);
        if (!mhlo_op) {
          return true;
        }
        return mhlo_op.getName() != "quant.fake_quant";
      });

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("Composite lowering pass failed.");
    signalPassFailure();
  }
}
}  // namespace
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerQuantAnnotationsPass() {
  return std::make_unique<LowerQuantAnnotationsPass>();
}
}  // namespace TFL
}  // namespace mlir
