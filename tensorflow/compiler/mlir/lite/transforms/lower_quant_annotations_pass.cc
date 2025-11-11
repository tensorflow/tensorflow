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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_LOWERQUANTANNOTATIONSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

/**
 * Replaces a quant.quantize composite op with a TFLite quantize op that outputs
 * a quantized tensor based on the composite op's attributes.
 */
class RewriteQuantizeCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.quantize") {
      return failure();
    }

    SmallVector<double, 4> scales;
    SmallVector<int64_t, 4> zero_points;
    int num_bits;
    bool is_signed;
    bool is_narrow_range;

    if (failed(FillCompositeParams(op, scales, zero_points, num_bits, is_signed,
                                   is_narrow_range))) {
      return op.emitError(
          "quantize composite does not contain the required attributes.");
    }

    ShapedType input_shaped_type = cast<ShapedType>(op.getOperand(0).getType());
    Type input_element_type = input_shaped_type.getElementType();

    Type quantized_element_type;
    if (scales.size() == 1) {
      quantized_element_type = GetPerTensorQuantizedTensorType(
          rewriter, scales[0], zero_points[0],
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          is_narrow_range, is_signed);
    } else {
      int32_t quantized_dimension;
      if (auto quantized_dimension_attr = llvm::dyn_cast_or_null<IntegerAttr>(
              op.getCompositeAttributes().get("quantization_dimension"))) {
        quantized_dimension =
            quantized_dimension_attr.getValue().getSExtValue();
      } else {
        return op.emitError(
            "quantization_dimension attribute is missing from the composite.");
      }
      quantized_element_type = GetPerAxisQuantizedTensorType(
          rewriter, scales, zero_points, quantized_dimension,
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          is_narrow_range, is_signed);
    }

    RankedTensorType output_type = RankedTensorType::get(
        input_shaped_type.getShape(), quantized_element_type);
    TFL::QuantizeOp tfl_quantize_op =
        TFL::QuantizeOp::create(rewriter, op.getLoc(), output_type,
                                /*input=*/op.getOperand(0),
                                /*qtype=*/TypeAttr::get(output_type));

    rewriter.replaceOp(op, tfl_quantize_op.getOutput());
    return success();
  }
};

static func::FuncOp GetEnclosingFunction(mlir::Operation* op) {
  while (op) {
    // Check if the current operation is a function.
    auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (func_op) {
      return func_op;
    }
    op = op->getParentOp();
  }
  return nullptr;  // Operation is not within a function.
}

/**
 * Replaces a quant.dequantize composite op with a TFLite dequantize op that
 * receives a quantized tensor based on the composite op's attributes.
 */
class RewriteDequantizeCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp composite_op,
                                PatternRewriter& rewriter) const final {
    if (composite_op.getName() != "quant.dequantize") {
      return failure();
    }
    Type output_type = composite_op.getType(0);

    SmallVector<double, 4> scales;
    SmallVector<int64_t, 4> zero_points;
    int num_bits;
    bool is_signed;
    bool is_narrow_range;

    if (failed(FillCompositeParams(composite_op, scales, zero_points, num_bits,
                                   is_signed, is_narrow_range))) {
      return failure();
    }

    auto composite_operand = composite_op.getOperand(0);

    ShapedType output_shaped_type = cast<ShapedType>(output_type);
    Type output_element_type = output_shaped_type.getElementType();

    Type quantized_element_type;
    if (scales.size() == 1) {
      quantized_element_type = GetPerTensorQuantizedTensorType(
          rewriter, scales[0], zero_points[0],
          /*expressed_type=*/output_element_type, num_bits,
          composite_op->getLoc(), is_narrow_range, is_signed);
    } else {
      int32_t quantized_dimension;
      if (auto quantized_dimension_attr = llvm::dyn_cast_or_null<IntegerAttr>(
              composite_op.getCompositeAttributes().get(
                  "quantization_dimension"))) {
        quantized_dimension =
            quantized_dimension_attr.getValue().getSExtValue();
      } else {
        return failure();
      }
      quantized_element_type = GetPerAxisQuantizedTensorType(
          rewriter, scales, zero_points, quantized_dimension,
          /*expressed_type=*/output_element_type, num_bits,
          composite_op->getLoc(), is_narrow_range, is_signed);
    }

    ShapedType input_shaped_type =
        cast<ShapedType>(composite_op.getOperand(0).getType());
    RankedTensorType qtensor_type = RankedTensorType::get(
        input_shaped_type.getShape(), quantized_element_type);

    Value tfl_quantize_input;
    if (mlir::dyn_cast<mlir::BlockArgument>(composite_operand)) {
      // Find the function enclosing this composite op.
      func::FuncOp func_op = GetEnclosingFunction(composite_op);
      if (func_op == nullptr) {
        return failure();
      }

      // Find the operand index of the input of the composite op.
      int arg_idx = -1;
      for (int i = 0; i < func_op.getNumArguments(); ++i) {
        if (func_op.getBody().front().getArgument(i) == composite_operand) {
          arg_idx = i;
          break;
        }
      }
      if (arg_idx == -1) {
        return failure();
      }

      // create a new set of operand types for the function with the type of
      // the operand that feeds the composite op changed.
      SmallVector<Type, 4> new_func_input_types;
      auto func_input_types = func_op.getFunctionType().getInputs();
      for (int i = 0; i < func_input_types.size(); ++i) {
        if (i != arg_idx) {
          new_func_input_types.push_back(func_input_types[i]);
        } else {
          new_func_input_types.push_back(qtensor_type);
        }
      }

      auto new_func_type =
          mlir::FunctionType::get(func_op.getContext(), new_func_input_types,
                                  func_op.getFunctionType().getResults());

      rewriter.startOpModification(func_op);
      // Update the function type.
      func_op.setType(new_func_type);

      // Update the block argument type.
      func_op.getBody().front().getArgument(arg_idx).setType(qtensor_type);
      rewriter.finalizeOpModification(func_op);

      tfl_quantize_input = func_op.getBody().front().getArgument(arg_idx);
    } else {
      // Using the last operand of the composite op as the input of the
      // dequantize op in case it's a dynamic shaped model.
      // TODO - b/422588785: Have proper support for dynamic shaped models.
      int num_operands = composite_op.getNumOperands();
      auto producer_op =
          composite_op.getOperand(num_operands - 1).getDefiningOp();
      rewriter.startOpModification(producer_op);
      producer_op->getResults().front().setType(qtensor_type);
      rewriter.finalizeOpModification(producer_op);

      tfl_quantize_input = composite_op.getOperand(num_operands - 1);
    }

    TFL::DequantizeOp tfl_dequantize_op =
        TFL::DequantizeOp::create(rewriter, composite_op.getLoc(), output_type,
                                  /*input=*/tfl_quantize_input);
    rewriter.replaceOp(composite_op, tfl_dequantize_op.getOutput());

    return success();
  }
};

class RewriteFakeQuantCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.fake_quant" || IsDrqFakeQuant(op)) {
      return failure();
    }

    SmallVector<double, 4> scales;
    SmallVector<int64_t, 4> zero_points;
    int num_bits;
    bool is_signed;
    bool is_narrow_range;

    if (failed(FillCompositeParams(op, scales, zero_points, num_bits, is_signed,
                                   is_narrow_range))) {
      return rewriter.notifyMatchFailure(op, "Failed to fill composite params");
    }

    // Using the last operand of the composite op as the input of the
    // dequantize op in case it's a dynamic shaped model.
    // TODO - b/422588785: Have proper support for dynamic shaped models.
    int num_operands = op.getNumOperands();
    ShapedType input_shaped_type =
        cast<ShapedType>(op.getOperand(num_operands - 1).getType());
    Type input_element_type = input_shaped_type.getElementType();
    Type quantized_element_type;
    if (scales.size() == 1) {
      quantized_element_type = GetPerTensorQuantizedTensorType(
          rewriter, scales[0], zero_points[0],
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          is_narrow_range, is_signed);
    } else {
      int32_t quantized_dimension;
      if (auto quantized_dimension_attr = llvm::dyn_cast_or_null<IntegerAttr>(
              op.getCompositeAttributes().get("quantization_dimension"))) {
        quantized_dimension =
            quantized_dimension_attr.getValue().getSExtValue();
      } else {
        return rewriter.notifyMatchFailure(
            op,
            "quantization_dimension attribute is missing from the composite.");
      }
      quantized_element_type = GetPerAxisQuantizedTensorType(
          rewriter, scales, zero_points, quantized_dimension,
          /*expressed_type=*/input_element_type, num_bits, op->getLoc(),
          is_narrow_range, is_signed);
    }
    RankedTensorType intermediate_type = RankedTensorType::get(
        input_shaped_type.getShape(), quantized_element_type);
    TFL::QuantizeOp tfl_quantize_op =
        TFL::QuantizeOp::create(rewriter, op.getLoc(), intermediate_type,
                                /*input=*/op.getOperand(num_operands - 1),
                                /*qtype=*/TypeAttr::get(intermediate_type));

    Type output_type = op.getType(0);
    TFL::DequantizeOp tfl_dequantize_op = TFL::DequantizeOp::create(
        rewriter, op.getLoc(), output_type, /*input=*/tfl_quantize_op);

    rewriter.replaceOp(op, tfl_dequantize_op.getOutput());
    return success();
  }
};

class RemovePreventGradient : public OpRewritePattern<TF::PreventGradientOp> {
  using OpRewritePattern<TF::PreventGradientOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::PreventGradientOp op,
                                PatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

class RemoveIdentity : public OpRewritePattern<TF::IdentityOp> {
  using OpRewritePattern<TF::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::IdentityOp op,
                                PatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

/**
 * When there is a quantize op at the output, the return op's operand is a
 * quantized tensor. However, the function's return type is still a simple
 * integer. This pattern makes sure the function's signature is updated so that
 * it's return type conforms the operand of its return op.
 */
class UpdateFunctionOutputType : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp return_op,
                                PatternRewriter& rewriter) const final {
    func::FuncOp func_op = GetEnclosingFunction(return_op);
    if (func_op == nullptr) {
      return rewriter.notifyMatchFailure(return_op,
                                         "Failed to find enclosing function");
    }

    auto return_op_types = return_op.getOperandTypes();
    auto current_func_type = func_op.getFunctionType();

    // If the function's result types already match the return op's
    // operand types, report failure so the rewriter converges.
    if (current_func_type.getResults() == return_op_types) {
      return failure();
    }

    rewriter.startOpModification(func_op);
    auto new_func_type = mlir::FunctionType::get(
        func_op.getContext(), current_func_type.getInputs(), return_op_types);
    func_op.setFunctionType(new_func_type);
    rewriter.finalizeOpModification(func_op);

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
  auto module = getOperation();

  RewritePatternSet prepare_patterns(&ctx);
  prepare_patterns.add<RemovePreventGradient, RemoveIdentity>(&ctx);

  GreedyRewriteConfig greedy_config;
  greedy_config.enableFolding(true);
  if (failed(applyPatternsGreedily(module, std::move(prepare_patterns),
                                   greedy_config))) {
    module.emitError(
        "Failed to apply lower_quant_annotations prepare patterns.");
    signalPassFailure();
  }

  RewritePatternSet patterns(&ctx);
  patterns.add<RewriteQuantizeCompositeOp, RewriteDequantizeCompositeOp,
               RewriteFakeQuantCompositeOp>(&ctx);

  if (failed(
          applyPatternsGreedily(module, std::move(patterns), greedy_config))) {
    getOperation().emitError("Composite lowering pass failed.");
    signalPassFailure();
  }

  RewritePatternSet cleanup_patterns(&ctx);
  cleanup_patterns.add<UpdateFunctionOutputType>(&ctx);

  // TODO(b/393642164) Somehow the following is returning failure even though
  // it is updating the function signature correctly. Investigate why failure is
  // returned and then add error-handling.
  (void)applyPatternsGreedily(module, std::move(cleanup_patterns),
                              greedy_config);
}
}  // namespace
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerQuantAnnotationsPass() {
  return std::make_unique<LowerQuantAnnotationsPass>();
}
}  // namespace TFL
}  // namespace mlir
