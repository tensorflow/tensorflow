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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_gpu.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/generated_transform_patterns.inc"
}  // namespace

OwningRewritePatternList GetHardwareRewritePatterns(
    MLIRContext* context, const std::string& hardware) {
  auto* devce_hardware = GetTargetHardware(hardware);
  if (devce_hardware == nullptr) return {context};
  return devce_hardware->GetTransformations(context);
}

bool IsSupported(Operation* op, const std::string& hardware) {
  auto* devce_hardware = GetTargetHardware(hardware);
  if (devce_hardware == nullptr) return {};
  return devce_hardware->IsOpSupported(op);
}

// ================== Convert Quantized Op ============================

// Walk through the func and convert the quantize ops to their float version.
void ConvertQuantizedOpToFloat(mlir::FuncOp func, OpBuilder* builder) {
  func.walk([&](Operation* op) {
    // TODO(renjieliu): Find a generic way to deal with const ops.
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp>(op) ||
        llvm::isa<TFL::QConstOp, TFL::ConstOp, TF::ConstOp, ConstOp>(op))
      return;

    bool int8_type_observed = false;
    bool uint8_type_observed = false;
    for (auto& input : op->getOpOperands()) {
      auto input_type = input.get().getType();
      if (IsQI8Type(input_type)) {
        int8_type_observed = true;
      } else if (IsQUI8Type(input_type)) {
        uint8_type_observed = true;
      }
    }

    // TODO(renjieliu): We probably should check whether the op supports float
    // execution to be safe. Although normally they should support float
    // execution. Not Quantized ops.
    if (!int8_type_observed && !uint8_type_observed) return;

    // Insert dequantize ops for every quantized input.
    SmallVector<Value, 4> dequantized_inputs;
    for (auto& input : op->getOpOperands()) {
      auto input_type = input.get().getType();
      if (IsQI8Type(input_type) || IsQUI8Type(input_type) ||
          IsQI32Type(input_type)) {
        auto dequantized_input_type =
            mlir::quant::QuantizedType::castToExpressedType(input_type);
        builder->setInsertionPoint(op);
        auto dequantize_op = builder->create<TFL::DequantizeOp>(
            op->getLoc(), dequantized_input_type, input.get());
        dequantized_inputs.push_back(dequantize_op);
      } else {
        dequantized_inputs.push_back(input.get());
      }
    }

    // Result types.
    SmallVector<Type, 4> result_types;
    for (auto result_type : op->getResultTypes()) {
      if (IsQI8Type(result_type) || IsQUI8Type(result_type)) {
        auto dequantized_result_type =
            mlir::quant::QuantizedType::castToExpressedType(result_type);
        result_types.push_back(dequantized_result_type);
      } else {
        result_types.push_back(result_type);
      }
    }

    // Build the new float-versioned op.
    OperationState state(op->getLoc(), op->getName());
    state.operands = dequantized_inputs;
    state.types = result_types;
    state.attributes = op->getAttrs();
    state.successors = op->getSuccessors();
    builder->setInsertionPoint(op);
    Operation* new_op = builder->createOperation(state);

    // Insert quantize ops for every outputs and rewrite.
    for (int i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      auto result_type = result.getType();

      Value new_result = new_op->getResult(i);
      if (IsQI8Type(result_type) || IsQUI8Type(result_type)) {
        builder->setInsertionPoint(op);
        TFL::QuantizeOp quant_op = builder->create<TFL::QuantizeOp>(
            op->getLoc(), result_type, new_result, TypeAttr::get(result_type));
        new_result = quant_op.getResult();
      }

      // Rewire the outputs.
      result.replaceAllUsesWith(new_result);
    }

    // Remove the old op.
    op->erase();
  });
}

// Fold quantized i32 (normally bias) into their float values.
struct FoldQuantizedI32ToFloat : public OpRewritePattern<TFL::DequantizeOp> {
  using OpRewritePattern<TFL::DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DequantizeOp dequant_op,
                                PatternRewriter& rewriter) const override {
    // We only fold i32 -> float pattern.
    auto input = dequant_op.input().getDefiningOp();
    if (!input) return failure();

    auto input_dequant = llvm::dyn_cast_or_null<TFL::QConstOp>(input);
    if (!input_dequant) return failure();

    if (!IsQI32Type(input_dequant.getType())) return failure();

    auto output_type =
        dequant_op.output().getType().dyn_cast_or_null<ShapedType>();
    if (!output_type || !output_type.getElementType().isF32()) return failure();

    auto input_type = input_dequant.getType().dyn_cast<ShapedType>();
    // TODO(renjieliu): support UniformQuantizedPerAxisType.
    auto q_type = input_type.getElementType()
                      .dyn_cast_or_null<quant::UniformQuantizedType>();
    if (!q_type) return failure();

    const float scale = q_type.getScale();
    const float zp = q_type.getZeroPoint();

    auto input_values = input_dequant.value();

    // mapValues always takes a function returning APInt, even when the output
    // is actually float.
    using DequantizeFuncType = llvm::APInt(const llvm::APInt&);
    auto dequantize_func = [&](const APInt& ap_int_value) -> APInt {
      const int64_t int_value = ap_int_value.getSExtValue();

      const float real = (int_value - zp) * scale;

      auto real_int = absl::bit_cast<int32_t>(real);
      return APInt(/*numBits=*/32, real_int);
    };

    auto dequant_values =
        input_values.cast<DenseIntOrFPElementsAttr>().mapValues(
            FloatType::getF32(rewriter.getContext()),
            llvm::function_ref<DequantizeFuncType>(dequantize_func));
    rewriter.replaceOpWithNewOp<TFL::ConstOp>(dequant_op, dequant_op.getType(),
                                              dequant_values);

    return success();
  }
};

// If the quant op has no consumer, we will remove them.
struct RemoveUnusedQuant : public OpRewritePattern<TFL::QuantizeOp> {
  using OpRewritePattern<TFL::QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::QuantizeOp quant_op,
                                PatternRewriter& rewriter) const override {
    if (!quant_op.getResult().use_empty()) return failure();

    rewriter.eraseOp(quant_op);
    return success();
  }
};

void OptimizeQuantizedOpToFloat(FuncOp func, MLIRContext* context) {
  OwningRewritePatternList patterns(func.getContext());
  patterns.insert<FoldQuantizedI32ToFloat, FoldQuantizeDequantize,
                  RemoveUnusedQuant>(context);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
