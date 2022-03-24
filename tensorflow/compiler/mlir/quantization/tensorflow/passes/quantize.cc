/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
// Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/transforms/quantize.cc
// This transformation pass applies quantization on TF dialect.
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/quant_spec.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcretTy,
          typename RootOp = DequantizeCastOp>
struct TFQuantizationBase
    : public QuantizationPattern<ConcretTy, QuantizeCastOp, DequantizeCastOp,
                                 /*VERIFIER=*/void, RootOp> {
  explicit TFQuantizationBase(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : QuantizationPattern<ConcretTy, QuantizeCastOp, DequantizeCastOp,
                            /*VERIFIER=*/void, RootOp>(ctx, quant_params) {}

  // Custom op quantization is not supported.
  static bool IsQuantizableCustomOp(Operation* op,
                                    const CustomMap& custom_op_map) {
    return false;
  }

  // Dynamic range quantization is not supported.
  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const CustomMap& custom_op_map) {
    return false;
  }

  // Dynamic range quantization is not supported.
  static bool AllowDynamicRangeQuantizedResult(Operation* quantized_op,
                                               const CustomMap& custom_op_map) {
    return false;
  }

  // Weight-only quantization is not supported.
  static bool IsWeightOnlyOp(Operation* quantized_op, StringSet& ops_blocklist,
                             bool weight_only_quantization,
                             const CustomMap& custom_op_map) {
    return false;
  }
};

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFFullQuantization
    : public TFQuantizationBase<kFullQuantization, TFFullQuantization> {
  explicit TFFullQuantization(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantization>(
            ctx, quant_params) {}
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFFullQuantizationReverse
    : public TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                                QuantizeCastOp> {
  explicit TFFullQuantizationReverse(MLIRContext* ctx,
                                     const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                           QuantizeCastOp>(ctx, quant_params) {}
};

// Removes quantize-dequantize pairs that are not used in the quantization.
// The benefit of this pattern is set to lower value than other patterns, so
// that the other patterns can work on quantize/dequantize ops first.
class RemoveUnusedQdqPattern : public OpRewritePattern<QuantizeCastOp> {
 public:
  explicit RemoveUnusedQdqPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context) {}
  LogicalResult matchAndRewrite(QuantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    if (!op->hasOneUse() ||
        !llvm::isa<DequantizeCastOp>(*op->getUsers().begin())) {
      return failure();
    }
    op->getUsers().begin()->getResult(0).replaceAllUsesWith(op.arg());
    return success();
  }
};

QuantizedType GetQuantizedTypeFromTensorType(Type type) {
  auto tensor_type = type.dyn_cast<TensorType>();
  if (!tensor_type) return {};
  return tensor_type.getElementType().dyn_cast<QuantizedType>();
}

class QuantizeSameScaleOpsPattern : public RewritePattern {
 public:
  explicit QuantizeSameScaleOpsPattern(
      MLIRContext* context, OpQuantScaleSpecGetter op_quant_scale_spec_getter)
      // Set the score to a large number so it is always preferred, after
      // quantization patterns.
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/200, context),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter) {}

  LogicalResult match(Operation* op) const override {
    if (!op_quant_scale_spec_getter_(op)->has_same_scale_requirement) {
      return failure();
    }

    bool should_match = false;
    for (const auto& operand : op->getOperands()) {
      Type operand_type = operand.getType();
      if (operand_type.isa<NoneType>()) {
        continue;
      }

      Type elem_type = operand_type.cast<TensorType>().getElementType();
      // If the operand is a non-float tensor, then it doesn't require the DQ op
      // in the pattern.
      if (isa_and_nonnull<DequantizeCastOp>(operand.getDefiningOp())) {
        should_match = true;
      } else if (elem_type.isa<FloatType>()) {
        // If the operand is a float tensor, it must be from a DQ op.
        return failure();
      }
    }

    for (const Value& result : op->getResults()) {
      Type result_type = result.getType();
      if (result_type.isa<NoneType>()) {
        continue;
      }

      auto result_tensor_type = result_type.cast<TensorType>();
      // If the user is the Quantize op, it must be the only user.
      if (result.hasOneUse() &&
          llvm::isa<QuantizeCastOp>(*result.user_begin())) {
        should_match = true;
      } else if (result_tensor_type.getElementType().isa<FloatType>()) {
        // If the result is a float tensor, it must only be consumed by a single
        // Q op.
        return failure();
      }
    }
    return success(should_match);
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    // Collect all the quantized inputs and "clone" the matched op by these
    // inputs.
    SmallVector<Value, 4> inputs;
    inputs.reserve(op->getNumOperands());
    for (const auto& operand : op->getOperands()) {
      Value new_operand = operand;
      // When the operand is a float tensor from a DQ op, replace DQ with scast
      // op that converts quantized type to raw integer type.
      if (auto dq_op =
              dyn_cast_or_null<DequantizeCastOp>(operand.getDefiningOp())) {
        auto dq_arg_type = dq_op.arg().getType().cast<TensorType>();
        auto qtype = GetQuantizedTypeFromTensorType(dq_arg_type);
        new_operand = rewriter.create<StorageCastOp>(
            dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
            dq_op.arg());
      }
      inputs.push_back(new_operand);
    }

    // Collect all the quantized outputs and replace them by the results of
    // the new quantized op.
    llvm::SmallDenseMap<Value, int> outputs_replaced;
    SmallVector<Type, 4> output_types;
    output_types.reserve(op->getNumResults());
    for (const auto& enumerated_result : llvm::enumerate(op->getResults())) {
      Value new_result = enumerated_result.value();
      Type new_result_type = new_result.getType();
      auto result_tensor_type = new_result_type.dyn_cast<TensorType>();

      // If the result is a float tensor that is consumed by a Q op, replace Q
      // op with scast that cnoverts raw integer value to a quantized type.
      if (result_tensor_type && new_result.hasOneUse() &&
          llvm::isa<QuantizeCastOp>(*new_result.user_begin())) {
        auto user = llvm::cast<QuantizeCastOp>(*new_result.user_begin());
        QuantizedType qtype = GetQuantizedTypeFromTensorType(user.getType());
        new_result = user.getResult();
        new_result_type = result_tensor_type.clone(qtype.getStorageType());
      }
      outputs_replaced.insert({new_result, enumerated_result.index()});
      output_types.push_back(new_result_type);
    }

    rewriter.setInsertionPointAfter(op);
    OperationState new_state(op->getLoc(), op->getName().getStringRef(), inputs,
                             output_types, op->getAttrs());
    for (int i = 0; i < op->getNumRegions(); ++i) {
      new_state.addRegion();
    }
    Operation* quantized_op = rewriter.createOperation(new_state);
    if (op->getNumRegions() != 0) {
      for (const auto& indexed_regions : llvm::enumerate(op->getRegions())) {
        BlockAndValueMapping mapping;
        indexed_regions.value().cloneInto(
            &quantized_op->getRegion(indexed_regions.index()), mapping);
      }
    }
    for (auto output_index_pair : outputs_replaced) {
      Value output = output_index_pair.getFirst();
      int output_index = output_index_pair.getSecond();
      Value new_output = quantized_op->getResult(output_index);
      if (GetQuantizedTypeFromTensorType(output.getType())) {
        new_output = rewriter.create<StorageCastOp>(
            output.getLoc(), output.getType(),
            quantized_op->getResult(output_index));
      }
      output.replaceAllUsesWith(new_output);
    }
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

 private:
  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;
};

// Applies quantization on the model in TF dialect.
struct QuantizePass : public PassWrapper<QuantizePass, OperationPass<FuncOp>> {
 public:
  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() { quant_specs.inference_type = tensorflow::DT_QINT8; }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs(quant_specs) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply quantization on models in TensorFlow dialect";
  }

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs;
};

void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();

  const QuantPassSpec quant_params = {
      {quant_specs.verify_numeric, /*error_tolerance=*/5.0f,
       quant_specs.whole_model_verify, /*enable_log_if_failed=*/false},
      quant_specs};

  patterns.add<TFFullQuantization, TFFullQuantizationReverse>(ctx,
                                                              quant_params);
  patterns.add<QuantizeSameScaleOpsPattern>(ctx, GetTfQuantScaleSpec);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<RemoveUnusedQdqPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow dialect Quantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass() {
  QuantizationSpecs quant_specs;
  return std::make_unique<QuantizePass>(quant_specs);
}

static PassRegistration<QuantizePass> pass;

}  // namespace quant
}  // namespace mlir
