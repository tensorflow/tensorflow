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

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//===----------------------------------------------------------------------===//
namespace {

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcreteT,
          typename RootOpT = quantfork::DequantizeCastOp>
struct TFQuantizationBase
    : public QuantizationPattern<ConcreteT, quantfork::QuantizeCastOp,
                                 quantfork::DequantizeCastOp,
                                 /*VerifierT=*/void, RootOpT> {
  explicit TFQuantizationBase(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : QuantizationPattern<ConcreteT, quantfork::QuantizeCastOp,
                            quantfork::DequantizeCastOp,
                            /*VerifierT=*/void, RootOpT>(ctx, quant_params) {}

  // Custom op quantization is not supported.
  static bool IsQuantizableCustomOp(Operation* op,
                                    const CustomMap& custom_op_map) {
    return false;
  }

  // All the quantized ops are supported if the quantization method is dynamic
  // range quantization.
  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const CustomMap& custom_op_map) {
    return quantization_trait == kDynamicRangeQuantization;
  }

  // All the quantized ops are supported if the quantization method is dynamic
  // range quantization.
  static bool AllowDynamicRangeQuantizedResult(Operation* quantized_op,
                                               const CustomMap& custom_op_map) {
    return quantization_trait == kDynamicRangeQuantization;
  }

  // If weight_only_quantization is true, the legacy weight-only quantization is
  // applied. The legacy weight-only graph has dequantization logic at the
  // front.
  static bool IsWeightOnlyOp(Operation* quantized_op,
                             absl::flat_hash_set<std::string>& ops_blocklist,
                             bool weight_only_quantization,
                             const CustomMap& custom_op_map) {
    return weight_only_quantization;
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
                                quantfork::QuantizeCastOp> {
  explicit TFFullQuantizationReverse(MLIRContext* ctx,
                                     const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                           quantfork::QuantizeCastOp>(ctx, quant_params) {}
};

// Dynamic range quantization rewrite pattern using DQ as the root op.
struct TFDynamicRangeQuantization
    : public TFQuantizationBase<kDynamicRangeQuantization,
                                TFDynamicRangeQuantization> {
  explicit TFDynamicRangeQuantization(MLIRContext* ctx,
                                      const quant::QuantPassSpec& quant_params)
      : TFQuantizationBase<kDynamicRangeQuantization,
                           TFDynamicRangeQuantization>(ctx, quant_params) {}
};

// Removes quantize-dequantize pairs that are not used in the quantization.
// The benefit of this pattern is set to lower value than other patterns, so
// that the other patterns can work on quantize/dequantize ops first.
class RemoveUnusedQdqPattern
    : public OpRewritePattern<quantfork::DequantizeCastOp> {
 public:
  explicit RemoveUnusedQdqPattern(MLIRContext* context)
      : OpRewritePattern<quantfork::DequantizeCastOp>(context) {}
  LogicalResult matchAndRewrite(quantfork::DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    auto q_op = dq_op.getArg().getDefiningOp<quantfork::QuantizeCastOp>();
    if (!q_op) return failure();

    dq_op.replaceAllUsesWith(q_op.getArg());
    return success();
  }
};

class QuantizeSameScaleOpsPattern
    : public OpRewritePattern<quantfork::DequantizeCastOp> {
 public:
  explicit QuantizeSameScaleOpsPattern(
      MLIRContext* context, OpQuantScaleSpecGetter op_quant_scale_spec_getter,
      OpSet target_opset)
      // Set the score to a large number so it is always preferred, after
      // quantization patterns.
      : OpRewritePattern<quantfork::DequantizeCastOp>(context, /*benefit=*/200),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter),
        target_opset_(target_opset) {}

  LogicalResult matchAndRewrite(quantfork::DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*, 4> quantizing_ops;
    auto users = op.getResult().getUsers();
    quantizing_ops.append(users.begin(), users.end());

    bool changed = false;
    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* quantizing_op : quantizing_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(
              quantizing_op)) {
        return failure();
      }

      // If the op is terminator, not quantizable or any ops from the mlir quant
      // ops dialect, we shouldn't rewrite.
      if (quantizing_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }

      if (!op_quant_scale_spec_getter_(quantizing_op)
               ->has_same_scale_requirement) {
        continue;
      }

      if (target_opset_ == OpSet::XLA &&
          !IsConnectedWithCompsiteFunction(quantizing_op)) {
        continue;
      }

      // Same scale op is not supported for Uniform Quantized ops.
      if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
        continue;
      }

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantizing_op->getNumOperands());
      for (const auto& operand : quantizing_op->getOperands()) {
        Type operand_type = operand.getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }

        Type elem_type = operand_type.cast<TensorType>().getElementType();
        if (auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
                operand.getDefiningOp())) {
          auto dq_arg_type = dq_op.getArg().getType().cast<TensorType>();
          auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
          auto scast_op = rewriter.create<quantfork::StorageCastOp>(
              dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
              dq_op.getArg());
          inputs.push_back(scast_op.getResult());
        } else if (!elem_type.isF32()) {
          // If the operand is an integer tensor, then it doesn't require the
          // DQ op in the pattern.
          inputs.push_back(operand);
        } else {
          return failure();
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantizing_op->getNumResults());
      for (const auto& enumerated_result :
           llvm::enumerate(quantizing_op->getResults())) {
        Value result = enumerated_result.value();
        Type result_type = result.getType();
        if (result_type.isa<NoneType>()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        auto result_tensor_type = result_type.cast<TensorType>();
        // If the user is the Quantize op, it must be the only user.
        if (result.hasOneUse() &&
            llvm::isa<quantfork::QuantizeCastOp>(*result.user_begin())) {
          auto user =
              llvm::cast<quantfork::QuantizeCastOp>(*result.user_begin());
          outputs_replaced.insert(
              {user.getResult(), enumerated_result.index()});
          auto qtype = user.getType()
                           .cast<TensorType>()
                           .getElementType()
                           .cast<QuantizedType>();
          output_types.push_back(
              result_tensor_type.clone(qtype.getStorageType()));
        } else if (!result_tensor_type.getElementType().isF32()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          // TODO(b/224691264): separate matching and rewriting clearly.
          return failure();
        }
      }

      rewriter.setInsertionPointAfter(quantizing_op);
      OperationState new_state(quantizing_op->getLoc(),
                               quantizing_op->getName().getStringRef(), inputs,
                               output_types, quantizing_op->getAttrs());
      for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* quantized_op = rewriter.create(new_state);
      if (quantizing_op->getNumRegions() != 0) {
        for (const auto& indexed_regions :
             llvm::enumerate(quantizing_op->getRegions())) {
          IRMapping mapping;
          indexed_regions.value().cloneInto(
              &quantized_op->getRegion(indexed_regions.index()), mapping);
        }
      }
      for (const auto& output_index_pair : outputs_replaced) {
        Value output = output_index_pair.getFirst();
        int output_index = output_index_pair.getSecond();
        auto scast_op = rewriter.create<quantfork::StorageCastOp>(
            output.getLoc(), output.getType(),
            quantized_op->getResult(output_index));
        output.replaceAllUsesWith(scast_op);
      }
      changed = true;
    }
    return success(changed);
  }

 private:
  // Checks whether the operation is connnected with a composite function.
  // If not, the same-scale op will not be quantized. This decision is based
  // on the current assumption that the performance gain of the same-scale
  // op itself could not beat the overhead of the quantize and dequantize
  // routines need to be added around that op. When the assumption changes,
  // this policy might change as well.
  bool IsConnectedWithCompsiteFunction(Operation* same_scale_op) const {
    for (const auto& operand : same_scale_op->getOperands()) {
      auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
          operand.getDefiningOp());
      if (!dq_op) continue;

      Operation* preceding_op = dq_op.getArg().getDefiningOp();
      if (!preceding_op) continue;

      // Check whether the preceding op is a quantized composite function.
      if (llvm::isa<TF::PartitionedCallOp>(preceding_op)) {
        auto call_op = llvm::cast<TF::PartitionedCallOp>(preceding_op);
        if (!IsCompositeFunction(call_op)) continue;
        return true;
      }

      // Check if the preceding op is a quantized same-scale op.
      if (llvm::isa<quantfork::StorageCastOp>(preceding_op)) {
        auto sc_op = llvm::cast<quantfork::StorageCastOp>(preceding_op);
        auto sc_arg_type = sc_op.getArg().getType().dyn_cast<TensorType>();
        if (sc_arg_type.getElementType().isInteger(8)) {
          return true;
        }
      }
    }

    for (const auto& result : same_scale_op->getResults()) {
      // If the user is the Quantize op, it must be the only user.
      if (!result.hasOneUse() ||
          !llvm::isa<quantfork::QuantizeCastOp>(*result.user_begin())) {
        continue;
      }

      auto q_op = llvm::cast<quantfork::QuantizeCastOp>(*result.user_begin());
      for (auto following_op : q_op->getUsers()) {
        // Check whether the preceding op is a quantized composite function.
        if (llvm::isa<TF::PartitionedCallOp>(following_op)) {
          auto call_op = llvm::cast<TF::PartitionedCallOp>(following_op);
          if (!IsCompositeFunction(call_op)) continue;
          return true;
        }

        // Check if the preceding op is a quantized same-scale op.
        if (llvm::isa<quantfork::StorageCastOp>(following_op)) {
          auto sc_op = llvm::cast<quantfork::StorageCastOp>(following_op);
          auto sc_arg_type = sc_op.getResult().getType().dyn_cast<TensorType>();
          if (sc_arg_type.getElementType().isInteger(8)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // Checks if op calls a composite function and all the inputs are quantized.
  bool IsCompositeFunction(TF::PartitionedCallOp call_op) const {
    if (!call_op->hasAttr(kQuantTraitAttrName)) {
      return false;
    }

    const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    if (!f_attr || !f_attr.getValue().startswith("composite_")) {
      return false;
    }

    bool has_quantized_types = false;
    for (Value input : call_op.getArgs()) {
      if (auto type = input.getType().dyn_cast<TensorType>()) {
        if (type.getElementType().isa<FloatType>()) {
          return false;
        }
        if (type.getElementType().isa<QuantizedType>()) {
          has_quantized_types = true;
        }
      }
    }
    for (Value output : call_op.getOutput()) {
      if (auto type = output.getType().dyn_cast<TensorType>()) {
        if (type.getElementType().isa<FloatType>()) {
          return false;
        }
        if (type.getElementType().isa<QuantizedType>()) {
          has_quantized_types = true;
        }
      }
    }
    return has_quantized_types;
  }

  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;
  OpSet target_opset_;
};

// The AvgPool op is a same-scale op but it doesn't have int8 kernel, so
// we cast its input to float and its output to int8 as a workaround.
// TODO(b/229183248): Remove this workaround after int8 kernels have been
// added to TF and XLA.
struct QuantizeAvgPoolOpPattern
    : public OpRewritePattern<quantfork::StorageCastOp> {
  explicit QuantizeAvgPoolOpPattern(MLIRContext* context)
      : OpRewritePattern<quantfork::StorageCastOp>(context, /*benefit=*/100) {}

  LogicalResult matchAndRewrite(quantfork::StorageCastOp sc_op,
                                PatternRewriter& rewriter) const override {
    auto avg_pool_op = sc_op.getArg().getDefiningOp<TF::AvgPoolOp>();
    if (!avg_pool_op) return failure();
    auto preceding_sc_op = dyn_cast_or_null<quantfork::StorageCastOp>(
        avg_pool_op.getValue().getDefiningOp());
    if (!preceding_sc_op) return failure();

    // Check if the same-scale requirement is met.
    auto dq_arg_type = preceding_sc_op.getArg().getType().cast<TensorType>();
    auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
    auto q_result_type = sc_op.getType().cast<TensorType>();
    auto out_qtype = q_result_type.getElementType().cast<QuantizedType>();
    if (qtype != out_qtype) {
      avg_pool_op.emitError(
          "The preceding StorageCastOp and the following "
          "StorageCastOp must have the same quantized type");
      return failure();
    }

    // Cast to float type before the AvgPool op.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(preceding_sc_op);
    auto fcast_op = rewriter.create<TF::CastOp>(
        preceding_sc_op->getLoc(), dq_arg_type.clone(rewriter.getF32Type()),
        preceding_sc_op.getResult());

    // Create a new AvgPool op with float type.
    TF::AvgPoolOp float_avg_pool_op = rewriter.create<TF::AvgPoolOp>(
        avg_pool_op->getLoc(),
        avg_pool_op.getType().clone(rewriter.getF32Type()),
        /*operands=*/fcast_op.getResult(),
        /*attributes=*/avg_pool_op->getAttrs());

    // Cast back to the storage type after AvgPool op.
    auto round_val = rewriter.create<TF::RoundOp>(
        sc_op.getLoc(), float_avg_pool_op.getOutput());
    auto icast_op = rewriter.create<TF::CastOp>(
        sc_op.getLoc(), q_result_type.clone(qtype.getStorageType()), round_val);
    avg_pool_op.getResult().replaceAllUsesWith(icast_op.getResult());
    return success();
  }
};

// Applies quantization on the model in TF dialect.
class QuantizePass
    : public PassWrapper<QuantizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)

  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const QuantizationSpecs& quant_specs,
                        OpSet target_opset)
      : quant_specs_(quant_specs) {
    weight_quantization_ = quant_specs.weight_quantization;
    target_opset_ = target_opset;
  }

  QuantizePass(const QuantizePass& other) : quant_specs_(other.quant_specs_) {
    weight_quantization_ = other.weight_quantization_;
    target_opset_ = other.target_opset_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply quantization on models in TensorFlow dialect";
  }

  // Determine if the unused Q-DQ pairs need to be removed. For weight-only
  // quantizable ops, Q-DQ ops need to be preserved.
  bool shouldKeepUnusedQdqPattern();

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs_;

  Option<bool> weight_quantization_{
      *this, "weight-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether to enable weight quantization.")};
  Option<OpSet> target_opset_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};
};

bool QuantizePass::shouldKeepUnusedQdqPattern() {
  return target_opset_ == OpSet::XLA &&
         (quant_specs_.weight_only_quantization ||
          quant_specs_.weight_quantization);
}

void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();

  quant_specs_.weight_quantization = weight_quantization_;
  const QuantPassSpec quant_params = {
      {quant_specs_.verify_numeric, /*error_tolerance=*/5.0f,
       quant_specs_.whole_model_verify, /*enable_log_if_failed=*/false},
      quant_specs_};

  if (quant_specs_.weight_quantization) {
    patterns.add<TFDynamicRangeQuantization>(ctx, quant_params);
  } else {
    patterns.add<TFFullQuantization, TFFullQuantizationReverse>(ctx,
                                                                quant_params);
    patterns.add<QuantizeSameScaleOpsPattern>(ctx, GetTfQuantScaleSpec,
                                              target_opset_);
    patterns.add<QuantizeAvgPoolOpPattern>(ctx);
  }
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitWarning("Failed to converge pattern at QuantizePass.");
  }

  if (!shouldKeepUnusedQdqPattern()) {
    RewritePatternSet patterns_2(&getContext());
    patterns_2.add<RemoveUnusedQdqPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns_2)))) {
      signalPassFailure();
    }
  }
}
}  // namespace

// Creates an instance of the TensorFlow dialect Quantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass() {
  QuantizationSpecs quant_specs;
  return std::make_unique<QuantizePass>(quant_specs, OpSet::TF);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    QuantizationSpecs quant_specs, OpSet target_opset) {
  return std::make_unique<QuantizePass>(quant_specs, target_opset);
}

static PassRegistration<QuantizePass> pass;

}  // namespace quant
}  // namespace mlir
