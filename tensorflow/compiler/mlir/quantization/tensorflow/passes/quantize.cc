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

class QuantizeSameScaleOpsPattern : public OpRewritePattern<DequantizeCastOp> {
 public:
  explicit QuantizeSameScaleOpsPattern(
      MLIRContext* context, OpQuantScaleSpecGetter op_quant_scale_spec_getter)
      // Set the score to a large number so it is always preferred, after
      // quantization patterns.
      : OpRewritePattern<DequantizeCastOp>(context, /*benefit=*/200),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter) {}

  LogicalResult matchAndRewrite(DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Operation*, 4> quantizing_ops;
    auto users = op.getResult().getUsers();
    quantizing_ops.append(users.begin(), users.end());

    bool changed = false;
    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* quantizing_op : quantizing_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (llvm::isa<QuantizeCastOp, DequantizeCastOp>(quantizing_op)) {
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
        if (auto dq_op =
                dyn_cast_or_null<DequantizeCastOp>(operand.getDefiningOp())) {
          auto dq_arg_type = dq_op.arg().getType().cast<TensorType>();
          auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
          auto scast_op = rewriter.create<StorageCastOp>(
              dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
              dq_op.arg());
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
            llvm::isa<QuantizeCastOp>(*result.user_begin())) {
          auto user = llvm::cast<QuantizeCastOp>(*result.user_begin());
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
          BlockAndValueMapping mapping;
          indexed_regions.value().cloneInto(
              &quantized_op->getRegion(indexed_regions.index()), mapping);
        }
      }
      for (const auto& output_index_pair : outputs_replaced) {
        Value output = output_index_pair.getFirst();
        int output_index = output_index_pair.getSecond();
        auto scast_op = rewriter.create<StorageCastOp>(
            output.getLoc(), output.getType(),
            quantized_op->getResult(output_index));
        output.replaceAllUsesWith(scast_op);
      }
      changed = true;
    }
    return success(changed);
  }

 private:
  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;
};

// The AvgPool op is a same-scale op but it doesn't have int8 kernel, so
// we cast its input to float and its output to int8 as a workaround.
// TODO(b/229183248): Remove this workaround after int8 kernels have been
// added to TF and XLA.
struct QuantizeAvgPoolOpPattern : public OpRewritePattern<QuantizeCastOp> {
  explicit QuantizeAvgPoolOpPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context, /*benefit=*/300) {}

  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    auto avg_pool_op = q_op.arg().getDefiningOp<TF::AvgPoolOp>();
    if (!avg_pool_op) return failure();
    auto dq_op =
        dyn_cast_or_null<DequantizeCastOp>(avg_pool_op.value().getDefiningOp());
    if (!dq_op) return failure();

    // Check if the same-scale requirement is met.
    auto dq_arg_type = dq_op.arg().getType().cast<TensorType>();
    auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
    auto q_result_type = q_op.getType().cast<TensorType>();
    auto out_qtype = q_result_type.getElementType().cast<QuantizedType>();
    if (qtype != out_qtype) {
      avg_pool_op.emitError(
          "The preceding DequantizeCastOp and the following "
          "QuantizeCastOp must have the same quantized type");
      return failure();
    }

    // Cast to float type before the AvgPool op.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(dq_op);
    auto scast_op = rewriter.create<StorageCastOp>(
        dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
        dq_op.arg());
    auto fcast_op = rewriter.create<TF::CastOp>(
        dq_op->getLoc(), dq_arg_type.clone(rewriter.getF32Type()),
        scast_op.getResult());
    dq_op.getResult().replaceUsesWithIf(fcast_op.y(), [&](OpOperand& operand) {
      return operand.getOwner() == avg_pool_op;
    });

    // Cast back to the storage type after AvgPool op.
    rewriter.setInsertionPointAfter(avg_pool_op);
    auto round_op =
        rewriter.create<TF::RoundOp>(q_op.getLoc(), avg_pool_op.output());
    auto icast_op = rewriter.create<TF::CastOp>(
        q_op.getLoc(), q_result_type.clone(qtype.getStorageType()),
        round_op.y());
    auto iscast_op = rewriter.create<StorageCastOp>(
        q_op.getLoc(), q_op.getType(), icast_op.y());
    q_op.getResult().replaceAllUsesWith(iscast_op.getResult());
    return success();
  }
};

// Applies quantization on the model in TF dialect.
struct QuantizePass
    : public PassWrapper<QuantizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)

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
  patterns.add<QuantizeAvgPoolOpPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<RemoveUnusedQdqPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow dialect Quantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass() {
  QuantizationSpecs quant_specs;
  return std::make_unique<QuantizePass>(quant_specs);
}

static PassRegistration<QuantizePass> pass;

}  // namespace quant
}  // namespace mlir
