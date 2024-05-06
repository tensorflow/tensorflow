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

// This transformation pass applies quantization on TFLite dialect.

#include <cstddef>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//===----------------------------------------------------------------------===//
namespace {
#define GEN_PASS_DEF_QUANTIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcreteT,
          typename RootOpT = DequantizeOp>
struct TFLQuantizationBase
    : public quant::QuantizationPattern<ConcreteT, QuantizeOp, DequantizeOp,
                                        NumericVerifyOp, RootOpT> {
  explicit TFLQuantizationBase(MLIRContext* ctx,
                               const quant::QuantPassSpec& quant_params)
      : quant::QuantizationPattern<ConcreteT, QuantizeOp, DequantizeOp,
                                   NumericVerifyOp, RootOpT>(ctx,
                                                             quant_params) {}

  static bool IsQuantizableCustomOp(Operation* op,
                                    const quant::CustomOpMap& custom_op_map) {
    // In some cases, ops may need to be quantized even though their op trait is
    // not quantizable. For example, for the case of custom op various ops can
    // be categorized as cusom ops despite each of them may require different
    // behaviors. In that case, these ops can be marked in the custom map and
    // treated separately in this pass.

    auto custom_op = llvm::dyn_cast_or_null<CustomOp>(op);
    if (!custom_op) return false;

    // Custom op which is marked in the custom op map is quantizable.
    std::string op_name = custom_op.getCustomCode().str();
    return (custom_op_map.find(op_name) == custom_op_map.end()) ? false : true;
  }

  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const quant::CustomOpMap& custom_op_map) {
    // Collect the input if dynamic range quantization is on and the op supports
    // it.
    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool AllowDynamicRangeQuantizedResult(
      Operation* quantized_op, const quant::CustomOpMap& custom_op_map) {
    // Collect the output if dynamic range quantization is on and the op
    // supports it.
    return quantization_trait == kDynamicRangeQuantization &&
           (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op) ||
            IsQuantizableCustomOp(quantized_op, custom_op_map));
  }

  static bool IsWeightOnlyOp(
      Operation* quantized_op,
      const absl::flat_hash_set<std::string>& ops_blocklist,
      const bool weight_only_quantization,
      const quant::CustomOpMap& custom_op_map) {
    // Check whether the quantized_op needs to be quantized in weight-only
    // manner.
    bool is_blocklisted = false;

    if (auto custom_op = dyn_cast_or_null<CustomOp>(quantized_op)) {
      std::string custom_op_name = custom_op.getCustomCode().str();
      auto custom_map_iter = custom_op_map.find(custom_op_name);

      is_blocklisted =
          ops_blocklist.find(custom_op_name) != ops_blocklist.end();

      bool weight_only_custom_op = custom_map_iter != custom_op_map.end()
                                       ? custom_map_iter->second.is_weight_only
                                       : false;

      return is_blocklisted || weight_only_custom_op ||
             weight_only_quantization;
    } else {
      auto dynamic_range_op =
          dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op);

      const auto op_name = quantized_op->getName().getStringRef().str();
      is_blocklisted = ops_blocklist.find(op_name) != ops_blocklist.end();

      bool kernel_support =
          dynamic_range_op.GetDynamicRangeQuantKernelSupport();

      return is_blocklisted || !kernel_support || weight_only_quantization;
    }
  }
};

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFLFullQuantization
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantization> {
  explicit TFLFullQuantization(MLIRContext* ctx,
                               const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantization>(
            ctx, quant_params) {}
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFLFullQuantizationReverse
    : public TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                                 QuantizeOp> {
  explicit TFLFullQuantizationReverse(MLIRContext* ctx,
                                      const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kFullQuantization, TFLFullQuantizationReverse,
                            QuantizeOp>(ctx, quant_params) {}
};

// Dynamic range quantization rewrite pattern using DQ as the root op.
struct TFLDynamicRangeQuantization
    : public TFLQuantizationBase<kDynamicRangeQuantization,
                                 TFLDynamicRangeQuantization> {
  explicit TFLDynamicRangeQuantization(MLIRContext* ctx,
                                       const quant::QuantPassSpec& quant_params)
      : TFLQuantizationBase<kDynamicRangeQuantization,
                            TFLDynamicRangeQuantization>(ctx, quant_params) {}
};

class QuantizeConstPattern : public OpRewritePattern<QuantizeOp> {
 public:
  explicit QuantizeConstPattern(MLIRContext* context, bool legacy_float_scale)
      : OpRewritePattern<QuantizeOp>(context),
        legacy_float_scale_(legacy_float_scale) {}
  LogicalResult matchAndRewrite(QuantizeOp op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (matchPattern(op.getInput(), m_Constant(&attr))) {
      auto qtype = op.getQtypeAttr();
      Attribute quantized_attr;
      if (legacy_float_scale_) {
        quantized_attr = quant::QuantizeLegacy(attr, qtype.getValue());
      } else {
        quantized_attr = quant::Quantize(attr, qtype.getValue());
      }
      if (quantized_attr) {
        rewriter.replaceOpWithNewOp<QConstOp>(op, qtype, quantized_attr);
        return success();
      }
    }
    return failure();
  }

 private:
  bool legacy_float_scale_;
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public impl::QuantizePassBase<QuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)

  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() { quant_specs.inference_type = tensorflow::DT_QINT8; }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const quant::QuantizationSpecs& quant_specs)
      : quant_specs(quant_specs) {
    enable_numeric_verify_ = quant_specs.verify_numeric;
    enable_whole_model_verify_ = quant_specs.whole_model_verify;
    enable_legacy_quantize_ = quant_specs.legacy_float_scale;
    enable_dynamic_range_quantization_ = quant_specs.weight_quantization;
    enable_weight_only_quantization_ = quant_specs.weight_only_quantization;
  }

  void runOnOperation() override;

 private:
  quant::QuantizationSpecs quant_specs;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  quant_specs.verify_numeric = enable_numeric_verify_;
  quant_specs.whole_model_verify = enable_whole_model_verify_;
  quant_specs.legacy_float_scale = enable_legacy_quantize_;
  quant_specs.weight_quantization = enable_dynamic_range_quantization_;
  quant_specs.weight_only_quantization = enable_weight_only_quantization_;
  if (!ops_blocklist_flag_.empty()) {
    quant_specs.ops_blocklist = absl::flat_hash_set<std::string>(
        ops_blocklist_flag_.begin(), ops_blocklist_flag_.end());
  }

  if (!nodes_blocklist_flag_.empty()) {
    quant_specs.nodes_blocklist = absl::flat_hash_set<std::string>(
        nodes_blocklist_flag_.begin(), nodes_blocklist_flag_.end());
  }

  if (!enable_custom_op_weight_only_.empty()) {
    ParseCustomOpSpecs(enable_custom_op_weight_only_,
                       quant::CustomOpUpdateOptions::kWeightOnly,
                       quant_specs.custom_map);
  }
  if (enable_float16_quantization_) {
    quant_specs.inference_type = tensorflow::DT_HALF;
  }

  const quant::QuantPassSpec quant_params = {
      {quant_specs.verify_numeric, error_tolerance_,
       quant_specs.whole_model_verify, enable_log_if_failed_},
      quant_specs};

  populateWithGenerated(patterns);

  if (quant_specs.weight_quantization || quant_specs.use_fake_quant_num_bits ||
      quant_specs.qdq_conversion_mode ==
          quant::QDQConversionMode::kQDQDynamic) {
    patterns.add<TFLDynamicRangeQuantization>(ctx, quant_params);
  } else {
    patterns.add<TFLFullQuantization, TFLFullQuantizationReverse>(ctx,
                                                                  quant_params);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(&getContext());
  patterns_2.add<QuantizeConstPattern>(ctx, quant_specs.legacy_float_scale);
  if (quant_params.numeric_verify_spec.whole_model_verify) {
    patterns_2.add<quant::RemoveDebugAttrPattern>(ctx);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    const quant::QuantizationSpecs& quant_specs,
    const absl::flat_hash_set<std::string>& ops_blocklist,
    const absl::flat_hash_set<std::string>& nodes_blocklist) {
  quant::QuantizationSpecs updated_quant_specs;
  updated_quant_specs = quant_specs;
  // If there's new blocklists given, update quant_specs to use the new one.
  if (!ops_blocklist.empty()) {
    updated_quant_specs.ops_blocklist = ops_blocklist;
  }
  if (!nodes_blocklist.empty()) {
    updated_quant_specs.nodes_blocklist = nodes_blocklist;
  }
  return std::make_unique<QuantizePass>(updated_quant_specs);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantizePass() {
  return std::make_unique<QuantizePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    const bool verify_numeric, const bool whole_model_verify,
    const bool legacy_float_scale,
    const absl::flat_hash_set<std::string>& ops_blocklist,
    const absl::flat_hash_set<std::string>& nodes_blocklist) {
  quant::QuantizationSpecs quant_specs;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = ops_blocklist;
  quant_specs.nodes_blocklist = nodes_blocklist;
  return std::make_unique<QuantizePass>(quant_specs);
}

}  // namespace TFL
}  // namespace mlir
