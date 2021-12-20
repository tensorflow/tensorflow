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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_numeric_verify(
    "tfl-numeric-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals at runtime."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<float> error_tolerance(
    "tfl-error-tolerance", llvm::cl::value_desc("float"),
    llvm::cl::desc("Error tolerance for numeric verify. Valid when "
                   "`-tfl-numeric-verify` is set."),
    llvm::cl::init(5.0));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_whole_model_verify(
    "tfl-whole-model-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals layer by layer or whole model. "
                   "Valid when `-tfl-numeric-verify` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_log_if_failed(
    "tfl-log-if-failed", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals with thresholding "
                   "tolerance. Valid when `-tfl-numeric-verify` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_dynamic_range_quantization(
    "tfl-enable-dynamic-range-quantization", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether run post-training dynamic range quantization pass"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_weight_only_quantization(
    "tfl-enable-weight-only-quantization", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether to run weight-only for post-training dynamic range "
                   "quantization pass"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_legacy_quantize(
    "tfl-legacy-quantize", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Use legacy quantize mode in test. Valid when"
                   "`-tfl-legacy-quantize` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::list<std::string> ops_blocklist_flag(
    "tfl-ops-blocklist",
    llvm::cl::desc("Names of ops to blocklist from quantization"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

// NOLINTNEXTLINE
static llvm::cl::list<std::string> nodes_blocklist_flag(
    "tfl-locs-blocklist",
    llvm::cl::desc("Names of location to blocklist from quantization"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

// Base struct for quantization.
template <QuantizationTrait quantization_trait, typename ConcretTy,
          typename RootOp = DequantizeOp>
struct TFLQuantizationBase
    : public quant::QuantizationPattern<ConcretTy, QuantizeOp, DequantizeOp,
                                        NumericVerifyOp, RootOp> {
  explicit TFLQuantizationBase(MLIRContext* ctx,
                               const quant::QuantPassSpec& quant_params)
      : quant::QuantizationPattern<ConcretTy, QuantizeOp, DequantizeOp,
                                   NumericVerifyOp, RootOp>(ctx, quant_params) {
  }

  static bool AllowDynamicRangeQuantizedOperand(Operation* quantized_op) {
    // Collect the input if dynamic range quantization is on and the op supports
    // it.

    return quantization_trait == kDynamicRangeQuantization &&
           dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op);
  }

  static bool AllowDynamicRangeQuantizedResult(Operation* quantized_op) {
    // Collect the output if dynamic range quantization is on and the op
    // supports it.

    return quantization_trait == kDynamicRangeQuantization &&
           dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(quantized_op);
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
    if (matchPattern(op.input(), m_Constant(&attr))) {
      auto qtype = op.qtypeAttr();
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

#define LIST_FLAG_OR_STRING_SET(list, set) \
  (!list.empty() ? StringSet(list.begin(), list.end()) : set)

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public PassWrapper<QuantizePass, FunctionPass> {
 public:
  // Constructor used by the PassRegistration and only used by test.
  explicit QuantizePass() {
    quant_specs.legacy_float_scale = enable_legacy_quantize;
    quant_specs.weight_only_quantization = enable_weight_only_quantization;
    ops_blocklist =
        StringSet(ops_blocklist_flag.begin(), ops_blocklist_flag.end());
    nodes_blocklist =
        StringSet(nodes_blocklist_flag.begin(), nodes_blocklist_flag.end());
  }

  // Constructor used by manually creating the pass.
  explicit QuantizePass(const QuantizationSpecs& quant_specs,
                        const StringSet& ops_blocklist_set = {},
                        const StringSet& nodes_blocklist_set = {})
      : quant_specs(quant_specs),
        ops_blocklist(
            LIST_FLAG_OR_STRING_SET(ops_blocklist_flag, ops_blocklist_set)),
        nodes_blocklist(LIST_FLAG_OR_STRING_SET(nodes_blocklist_flag,
                                                nodes_blocklist_set)) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply quantization on models in TensorFlow Lite dialect";
  }

  void runOnFunction() override;

 private:
  QuantizationSpecs quant_specs;
  StringSet ops_blocklist;
  StringSet nodes_blocklist;
};

#undef LIST_FLAG_OR_STRING_SET

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  auto* ctx = func.getContext();

  const quant::QuantPassSpec quant_params = {
      {enable_numeric_verify || quant_specs.verify_numeric, error_tolerance,
       enable_whole_model_verify || quant_specs.whole_model_verify,
       enable_log_if_failed},
      enable_weight_only_quantization || quant_specs.weight_only_quantization,
      ops_blocklist,
      nodes_blocklist};

  TFL::populateWithGenerated(patterns);

  // TODO(b/202451048): separate full and weight-only post-training dynamic
  // range quantization
  if (quant_specs.weight_quantization || enable_dynamic_range_quantization ||
      quant_specs.use_fake_quant_num_bits) {
    patterns.insert<TFLDynamicRangeQuantization>(ctx, quant_params);
  } else {
    patterns.insert<TFLFullQuantization, TFLFullQuantizationReverse>(
        ctx, quant_params);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  OwningRewritePatternList patterns_2(&getContext());
  patterns_2.insert<QuantizeConstPattern>(
      ctx, quant_specs.legacy_float_scale || enable_legacy_quantize);
  if (quant_params.numeric_verify.whole_model_verify) {
    patterns_2.insert<quant::RemoveDebugAttrPattern>(ctx);
  }
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    const QuantizationSpecs& quant_specs, const StringSet& ops_blocklist,
    const StringSet& nodes_blocklist) {
  return std::make_unique<QuantizePass>(quant_specs, ops_blocklist,
                                        nodes_blocklist);
}

std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    bool verify_numeric, bool whole_model_verify, bool legacy_float_scale,
    const StringSet& ops_blocklist, const StringSet& nodes_blocklist) {
  QuantizationSpecs quant_specs;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  return std::make_unique<QuantizePass>(quant_specs, ops_blocklist,
                                        nodes_blocklist);
}
static PassRegistration<QuantizePass> pass;

}  // namespace TFL
}  // namespace mlir
