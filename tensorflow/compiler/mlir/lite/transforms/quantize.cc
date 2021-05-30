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
static llvm::cl::opt<bool> enable_single_layer_verify(
    "tfl-single-layer-verify", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals layer by layer. Valid when "
                   "`-tfl-numeric-verify` is set."),
    llvm::cl::init(true));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_log_if_failed(
    "tfl-log-if-failed", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether verify numericals with thresholding "
                   "tolerance. Valid when `-tfl-numeric-verify` is set."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> enable_legacy_quantize(
    "tfl-legacy-quantize", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Use legacy quantize mode in test. Valid when"
                   "`-tfl-legacy-quantize` is set."),
    llvm::cl::init(false));

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Quantize Pass.
//
namespace {

// Full integer quantization rewrite pattern using DQ as the root op.
struct TFLFullQuantization
    : public quant::QuantizationPattern<TFLFullQuantization, QuantizeOp,
                                        DequantizeOp, NumericVerifyOp> {
  explicit TFLFullQuantization(MLIRContext* ctx, bool verify_numeric_flag,
                               float tolerance, bool verify_single_layer,
                               bool log_if_failed_flag = false)
      : BaseType(ctx, verify_numeric_flag, tolerance, verify_single_layer,
                 log_if_failed_flag) {}
  static bool AllowHybridOperand() { return false; }
  static bool AllowHybridResult() { return false; }
};

// Full integer quantization rewrite pattern using Q as the root op. This is for
// the quantizable ops without floating-point operands.
struct TFLFullQuantizationReverse
    : public quant::QuantizationPattern<TFLFullQuantizationReverse, QuantizeOp,
                                        DequantizeOp, NumericVerifyOp,
                                        QuantizeOp> {
  explicit TFLFullQuantizationReverse(MLIRContext* ctx,
                                      bool verify_numeric_flag, float tolerance,
                                      bool verify_single_layer,
                                      bool log_if_failed_flag = false)
      : BaseType(ctx, verify_numeric_flag, tolerance, verify_single_layer,
                 log_if_failed_flag) {}
  static bool AllowHybridOperand() { return false; }
  static bool AllowHybridResult() { return false; }
};

struct QuantizeConstPattern : public OpRewritePattern<QuantizeOp> {
  explicit QuantizeConstPattern(MLIRContext* context, bool legacy_float_scale)
      : OpRewritePattern<QuantizeOp>(context),
        legacy_float_scale(legacy_float_scale) {}
  LogicalResult matchAndRewrite(QuantizeOp op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (matchPattern(op.input(), m_Constant(&attr))) {
      auto qtype = op.qtypeAttr();
      Attribute quantized_attr;
      if (legacy_float_scale) {
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
  bool legacy_float_scale;
};

// Applies quantization on the model in TFL dialect.
struct QuantizePass : public PassWrapper<QuantizePass, FunctionPass> {
 public:
  // Constructor used by manually creating the pass.
  explicit QuantizePass(bool verify_numeric_flag = false,
                        bool legacy_float_scale = false)
      : verify_numeric(verify_numeric_flag),
        legacy_float_scale(legacy_float_scale) {}

  void runOnFunction() override;

 private:
  bool verify_numeric;
  bool legacy_float_scale;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_quantize.inc"

void QuantizePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  auto* ctx = func.getContext();

  TFL::populateWithGenerated(patterns);
  patterns.insert<TFLFullQuantization, TFLFullQuantizationReverse>(
      ctx, enable_numeric_verify || verify_numeric, error_tolerance,
      enable_single_layer_verify, enable_log_if_failed);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  OwningRewritePatternList patterns_2(&getContext());
  patterns_2.insert<QuantizeConstPattern>(
      ctx, legacy_float_scale || enable_legacy_quantize);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns_2));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect QuantizeTFL pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    bool verify_numeric, bool legacy_float_scale) {
  return std::make_unique<QuantizePass>(verify_numeric, legacy_float_scale);
}

static PassRegistration<QuantizePass> pass(
    "tfl-quantize", "Apply quantization on models in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
