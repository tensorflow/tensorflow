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
