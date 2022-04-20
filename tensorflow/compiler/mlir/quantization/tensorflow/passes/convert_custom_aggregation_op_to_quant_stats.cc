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
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class ConvertCustomAggregationOpToQuantStatsPass
    : public PassWrapper<ConvertCustomAggregationOpToQuantStatsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertCustomAggregationOpToQuantStatsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-convert-tf-custom-aggregator-op-to-quant-stats";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert tf.CustomAggregator op to quant.Stats";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<QuantizationDialect>();
  }

  void runOnOperation() override;
};

class ConvertCustomAggregationOpToQuantStats : public RewritePattern {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit ConvertCustomAggregationOpToQuantStats(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Return early if the given operator isn't the custom aggregator op.
    if (op->getName().getStringRef() != "tf.CustomAggregator") return failure();

    FloatAttr min = op->getAttr("min").dyn_cast_or_null<FloatAttr>();
    FloatAttr max = op->getAttr("max").dyn_cast_or_null<FloatAttr>();

    // When there are no min and max attributes, remove op.
    if (min == nullptr || max == nullptr) {
      op->replaceAllUsesWith(op->getOperands());
      rewriter.eraseOp(op);
      return success();
    }

    // The layer stats contain only the first min/max pairs.
    ElementsAttr layer_stats = DenseFPElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getF32Type()),
        {static_cast<float>(min.getValueAsDouble()),
         static_cast<float>(max.getValueAsDouble())});
    ElementsAttr axis_stats;
    IntegerAttr axis;

    rewriter.replaceOpWithNewOp<StatisticsOp>(op, op->getOperand(0),
                                              layer_stats, axis_stats, axis);
    return success();
  }
};

static PassRegistration<ConvertCustomAggregationOpToQuantStatsPass> pass;

void ConvertCustomAggregationOpToQuantStatsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<ConvertCustomAggregationOpToQuantStats>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError()
        << "quant-convert-tf-custom-aggregator-op-to-quant-stats failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass() {
  return std::make_unique<ConvertCustomAggregationOpToQuantStatsPass>();
}

}  // namespace quant
}  // namespace mlir
