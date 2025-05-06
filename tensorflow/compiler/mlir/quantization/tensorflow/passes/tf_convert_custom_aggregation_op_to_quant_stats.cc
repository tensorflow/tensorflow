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
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace tf_quant {
namespace {

class TFConvertCustomAggregationOpToQuantStatsPass
    : public PassWrapper<TFConvertCustomAggregationOpToQuantStatsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TFConvertCustomAggregationOpToQuantStatsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "tf-quant-convert-tf-custom-aggregator-op-to-quant-stats";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert tf.CustomAggregator op to quant.Stats";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<quant::QuantDialect>();
    registry.insert<mlir::quant::ir::TFQuantDialect>();
  }

  void runOnOperation() override;
};

class ConvertCustomAggregationOpToQuantStats
    : public OpRewritePattern<TF::CustomAggregatorOp> {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit ConvertCustomAggregationOpToQuantStats(MLIRContext *context)
      : OpRewritePattern<TF::CustomAggregatorOp>(context) {}

  LogicalResult matchAndRewrite(TF::CustomAggregatorOp op,
                                PatternRewriter &rewriter) const override {
    FloatAttr min = mlir::dyn_cast_or_null<FloatAttr>(op->getAttr("min"));
    FloatAttr max = mlir::dyn_cast_or_null<FloatAttr>(op->getAttr("max"));

    // When there are no min and max attributes, remove op.
    if (min == nullptr || max == nullptr) {
      op.getOutput().replaceAllUsesWith(op.getInput());
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

    mlir::quant::ir::StatisticsOp stats_op =
        rewriter.create<mlir::quant::ir::StatisticsOp>(
            op->getLoc(), op.getInput(), layer_stats, axis_stats, axis);
    op.getOutput().replaceAllUsesWith(stats_op.getResult());
    return success();
  }
};

static PassRegistration<TFConvertCustomAggregationOpToQuantStatsPass> pass;

void TFConvertCustomAggregationOpToQuantStatsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<ConvertCustomAggregationOpToQuantStats>(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    func.emitError()
        << "tf-quant-convert-tf-custom-aggregator-op-to-quant-stats failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFConvertCustomAggregationOpToQuantStatsPass() {
  return std::make_unique<TFConvertCustomAggregationOpToQuantStatsPass>();
}

}  // namespace tf_quant
}  // namespace mlir
