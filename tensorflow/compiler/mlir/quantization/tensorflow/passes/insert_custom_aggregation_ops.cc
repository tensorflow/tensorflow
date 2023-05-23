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

constexpr StringRef kCustomAggregatorOpName = "tf.CustomAggregator";
constexpr StringRef kQuantTraitAttrName = "_tfl_quant_trait";

class InsertCustomAggregationOpsPass
    : public PassWrapper<InsertCustomAggregationOpsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertCustomAggregationOpsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-insert-custom-aggregation-ops";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Insert custom aggregation ops for the calibration procedure";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

static PassRegistration<InsertCustomAggregationOpsPass> pass;

class AddCustomAggregationOp : public RewritePattern {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit AddCustomAggregationOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Return early if the given operator is the custom aggregator op.
    if (op->getName().getStringRef() == kCustomAggregatorOpName)
      return failure();

    // Return early if the given op is a non-quantizable op.
    auto call_op = dyn_cast_or_null<TF::PartitionedCallOp>(op);
    if (call_op && !op->hasAttr(kQuantTraitAttrName)) {
      return failure();
    }

    bool mutated = false;
    for (Value input : op->getOperands()) {
      Type element_type = getElementTypeOrSelf(input.getType());
      // Non-float cases won't be calibrated.
      if (!element_type.isF32()) {
        continue;
      }
      // Skip when there is any already existing StatisticsOp found.
      Operation *defining_op = input.getDefiningOp();
      if (defining_op != nullptr &&
          defining_op->getName().getStringRef() == kCustomAggregatorOpName) {
        continue;
      }

      // Skip calibration when the given operand comes from a constant.
      if (defining_op != nullptr &&
          defining_op->hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }

      SmallVector<NamedAttribute, 1> attributes{
          rewriter.getNamedAttr("id", rewriter.getStringAttr(""))};

      // Insert custom aggregation op between operand and operator.
      rewriter.setInsertionPointAfterValue(input);
      // ID attribute will have empty value for now.
      OperationState state(
          op->getLoc(), kCustomAggregatorOpName, /*operands=*/ValueRange{input},
          /*types=*/TypeRange{input.getType()}, /*attributes=*/attributes);
      Operation *aggregator_op = Operation::create(state);
      rewriter.insert(aggregator_op);
      Value aggregator_op_result = aggregator_op->getOpResult(0);
      input.replaceAllUsesWith(aggregator_op_result);
      aggregator_op->replaceUsesOfWith(aggregator_op_result, input);

      // Mark mutated.
      mutated = true;
    }

    // Return failure when there is no matching operand.
    return mutated ? success() : failure();
  }
};

void InsertCustomAggregationOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<AddCustomAggregationOp>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-insert-custom-aggregation-ops failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateInsertCustomAggregationOpsPass() {
  return std::make_unique<InsertCustomAggregationOpsPass>();
}

}  // namespace quant
}  // namespace mlir
