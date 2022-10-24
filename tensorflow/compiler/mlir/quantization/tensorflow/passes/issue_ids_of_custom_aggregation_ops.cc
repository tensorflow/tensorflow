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
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace quant {
namespace {

class IssueIDsOfCustomAggregationOpsPass
    : public PassWrapper<IssueIDsOfCustomAggregationOpsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      IssueIDsOfCustomAggregationOpsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-issues-ids-of-custom-aggregation-ops";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Issue IDs of custom aggregation ops for the calibration procedure";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

 private:
  void runOnOperation() override;

  void issueIdToCustomAggregator(Operation* op);

  // Count of aggregator ops encountered;
  int aggregator_count_;
};

static PassRegistration<IssueIDsOfCustomAggregationOpsPass> pass;

void IssueIDsOfCustomAggregationOpsPass::issueIdToCustomAggregator(
    Operation* op) {
  // Return early when only aggregator operators are given.
  if (op->getName().getStringRef() != "tf.CustomAggregator") return;

  // Issue id based on the number of aggregators found.
  OpBuilder builder(op);
  op->setAttr("id", builder.getStringAttr(std::to_string(aggregator_count_)));
  ++aggregator_count_;
}

void IssueIDsOfCustomAggregationOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([&](Operation* op) { issueIdToCustomAggregator(op); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateIssueIDsOfCustomAggregationOpsPass() {
  return std::make_unique<IssueIDsOfCustomAggregationOpsPass>();
}

}  // namespace quant
}  // namespace mlir
