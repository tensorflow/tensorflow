/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
// Attribute name to be added on the module to identify whether
// variables should be legalized to TFLite or not.
const char kLegalizeTflVariables[] = "tfl._legalize_tfl_variables";

// Returns true if 'op' is TF op that accepts resource type, but is
// supported by TFLite.
bool IsSupportedTFLiteResourceOp(Operation* op) {
  return llvm::isa<TF::ReadVariableOp, TF::AssignVariableOp, TF::VarHandleOp,
                   TF::LookupTableFindV2Op, TF::LookupTableImportV2Op,
                   TF::LookupTableSizeV2Op>(op);
}

// Returns true if 'op' is TF/TFLite control flow op that can accept resource
// type. Usually these ops are just pass through, they call another subgraph and
// pass the operands to.
bool IsSupportedTFLiteControlFlow(Operation* op) {
  return llvm::isa<TFL::WhileOp, TFL::IfOp, TFL::CallOnceOp>(op);
}

// Returns true if the 'op' is one of the supported TF control flow ops or
// dataset ops. Those ops just forward the operands to other subgraphs.
bool IsSupportedTFDataForwardingOp(Operation* op) {
  return llvm::isa<TF::MapDatasetOp, TF::ReduceDatasetOp,
                   TF::TakeWhileDatasetOp, TF::IfOp, TF::WhileOp>(op);
}

}  // namespace

// Pass which analyzes the variables in the graph and add an attribute whether
// variables should be legalized to TFLite native ones.
// This pass needs to run post TF->TFL legalization and before variable
// legalization.
class AnalyzeVariablesPass
    : public PassWrapper<AnalyzeVariablesPass, OperationPass<ModuleOp>> {
 public:
  AnalyzeVariablesPass() = default;
  AnalyzeVariablesPass(const AnalyzeVariablesPass&) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-analyze-variables-pass";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Analyze variables in the graph.";
  }

  void runOnOperation() override {
    auto* context = &getContext();
    auto module = getOperation();
    bool legalize_to_tfl = true;

    module.walk([&](Operation* op) {
      // Skip ops that are supported natively by TFLite.
      if (IsSupportedTFLiteResourceOp(op)) return WalkResult::advance();
      if (IsSupportedTFLiteControlFlow(op)) return WalkResult::advance();

      // Check for ops that are legalized to TFLite.
      if (op->getDialect()->getNamespace() == "tfl") {
        return WalkResult::advance();
      }
      // Check for ops that are not legalized to TFLite.
      if (IsSupportedTFDataForwardingOp(op)) {
        return WalkResult::advance();
      }

      // If any of the operands is a resource type, then we break
      // and mark the module as not valid for TFLite legalization.
      // Note: this might disable native variables in more than needed cases.
      // TODO(b/189370197): Enhance variable analysis.
      for (auto operand : op->getOperands()) {
        if (getElementTypeOrSelf(operand.getType()).isa<TF::ResourceType>()) {
          legalize_to_tfl = false;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    module->setAttr(kLegalizeTflVariables,
                    BoolAttr::get(context, legalize_to_tfl));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> CreateAnalyzeVariablesPass() {
  return std::make_unique<AnalyzeVariablesPass>();
}

static PassRegistration<AnalyzeVariablesPass> pass([] {
  return CreateAnalyzeVariablesPass();
});

}  // namespace TFL
}  // namespace mlir
