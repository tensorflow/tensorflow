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
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace tf_executor {
namespace {

class UpdateControlDependenciesPass
    : public TF::ExecutorUpdateControlDependenciesPassBase<
          UpdateControlDependenciesPass> {
 public:
  void runOnOperation() override;
};

void UpdateAllControlDependencies(
    func::FuncOp func, const TF::SideEffectAnalysis::Info& analysis_for_func) {
  int control_inputs_added = 0;
  std::vector<Value> new_control_inputs{};

  auto graph_op = cast<GraphOp>(func.front().front());
  graph_op.walk([&](IslandOp island) {
    // Collect control inputs by querying side effect analysis.

    for (Operation* control_predecessor :
         analysis_for_func.DirectControlPredecessors(island)) {
      if (auto control_input_island =
              dyn_cast<mlir::tf_executor::IslandOp>(control_predecessor)) {
        new_control_inputs.push_back(control_input_island.control());
      }
    }
    // None of the originally given control deps are necessary.
    island.controlInputsMutable().clear();
    island.controlInputsMutable().append(new_control_inputs);
    control_inputs_added += new_control_inputs.size();
    new_control_inputs.clear();
  });

  FetchOp fetch_op = graph_op.GetFetch();

  // None of the originally given control deps are necessary.
  int num_control_fetches =
      fetch_op.getNumOperands() - graph_op.getNumResults();
  if (num_control_fetches > 0) {
    fetch_op.fetchesMutable().erase(graph_op.getNumResults(),
                                    num_control_fetches);
  }

  // Collect control inputs for `fetch_op` by querying side effect analysis.
  for (Operation* control_predecessor :
       analysis_for_func.DirectControlPredecessors(fetch_op)) {
    if (auto control_input_island =
            dyn_cast<tf_executor::IslandOp>(control_predecessor)) {
      new_control_inputs.push_back(control_input_island.control());
    }
  }
  control_inputs_added += new_control_inputs.size();
  fetch_op.fetchesMutable().append(new_control_inputs);

  VLOG(2) << "Number of control inputs added: " << control_inputs_added;
}

void UpdateControlDependenciesPass::runOnOperation() {
  ModuleOp module = getOperation();
  // This pass assumes that all functions are suitable for export, i.e., each
  // function has a single tf_executor.graph op and all islands wrap single
  // ops.
  if (failed(tensorflow::VerifyExportSuitable(module))) {
    signalPassFailure();
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module);

  // Recompute control dependencies between all islands.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    const auto& analysis_for_func =
        side_effect_analysis.GetAnalysisForFunc(func);
    UpdateAllControlDependencies(func, analysis_for_func);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorUpdateControlDependenciesPass() {
  return std::make_unique<UpdateControlDependenciesPass>();
}

}  // namespace tf_executor
}  // namespace mlir
