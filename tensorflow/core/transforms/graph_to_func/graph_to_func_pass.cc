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

#include "tensorflow/core/transforms/graph_to_func/graph_to_func_pass.h"

#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/graph_to_func/graph_to_func.h"

namespace mlir {
namespace tfg {

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/core/transforms/passes.h.inc"

// A pass that lift a graph into a function based on provided feeds and fetches.
struct GraphToFuncPass : GraphToFuncBase<GraphToFuncPass> {
  GraphToFuncPass(ArrayRef<std::string> feeds, ArrayRef<std::string> fetches) {
    feeds_ = feeds;
    fetches_ = fetches;
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto ops_list = module.getOps<GraphOp>();
    if (ops_list.empty()) return;
    if (std::next(ops_list.begin()) != ops_list.end()) {
      emitError((*std::next(ops_list.begin())).getLoc())
          << "expects a single GraphOp in the module";
      signalPassFailure();
      return;
    }
    GraphOp graph = *ops_list.begin();
    auto status = GraphToFunc(graph, feeds_, fetches_, control_rets_,
                              "_mlir_lifted_graph");
    if (!status.ok()) {
      emitError(graph.getLoc())
          << "GraphToFunc failed: " << status.error_message();
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateGraphToFuncPass(ArrayRef<std::string> feeds,
                                            ArrayRef<std::string> fetches) {
  return std::make_unique<GraphToFuncPass>(feeds, fetches);
}
}  // namespace tfg
}  // namespace mlir
