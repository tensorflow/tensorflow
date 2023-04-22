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

#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace tensorflow {
namespace {

constexpr char kInvalidExecutorGraphMsg[] =
    "functions must be of a single Graph with single op Islands: ";

}  // namespace

mlir::LogicalResult VerifyExportSuitable(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](mlir::FuncOp function) {
    if (!llvm::hasSingleElement(function)) {
      function.emitError(kInvalidExecutorGraphMsg)
          << "only single block functions are supported";
      return mlir::WalkResult::interrupt();
    }

    auto block = function.front().without_terminator();
    auto graph = llvm::dyn_cast<mlir::tf_executor::GraphOp>(block.begin());
    if (!graph) {
      block.begin()->emitError(kInvalidExecutorGraphMsg)
          << "first op in function is not a tf_executor.graph";
      return mlir::WalkResult::interrupt();
    }

    if (!hasSingleElement(block)) {
      function.emitError(kInvalidExecutorGraphMsg)
          << "function does not only contain a single tf_executor.graph";
      return mlir::WalkResult::interrupt();
    }

    for (mlir::Operation& op : graph.GetBody()) {
      auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(op);
      if (!island) continue;

      if (!island.WrapsSingleOp()) {
        island.emitError(kInvalidExecutorGraphMsg)
            << "tf_executor.island must perfectly wrap a single op";
        return mlir::WalkResult::interrupt();
      }
    }

    return mlir::WalkResult::advance();
  });

  return mlir::failure(result.wasInterrupted());
}

}  // namespace tensorflow
