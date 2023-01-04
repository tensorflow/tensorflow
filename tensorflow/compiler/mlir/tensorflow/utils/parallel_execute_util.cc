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

#include "tensorflow/compiler/mlir/tensorflow/utils/parallel_execute_util.h"

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace mlir {
namespace TFTPU {

tf_device::ParallelExecuteOp BuildParallelExecuteOp(
    tf_device::ClusterFuncOp cluster_func, OpBuilder* builder) {
  const auto output_types = cluster_func.getResultTypes();
  builder->setInsertionPoint(cluster_func);
  auto parallel_execute = builder->create<tf_device::ParallelExecuteOp>(
      cluster_func.getLoc(), 1, output_types);
  cluster_func->remove();
  auto& block = parallel_execute.GetRegionBlockWithIndex(0);
  builder->setInsertionPointToEnd(&block);
  builder->insert(cluster_func);
  cluster_func.replaceAllUsesWith(parallel_execute);
  builder->create<tf_device::ReturnOp>(block.getParent()->getLoc(),
                                       cluster_func.getResults());
  return parallel_execute;
}

LogicalResult RemoveSingletonParallelExecuteOp(
    tf_device::ParallelExecuteOp parallel_execute, OpBuilder* builder) {
  if (parallel_execute.getRegions().size() == 1) {
    builder->setInsertionPoint(parallel_execute);
    auto& block = parallel_execute.GetRegionBlockWithIndex(0);
    llvm::SmallVector<Operation*, 2> ops_move;
    for (Operation& op : block) {
      ops_move.push_back(&op);
    }
    if (ops_move.size() != 2) {
      parallel_execute.emitError() << "Expected 2 ops in parallel_execute.";
      return failure();
    }
    ops_move[0]->remove();
    builder->insert(ops_move[0]);
    parallel_execute.replaceAllUsesWith(ops_move[0]);
    parallel_execute.erase();
  }
  return success();
}

}  // namespace TFTPU
}  // namespace mlir
