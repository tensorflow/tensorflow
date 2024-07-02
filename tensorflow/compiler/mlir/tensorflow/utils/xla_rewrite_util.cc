/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/xla_rewrite_util.h"

#include "absl/log/log.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
mlir::LogicalResult EraseClusterFuncs(
    llvm::MutableArrayRef<mlir::tf_device::ClusterFuncOp> to_be_erased) {
  for (auto cluster : to_be_erased) {
    auto old_parallel_execute =
        cluster->getParentOfType<mlir::tf_device::ParallelExecuteOp>();
    if (!old_parallel_execute) {
      LOG(ERROR) << "Parent op of cluster " << cluster.getOperationName().str()
                 << " is not ParallelExecuteOp.";
      return mlir::failure();
    }

    for (auto result : old_parallel_execute.getExecuteOutputs()) {
      for (mlir::Operation* user :
           llvm::make_early_inc_range(result.getUsers())) {
        if (llvm::isa<mlir::TF::TPUPartitionedOutputV2Op>(user)) {
          assert(user->use_empty());
          user->erase();
        }
      }
    }

    for (auto operand : cluster.getOperands()) {
      mlir::Operation* def = operand.getDefiningOp();
      if (operand.hasOneUse() &&
          llvm::isa_and_nonnull<mlir::TF::TPUPartitionedInputV2Op>(def)) {
        operand.dropAllUses();
        def->erase();
      }
    }
    if (!old_parallel_execute->use_empty()) {
      LOG(ERROR) << "Use of parallel execute op "
                 << old_parallel_execute.getOperationName().str()
                 << " is not empty.";
      return mlir::failure();
    }
    old_parallel_execute->erase();
  }
  return mlir::success();
}

int MovePreservedParallelExecuteChildren(
    int num_cores_per_replica,
    llvm::SmallVector<mlir::Type, 8>& concatenated_output_types,
    mlir::OpBuilder* builder, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute,
    mlir::tf_device::ParallelExecuteOp* new_parallel_execute) {
  // `num_moved_children` is the number of children that will be preserved.
  const size_t num_moved_children =
      old_parallel_execute.getRegions().size() - 1;
  *new_parallel_execute = builder->create<mlir::tf_device::ParallelExecuteOp>(
      old_parallel_execute->getLoc(),
      num_moved_children + num_cores_per_replica, concatenated_output_types);

  // `cluster_idx` is the index of the child with the `ClusterFuncOp`, which
  // will be replaced.
  int cluster_idx = -1;
  for (size_t child_idx = 0;
       child_idx < old_parallel_execute.getRegions().size(); ++child_idx) {
    auto& block = old_parallel_execute.GetRegionBlockWithIndex(child_idx);
    if (cluster_func->getBlock() == &block) {
      assert(cluster_idx == -1);
      cluster_idx = child_idx;
    }
  }
  assert(cluster_idx != -1);

  // Move children that are preserved.
  for (int child_idx = 0; child_idx < num_moved_children; ++child_idx) {
    int old_idx = child_idx >= cluster_idx ? child_idx + 1 : child_idx;
    int new_idx = child_idx >= cluster_idx ? child_idx + num_cores_per_replica
                                           : child_idx;
    new_parallel_execute->getRegions()[new_idx].takeBody(
        old_parallel_execute.getRegions()[old_idx]);
  }

  return cluster_idx;
}

mlir::tf_device::LaunchOp WrapOpInLaunch(mlir::OpBuilder* builder,
                                         mlir::Location loc,
                                         mlir::Operation* op,
                                         llvm::StringRef device) {
  mlir::OpBuilder::InsertPoint insert_point = builder->saveInsertionPoint();

  auto launch = builder->create<mlir::tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.getBody().push_back(new mlir::Block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<mlir::tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside cluster.
  op->moveBefore(launch.GetBody().getTerminator());

  builder->restoreInsertionPoint(insert_point);

  return launch;
}

}  // namespace tensorflow
