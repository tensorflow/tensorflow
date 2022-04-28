/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {

namespace {

// A pass that moves `tf.AssignVariableOp` into a `tf_device.parallel_execute`
// region if the `tf.AssignVariableOp` is the only consumer of a
// `tf_device.parallel_execute` result. This will allow
// TPUMergeVariablesWithExecute to merge resource writes without special
// handling for `tf_device.parallel_execute`.
struct TPUParallelExecuteSinkResourceWrite
    : public TF::TPUParallelExecuteSinkResourceWritePassBase<
          TPUParallelExecuteSinkResourceWrite> {
  void runOnOperation() override;
};

// Finds an AssignVariableOp that can be moved into the parallel_execute region.
// These AssignVariableOps must be the only consumer of the respective
// parallel_execute result, and the resource handle producer must be from an op
// before or above the parallel_execute.
TF::AssignVariableOp GetSingleUseResourceWrite(
    tf_device::ParallelExecuteOp parallel_execute, Value result) {
  if (!result.hasOneUse()) return nullptr;

  OpOperand& use = *result.getUses().begin();
  auto assign_var = dyn_cast<TF::AssignVariableOp>(use.getOwner());
  if (!assign_var) return nullptr;

  if (use.get() != assign_var.value()) return nullptr;

  auto* resource_handle_op = assign_var.resource().getDefiningOp();
  if (resource_handle_op == parallel_execute) return nullptr;

  if (resource_handle_op &&
      resource_handle_op->getBlock() ==
          parallel_execute.getOperation()->getBlock() &&
      parallel_execute.getOperation()->isBeforeInBlock(resource_handle_op))
    return nullptr;

  return assign_var;
}

// Finds AssignVariableOps that can be moved into a parallel_execute region and
// moves them. Leftover parallel_execute results that were used by the
// such AssignVariableOp are also pruned.
void SinkResourceWritesIntoParallelExecute(
    tf_device::ParallelExecuteOp parallel_execute) {
  bool rewrite = false;
  const int num_regions = parallel_execute.getNumRegions();
  llvm::SmallVector<Value, 4> results_to_remap;

  // Go through each region and find AssignVariableOps that can be moved into
  // the parallel_execute region. Result indices by region index are collected,
  // so they can be removed afterwards.
  llvm::SmallVector<llvm::SmallVector<int, 4>, 4> results_to_remove_by_region;
  results_to_remove_by_region.resize(num_regions);
  for (int i = 0; i < num_regions; ++i) {
    Block& block = parallel_execute.GetRegionBlockWithIndex(i);
    auto results = parallel_execute.GetRegionOutputs(i);
    auto& results_to_remove = results_to_remove_by_region[i];
    results_to_remove.reserve(results.size());
    Operation* terminator = block.getTerminator();
    for (auto result : llvm::enumerate(results)) {
      TF::AssignVariableOp assign_var =
          GetSingleUseResourceWrite(parallel_execute, result.value());
      if (!assign_var) {
        results_to_remap.push_back(result.value());
        continue;
      }

      // Move AssignVariableOp and update the value to be written to the
      // resource variable to be the non forwarded value from within the
      // parallel_execute region.
      assign_var.getOperation()->moveBefore(terminator);
      assign_var.valueMutable().assign(terminator->getOperand(result.index()));
      results_to_remove.push_back(result.index());
    }

    rewrite |= !results_to_remove.empty();
  }

  if (!rewrite) return;

  // Remove leftover unused results (terminator operands) from moving
  // AssignVariabeOps into the parallel_execute region.
  for (auto results_to_remove : llvm::enumerate(results_to_remove_by_region)) {
    Block& block =
        parallel_execute.GetRegionBlockWithIndex(results_to_remove.index());
    Operation* terminator = block.getTerminator();
    for (int index_to_remove : llvm::reverse(results_to_remove.value()))
      terminator->eraseOperand(index_to_remove);
  }

  // Replace old parallel_execute with new parallel_execute by moving the
  // regions to a new parallel_execute and remapping the results.
  llvm::SmallVector<Type, 4> new_result_types;
  new_result_types.reserve(results_to_remap.size());
  for (Value old_result : results_to_remap)
    new_result_types.push_back(old_result.getType());

  OpBuilder builder(parallel_execute);
  auto new_parallel_execute = builder.create<tf_device::ParallelExecuteOp>(
      parallel_execute.getLoc(), num_regions, new_result_types);

  for (auto region : llvm::zip(new_parallel_execute.getRegions(),
                               parallel_execute.getRegions()))
    std::get<0>(region)->takeBody(*std::get<1>(region));

  for (auto result :
       llvm::zip(results_to_remap, new_parallel_execute.getResults()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  parallel_execute.erase();
}

void TPUParallelExecuteSinkResourceWrite::runOnOperation() {
  llvm::SmallVector<tf_device::ParallelExecuteOp, 4> parallel_executes;
  getOperation().walk([&](tf_device::ParallelExecuteOp parallel_execute) {
    parallel_executes.push_back(parallel_execute);
  });

  for (tf_device::ParallelExecuteOp parallel_execute : parallel_executes)
    SinkResourceWritesIntoParallelExecute(parallel_execute);
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUParallelExecuteSinkResourceWritePass() {
  return std::make_unique<TPUParallelExecuteSinkResourceWrite>();
}

}  // namespace TFTPU
}  // namespace mlir
