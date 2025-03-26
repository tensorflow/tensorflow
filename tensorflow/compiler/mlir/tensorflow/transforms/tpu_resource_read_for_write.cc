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
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TFTPU {

#define GEN_PASS_DEF_TPURESOURCEREADFORWRITEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// A pass that finds TPU clusters with write only resource access and adds an
// associated resource read, so the resource can later be fused into TPUExecute.
namespace {
struct TPUResourceReadForWritePass
    : public impl::TPUResourceReadForWritePassBase<
          TPUResourceReadForWritePass> {
  void runOnOperation() override;
};

// Helper struct holding a resource value and its associated type.
struct ResourceValueAndSubtype {
  Value resource;
  Type subtype;
};

// Finds resource handle and type for result if result writes to a resource.
ResourceValueAndSubtype GetResourceWriteResult(
    tf_device::ClusterFuncOp cluster_func, Value result) {
  ResourceValueAndSubtype resource;
  if (!result.hasOneUse()) return resource;
  Operation* result_user = *result.getUsers().begin();
  auto assign_var = dyn_cast<TF::AssignVariableOp>(result_user);
  if (!assign_var) return resource;

  auto handle = assign_var.getResource();
  // Skip result if cluster writes to the same variable via multiple results.
  for (Operation* handle_user : handle.getUsers()) {
    if (handle_user == assign_var) continue;
    auto assign_var_user = dyn_cast<TF::AssignVariableOp>(handle_user);
    if (!assign_var_user) continue;
    if (assign_var_user.getValue().getDefiningOp() == cluster_func)
      return resource;
  }

  resource.resource = assign_var.getResource();
  resource.subtype = assign_var.getValue().getType();
  return resource;
}

// Checks if resource is read by TPU cluster.
bool ClusterFuncHasResourceRead(tf_device::ClusterFuncOp cluster_func,
                                Value resource) {
  for (Operation* resource_user : resource.getUsers()) {
    if (auto read = dyn_cast<TF::ReadVariableOp>(resource_user)) {
      for (Operation* read_user : read.getValue().getUsers()) {
        if (read_user == cluster_func) return true;
        if (isa<tf_device::ReplicateOp>(read_user)) return true;
      }
    }
  }

  return false;
}

void TPUResourceReadForWritePass::runOnOperation() {
  SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func) {
    cluster_funcs.push_back(cluster_func);
  });

  OpBuilder builder(&getContext());
  // Add resource reads for resource writes from TPU cluster where for such
  // resources the TPU cluster does not read from.
  for (tf_device::ClusterFuncOp cluster_func : cluster_funcs) {
    builder.setInsertionPoint(cluster_func);

    SmallVector<Value, 4> read_operands;
    for (Value result : cluster_func.getResults()) {
      // TODO(lyandy): Update pass to use resource alias analysis.
      auto resource_and_type = GetResourceWriteResult(cluster_func, result);
      if (!resource_and_type.resource) continue;
      if (ClusterFuncHasResourceRead(cluster_func, resource_and_type.resource))
        continue;
      auto new_read = builder.create<TF::ReadVariableOp>(
          resource_and_type.resource.getLoc(), resource_and_type.subtype,
          resource_and_type.resource);
      read_operands.push_back(new_read.getValue());
    }

    if (read_operands.empty()) continue;

    // Update caller and function types with new read operands.
    auto operands = llvm::to_vector<4>(cluster_func.getOperands());
    operands.append(read_operands.begin(), read_operands.end());

    auto loc = cluster_func.getLoc();
    auto new_cluster_func = builder.create<tf_device::ClusterFuncOp>(
        loc, cluster_func.getResultTypes(), operands, cluster_func->getAttrs());
    cluster_func.replaceAllUsesWith(new_cluster_func);
    func::FuncOp func = cluster_func.getFuncOp();
    Block& block = func.front();
    for (Value read_operand : read_operands)
      block.addArgument(read_operand.getType(), loc);

    func.setType(FunctionType::get(&getContext(), block.getArgumentTypes(),
                                   func.getResultTypes()));
    cluster_func.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUResourceReadForWritePass() {
  return std::make_unique<TPUResourceReadForWritePass>();
}

}  // namespace TFTPU
}  // namespace mlir
