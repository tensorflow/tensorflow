/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This pass outlines regions of `tf_device.cluster` into functions and replaces
// `tf_device.cluster` with equivalent `tf_device.cluster_func` operations.

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kFuncAttr[] = "func";

struct ClusterOutliningPass
    : public PassWrapper<ClusterOutliningPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void ReplaceClusterReturnWithReturn(tf_device::ReturnOp cluster_return_op,
                                    OpBuilder* builder) {
  builder->create<ReturnOp>(cluster_return_op.getLoc(),
                            cluster_return_op.getOperands());
  cluster_return_op.erase();
}

// Builds a function that outlines region attached to cluster_op and inserts
// built function into given module.
FuncOp BuildFunction(llvm::ArrayRef<Value> live_ins,
                     tf_device::ClusterOp cluster_op, SymbolTable* symbol_table,
                     OpBuilder* builder) {
  llvm::SmallVector<Type, 4> operand_types;
  operand_types.reserve(live_ins.size());
  for (Value v : live_ins) operand_types.emplace_back(v.getType());

  auto func_type = FunctionType::get(operand_types, cluster_op.getResultTypes(),
                                     builder->getContext());

  // TODO(lyandy): Define better name for outlined function. Potentially some
  // name can be added during cluster formation.
  FuncOp outlined_func =
      FuncOp::create(cluster_op.getLoc(), "_func", func_type);

  // This function is not externally visible and marking it private would allow
  // symbol-dce pass to remove it when it is not referenced anymore.
  outlined_func.setVisibility(FuncOp::Visibility::Private);

  // Create function body.
  Block* outlined_func_block = outlined_func.addEntryBlock();

  // Replace uses of live-in values within cluster_op region with function
  // arguments.
  Region& cluster_op_region = cluster_op.body();
  for (auto p : llvm::zip(live_ins, outlined_func_block->getArguments())) {
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               cluster_op_region);
  }

  // Move all instructions in cluster_op into outlined_function's only block.
  auto& cluster_op_body = cluster_op.GetBody().getOperations();
  outlined_func_block->getOperations().splice(
      outlined_func_block->end(), cluster_op_body, cluster_op_body.begin(),
      cluster_op_body.end());

  // Replace `tf_device.return` terminator with `std.return` in function
  // body.
  auto cluster_return_op =
      cast<tf_device::ReturnOp>(outlined_func_block->getTerminator());
  builder->setInsertionPoint(cluster_return_op);
  ReplaceClusterReturnWithReturn(cluster_return_op, builder);

  symbol_table->insert(outlined_func);
  return outlined_func;
}

// Outlines body of `tf_device.cluster` into a function and create a
// `tf_device.cluster_func` to invoke that function. `tf_device.cluster` is
// removed afterwards.`
void OutlineCluster(tf_device::ClusterOp cluster_op, SymbolTable* symbol_table,
                    OpBuilder* builder) {
  llvm::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(cluster_op.body(), cluster_op.body(), live_ins);

  FuncOp outlined_func =
      BuildFunction(live_ins.getArrayRef(), cluster_op, symbol_table, builder);
  cluster_op.setAttr(builder->getIdentifier(kFuncAttr),
                     builder->getSymbolRefAttr(outlined_func.getName()));

  builder->setInsertionPoint(cluster_op);
  auto cluster_func_op = builder->create<tf_device::ClusterFuncOp>(
      cluster_op.getLoc(), outlined_func.getType().getResults(),
      live_ins.getArrayRef(), cluster_op.getAttrs());

  cluster_op.replaceAllUsesWith(cluster_func_op);
  cluster_op.erase();
}

void ClusterOutliningPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);
  OpBuilder builder(module.getContext());
  module.walk([&](tf_device::ClusterOp cluster) {
    OutlineCluster(cluster, &symbol_table, &builder);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateClusterOutliningPass() {
  return std::make_unique<ClusterOutliningPass>();
}

static PassRegistration<ClusterOutliningPass> pass(
    "tf-device-cluster-outlining",
    "Outline regions of tf_device.cluster operations.");

}  // namespace TFDevice
}  // namespace mlir
