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

// This pass outlines regions of `tf_device.launch` into functions and replaces
// `tf_device.launch` with equivalent `tf_device.launch_func` operations.

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncAttr[] = "func";

struct ClusterOutliningPass : public ModulePass<ClusterOutliningPass> {
  void runOnModule() override;
};

void ReplaceLaunchReturnWithReturn(tf_device::ReturnOp launch_return_op,
                                   OpBuilder* builder) {
  llvm::SmallVector<Value*, 4> operands(launch_return_op.getOperands());
  builder->create<ReturnOp>(launch_return_op.getLoc(), operands);
  launch_return_op.erase();
}

// Builds a function that outlines region attached to launch_op and inserts
// built function into given module.
FuncOp BuildFunction(StringRef device, llvm::ArrayRef<Value*> live_ins,
                     tf_device::LaunchOp launch_op,
                     ModuleManager* module_manager, OpBuilder* builder) {
  llvm::SmallVector<Type, 4> operand_types;
  operand_types.reserve(live_ins.size());
  for (Value* v : live_ins) operand_types.emplace_back(v->getType());

  llvm::SmallVector<Type, 4> result_types(launch_op.getResultTypes());

  auto func_type =
      FunctionType::get(operand_types, result_types, builder->getContext());

  std::string func_name_prefix = Twine(device, "_func").str();
  FuncOp outlined_func =
      FuncOp::create(launch_op.getLoc(), func_name_prefix, func_type);

  // Create function body.
  Block* outlined_func_block = outlined_func.addEntryBlock();

  // Replace uses of live-in values within launch_op region with function
  // arguments.
  Region& launch_op_region = launch_op.body();
  for (const auto& p :
       llvm::zip(live_ins, outlined_func_block->getArguments())) {
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               launch_op_region);
  }

  // Move all instructions in launch_op into outlined_function's only block.
  auto& launch_op_body = launch_op_region.front().getOperations();
  outlined_func_block->getOperations().splice(
      outlined_func_block->end(), launch_op_body, launch_op_body.begin(),
      launch_op_body.end());

  // Replace `tf_device.launch_return` terminator with `std.return` in function
  // body.
  auto launch_return_op =
      cast<tf_device::ReturnOp>(outlined_func_block->getTerminator());
  builder->setInsertionPoint(launch_return_op);
  ReplaceLaunchReturnWithReturn(launch_return_op, builder);

  module_manager->insert(outlined_func);
  return outlined_func;
}

// Outlines body of `tf_device.launch` into a function and create a
// `tf_device.launch_func` to invoke that function. `tf_device.launch` is
// removed afterwards.`
void OutlineLaunch(tf_device::LaunchOp launch_op, ModuleManager* module_manager,
                   OpBuilder* builder) {
  llvm::SetVector<Value*> live_ins;
  getUsedValuesDefinedAbove(launch_op.body(), launch_op.body(), live_ins);

  StringRef device =
      launch_op.getAttrOfType<StringAttr>(kDeviceAttr).getValue();

  FuncOp outlined_func = BuildFunction(device, live_ins.getArrayRef(),
                                       launch_op, module_manager, builder);
  launch_op.setAttr(builder->getIdentifier(kFuncAttr),
                    builder->getSymbolRefAttr(outlined_func.getName()));

  builder->setInsertionPoint(launch_op);
  tf_device::LaunchFuncOp launch_func_op =
      builder->create<tf_device::LaunchFuncOp>(
          launch_op.getLoc(), outlined_func.getType().getResults(),
          live_ins.getArrayRef(), launch_op.getAttrs());

  launch_op.replaceAllUsesWith(launch_func_op);
  launch_op.erase();
}

void ClusterOutliningPass::runOnModule() {
  ModuleOp m = getModule();
  ModuleManager module_manager(m);
  OpBuilder builder(m.getContext());
  m.walk([&](tf_device::LaunchOp launch) {
    OutlineLaunch(launch, &module_manager, &builder);
  });
}

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateClusterOutliningPass() {
  return std::make_unique<ClusterOutliningPass>();
}

static PassRegistration<ClusterOutliningPass> pass(
    "tf-device-cluster-outlining",
    "Outline regions of tf_device.launch operations.");

}  // namespace TFDevice
}  // namespace mlir
