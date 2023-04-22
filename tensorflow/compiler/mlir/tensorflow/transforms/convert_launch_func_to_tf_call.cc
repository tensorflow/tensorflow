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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFDevice {

namespace {

// Rewrites tf_device::LaunchFuncOp into TF::PartitionedCallOp.
struct ConvertLaunchFuncToTFCallPass
    : public TF::ConvertLaunchFuncToTFCallPassBase<
          ConvertLaunchFuncToTFCallPass> {
  void runOnOperation() override;
};

void ConvertLaunchFuncToTFCallPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([&](tf_device::LaunchFuncOp launch) {
    OpBuilder builder(launch);
    auto call_op = builder.create<TF::PartitionedCallOp>(
        module.getLoc(), launch.getResultTypes(), launch.operands(),
        builder.getSymbolRefAttr(launch.func()),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    call_op->setAttr("device", launch->getAttrOfType<StringAttr>("device"));
    launch.replaceAllUsesWith(call_op);
    launch.erase();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertLaunchFuncToTFCallPass() {
  return std::make_unique<ConvertLaunchFuncToTFCallPass>();
}

}  // namespace TFDevice
}  // namespace mlir
