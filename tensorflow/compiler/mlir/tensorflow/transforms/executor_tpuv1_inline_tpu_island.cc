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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Visitors.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Transforms/InliningUtils.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#define DEBUG_TYPE "tf-executor-tpu-v1-island-inlining"

namespace mlir {
namespace tf_executor {

namespace {
constexpr llvm::StringRef kNestedModule = "_tpu_v1_compat_outlined";

// Inlining the islands calling into the nested module that was outlined.
// This is the end of the TPU bridge in V1 compatibility mode.
struct TPUBridgeExecutorIslandInlining
    : public ModulePass<TPUBridgeExecutorIslandInlining> {
  void runOnModule() override;
};

void TPUBridgeExecutorIslandInlining::runOnModule() {
  SymbolTable symbol_table(getModule());
  Operation *nested_module = symbol_table.lookup(kNestedModule);
  if (!nested_module) return;

  InlinerInterface inliner(&getContext());
  auto walk_result = getModule().walk([&](TF::PartitionedCallOp call_op) {
    if (!call_op.f().getRootReference().startswith(kNestedModule))
      return WalkResult::advance();
    // This is a call we need to inline!
    LLVM_DEBUG(llvm::dbgs()
               << "Found call to inline: " << *call_op.getOperation() << "\n");

    FuncOp called_func = dyn_cast_or_null<FuncOp>(
        symbol_table.lookupSymbolIn(getModule(), call_op.f()));

    if (failed(inlineCall(inliner,
                          cast<CallOpInterface>(call_op.getOperation()),
                          cast<CallableOpInterface>(called_func.getOperation()),
                          called_func.getCallableRegion(),
                          /* shouldCloneInlinedRegion = */ false))) {
      call_op.emitOpError() << "Failed to inline\n";
      return WalkResult::interrupt();
    }
    called_func.erase();
    call_op.erase();
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return signalPassFailure();
  // Move all remaining nested functions back into the parent module.
  Block &nested_block = nested_module->getRegion(0).front();
  for (FuncOp func_op :
       llvm::make_early_inc_range(nested_block.getOps<FuncOp>())) {
    if (!symbol_table.lookupSymbolIn(getModule(), func_op.getName())) {
      nested_block.getOperations().remove(func_op.getOperation());
      symbol_table.insert(func_op.getOperation());
    }
  }
  nested_module->erase();
}

PassRegistration<TPUBridgeExecutorIslandInlining> tpu_pass(
    "tf-executor-tpu-v1-island-inlining",
    "Inline calls to the nested TPU module, this reverses the effect of the "
    "-tf-executor-tpu-v1-island-outlining pass");

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>>
CreateTFExecutorTPUV1IslandInliningPass() {
  return std::make_unique<TPUBridgeExecutorIslandInlining>();
}

}  // namespace tf_executor
}  // namespace mlir
