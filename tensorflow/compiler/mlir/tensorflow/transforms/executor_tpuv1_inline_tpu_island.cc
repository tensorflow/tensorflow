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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#define DEBUG_TYPE "tf-executor-tpu-v1-island-inlining"

namespace mlir {
namespace tf_executor {

namespace {
constexpr llvm::StringRef kNestedModule = "_tpu_v1_compat_outlined";

#define GEN_PASS_DEF_EXECUTORTPUV1ISLANDINLININGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ExecutorTPUV1IslandInliningPass
    : public impl::ExecutorTPUV1IslandInliningPassBase<
          ExecutorTPUV1IslandInliningPass> {
  void runOnOperation() override;
};

void ExecutorTPUV1IslandInliningPass::runOnOperation() {
  SymbolTable symbol_table(getOperation());
  Operation *nested_module = symbol_table.lookup(kNestedModule);
  if (!nested_module) return;

  InlinerInterface inliner(&getContext());
  auto walk_result = getOperation().walk([&](TF::PartitionedCallOp call_op) {
    if (!call_op.getF().getRootReference().getValue().starts_with(
            kNestedModule))
      return WalkResult::advance();
    // This is a call we need to inline!
    LLVM_DEBUG(llvm::dbgs()
               << "Found call to inline: " << *call_op.getOperation() << "\n");

    auto call_interface = cast<CallOpInterface>(call_op.getOperation());
    auto called_func =
        dyn_cast_or_null<func::FuncOp>(call_interface.resolveCallable());

    if (failed(inlineCall(inliner, call_interface,
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
  for (func::FuncOp func_op :
       llvm::make_early_inc_range(nested_block.getOps<func::FuncOp>())) {
    if (!symbol_table.lookupSymbolIn(getOperation(), func_op.getName())) {
      nested_block.getOperations().remove(func_op.getOperation());
      symbol_table.insert(func_op.getOperation());
    }
  }
  nested_module->erase();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandInliningPass() {
  return std::make_unique<ExecutorTPUV1IslandInliningPass>();
}

}  // namespace tf_executor
}  // namespace mlir
