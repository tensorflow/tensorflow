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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TFTPU {

// Creates a pass that rewrites `tf_device.launch_func` on TPUs into TPU runtime
// ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateTPURewritePass(
    llvm::StringRef module_name = llvm::StringRef());

// Creates a pass that adds ops which perform formatting on variables at
// run-time according to compilation result.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUVariableRuntimeReformattingPass();

// Creates a pass that merges device variable reads/updates into the surrounded
// TPUExecute node. This allows the execute node to perform in-place variable
// updates.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUMergeVariablesWithExecutePass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_TPUMERGEVARIABLESWITHEXECUTEPASS
#define GEN_PASS_DECL_TPUREWRITEPASS
#define GEN_PASS_DECL_TPUVARIABLERUNTIMEREFORMATTINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/runtime_passes.h.inc"

}  // namespace TFTPU
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_RUNTIME_PASSES_H_
