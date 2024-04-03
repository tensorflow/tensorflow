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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_

#include <memory>

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace ifrt_serving {

// Create a pass to convert tf_device.cluster_func to tf.ifrt_program_call.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteClusterToIfrtCallPass();

// Creates a pass that sinks variable tensor argument to `tf.IfrtCall` as named
// arrays and lowers `tf.ReadVariableOp` to `tf.IfrtLoadVariableOp`.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkVariableAsNamedArrayPass();

// Creates a pass that splits `tf.RestoreV2` ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestoreSplittingPass();

// Creates a pass that merges `tf.RestoreV2` ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestoreMergingPass();

// Creates a pass that propagates inputs of no-op identity ops to their outputs.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfIdentityPropagationPass();

// Creates a pass that prunes unused `tf.RestoreV2` ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestorePruningPass();

// Creates a pass that lower `tf.RestoreVariableOp` to
// `tf.IfrtRestoreVariableOp`.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerToIfrtRestoreVariablePass();

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

// Register all passes.
void RegisterTfIfrtPasses();

// Convert tf_device.cluster_func to tf.ifrt_program_call.
// The callee function is converted to a ifrt_program.
absl::Status RunClusterToIfrtRuntimeOpsPassPipeline(
    mlir::ModuleOp module, llvm::StringRef module_name = llvm::StringRef());

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF_IFRT_PASSES_H_
