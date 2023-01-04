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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

// Creates a pass that optimizes tf_saved_model.global_tensor ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeGlobalTensorsPass();

// Creates a pass that freezes tf_saved_model.global_tensor ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeGlobalTensorsPass(
    bool allow_mutable_tensors = false);

// Creates a pass that freezes tf_saved_model.asset ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeAssetsPass(
    std::string saved_model_dir = "");

// Creates as pass that removes variables in the session initializer.
// This job is required with lifting variable passes. Originally, the session
// initializer function does assigning variables. However, the read-only
// variable assignments will be done via lifting variables pass by converting
// the read-only variables to constant ops, instead. This pass removes the
// redundant operations. This pass should be located in front of the pass for
// lifting read-only variables.
std::unique_ptr<OperationPass<ModuleOp>>
CreateRemoveVariablesInSessionInitializerPass();

// Creates a pass that removes duplicate 'tf_saved_model.bound_input' bindings.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDedupBoundInputBindingPass();

// Create a pass that removes function arguments that map to global tensors.
std::unique_ptr<Pass> CreateLowerGlobalsToMlProgramPass();

// Create a pass that lowers variable read/write ops to ml_program ops.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLowerVariableOpsToMlProgramPass();

// Strips saved_model attributes from a module and its functions.
std::unique_ptr<OperationPass<ModuleOp>> CreateStripSavedModuleMetadataPass();

// Convert the session initializer to a function.
std::unique_ptr<OperationPass<ModuleOp>>
CreateConvertSessionInitializerToFunctionPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_DEDUPBOUNDINPUTBINDINGPASS
#define GEN_PASS_DECL_FREEZEASSETSPASS
#define GEN_PASS_DECL_FREEZEGLOBALTENSORSPASS
#define GEN_PASS_DECL_LOWERGLOBALSTOMLPROGRAMPASS
#define GEN_PASS_DECL_LOWERVARIABLEOPSTOMLPROGRAMPASS
#define GEN_PASS_DECL_OPTIMIZEGLOBALTENSORSPASS
#define GEN_PASS_DECL_REMOVEVARIABLESINSESSIONINITIALIZERPASS
#define GEN_PASS_DECL_STRIPSAVEDMODULEMETADATAPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

}  // namespace tf_saved_model

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_PASSES_H_
