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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_ASSET_SINKING_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_ASSET_SINKING_PASS_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace tf_saved_model {

// Helper function that sets up a module for an AssetSinkingPass. The sole
// argument of the main function of `module` is prepared to be inlined with
// the value `checkpoint_path`.
// Also adds SessionInitializer op.
absl::Status AddSessionInitializerAndInlineCheckpoint(
    ModuleOp module, absl::string_view checkpoint_path);

// Creates a pass that sinks SavedModel asset filenames to constants.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAssetSinkingPass(
    llvm::StringRef saved_model_dir);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_ASSET_SINKING_PASS_H_
