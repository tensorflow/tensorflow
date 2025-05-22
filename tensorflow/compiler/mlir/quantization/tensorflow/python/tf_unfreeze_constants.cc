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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/tf_unfreeze_constants.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace tf_quantization {

// Unfreezes constants into variables and saves them to a checkpoint files under
// `checkpoint_dir`. `checkpoint_dir` will be created within this function. It
// will return a non-OK status if it already exists or permission is denied.
// TODO(b/261652258): Make sure this works for when there are non-frozen
// variables in the model.
absl::Status UnfreezeConstantsAndSaveVariables(
    const absl::string_view checkpoint_dir, mlir::MLIRContext &ctx,
    mlir::ModuleOp module_op) {
  TF_RETURN_IF_ERROR(quantization::RunPasses(
      /*name=*/kTfQuantConstantUnfreezingStepName, /*add_passes_func=*/
      [](mlir::PassManager &pm) {
        pm.addPass(mlir::tf_quant::CreateUnfreezeConstantsPass());
      },
      ctx, module_op));

  if (const absl::Status create_dir_status =
          Env::Default()->CreateDir(std::string(checkpoint_dir));
      !create_dir_status.ok()) {
    LOG(ERROR) << "Failed to create checkpoint directory at: "
               << checkpoint_dir;
    return create_dir_status;
  }

  TF_ASSIGN_OR_RETURN(
      const auto unused_variable_names,
      quantization::SaveVariablesToCheckpoint(checkpoint_dir, module_op));

  return quantization::RunPasses(
      /*name=*/kTfQuantInsertRestoreOpStepName,
      /*add_passes_func=*/
      [](mlir::PassManager &pm) {
        pm.addPass(mlir::tf_quant::CreateInsertRestoreOpPass());
        pm.addPass(mlir::tf_quant::CreateInsertSaveOpPass());
        // Initialization by `tf.ConstOp` is no longer required as there is
        // a `tf.RestoreV2Op` now.
        pm.addPass(
            mlir::tf_quant::CreateRemoveVariableInitializationByConstPass());
      },
      ctx, module_op);
}
}  // namespace tf_quantization
}  // namespace tensorflow
