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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_RUN_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_RUN_PASSES_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace tensorflow {
namespace quantization {

// Runs MLIR passes with `module_op`. The passes are added by calling
// `add_passes_func`, which is a callable receiving mlir::PassManager& as its
// only argument. `name` identifies the set of passes added by `add_passes_func`
// and is used for debugging. Changing the `name` does not modify the behavior
// of the passes.
//
// It will try to dump intermediate MLIRs if certain conditions are met. See the
// description from `MaybeEnableIrPrinting` for the details about the
// conditions.
//
// Returns a non-OK status when the pass run fails or it fails to create an MLIR
// dump file.
template <typename FuncT>
absl::Status RunPasses(const absl::string_view name, FuncT add_passes_func,
                       mlir::MLIRContext& ctx, mlir::ModuleOp module_op) {
  mlir::PassManager pm{&ctx};
  add_passes_func(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler{&ctx};
  TF_RETURN_IF_ERROR(MaybeEnableIrPrinting(pm, name));

  if (failed(pm.run(module_op))) {
    return absl::InternalError(
        absl::StrFormat("Failed to run pass: %s. %s", name,
                        diagnostic_handler.ConsumeStatus().message()));
  }

  return absl::OkStatus();
}

// Runs MLIR passes with `module_op` on a `pass_manager`.
//
// It will try to dump intermediate MLIRs if certain conditions are met. See the
// description from `MaybeEnableIrPrinting` for the details about the
// conditions.
//
// Returns a non-OK status when the pass run fails or it fails to create an MLIR
// dump file.
absl::Status RunPassesOnModuleOp(
    std::optional<absl::string_view> mlir_dump_file_name,
    mlir::PassManager& pass_manager, mlir::ModuleOp module_op);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_RUN_PASSES_H_
