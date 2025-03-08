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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/tsl/platform/errors.h"

namespace tensorflow {
namespace quantization {

absl::Status RunPassesOnModuleOp(
    std::optional<absl::string_view> mlir_dump_file_name,
    mlir::PassManager& pass_manager, mlir::ModuleOp module_op) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module_op.getContext(),
                                                    /*propagate=*/true);

  absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> dump_file;
  if (mlir_dump_file_name) {
    TF_RETURN_IF_ERROR(tensorflow::quantization::MaybeEnableIrPrinting(
        pass_manager, mlir_dump_file_name.value()));
  }

  if (failed(pass_manager.run(module_op))) {
    return statusHandler.ConsumeStatus();
  }

  return absl::OkStatus();
}

}  // namespace quantization
}  // namespace tensorflow
