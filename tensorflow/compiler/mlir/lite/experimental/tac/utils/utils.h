/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_UTILS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_UTILS_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

// Import the file as mlir module, the input maybe flatbuffer or mlir file.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportFlatbufferOrMlir(
    const std::string& input_filename, bool input_mlir,
    bool experimental_prune_unreachable_nodes_unconditionally,
    llvm::SourceMgr* source_mgr, mlir::MLIRContext* context);

// Export the module to file, can be either mlir or flatbuffer.
absl::Status ExportFlatbufferOrMlir(
    const std::string& output_filename, bool output_mlir, mlir::ModuleOp module,
    bool enable_select_tf_ops,
    std::optional<int> custom_option_alignment = std::nullopt);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_UTILS_UTILS_H_
