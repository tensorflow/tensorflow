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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_IMPORT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_IMPORT_MODEL_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {
namespace mlrt_compiler {

// Converts an MLIR `module` in TF dialect to MLRT's bytecode format. If
// `module_with_op_keys` is non-null, the intermediate module on which passes
// until (including) AssignOpKeyPass have run will be cloned to it.
//
// This is for initial conversion.
absl::StatusOr<mlrt::bc::Buffer> ConvertTfMlirToBytecode(
    const TfrtCompileOptions& options, tfrt_stub::FallbackState& fallback_state,
    mlir::ModuleOp module, tfrt_stub::ModelRuntimeContext& model_context,
    mlir::OwningOpRef<mlir::ModuleOp>* module_with_op_keys = nullptr,
    std::vector<std::string>* added_xla_function_names = nullptr);

// Converts an MLIR `module_with_op_keys` in TF dialect to MLRT's bytecode
// format, with op costs from `cost_recorder`.
//
// This is for re-conversion.
absl::StatusOr<mlrt::bc::Buffer> ConvertTfMlirWithOpKeysToBytecode(
    const TfrtCompileOptions& options,
    const tfrt_stub::FallbackState& fallback_state,
    mlir::ModuleOp module_with_op_keys,
    const tfrt_stub::CostRecorder& cost_recorder);

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_IMPORT_MODEL_H_
