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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow {

struct FunctionBody;

// Converts an MLIR `module` in TF dialect to TFRT's Binary Executable Format.
// If `fallback_state` is not null, the MLIR functions for XLA clusters in
// the form of XlaLaunch will be exported and added to the function library when
// needed. The nested functions will also be exported. If
// `added_xla_function_names` is not null, it will be populated with the names
// of the added XLA functions.
Status ConvertTfMlirToBef(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    tfrt::BefBuffer* bef_buffer, tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state = nullptr,
    std::vector<std::string>* added_xla_function_names = nullptr);

Status ConvertTfMlirToRuntimeExecutable(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    absl::FunctionRef<Status(mlir::PassManager&, mlir::ModuleOp,
                             const tensorflow::TfrtPipelineOptions& options)>
        emit_executable,
    tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state = nullptr,
    std::vector<std::string>* added_xla_function_names = nullptr);

std::unique_ptr<tensorflow::TfrtPipelineOptions> GetTfrtPipelineOptions(
    const TfrtCompileOptions& options);

// Adds MLIR functions for XLA clusters to the function library.
tensorflow::Status AddXlaFunctions(
    tfrt_stub::FallbackState* fallback_state, mlir::ModuleOp mlir_module,
    std::vector<std::string>* added_xla_function_names = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_
