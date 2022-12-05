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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow {

struct FunctionBody;

// Converts FunctionDef to TFRT's Binary Executable Format. This is the entry
// point of tf.function to TFRT. function_name and device_name are given from
// the Python context. The lowered BEF will be stored in an external buffer
// pointed by bef_buffer.
Status ConvertFunctionToBef(
    mlir::StringRef function_name, const tensorflow::FunctionBody* fbody,
    const FunctionLibraryDefinition& flib_def,
    tfrt::ArrayRef<tfrt::string_view> devices,
    const tensorflow::TfrtFunctionCompileOptions& options,
    tfrt::BefBuffer* bef_buffer);

// Converts an MLIR `module` in TF dialect to TFRT's Binary Executable Format.
Status ConvertTfMlirToBef(const TfrtCompileOptions& options,
                          mlir::ModuleOp module, tfrt::BefBuffer* bef_buffer);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_IMPORT_MODEL_H_
