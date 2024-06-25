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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {

class BackendCompiler {
 public:
  virtual ~BackendCompiler();

  virtual void GetDependentDialects(mlir::DialectRegistry& registry) const {}

  // Compile the `module` in TF dialect. The result module should be also in TF
  // dialect.
  virtual absl::Status CompileTensorflow(
      tfrt_stub::ModelRuntimeContext& model_context,
      mlir::ModuleOp module) const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BACKEND_COMPILER_H_
