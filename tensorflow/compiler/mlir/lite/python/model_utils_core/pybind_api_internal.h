/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_INTERNAL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_INTERNAL_H_

#include <memory>

#include "absl/status/status.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace model_utils {

std::unique_ptr<mlir::MLIRContext> CreateIRContext();
absl::Status MlirVerify(mlir::Operation* op);

}  // namespace model_utils
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_INTERNAL_H_
