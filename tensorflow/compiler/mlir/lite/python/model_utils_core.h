/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_H_
#include <Python.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/CAPI/IR.h"  // from @llvm-project

namespace tflite {
namespace model_utils {

void RegisterMlirPasses();
MlirContext CreateIRContext();
void RegisterDialects(MlirContext context);
MlirModule FlatBufferToMlir(absl::string_view buffer, MlirContext context);
std::string MlirToFlatbuffer(MlirOperation c_op);
std::vector<std::string> GetOperationAttributeNames(MlirOperation c_op);

}  // namespace model_utils

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_H_