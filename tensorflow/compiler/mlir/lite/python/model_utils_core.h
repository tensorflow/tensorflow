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

#include "absl/status/status.h"
#include "mlir-c/IR.h"  // from @llvm-project
#include "mlir/Bindings/Python/PybindAdaptors.h"  // from @llvm-project  // IWYU pragma: keep
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
// #include "tensorflow/core/framework/tensor.h"

namespace tflite {
namespace model_utils {

namespace py = pybind11;

void RegisterMlirPasses();
MlirContext CreateIRContext();
void RegisterDialects(MlirContext context);
MlirModule FlatBufferToMlir(py::bytes buffer, MlirContext context);
py::bytes MlirToFlatbuffer(MlirOperation c_op);
std::vector<std::string> GetOperationAttributeNames(MlirOperation c_op);
// absl::StatusOr<tensorflow::Tensor> GetElementsAttrTensor(MlirAttribute
// c_attr);

}  // namespace model_utils

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_H_
