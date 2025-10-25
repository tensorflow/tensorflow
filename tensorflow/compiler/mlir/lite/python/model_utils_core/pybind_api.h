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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_H_

#include "pybind11/pybind11.h"  // from @pybind11

void PopulateModelUtilsCoreApis(pybind11::module& m);

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MODEL_UTILS_CORE_PYBIND_API_H_
