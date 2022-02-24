/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TFRT_FALLBACK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TFRT_FALLBACK_H_

#include <string>
#include <vector>

#include "pybind11/numpy.h"

namespace tensorflow {

// PyBind integration to compile and execute Tensorflow MLIR modules with the
// Tfrt fallback to Tensorflow. The only intended use case is testing tests for
// tf_jitrt in python (the fallback to Tensorflow provides the result to compare
// against).
std::vector<pybind11::array> RunTfrtFallback(
    const std::string& module_ir, const std::string& entrypoint,
    const std::vector<pybind11::array>& arguments);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TFRT_FALLBACK_H_
