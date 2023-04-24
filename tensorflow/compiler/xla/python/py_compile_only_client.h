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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_COMPILE_ONLY_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_COMPILE_ONLY_CLIENT_H_

#include <memory>

#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "tensorflow/compiler/xla/python/py_client.h"

namespace xla {

// This is a workaround for AOT compilation until topologies and device
// descriptions are better integrated into jax's Python code. It returns a
// PyClient that will return errors for all non-AOT methods. It also exposes a
// different compile method that returns an unloaded executable (vs. PyClient
// usually returns a loaded executable). RegisterCompileOnlyClient() overloads
// the Python "compile" method to return the unloaded executable, and we rely on
// Python duck typing to treat the unloaded executable like a loaded executable
// (except it will raise errors if you try to run it, which is what we want for
// AOT environments).
std::shared_ptr<PyClient> MakeCompileOnlyClient(
    std::shared_ptr<PjRtTopologyDescription>);

void RegisterCompileOnlyClient(pybind11::module& m);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_COMPILE_ONLY_CLIENT_H_
