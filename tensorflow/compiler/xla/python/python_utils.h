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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_

#include <Python.h>

#include <optional>
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace jax {

// This file contains utilities to write Python wrapers using the C API.
// It's used for performance critical code such as PyBuffer, jax.jit or
// jax.pmap.

// Helpers for building Python properties
template <typename Func>
pybind11::object property_readonly(Func&& get) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property(pybind11::cpp_function(std::forward<Func>(get)),
                  pybind11::none(), pybind11::none(), "");
}

template <typename GetFunc, typename SetFunc>
pybind11::object property(GetFunc&& get, SetFunc&& set) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property(pybind11::cpp_function(std::forward<GetFunc>(get)),
                  pybind11::cpp_function(std::forward<SetFunc>(set)),
                  pybind11::none(), "");
}

template <typename Constructor>
pybind11::object def_static(Constructor&& constructor) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return pybind11::staticmethod(
      pybind11::cpp_function(std::forward<Constructor>(constructor)));
}

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_
