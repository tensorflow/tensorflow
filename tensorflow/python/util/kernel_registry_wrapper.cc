/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"
#include "tensorflow/python/util/kernel_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_kernel_registry, m) {
  m.def("TryFindKernelClass", [](const std::string& serialized_node_def) {
    return py::bytes(tensorflow::swig::TryFindKernelClass(serialized_node_def));
  });
}
