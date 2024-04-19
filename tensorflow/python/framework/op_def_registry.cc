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

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_op_def_registry, m) {
  m.def("get", [](const std::string& name) {
    const tensorflow::OpDef* op_def = nullptr;
    auto status = tensorflow::OpRegistry::Global()->LookUpOpDef(name, &op_def);
    if (!status.ok()) return py::reinterpret_borrow<py::object>(py::none());

    tensorflow::OpDef stripped_op_def = *op_def;
    tensorflow::RemoveNonDeprecationDescriptionsFromOpDef(&stripped_op_def);

    tensorflow::MaybeRaiseFromStatus(status);
    std::string serialized_op_def;
    if (!stripped_op_def.SerializeToString(&serialized_op_def)) {
      throw std::runtime_error("Failed to serialize OpDef to string");
    }

    // Explicitly convert to py::bytes because std::string is implicitly
    // convertable to py::str by default.
    return py::reinterpret_borrow<py::object>(py::bytes(serialized_op_def));
  });
}
