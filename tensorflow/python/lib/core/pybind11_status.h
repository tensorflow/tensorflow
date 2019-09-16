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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_

#include <Python.h>

#include "pybind11/pybind11.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"

namespace tensorflow {

namespace py = ::pybind11;

namespace pybind11 {

inline void MaybeRaiseFromStatus(const Status& status) {
  if (!status.ok()) {
    // TODO(slebedev): translate to builtin exception classes instead?
    auto* exc_type = PyExceptionRegistry::Lookup(status.code());
    PyErr_SetObject(
        exc_type,
        py::make_tuple(nullptr, nullptr, status.error_message()).ptr());
    throw py::error_already_set();
  }
}

}  // namespace pybind11
}  // namespace tensorflow

namespace pybind11 {
namespace detail {

// Raise an exception if a given status is not OK, otherwise return None.
//
// The correspondence between status codes and exception classes is given
// by PyExceptionRegistry. Note that the registry should be initialized
// in order to be used, see PyExceptionRegistry::Init.
template <>
struct type_caster<::tensorflow::Status> {
 public:
  PYBIND11_TYPE_CASTER(::tensorflow::Status, _("Status"));
  static handle cast(::tensorflow::Status status, return_value_policy, handle) {
    tensorflow::pybind11::MaybeRaiseFromStatus(status);
    return none();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
