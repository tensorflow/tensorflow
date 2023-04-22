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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_PROTO_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_PROTO_H_

#include "absl/strings/str_cat.h"
#include "pybind11/pybind11.h"

namespace tensorflow {

inline void CheckProtoType(const pybind11::handle& py_object,
                           const std::string expected_proto_type) {
  // Check the name of the proto object.
  if (pybind11::hasattr(py_object, "DESCRIPTOR")) {
    pybind11::handle descriptor = pybind11::getattr(py_object, "DESCRIPTOR");
    std::string py_object_type =
        pybind11::cast<std::string>(pybind11::getattr(descriptor, "full_name"));

    if (py_object_type == expected_proto_type) {
      return;
    }
    throw pybind11::type_error(absl::StrCat("Expected an ", expected_proto_type,
                                            " proto, but got ",
                                            py_object_type));
  }
  throw pybind11::type_error(absl::StrCat(
      std::string(py_object.get_type().str()), " is not a valid proto."));
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_PROTO_H_
