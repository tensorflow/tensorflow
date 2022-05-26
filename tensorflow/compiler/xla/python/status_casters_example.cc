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

#include <stdexcept>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/status_casters_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace py = ::pybind11;

namespace {

class XlaTestError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void ThrowXlaTestError(xla::Status status) {
  DCHECK(!status.ok());
  throw XlaTestError("XlaTestError");
}

xla::Status GetXlaStatus() {
  return xla::Status(tensorflow::error::Code::UNKNOWN, "XlaStatus");
}

xla::StatusOr<int> GetXlaStatusOr() { return GetXlaStatus(); }

xla::Status GetXlaStatusWithXlaTestError() {
  xla::Status status(tensorflow::error::Code::UNKNOWN, "XlaTestError");
  status_casters_util::SetFunctionPointerAsPayload(status, &ThrowXlaTestError);

  return status;
}

xla::StatusOr<int> GetXlaStatusOrWithXlaTestError() {
  return GetXlaStatusWithXlaTestError();
}

}  // namespace

PYBIND11_MODULE(status_casters_example, m) {
  py::register_exception<XlaTestError>(m, "XlaTestError", PyExc_RuntimeError);

  m.def("raise_xla_status", &GetXlaStatus);
  m.def("raise_xla_status_or", &GetXlaStatusOr);

  m.def("raise_xla_status_with_xla_test_error", &GetXlaStatusWithXlaTestError);
  m.def("raise_xla_status_or_with_xla_test_error",
        &GetXlaStatusOrWithXlaTestError);
}

}  // namespace xla
