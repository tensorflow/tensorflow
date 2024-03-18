/* Copyright 2019 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"

namespace xla {

namespace nb = ::nanobind;

namespace {

absl::Status MyFunc() { return absl::OkStatus(); }

class MyClass {
 public:
  absl::Status MyMethod(int a, int b) { return absl::OkStatus(); }
  absl::Status MyMethodConst(int a, int b) const { return absl::OkStatus(); }

  absl::StatusOr<int> MyStatusOrMethod(int a, int b) { return a + b; }
  absl::StatusOr<int> MyStatusOrMethodConst(int a, int b) const {
    return a + b;
  }
};

absl::StatusOr<int> StatusOrIdentity(int i) { return i; }

NB_MODULE(status_casters_ext, m) {
  // Exceptions
  nb::exception<xla::XlaRuntimeError>(m, "XlaRuntimeError", PyExc_RuntimeError);

  m.def("my_lambda",
        xla::ThrowIfErrorWrapper([]() { return absl::OkStatus(); }));
  m.def("my_lambda2", xla::ThrowIfErrorWrapper(MyFunc));

  m.def("my_lambda_statusor",
        xla::ValueOrThrowWrapper([]() -> absl::StatusOr<int> { return 1; }));
  m.def("status_or_identity", xla::ValueOrThrowWrapper(StatusOrIdentity));

  nb::class_<MyClass> my_class(m, "MyClass");
  my_class.def(nb::init<>());
  my_class.def("my_method", xla::ThrowIfErrorWrapper(&MyClass::MyMethod));
  my_class.def("my_method_const", xla::ThrowIfErrorWrapper(&MyClass::MyMethod));
  my_class.def("my_method_status_or",
               xla::ValueOrThrowWrapper(&MyClass::MyStatusOrMethod));
  my_class.def("my_method_status_or_const",
               xla::ValueOrThrowWrapper(&MyClass::MyStatusOrMethodConst));
}

}  // namespace

}  // namespace xla
