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
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/status_casters.h"

namespace xla {

namespace py = ::pybind11;

namespace {

xla::Status MyFunc() { return xla::OkStatus(); }

class MyClass {
 public:
  xla::Status MyMethod(int a, int b) { return xla::OkStatus(); }
};

PYBIND11_MODULE(status_casters_ext, m) {
  // Exceptions
  py::register_exception<xla::XlaRuntimeError>(m, "XlaRuntimeError",
                                               PyExc_RuntimeError);

  m.def("my_lambda",
        xla::ThrowIfErrorWrapper([]() { return xla::OkStatus(); }));
  m.def("my_lambda2", xla::ThrowIfErrorWrapper(MyFunc));

  py::class_<MyClass> my_class(m, "MyClass");
  my_class.def(py::init<>());
  my_class.def("my_method", xla::ThrowIfErrorWrapper(&MyClass::MyMethod));
}

}  // namespace

}  // namespace xla
