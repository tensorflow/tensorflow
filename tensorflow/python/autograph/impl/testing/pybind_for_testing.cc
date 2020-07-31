// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace autograph {

namespace py = pybind11;

class TestClassDef {
 public:
  TestClassDef() = default;

  py::object Method() const;
};

py::object TestClassDef::Method() const { return py::none(); }

PYBIND11_MODULE(pybind_for_testing, m) {
  py::class_<TestClassDef>(m, "TestClassDef")
      .def(py::init<>())
      .def("method", &TestClassDef::Method);
}

}  // namespace autograph
