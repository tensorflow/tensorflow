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

#include "pybind11/attr.h"
#include "pybind11/pybind11.h"
#include "tensorflow/python/profiler/internal/traceme_context_manager.h"

using ::tensorflow::profiler::TraceMeContextManager;

PYBIND11_MODULE(_pywrap_traceme, m) {
  py::class_<TraceMeContextManager> traceme_class(m, "TraceMe",
                                                  py::module_local());
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("Enter", &TraceMeContextManager::Enter)
      .def("Exit", &TraceMeContextManager::Exit)
      .def("SetMetadata", &TraceMeContextManager::SetMetadata)
      .def_static("IsEnabled", &TraceMeContextManager::IsEnabled);
};
