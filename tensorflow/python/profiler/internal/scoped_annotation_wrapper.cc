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

#include <utility>

#include "absl/types/optional.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"

namespace py = pybind11;

namespace {

// Helper to implement ScopedAnnotation as a context manager in Python.
class ScopedAnnotationWrapper {
 public:
  explicit ScopedAnnotationWrapper(const tensorflow::string& name)
      : name_(name) {}

  void Enter() { annotation_.emplace(std::move(name_)); }

  void Exit() { annotation_.reset(); }

  static bool IsEnabled() {
    return tensorflow::profiler::ScopedAnnotation::IsEnabled();
  }

 private:
  tensorflow::string name_;
  absl::optional<tensorflow::profiler::ScopedAnnotation> annotation_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_scoped_annotation, m) {
  py::class_<ScopedAnnotationWrapper> scoped_annotation_class(
      m, "ScopedAnnotation");
  scoped_annotation_class.def(py::init<const tensorflow::string&>())
      .def("Enter", &ScopedAnnotationWrapper::Enter)
      .def("Exit", &ScopedAnnotationWrapper::Exit)
      .def_static("IsEnabled", &ScopedAnnotationWrapper::IsEnabled);
};
