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

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace py = pybind11;

namespace {

// Helper to implement TraceMe as a context manager in Python.
class TraceMeWrapper {
 public:
  explicit TraceMeWrapper(const std::string& name) : name_(name) {}

  void Enter() { traceme_.emplace(std::move(name_)); }

  void SetMetadata(const std::string& new_metadata) {
    if (TF_PREDICT_TRUE(traceme_)) {
      traceme_->AppendMetadata(absl::string_view(new_metadata));
    }
  }

  void Exit() { traceme_.reset(); }

  static bool IsEnabled() { return tensorflow::profiler::TraceMe::Active(); }

 private:
  tensorflow::string name_;
  absl::optional<tensorflow::profiler::TraceMe> traceme_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_traceme, m) {
  py::class_<TraceMeWrapper> traceme_class(m, "TraceMe");
  traceme_class.def(py::init<const std::string&>())
      .def("Enter", &TraceMeWrapper::Enter)
      .def("Exit", &TraceMeWrapper::Exit)
      .def("SetMetadata", &TraceMeWrapper::SetMetadata)
      .def_static("IsEnabled", &TraceMeWrapper::IsEnabled);
};
