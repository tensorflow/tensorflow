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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace py = pybind11;

namespace {

// Converts kwargs to strings and appends them to name encoded as TraceMe
// metadata.
TF_ATTRIBUTE_ALWAYS_INLINE inline void AppendMetadata(
    std::string* name, const py::kwargs& kwargs) {
  name->push_back('#');
  for (const auto& kv : kwargs) {
    absl::StrAppend(name, std::string(py::str(kv.first)), "=",
                    std::string(py::str(kv.second)), ",");
  }
  name->back() = '#';
}

// Helper to implement TraceMe as a context manager in Python.
class TraceMeWrapper {
 public:
  explicit TraceMeWrapper(py::str name, py::kwargs kwargs)
      : name_(std::move(name)), kwargs_(std::move(kwargs)) {}

  void Enter() {
    traceme_.emplace([this]() {
      std::string name(name_);
      if (!kwargs_.empty()) {
        AppendMetadata(&name, kwargs_);
      }
      return name;
    });
  }

  void SetMetadata(py::kwargs kwargs) {
    if (TF_PREDICT_TRUE(traceme_.has_value() && !kwargs.empty())) {
      traceme_->AppendMetadata([&kwargs]() {
        std::string metadata;
        AppendMetadata(&metadata, kwargs);
        return metadata;
      });
    }
  }

  void Exit() { traceme_.reset(); }

  static bool IsEnabled() { return tensorflow::profiler::TraceMe::Active(); }

 private:
  py::str name_;
  py::kwargs kwargs_;
  absl::optional<tensorflow::profiler::TraceMe> traceme_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_traceme, m) {
  py::class_<TraceMeWrapper> traceme_class(m, "TraceMe");
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("Enter", &TraceMeWrapper::Enter)
      .def("Exit", &TraceMeWrapper::Exit)
      .def("SetMetadata", &TraceMeWrapper::SetMetadata)
      .def_static("IsEnabled", &TraceMeWrapper::IsEnabled);
};
