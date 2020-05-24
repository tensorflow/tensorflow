/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_CONTEXT_MANAGER_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_CONTEXT_MANAGER_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace py = pybind11;

namespace tensorflow {
namespace profiler {

// Helper to implement TraceMe as a context manager in Python.
class TraceMeContextManager {
 public:
  explicit TraceMeContextManager(py::str name, py::kwargs kwargs)
      : name_(std::move(name)), kwargs_(std::move(kwargs)) {}

  void Enter() {
    if (IsEnabled()) {
      traceme_.emplace([this]() {
        std::string name(name_);
        if (!kwargs_.empty()) {
          AppendMetadata(&name, kwargs_);
        }
        return name;
      });
    }
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
  // Converts kwargs to strings and appends them to name encoded as TraceMe
  // metadata.
  static void AppendMetadata(std::string* name, const py::kwargs& kwargs) {
    name->push_back('#');
    for (const auto& kv : kwargs) {
      absl::StrAppend(name, std::string(py::str(kv.first)), "=",
                      std::string(py::str(kv.second)), ",");
    }
    name->back() = '#';
  }

  py::str name_;
  py::kwargs kwargs_;
  absl::optional<tensorflow::profiler::TraceMe> traceme_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_TRACEME_CONTEXT_MANAGER_
