/* Copyright 2020 The OpenXLA Authors.

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
#ifndef XLA_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_H_
#define XLA_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_H_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "pybind11/pytypes.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace profiler {

// Wraps TraceMe with an interface that takes python types.
class TraceMeWrapper {
 public:
  // pybind11::str and pybind11::kwargs are taken by const reference to avoid
  // python reference-counting overhead.
  TraceMeWrapper(const pybind11::str& name, const pybind11::kwargs& kwargs)
      : traceme_(
            [&]() {
              std::string name_and_metadata(name);
              if (!kwargs.empty()) {
                AppendMetadata(&name_and_metadata, kwargs);
              }
              return name_and_metadata;
            },
            /*level=*/1) {}

  // pybind11::kwargs is taken by const reference to avoid python
  // reference-counting overhead.
  void SetMetadata(const pybind11::kwargs& kwargs) {
    if (TF_PREDICT_FALSE(!kwargs.empty())) {
      traceme_.AppendMetadata([&]() {
        std::string metadata;
        AppendMetadata(&metadata, kwargs);
        return metadata;
      });
    }
  }

  void Stop() { traceme_.Stop(); }

 private:
  // Converts kwargs to strings and appends them to name encoded as TraceMe
  // metadata.
  static void AppendMetadata(std::string* name,
                             const pybind11::kwargs& kwargs) {
    name->push_back('#');
    for (const auto& kv : kwargs) {
      absl::StrAppend(name, std::string(pybind11::str(kv.first)), "=",
                      EncodePyObject(kv.second), ",");
    }
    name->back() = '#';
  }

  static std::string EncodePyObject(const pybind11::handle& handle) {
    if (pybind11::isinstance<pybind11::bool_>(handle)) {
      return handle.cast<bool>() ? "1" : "0";
    }
    return std::string(pybind11::str(handle));
  }

  tsl::profiler::TraceMe traceme_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_PYTHON_PROFILER_INTERNAL_TRACEME_WRAPPER_H_
