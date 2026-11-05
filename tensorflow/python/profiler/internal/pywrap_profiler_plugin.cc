/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "xla/pjrt/status_casters.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/utils/session_manager.h"

namespace py = ::pybind11;

namespace {

using ToolOptions =
    absl::flat_hash_map<std::string, std::variant<bool, int, std::string>>;

// These must be called under GIL because it reads Python objects. Reading
// Python objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
ToolOptions ToolOptionsFromPythonDict(const py::dict& dictionary) {
  ToolOptions map;
  // Copy keys to avoid iterator invalidation if the dict is mutated
  // during implicit casts of values which can run Python user code.
  std::vector<py::object> keys;
  keys.reserve(dictionary.size());
  for (const auto& item : dictionary) {
    keys.push_back(py::reinterpret_borrow<py::object>(item.first));
  }

  for (const auto& key : keys) {
    if (!dictionary.contains(key)) {
      continue;
    }

    py::object py_value = dictionary[key];
    std::variant<bool, int, std::string> value;
    try {
      value = py_value.cast<bool>();
    } catch (...) {
      try {
        value = py_value.cast<int>();
      } catch (...) {
        try {
          value = py_value.cast<std::string>();
        } catch (...) {
          continue;
        }
      }
    }
    try {
      map.emplace(key.cast<std::string>(), value);
    } catch (...) {
      // Ignore keys that can't be cast to string
    }
  }
  return map;
}

}  // namespace

PYBIND11_MODULE(_pywrap_profiler_plugin, m, pybind11::mod_gil_not_used()) {
  m.def("trace",
        [](const char* service_addr, const char* logdir,
           const char* worker_list, bool include_dataset_ops, int duration_ms,
           int num_tracing_attempts, py::dict options) {
          absl::Status status;
          ToolOptions tool_options = ToolOptionsFromPythonDict(options);
          {
            py::gil_scoped_release release;
            status = tsl::profiler::CaptureRemoteTrace(
                service_addr, logdir, worker_list, include_dataset_ops,
                duration_ms, num_tracing_attempts, tool_options);
          }
          // Py_INCREF and Py_DECREF must be called holding the GIL.
          xla::ThrowIfError(status);
        });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    std::string content;
    absl::Status status;
    {
      py::gil_scoped_release release;
      status = tsl::profiler::ValidateHostPortPair(service_addr);
      if (status.ok()) {
        status =
            tsl::profiler::Monitor(service_addr, duration_ms, monitoring_level,
                                   display_timestamp, &content);
      }
    }
    // Py_INCREF and Py_DECREF must be called holding the GIL.
    xla::ThrowIfError(status);
    return content;
  });
};
