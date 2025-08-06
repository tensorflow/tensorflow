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
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "xla/pjrt/status_casters.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xprof/convert/tool_options.h"  // from @org_xprof
#include "xprof/pywrap/profiler_plugin_impl.h"  // from @org_xprof

namespace py = ::pybind11;

namespace {

using ::tensorflow::profiler::ToolOptions;

// These must be called under GIL because it reads Python objects. Reading
// Python objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
ToolOptions ToolOptionsFromPythonDict(const py::dict& dictionary) {
  ToolOptions map;
  for (const auto& item : dictionary) {
    std::variant<bool, int, std::string> value;
    try {
      value = item.second.cast<bool>();
    } catch (...) {
      try {
        value = item.second.cast<int>();
      } catch (...) {
        try {
          value = item.second.cast<std::string>();
        } catch (...) {
          continue;
        }
      }
    }
    map.emplace(item.first.cast<std::string>(), value);
  }
  return map;
}

}  // namespace

PYBIND11_MODULE(_pywrap_profiler_plugin, m) {
  m.def(
      "trace", [](const char* service_addr, const char* logdir,
                  const char* worker_list, bool include_dataset_ops,
                  int duration_ms, int num_tracing_attempts, py::dict options) {
        absl::Status status;
        ToolOptions tool_options = ToolOptionsFromPythonDict(options);
        {
          py::gil_scoped_release release;
          status = tsl::profiler::CaptureRemoteTraceWithBoolOpts(
              service_addr, logdir, worker_list, include_dataset_ops,
              duration_ms, num_tracing_attempts, tool_options);
        }
        // Py_INCREF and Py_DECREF must be called holding the GIL.
        xla::ThrowIfError(status);
      });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    tsl::string content;
    absl::Status status;
    {
      py::gil_scoped_release release;
      status =
          xprof::pywrap::Monitor(service_addr, duration_ms, monitoring_level,
                                 display_timestamp, &content);
    }
    // Py_INCREF and Py_DECREF must be called holding the GIL.
    xla::ThrowIfError(status);
    return content;
  });

  m.def(
      "xspace_to_tools_data",
      [](const py::list& xspace_path_list, const py::str& py_tool_name,
         const py::dict options = py::dict()) {
        std::vector<std::string> xspace_paths;
        xspace_paths.reserve(xspace_path_list.size());
        for (py::handle obj : xspace_path_list) {
          std::string xspace_path = std::string(py::cast<py::str>(obj));
          xspace_paths.push_back(xspace_path);
        }

        std::string tool_name = std::string(py_tool_name);
        ToolOptions tool_options = ToolOptionsFromPythonDict(options);
        absl::StatusOr<std::pair<std::string, bool>> result;

        {
          py::gil_scoped_release release;
          result = xprof::pywrap::XSpaceToToolsData(xspace_paths, tool_name,
                                                    tool_options);
        }

        if (!result.ok()) {
          xla::ThrowIfError(result.status());
        }
        return py::make_tuple(py::bytes(result->first),
                              py::bool_(result->second));
      },
      // TODO: consider defaulting `xspace_path_list` to empty list, since
      // this parameter is only used for two of the tools.
      py::arg(), py::arg(), py::arg() = py::dict());

  m.def(
      "xspace_to_tools_data_from_byte_string",
      [](const py::list& xspace_string_list, const py::list& filenames_list,
         const py::str& py_tool_name, const py::dict options = py::dict()) {
        std::vector<std::string> xspace_strings;
        xspace_strings.reserve(xspace_string_list.size());
        for (py::handle obj : xspace_string_list) {
          xspace_strings.push_back(std::string(py::cast<py::bytes>(obj)));
        }

        std::vector<std::string> xspace_paths;
        xspace_paths.reserve(filenames_list.size());
        for (py::handle obj : filenames_list) {
          xspace_paths.push_back(std::string(py::cast<py::str>(obj)));
        }

        std::string tool_name = std::string(py_tool_name);
        ToolOptions tool_options = ToolOptionsFromPythonDict(options);

        absl::StatusOr<std::pair<std::string, bool>> result;
        {
          py::gil_scoped_release release;
          result = xprof::pywrap::XSpaceToToolsDataFromByteString(
              xspace_strings, xspace_paths, tool_name, tool_options);
        }

        if (!result.ok()) {
          xla::ThrowIfError(result.status());
        }
        return py::make_tuple(py::bytes(result->first),
                              py::bool_(result->second));
      },
      py::arg(), py::arg(), py::arg(), py::arg() = py::dict());
};
