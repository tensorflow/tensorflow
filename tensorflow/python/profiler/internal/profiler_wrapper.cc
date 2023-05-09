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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"

namespace py = ::pybind11;

namespace {

using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::pywrap::ProfilerSessionWrapper;

// These must be called under GIL because it reads Python objects. Reading
// Python objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
ToolOptions ToolOptionsFromPythonDict(const py::dict& dictionary) {
  ToolOptions map;
  for (const auto& item : dictionary) {
    std::variant<int, std::string> value;
    try {
      value = item.second.cast<int>();
    } catch (...) {
      try {
        value = item.second.cast<std::string>();
      } catch (...) {
        continue;
      }
    }
    map.emplace(item.first.cast<std::string>(), value);
  }
  return map;
}

}  // namespace

PYBIND11_MODULE(_pywrap_profiler, m) {
  py::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class.def(py::init<>())
      .def("start",
           [](ProfilerSessionWrapper& wrapper, const char* logdir,
              const py::dict& options) {
             tensorflow::Status status;
             ToolOptions tool_options = ToolOptionsFromPythonDict(options);
             {
               py::gil_scoped_release release;
               status = wrapper.Start(logdir, tool_options);
             }
             // Py_INCREF and Py_DECREF must be called holding the GIL.
             tensorflow::MaybeRaiseRegisteredFromStatus(status);
           })
      .def("stop",
           [](ProfilerSessionWrapper& wrapper) {
             tensorflow::string content;
             tensorflow::Status status;
             {
               py::gil_scoped_release release;
               status = wrapper.Stop(&content);
             }
             // Py_INCREF and Py_DECREF must be called holding the GIL.
             tensorflow::MaybeRaiseRegisteredFromStatus(status);
             // The content is not valid UTF-8. It must be converted to bytes.
             return py::bytes(content);
           })
      .def("export_to_tb", [](ProfilerSessionWrapper& wrapper) {
        tensorflow::Status status;
        {
          py::gil_scoped_release release;
          status = wrapper.ExportToTensorBoard();
        }
        // Py_INCREF and Py_DECREF must be called holding the GIL.
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
      });

  m.def("start_server", [](int port) {
    auto profiler_server =
        std::make_unique<tensorflow::profiler::ProfilerServer>();
    profiler_server->StartProfilerServer(port);
    // Intentionally release profiler server. Should transfer ownership to
    // caller instead.
    profiler_server.release();
  });

  m.def("trace",
        [](const char* service_addr, const char* logdir,
           const char* worker_list, bool include_dataset_ops, int duration_ms,
           int num_tracing_attempts, py::dict options) {
          tensorflow::Status status;
          ToolOptions tool_options = ToolOptionsFromPythonDict(options);
          {
            py::gil_scoped_release release;
            status = tensorflow::profiler::pywrap::Trace(
                service_addr, logdir, worker_list, include_dataset_ops,
                duration_ms, num_tracing_attempts, tool_options);
          }
          // Py_INCREF and Py_DECREF must be called holding the GIL.
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
        });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    tensorflow::string content;
    tensorflow::Status status;
    {
      py::gil_scoped_release release;
      status = tensorflow::profiler::pywrap::Monitor(
          service_addr, duration_ms, monitoring_level, display_timestamp,
          &content);
    }
    // Py_INCREF and Py_DECREF must be called holding the GIL.
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
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
        auto status_or_session_snapshot =
            tensorflow::profiler::SessionSnapshot::Create(
                std::move(xspace_paths),
                /*xspaces=*/std::nullopt);
        if (!status_or_session_snapshot.ok()) {
          LOG(ERROR) << status_or_session_snapshot.status().message();
          return py::make_tuple(py::bytes(""), py::bool_(false));
        }

        std::string tool_name = std::string(py_tool_name);
        ToolOptions tool_options = ToolOptionsFromPythonDict(options);
        ::tensorflow::StatusOr<std::string> status_or_tool_data;
        {
          py::gil_scoped_release release;
          status_or_tool_data =
              tensorflow::profiler::ConvertMultiXSpacesToToolData(
                  status_or_session_snapshot.value(), tool_name, tool_options);
        }
        if (!status_or_tool_data.ok()) {
          LOG(ERROR) << status_or_tool_data.status().message();
          return py::make_tuple(py::bytes(""), py::bool_(false));
        }
        return py::make_tuple(py::bytes(status_or_tool_data.value()),
                              py::bool_(true));
      },
      // TODO: consider defaulting `xspace_path_list` to empty list, since
      // this parameter is only used for two of the tools...
      py::arg(), py::arg(), py::arg() = py::dict());

  m.def("xspace_to_tools_data_from_byte_string",
        [](const py::list& xspace_string_list, const py::list& filenames_list,
           const py::str& py_tool_name) {
          std::vector<std::unique_ptr<tensorflow::profiler::XSpace>> xspaces;
          xspaces.reserve(xspace_string_list.size());
          std::vector<std::string> xspace_paths;
          xspace_paths.reserve(filenames_list.size());

          // XSpace string inputs
          for (py::handle obj : xspace_string_list) {
            std::string xspace_string = std::string(py::cast<py::bytes>(obj));
            auto xspace = std::make_unique<tensorflow::profiler::XSpace>();
            if (!xspace->ParseFromString(xspace_string)) {
              return py::make_tuple(py::bytes(""), py::bool_(false));
            }
            for (int i = 0; i < xspace->hostnames_size(); ++i) {
              std::string hostname = xspace->hostnames(i);
              std::replace(hostname.begin(), hostname.end(), ':', '_');
              xspace->mutable_hostnames(i)->swap(hostname);
            }
            xspaces.push_back(std::move(xspace));
          }

          // XSpace paths.
          for (py::handle obj : filenames_list) {
            xspace_paths.push_back(std::string(py::cast<py::str>(obj)));
          }

          auto status_or_session_snapshot =
              tensorflow::profiler::SessionSnapshot::Create(
                  std::move(xspace_paths), std::move(xspaces));
          if (!status_or_session_snapshot.ok()) {
            LOG(ERROR) << status_or_session_snapshot.status().message();
            return py::make_tuple(py::bytes(""), py::bool_(false));
          }

          std::string tool_name = std::string(py_tool_name);
          auto status_or_tool_data =
              tensorflow::profiler::ConvertMultiXSpacesToToolData(
                  status_or_session_snapshot.value(), tool_name,
                  /*options=*/{});
          if (!status_or_tool_data.ok()) {
            LOG(ERROR) << status_or_tool_data.status().message();
            return py::make_tuple(py::bytes(""), py::bool_(false));
          }
          return py::make_tuple(py::bytes(status_or_tool_data.value()),
                                py::bool_(true));
        });
};
