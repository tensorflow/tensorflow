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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/variant.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"

namespace py = ::pybind11;

using ::tensorflow::profiler::pywrap::ProfilerSessionWrapper;

namespace {

// This must be called under GIL because it reads Python objects. Reading Python
// objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
absl::flat_hash_map<std::string, absl::variant<int>> ConvertDictToMap(
    const py::dict& dict) {
  absl::flat_hash_map<std::string, absl::variant<int>> map;
  for (const auto& kw : dict) {
    if (!kw.second.is_none()) {
      map.emplace(kw.first.cast<std::string>(), kw.second.cast<int>());
    }
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
             absl::flat_hash_map<std::string, absl::variant<int>> opts =
                 ConvertDictToMap(options);
             {
               py::gil_scoped_release release;
               status = wrapper.Start(logdir, opts);
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
        absl::make_unique<tensorflow::profiler::ProfilerServer>();
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
          absl::flat_hash_map<std::string, absl::variant<int>> opts =
              ConvertDictToMap(options);
          {
            py::gil_scoped_release release;
            status = tensorflow::profiler::pywrap::Trace(
                service_addr, logdir, worker_list, include_dataset_ops,
                duration_ms, num_tracing_attempts, opts);
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

  m.def("xspace_to_tools_data",
        [](const py::list& xspace_path_list, const py::str& py_tool_name) {
          std::vector<tensorflow::profiler::XSpace> xspaces;
          xspaces.reserve(xspace_path_list.size());
          std::vector<std::string> filenames;
          filenames.reserve(xspace_path_list.size());
          for (py::handle obj : xspace_path_list) {
            std::string filename = std::string(py::cast<py::str>(obj));

            tensorflow::profiler::XSpace xspace;
            tensorflow::Status status;

            status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &xspace);

            if (!status.ok()) {
              return py::make_tuple(py::bytes(""), py::bool_(false));
            }

            xspaces.push_back(xspace);
            filenames.push_back(filename);
          }
          std::string tool_name = std::string(py_tool_name);
          auto tool_data_and_success =
              tensorflow::profiler::ConvertMultiXSpacesToToolData(
                  xspaces, filenames, tool_name);
          return py::make_tuple(py::bytes(tool_data_and_success.first),
                                py::bool_(tool_data_and_success.second));
        });

  m.def("xspace_to_tools_data_from_byte_string",
        [](const py::list& xspace_string_list, const py::list& filenames_list,
           const py::str& py_tool_name) {
          std::vector<tensorflow::profiler::XSpace> xspaces;
          xspaces.reserve(xspace_string_list.size());
          std::vector<std::string> filenames;
          filenames.reserve(filenames_list.size());

          // XSpace string inputs
          for (py::handle obj : xspace_string_list) {
            std::string xspace_string = std::string(py::cast<py::bytes>(obj));

            tensorflow::profiler::XSpace xspace;

            if (!xspace.ParseFromString(xspace_string)) {
              return py::make_tuple(py::bytes(""), py::bool_(false));
            }

            xspaces.push_back(xspace);
          }

          // Filenames
          for (py::handle obj : filenames_list) {
            filenames.push_back(std::string(py::cast<py::str>(obj)));
          }

          std::string tool_name = std::string(py_tool_name);
          auto tool_data_and_success =
              tensorflow::profiler::ConvertMultiXSpacesToToolData(
                  xspaces, filenames, tool_name);
          return py::make_tuple(py::bytes(tool_data_and_success.first),
                                py::bool_(tool_data_and_success.second));
        });
};
