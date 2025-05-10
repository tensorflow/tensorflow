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
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"
#include "xprof/convert/repository.h"  // from @org_xprof
#include "xprof/convert/tool_options.h"  // from @org_xprof
#include "xprof/convert/xplane_to_tools_data.h"  // from @org_xprof

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
    std::variant<bool, int, std::string> value;
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
             absl::Status status;
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
             absl::Status status;
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
        absl::Status status;
        {
          py::gil_scoped_release release;
          status = wrapper.ExportToTensorBoard();
        }
        // Py_INCREF and Py_DECREF must be called holding the GIL.
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
      });

  m.def("start_server", [](int port) {
    auto profiler_server = std::make_unique<tsl::profiler::ProfilerServer>();
    profiler_server->StartProfilerServer(port);
    // Intentionally release profiler server. Should transfer ownership to
    // caller instead.
    profiler_server.release();
  });
};
