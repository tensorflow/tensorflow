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

#include "tensorflow/compiler/xla/python/profiler.h"

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/profiler/internal/traceme_wrapper.h"

namespace xla {

namespace py = pybind11;

namespace {
// Adds a trivial forwarding class so these Python bindings and TensorFlow's
// bindings of the same thing don't register the same class with pybind11.
class TraceMeWrapper : public tensorflow::profiler::TraceMeWrapper {
 public:
  using tensorflow::profiler::TraceMeWrapper::TraceMeWrapper;
};
}  // namespace

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler =
      m->def_submodule("profiler", "TensorFlow profiler integration");
  py::class_<tensorflow::profiler::ProfilerServer,
             std::unique_ptr<tensorflow::profiler::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def(
      "start_server",
      [](int port) -> std::unique_ptr<tensorflow::profiler::ProfilerServer> {
        auto server = absl::make_unique<tensorflow::profiler::ProfilerServer>();
        server->StartProfilerServer(port);
        return server;
      },
      py::arg("port"));

  py::class_<tensorflow::ProfilerSession> profiler_session_class(
      profiler, "ProfilerSession");
  profiler_session_class
      .def(py::init([]() {
        return tensorflow::ProfilerSession::Create(
            tensorflow::ProfilerSession::DefaultOptions());
      }))
      .def("stop_and_export",
           [](tensorflow::ProfilerSession* sess,
              const std::string& tensorboard_dir) -> xla::Status {
             tensorflow::profiler::XSpace xspace;
             // Disables the ProfilerSession
             TF_RETURN_IF_ERROR(sess->CollectData(&xspace));
             xspace.add_hostnames(tensorflow::port::Hostname());
             return tensorflow::profiler::ExportToTensorBoard(xspace,
                                                              tensorboard_dir);
           });

  py::class_<TraceMeWrapper> traceme_class(profiler, "TraceMe",
                                           py::module_local());
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("__enter__", [](py::object self) -> py::object { return self; })
      .def("__exit__",
           [](py::object self, const py::object& ex_type,
              const py::object& ex_value,
              const py::object& traceback) -> py::object {
             py::cast<TraceMeWrapper*>(self)->Stop();
             return py::none();
           })
      .def("set_metadata", &TraceMeWrapper::SetMetadata)
      .def_static("is_enabled", &TraceMeWrapper::IsEnabled);
}

}  // namespace xla
