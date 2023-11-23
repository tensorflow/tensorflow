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

#include "xla/python/profiler.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/python/exceptions.h"
#include "xla/python/profiler/internal/traceme_wrapper.h"
#include "xla/python/status_casters.h"
#include "xla/python/types.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "xla/status.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/rpc/client/capture_profile.h"
#include "tsl/profiler/rpc/profiler_server.h"

namespace xla {

namespace py = pybind11;

namespace {
// Adds a trivial forwarding class so these Python bindings and TensorFlow's
// bindings of the same thing don't register the same class with pybind11.
class TraceMeWrapper : public xla::profiler::TraceMeWrapper {
 public:
  using xla::profiler::TraceMeWrapper::TraceMeWrapper;
};

tensorflow::ProfileOptions DefaultPythonProfileOptions() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return options;
}

const PLUGIN_Profiler_Api* FindProfilerApi(const PJRT_Api* pjrt_api) {
  const PJRT_Structure_Base* next =
      reinterpret_cast<const PJRT_Structure_Base*>(pjrt_api->extension_start);
  while (next != nullptr &&
         next->type != PJRT_Structure_Type::PJRT_Structure_Type_Profiler) {
    next = next->next;
  }
  if (next == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<const PJRT_Profiler_Extension*>(next)->profiler_api;
}
}  // namespace

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler =
      m->def_submodule("profiler", "TensorFlow profiler integration");
  py::class_<tsl::profiler::ProfilerServer,
             std::unique_ptr<tsl::profiler::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def(
      "start_server",
      [](int port) -> std::unique_ptr<tsl::profiler::ProfilerServer> {
        auto server = std::make_unique<tsl::profiler::ProfilerServer>();
        server->StartProfilerServer(port);
        return server;
      },
      py::arg("port"));
  profiler.def("register_plugin_profiler", [](py::capsule c_api) -> void {
    if (absl::string_view(c_api.name()) != "pjrt_c_api") {
      throw xla::XlaRuntimeError(
          "Argument to register_plugin_profiler was not a pjrt_c_api capsule.");
    }
    const PLUGIN_Profiler_Api* profiler_api =
        FindProfilerApi(static_cast<const PJRT_Api*>(c_api));
    std::function<std::unique_ptr<tsl::profiler::ProfilerInterface>(
        const tensorflow::ProfileOptions&)>
        create_func = [profiler_api = profiler_api](
                          const tensorflow::ProfileOptions& options) mutable {
          return std::make_unique<xla::profiler::PluginTracer>(profiler_api,
                                                               options);
        };
    tsl::profiler::RegisterProfilerFactory(std::move(create_func));
  });

  py::class_<tsl::ProfilerSession> profiler_session_class(profiler,
                                                          "ProfilerSession");
  profiler_session_class
      .def(py::init([]() {
        return tsl::ProfilerSession::Create(DefaultPythonProfileOptions());
      }))
      .def(py::init([](const tensorflow::ProfileOptions& options) {
        return tsl::ProfilerSession::Create(options);
      }))
      .def("stop_and_export",
           [](tsl::ProfilerSession* sess,
              const std::string& tensorboard_dir) -> void {
             tensorflow::profiler::XSpace xspace;
             // Disables the ProfilerSession
             xla::ThrowIfError(sess->CollectData(&xspace));
             xla::ThrowIfError(tsl::profiler::ExportToTensorBoard(
                 xspace, tensorboard_dir, /* also_export_trace_json= */ true));
           })
      .def("stop",
           [](tsl::ProfilerSession* sess) -> pybind11::bytes {
             tensorflow::profiler::XSpace xspace;
             // Disables the ProfilerSession
             xla::ThrowIfError(sess->CollectData(&xspace));
             return xspace.SerializeAsString();
           })
      .def("export",
           [](tsl::ProfilerSession* sess, const std::string& xspace,
              const std::string& tensorboard_dir) -> void {
             tensorflow::profiler::XSpace xspace_proto;
             xspace_proto.ParseFromString(xspace);
             xla::ThrowIfError(tsl::profiler::ExportToTensorBoard(
                 xspace_proto, tensorboard_dir,
                 /* also_export_trace_json= */ true));
           });

  py::class_<tensorflow::ProfileOptions> profile_options_class(
      profiler, "ProfileOptions");
  profile_options_class.def(py::init(&DefaultPythonProfileOptions))
      .def_property("include_dataset_ops",
                    &tensorflow::ProfileOptions::include_dataset_ops,
                    &tensorflow::ProfileOptions::set_include_dataset_ops)
      .def_property("host_tracer_level",
                    &tensorflow::ProfileOptions::host_tracer_level,
                    &tensorflow::ProfileOptions::set_host_tracer_level)
      .def_property("python_tracer_level",
                    &tensorflow::ProfileOptions::python_tracer_level,
                    &tensorflow::ProfileOptions::set_python_tracer_level)
      .def_property("enable_hlo_proto",
                    &tensorflow::ProfileOptions::enable_hlo_proto,
                    &tensorflow::ProfileOptions::set_enable_hlo_proto)
      .def_property("start_timestamp_ns",
                    &tensorflow::ProfileOptions::start_timestamp_ns,
                    &tensorflow::ProfileOptions::set_start_timestamp_ns)
      .def_property("duration_ms", &tensorflow::ProfileOptions::duration_ms,
                    &tensorflow::ProfileOptions::set_duration_ms)
      .def_property(
          "repository_path", &tensorflow::ProfileOptions::repository_path,
          [](tensorflow::ProfileOptions* options, const std::string& path) {
            options->set_repository_path(path);
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

  profiler.def(
      "get_profiled_instructions_proto",
      [](py::str tensorboard_dir) -> pybind11::bytes {
        tensorflow::profiler::ProfiledInstructionsProto profile_proto;
        xla::ThrowIfError(
            xla::ConvertXplaneUnderLogdirToProfiledInstructionsProto(
                tensorboard_dir, &profile_proto));
        return profile_proto.SerializeAsString();
      },
      py::arg("tensorboard_dir"));

  profiler.def(
      "get_fdo_profile", [](const std::string& xspace) -> pybind11::bytes {
        tensorflow::profiler::XSpace xspace_proto;
        xspace_proto.ParseFromString(xspace);
        tensorflow::profiler::ProfiledInstructionsProto fdo_profile;
        xla::ThrowIfError(xla::ConvertXplaneToProfiledInstructionsProto(
            {xspace_proto}, &fdo_profile));
        return fdo_profile.SerializeAsString();
      });
}

}  // namespace xla
