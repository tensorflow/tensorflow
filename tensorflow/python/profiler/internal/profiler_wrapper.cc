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

#include "absl/memory/memory.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

tensorflow::ProfileRequest MakeProfileRequest(
    const tensorflow::string& logdir, const tensorflow::string& session_id,
    const tensorflow::string& host) {
  tensorflow::ProfileRequest request;
  request.add_tools("trace_viewer");
  request.add_tools("overview_page");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("tensorflow_stats");
  request.set_host_name(host);
  request.set_repository_root(logdir);
  request.set_session_id(session_id);
  return request;
}

tensorflow::ProfileOptions GetOptions(const py::dict& opts) {
  tensorflow::ProfileOptions options =
      tensorflow::ProfilerSession::DefaultOptions();
  for (const auto& kw : opts) {
    std::string key = py::cast<std::string>(kw.first);
    if (key == "host_tracer_level") {
      options.set_host_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "host_tracer_level set to " << options.host_tracer_level();
    } else if (key == "device_tracer_level") {
      options.set_device_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "device_tracer_level set to " << options.device_tracer_level();
    } else if (key == "python_tracer_level") {
      options.set_python_tracer_level(py::cast<int>(kw.second));
      VLOG(1) << "python_tracer_level set to " << options.python_tracer_level();
    }
  }
  return options;
}

class ProfilerSessionWrapper {
 public:
  void Start(const char* logdir, const py::dict& options) {
    session_ = tensorflow::ProfilerSession::Create(GetOptions(options));
    logdir_ = logdir;
    tensorflow::MaybeRaiseRegisteredFromStatus(session_->Status());
  }

  py::bytes Stop() {
    tensorflow::string content;
    if (session_ != nullptr) {
      tensorflow::profiler::XSpace xspace;
      tensorflow::Status status = session_->CollectData(&xspace);
      session_.reset();
      tensorflow::profiler::ConvertXSpaceToTraceEventsString(xspace, &content);
      tensorflow::MaybeRaiseRegisteredFromStatus(status);
    }
    // The content is not valid UTF-8, so it must be converted to bytes.
    return py::bytes(content);
  }

  void ExportToTensorBoard() {
    if (!session_ || logdir_.empty()) return;
    tensorflow::profiler::XSpace xspace;
    tensorflow::Status status;
    status = session_->CollectData(&xspace);
    session_.reset();
    tensorflow::MaybeRaiseRegisteredFromStatus(status);

    tensorflow::ProfileResponse response;
    tensorflow::ProfileRequest request = MakeProfileRequest(
        logdir_, tensorflow::profiler::GetCurrentTimeStampAsString(),
        tensorflow::port::Hostname());
    status = tensorflow::profiler::ConvertXSpaceToProfileResponse(
        xspace, request, &response);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);

    std::stringstream ss;  // Record LOG messages.
    status = tensorflow::profiler::SaveTensorboardProfile(
        request.repository_root(), request.session_id(), request.host_name(),
        response, &ss);
    LOG(INFO) << ss.str();
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  }

 private:
  std::unique_ptr<tensorflow::ProfilerSession> session_;
  tensorflow::string logdir_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_profiler, m) {
  py::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class.def(py::init<>())
      .def("start", &ProfilerSessionWrapper::Start)
      .def("stop", &ProfilerSessionWrapper::Stop)
      .def("export_to_tb", &ProfilerSessionWrapper::ExportToTensorBoard);

  m.def("start_server", [](int port) {
    auto profiler_server = absl::make_unique<tensorflow::ProfilerServer>();
    profiler_server->StartProfilerServer(port);
    // Intentionally release profiler server. Should transfer ownership to
    // caller instead.
    profiler_server.release();
  });

  m.def("trace", [](const char* service_addr, const char* logdir,
                    const char* worker_list, bool include_dataset_ops,
                    int duration_ms, int num_tracing_attempts,
                    py::dict options) {
    tensorflow::Status status =
        tensorflow::profiler::ValidateHostPortPair(service_addr);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    tensorflow::ProfileOptions opts = GetOptions(options);
    opts.set_include_dataset_ops(include_dataset_ops);
    status =
        tensorflow::profiler::Trace(service_addr, logdir, worker_list,
                                    duration_ms, num_tracing_attempts, opts);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    tensorflow::Status status =
        tensorflow::profiler::ValidateHostPortPair(service_addr);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    tensorflow::string content;
    status = tensorflow::profiler::Monitor(service_addr, duration_ms,
                                           monitoring_level, display_timestamp,
                                           &content);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return content;
  });
};
