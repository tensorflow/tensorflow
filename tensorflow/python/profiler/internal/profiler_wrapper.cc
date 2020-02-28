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
#include "absl/time/time.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

tensorflow::string GetCurrentTimeStampAsString() {
  return absl::FormatTime("%E4Y-%m-%d_%H:%M:%S", absl::Now(),
                          absl::LocalTimeZone());
}

tensorflow::ProfileRequest MakeProfileRequest() {
  tensorflow::ProfileRequest request;
  request.add_tools("overview_page");
  request.add_tools("input_pipeline");
  request.add_tools("tensorflow_stats");
  return request;
}

class ProfilerSessionWrapper {
 public:
  void Start(const char* logdir) {
    session_ = tensorflow::ProfilerSession::Create();
    logdir_ = logdir;
    tensorflow::MaybeRaiseRegisteredFromStatus(session_->Status());
  }

  py::bytes Stop() {
    tensorflow::string content;
    if (session_ != nullptr) {
      tensorflow::Status status = session_->SerializeToString(&content);
      session_.reset();
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
    if (!status.ok()) {
      tensorflow::MaybeRaiseRegisteredFromStatus(status);
      return;
    }
    tensorflow::ProfileResponse response;
    tensorflow::profiler::ConvertXSpaceToProfileResponse(
        xspace, MakeProfileRequest(), &response);

    std::stringstream ss;  // Record LOG messages.
    status = tensorflow::profiler::SaveTensorboardProfile(
        logdir_, GetCurrentTimeStampAsString(), tensorflow::port::Hostname(),
        response, &ss);
    LOG(INFO) << ss.str();
    tensorflow::MaybeRaiseRegisteredFromStatus(tensorflow::Status::OK());
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
                    int duration_ms, int num_tracing_attempts) {
    tensorflow::Status status =
        tensorflow::profiler::ValidateHostPortPair(service_addr);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    status = tensorflow::profiler::Trace(service_addr, logdir, worker_list,
                                         include_dataset_ops, duration_ms,
                                         num_tracing_attempts);
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
