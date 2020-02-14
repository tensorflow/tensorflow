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
#include "include/pybind11/pybind11.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

class ProfilerSessionWrapper {
 public:
  void Start() {
    session_ = tensorflow::ProfilerSession::Create();
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

 private:
  std::unique_ptr<tensorflow::ProfilerSession> session_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_profiler, m) {
  py::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class.def(py::init<>())
      .def("start", &ProfilerSessionWrapper::Start)
      .def("stop", &ProfilerSessionWrapper::Stop);

  m.def("start_profiler_server", [](int port) {
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
