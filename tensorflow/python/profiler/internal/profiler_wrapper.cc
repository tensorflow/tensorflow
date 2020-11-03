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

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

using ::tensorflow::RemoteProfilerSessionManagerOptions;

// Profiler gives grace after profiling duration to terminate.
constexpr absl::Duration kMinSessionGraceTime = absl::Seconds(60);

tensorflow::Status ValidateHostPortPair(absl::string_view host_port) {
  tensorflow::uint32 port;
  std::vector<absl::string_view> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      absl::StrContains(parts[0], "/") || parts[0].empty()) {
    return tensorflow::errors::InvalidArgument(
        "Could not interpret \"", host_port, "\" as a host-port pair.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateOptions(
    const RemoteProfilerSessionManagerOptions& options) {
  if (options.service_addresses().empty()) {
    return tensorflow::errors::InvalidArgument("No service address provided.");
  }

  if (options.profiler_options().duration_ms() == 0) {
    return tensorflow::errors::InvalidArgument(
        "duration_ms must be greater than zero.");
  }

  for (absl::string_view host_port : options.service_addresses()) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(host_port));
  }

  if (options.max_session_duration_ms() <
      options.profiler_options().duration_ms()) {
    return tensorflow::errors::InvalidArgument(
        "The maximum profiling session duration must be greater than or equal "
        "to the local profiler duration.");
  }

  return tensorflow::Status::OK();
}

// Receives a comma delimited list of service_addresses and adds them to
// RemoteProfilerSessionManagerOptions::service_addresses.
void AddServiceAddresses(absl::string_view service_addresses,
                         RemoteProfilerSessionManagerOptions* options) {
  for (absl::string_view server : absl::StrSplit(service_addresses, ',')) {
    options->add_service_addresses(server.data(), server.size());
  }
}

// Sets gRPC deadline to a grace period based on the profiling duration.
void UpdateMaxSessionDuration(RemoteProfilerSessionManagerOptions& options) {
  auto local_profiler_duration = options.profiler_options().duration_ms();
  auto session_creation_ts = options.session_creation_timestamp_ns();
  auto requested_start_ts = options.profiler_options().start_timestamp_ns();
  // User only needs to set maximal session duration if the profiling duration
  // is bounded.
  DCHECK_GT(local_profiler_duration, 0);
  VLOG(3) << "duration_ms was given as " << local_profiler_duration;
  // Max session duration is the profiling session with grace time.
  auto profile_duration = std::max(
      kMinSessionGraceTime, absl::Milliseconds(local_profiler_duration) * 2);
  absl::Duration delay_duration;
  // When requested start timestamp is 0, profiling starts immediately.
  if (requested_start_ts > 0) {
    delay_duration =
        absl::Nanoseconds(requested_start_ts - session_creation_ts);
  }

  auto max_session_duration = profile_duration + delay_duration;
  options.set_max_session_duration_ms(
      absl::ToInt64Milliseconds(max_session_duration));
  VLOG(1) << "max_session_duration set to " << max_session_duration;
}

// Takes profiler options in a py::dict and returns a
// RemoteProfilerSessionManagerOptions.
// This must be called under GIL because it reads Python objects. Reading Python
// objects require GIL because the objects can be mutated by other Python
// threads. In addition, Python objects are reference counted; reading py::dict
// will increase its reference count.
RemoteProfilerSessionManagerOptions GetOptionsLocked(absl::string_view logdir,
                                                     const py::dict& opts) {
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() =
      tensorflow::ProfilerSession::DefaultOptions();
  // Store a timestamp of when this session was created. This will be the basis
  // of gRPC deadline afterwards.
  auto now = absl::Now();
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(now));
  VLOG(2) << "set_session_creation_timestamp_ns set to "
          << options.session_creation_timestamp_ns() << " [" << now << "]";

  // Set the path of where to store XSpaces.
  options.mutable_profiler_options()->set_repository_path(logdir.data(),
                                                          logdir.size());
  VLOG(2) << "repository_path set to "
          << options.profiler_options().repository_path();

  for (const auto& kw : opts) {
    std::string key = py::cast<std::string>(kw.first);
    if (key == "host_tracer_level") {
      auto value = py::cast<int>(kw.second);
      options.mutable_profiler_options()->set_host_tracer_level(value);
      VLOG(1) << "host_tracer_level set to " << value;
    } else if (key == "device_tracer_level") {
      auto value = py::cast<int>(kw.second);
      options.mutable_profiler_options()->set_device_tracer_level(value);
      VLOG(1) << "device_tracer_level set to " << value;
    } else if (key == "python_tracer_level") {
      auto value = py::cast<int>(kw.second);
      options.mutable_profiler_options()->set_python_tracer_level(value);
      VLOG(1) << "python_tracer_level set to " << value;
    } else if (key == "delay_ms") {
      if (!kw.second.is_none()) {
        auto value = py::cast<int>(kw.second);
        options.set_delay_ms(value);
        VLOG(1) << "delay_ms was set to " << value;
      }
    } else {
      LOG(WARNING) << "Unrecognised key: " << key;
    }
  }

  return options;
}

RemoteProfilerSessionManagerOptions GetOptionsLocked(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    tensorflow::int32 duration_ms, py::dict opts, bool* is_cloud_tpu_session) {
  RemoteProfilerSessionManagerOptions options = GetOptionsLocked(logdir, opts);

  // Remote profiling does not support any use cases where the following options
  // are set by `py::dict opts`. e.g. `opts['service_addrs']` will not happen.
  DCHECK(options.service_addresses().empty());
  // In remote profiling, duration is always passed by value explicitly and not
  // set in py::dict opts.
  DCHECK_EQ(options.profiler_options().duration_ms(), 0);
  // Because duration_ms is not set from py::dict opts, it follows that
  // max_session_duration_ms must be unset as well.
  DCHECK_EQ(options.max_session_duration_ms(), 0);

  // Worker_list is only used for TensorBoard TPU capture cases. For a TPU
  // cluster, service_address is the Master, which can already be found in the
  // list of workers. These sessions will be used with the ProfileAnalysis
  // service.
  *is_cloud_tpu_session = !worker_list.empty();
  AddServiceAddresses(*is_cloud_tpu_session ? worker_list : service_addresses,
                      &options);

  // Set local profiler duration and profiler session durations.
  options.mutable_profiler_options()->set_include_dataset_ops(
      include_dataset_ops);
  options.mutable_profiler_options()->set_duration_ms(duration_ms);
  UpdateMaxSessionDuration(options);

  for (int idx = 0; idx < options.service_addresses_size(); ++idx) {
    VLOG(1) << "service_addr " << idx << " set to "
            << options.service_addresses(idx);
  }
  VLOG(1) << "include_dataset_ops set to " << include_dataset_ops;
  VLOG(1) << "duration_ms set to " << duration_ms;

  return options;
}

class ProfilerSessionWrapper {
 public:
  void Start(const char* logdir, const py::dict& options) {
    auto opts = GetOptionsLocked(logdir, options);
    session_ = tensorflow::ProfilerSession::Create(opts.profiler_options());
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
    xspace.add_hostnames(tensorflow::port::Hostname());
    session_.reset();
    status = tensorflow::profiler::ExportToTensorBoard(xspace, logdir_);
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
    auto profiler_server =
        absl::make_unique<tensorflow::profiler::ProfilerServer>();
    profiler_server->StartProfilerServer(port);
    // Intentionally release profiler server. Should transfer ownership to
    // caller instead.
    profiler_server.release();
  });

  m.def("trace", [](const char* service_addr, const char* logdir,
                    const char* worker_list, bool include_dataset_ops,
                    int duration_ms, int num_tracing_attempts,
                    py::dict options) {
    // TPU capture is true if the user sets worker_list.
    bool is_cloud_tpu_session = false;
    // Normalize py::dict into a well defined and validated proto.
    tensorflow::RemoteProfilerSessionManagerOptions opts =
        GetOptionsLocked(service_addr, logdir, worker_list, include_dataset_ops,
                         duration_ms, options, &is_cloud_tpu_session);
    tensorflow::Status status = ValidateOptions(opts);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);

    {
      // Release the lock to keep the lock scope to a minimum, and allow
      // other threads to proceed.
      py::gil_scoped_release release;
      status = tensorflow::profiler::Trace(logdir, num_tracing_attempts, opts,
                                           is_cloud_tpu_session);
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });

  m.def("monitor", [](const char* service_addr, int duration_ms,
                      int monitoring_level, bool display_timestamp) {
    tensorflow::Status status = ValidateHostPortPair(service_addr);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    tensorflow::string content;
    {
      // Release the lock to keep the lock scope to a minimum, and allow
      // other threads to proceed.
      py::gil_scoped_release release;
      status = tensorflow::profiler::Monitor(service_addr, duration_ms,
                                             monitoring_level,
                                             display_timestamp, &content);
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return content;
  });

  m.def("xspace_to_tools_data",
        [](const py::list& xspace_path_list, const py::str& py_tool_name) {
          std::vector<std::string> xspace_paths;
          for (py::handle obj : xspace_path_list) {
            xspace_paths.push_back(std::string(py::cast<py::str>(obj)));
          }
          std::string tool_name = std::string(py_tool_name);
          auto tool_data_and_success =
              tensorflow::profiler::ConvertMultiXSpacesToToolData(xspace_paths,
                                                                  tool_name);
          return py::make_tuple(py::bytes(tool_data_and_success.first),
                                py::bool_(tool_data_and_success.second));
        });
};
