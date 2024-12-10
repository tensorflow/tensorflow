/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
#include "xla/tsl/profiler/rpc/client/capture_profile.h"

#include <iostream>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"
#include "xla/tsl/profiler/rpc/client/remote_profiler_session_manager.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "xla/tsl/profiler/utils/session_manager.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/profiler_analysis.pb.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::profiler::RemoteProfilerSessionManager;
using Response = ::tsl::profiler::RemoteProfilerSessionManager::Response;

using tensorflow::MonitorRequest;
using tensorflow::MonitorResponse;
using tensorflow::NewProfileSessionRequest;
using tensorflow::NewProfileSessionResponse;
using tensorflow::ProfileRequest;
using tensorflow::ProfileResponse;
using tensorflow::RemoteProfilerSessionManagerOptions;
using tensorflow::profiler::XSpace;

constexpr uint64 kMaxEvents = 1000000;
const absl::string_view kXPlanePb = "xplane.pb";

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level,
                                      bool timestamp) {
  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  request.set_timestamp(timestamp);
  return request;
}

ProfileRequest PopulateProfileRequest(
    absl::string_view repository_root, absl::string_view session_id,
    absl::string_view host_name,
    const RemoteProfilerSessionManagerOptions& options) {
  ProfileRequest request;
  // TODO(b/169976117) Remove duration from request.
  request.set_duration_ms(options.profiler_options().duration_ms());
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  request.set_host_name(host_name.data(), host_name.size());
  // These tools are only used by TPU profiler.
  request.add_tools("trace_viewer");
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("memory_viewer");
  request.add_tools("memory_profile");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  request.add_tools("tensorflow_stats");
  // XPlane tool is only used by OSS profiler and safely ignored by TPU
  // profiler.
  request.add_tools(kXPlanePb.data(), kXPlanePb.size());
  *request.mutable_opts() = options.profiler_options();
  return request;
}

NewProfileSessionRequest PopulateNewProfileSessionRequest(
    absl::string_view repository_root, absl::string_view session_id,
    const RemoteProfilerSessionManagerOptions& opts) {
  NewProfileSessionRequest request;
  std::vector<absl::string_view> parts =
      absl::StrSplit(opts.service_addresses(0), ':');
  DCHECK(!parts.empty());

  *request.mutable_request() =
      PopulateProfileRequest(repository_root, session_id, parts[0], opts);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  for (const auto& hostname : opts.service_addresses()) {
    request.add_hosts(hostname);
  }
  return request;
}

inline bool ShouldRetryTracing(absl::Status status) {
  return status.code() == error::Code::UNAVAILABLE ||
         status.code() == error::Code::ALREADY_EXISTS ||
         // When auto-reconnecting to a remote TensorFlow worker after it
         // restarts, gRPC can return an UNKNOWN error code with a "Stream
         // removed" error message. This should not be treated as an
         // unrecoverable error.
         (status.code() == error::Code::UNKNOWN &&
          status.message() == "Stream removed");
}

absl::Status Profile(const std::string& repository_root,
                     const std::string& session_id,
                     const RemoteProfilerSessionManagerOptions& opts) {
  absl::Status status;
  // Host name will be overwritten by RemoteProfilerSessionManager later.
  ProfileRequest request = PopulateProfileRequest(repository_root, session_id,
                                                  /*host_name=*/"", opts);
  auto session = RemoteProfilerSessionManager::Create(opts, request, status);
  TF_RETURN_IF_ERROR(status);
  // Expect one or more service addresses.
  DCHECK_GT(opts.service_addresses_size(), 0);
  std::vector<Response> responses = session->WaitForCompletion();
  // Expect responses to have the same size as clients.
  DCHECK_EQ(responses.size(), opts.service_addresses_size());

  bool has_trace_data = false;
  for (const auto& client_response : responses) {
    ProfileResponse& response = *client_response.profile_response;
    if (response.empty_trace()) {
      LOG(WARNING) << "No trace event is collected from "
                   << client_response.service_address;
    } else {
      has_trace_data = true;
      // If server side returns tool data in the response, saves that into the
      // repository. This improves backward compatibility by reducing assumption
      // of what server side does.
      TF_RETURN_IF_ERROR(SaveProfile(repository_root, session_id,
                                     client_response.service_address, response,
                                     &std::cout));
    }
    if (!client_response.status.ok()) {
      LOG(WARNING) << client_response.service_address << " returned "
                   << client_response.status;
    }
  }

  if (!has_trace_data) {
    return absl::Status(
        absl::StatusCode::kUnavailable,
        "No trace event was collected because there were no responses"
        " from clients or the responses did not have trace data.");
  }
  return absl::OkStatus();
}

// Start a new profiling session that include all the hosts included in
// hostnames, for the time interval of duration_ms. Possibly save the profiling
// result in the directory specified by repository_root and session_id.
absl::Status NewSession(absl::string_view repository_root,
                        absl::string_view session_id,
                        const RemoteProfilerSessionManagerOptions& opts) {
  NewProfileSessionRequest request =
      PopulateNewProfileSessionRequest(repository_root, session_id, opts);
  NewProfileSessionResponse response;
  TF_RETURN_IF_ERROR(
      NewSessionGrpc(opts.service_addresses(0), request, &response));

  std::cout << "Profile session succeed for host(s):"
            << absl::StrJoin(opts.service_addresses(), ",") << std::endl;
  if (response.empty_trace()) {
    return errors::Unavailable("No trace event is collected");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status CaptureRemoteTrace(const std::string& logdir,
                                int num_tracing_attempts,
                                RemoteProfilerSessionManagerOptions& opts,
                                bool is_cloud_tpu_session) {
  DCHECK_GT(opts.profiler_options().duration_ms(), 0);
  DCHECK(!opts.service_addresses().empty());

  // Use the current timestamp as the run name.
  std::string session_id = GetCurrentTimeStampAsString();
  std::string repository_root = GetTensorBoardProfilePluginDir(logdir);
  auto duration_ms = opts.profiler_options().duration_ms();

  absl::Status status;
  int remaining_attempts = num_tracing_attempts;
  while (true) {
    auto start_timestamp = absl::Now() + absl::Milliseconds(opts.delay_ms());
    opts.mutable_profiler_options()->set_start_timestamp_ns(
        absl::ToUnixNanos(start_timestamp));
    LOG(INFO) << "Profiler delay_ms was " << opts.delay_ms()
              << ", start_timestamp_ns set to "
              << opts.profiler_options().start_timestamp_ns() << " ["
              << start_timestamp << "]";

    std::cout << "Starting to trace for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << --remaining_attempts << std::endl;

    if (is_cloud_tpu_session) {
      status = NewSession(repository_root, session_id, opts);
    } else {
      status = Profile(repository_root, session_id, opts);
    }
    if (remaining_attempts <= 0 || status.ok() || !ShouldRetryTracing(status))
      break;
    std::cout << "No trace event is collected. Automatically retrying.\n"
              << std::endl;
  }

  if (ShouldRetryTracing(status)) {
    std::cout << "No trace event is collected after " << num_tracing_attempts
              << " attempt(s). "
              << "Perhaps, you want to try again (with more attempts?).\n"
              << "Tip: increase number of attempts with --num_tracing_attempts."
              << std::endl;
  }
  return status;
}

absl::Status Monitor(const std::string& service_addr, int duration_ms,
                     int monitoring_level, bool display_timestamp,
                     std::string* result) {
  MonitorRequest request =
      PopulateMonitorRequest(duration_ms, monitoring_level, display_timestamp);
  MonitorResponse response;
  TF_RETURN_IF_ERROR(MonitorGrpc(service_addr, request, &response));
  *result = response.data();
  return absl::OkStatus();
}

absl::Status ExportToTensorBoard(const XSpace& xspace,
                                 const std::string& logdir,
                                 bool also_export_trace_json) {
  std::string repository_root =
      tsl::profiler::GetTensorBoardProfilePluginDir(logdir);
  std::string run = tsl::profiler::GetCurrentTimeStampAsString();
  std::string host = tsl::port::Hostname();
  TF_RETURN_IF_ERROR(
      tsl::profiler::SaveXSpace(repository_root, run, host, xspace));
  if (also_export_trace_json) {
    tsl::profiler::TraceContainer container =
        tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
    return tsl::profiler::SaveGzippedToolData(
        repository_root, run, host, "trace.json.gz",
        tsl::profiler::TraceContainerToJson(container));
  }
  return absl::OkStatus();
}

absl::Status CaptureRemoteTrace(
    const char* service_addr, const char* logdir, const char* worker_list,
    bool include_dataset_ops, int duration_ms, int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options) {
  // TPU capture is true if the user sets worker_list.
  bool is_cloud_tpu_session = false;
  RemoteProfilerSessionManagerOptions opts =
      GetRemoteSessionManagerOptionsLocked(service_addr, logdir, worker_list,
                                           include_dataset_ops, duration_ms,
                                           options, &is_cloud_tpu_session);
  TF_RETURN_IF_ERROR(ValidateRemoteProfilerSessionManagerOptions(opts));

  {
    TF_RETURN_IF_ERROR(CaptureRemoteTrace(logdir, num_tracing_attempts, opts,
                                          is_cloud_tpu_session));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl
