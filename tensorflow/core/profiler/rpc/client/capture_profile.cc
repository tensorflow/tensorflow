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
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/profiler_analysis.pb.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"

namespace tensorflow {
namespace profiler {
namespace {

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

ProfileRequest PopulateProfileRequest(int duration_ms,
                                      const std::string& repository_root,
                                      const std::string& session_id,
                                      const std::string& host_name,
                                      const ProfileOptions& opts) {
  ProfileRequest request;
  request.set_duration_ms(duration_ms);
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root);
  request.set_session_id(session_id);
  request.set_host_name(host_name);
  request.add_tools("trace_viewer");
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("memory_viewer");
  request.add_tools("memory_profile");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  request.add_tools("tensorflow_stats");
  *request.mutable_opts() = opts;
  return request;
}

NewProfileSessionRequest PopulateNewProfileSessionRequest(
    const std::string& service_addr, const std::string& repository_root,
    const std::vector<string>& hostnames, int duration_ms,
    const std::string& session_id, const ProfileOptions& opts) {
  NewProfileSessionRequest request;
  std::vector<std::string> parts = absl::StrSplit(service_addr, ':');
  *request.mutable_request() = PopulateProfileRequest(
      duration_ms, repository_root, session_id, parts[0], opts);
  request.set_repository_root(repository_root);
  request.set_session_id(session_id);
  for (const auto& hostname : hostnames) {
    request.add_hosts(hostname);
  }
  return request;
}

inline bool ShouldRetryTracing(Status status) {
  return status.code() == error::Code::UNAVAILABLE ||
         status.code() == error::Code::ALREADY_EXISTS ||
         // When auto-reconnecting to a remote TensorFlow worker after it
         // restarts, gRPC can return an UNKNOWN error code with a "Stream
         // removed" error message. This should not be treated as an
         // unrecoverable error.
         (status.code() == error::Code::UNKNOWN &&
          status.error_message() == "Stream removed");
}

// If the ProfileResponse has single 'xplane.pb' tool, convert the xplane to
// other tools and add in ProfileResponse. Otherwise, the ProfileResponse is
// already converted, simply return.
Status ConvertXSpaceToToolsInProfileResponse(const ProfileRequest& request,
                                             ProfileResponse* response) {
  if (response->tool_data_size() != 1) return Status::OK();
  if (response->tool_data(0).name() != kXPlanePb) return Status::OK();
  XSpace xspace;
  xspace.ParseFromString(response->tool_data(0).data());
  TF_RETURN_IF_ERROR(ConvertXSpaceToProfileResponse(xspace, request, response));
  return Status::OK();
}

Status Profile(const std::string& service_addr,
               const std::string& repository_root, int duration_ms,
               const std::string& session_id, const ProfileOptions& opts) {
  std::vector<std::string> parts = absl::StrSplit(service_addr, ':');
  ProfileRequest request = PopulateProfileRequest(duration_ms, repository_root,
                                                  session_id, parts[0], opts);
  ProfileResponse response;
  TF_RETURN_IF_ERROR(ProfileGrpc(service_addr, request, &response));

  if (!response.empty_trace()) {
    TF_RETURN_IF_ERROR(
        ConvertXSpaceToToolsInProfileResponse(request, &response));
    TF_RETURN_IF_ERROR(SaveProfile(repository_root, session_id,
                                   request.host_name(), response, &std::cout));
    // Print this at the end so that it's not buried in irrelevant LOG messages.
    std::cout
        << "NOTE: using the trace duration " << duration_ms << "ms.\n"
        << "Set an appropriate duration (with --duration_ms) if you "
           "don't see a full step in your trace or the captured trace is too "
           "large."
        << std::endl;
  }

  if (response.empty_trace()) {
    return Status(error::Code::UNAVAILABLE, "No trace event is collected");
  }
  return Status::OK();
}

// Start a new profiling session that include all the hosts included in
// hostnames, for the time interval of duration_ms. Possibly save the profiling
// result in the directory specified by repository_root and session_id.
Status NewSession(const std::string& service_addr,
                  const std::string& repository_root,
                  const std::vector<string>& hostnames, int duration_ms,
                  const std::string& session_id, const ProfileOptions& opts) {
  NewProfileSessionRequest request = PopulateNewProfileSessionRequest(
      service_addr, repository_root, hostnames, duration_ms, session_id, opts);
  NewProfileSessionResponse response;
  TF_RETURN_IF_ERROR(NewSessionGrpc(service_addr, request, &response));

  std::cout << "Profile session succeed for host(s):"
            << absl::StrJoin(hostnames, ",") << std::endl;
  if (response.empty_trace()) {
    return Status(error::Code::UNAVAILABLE, "No trace event is collected");
  }
  return Status::OK();
}

}  // namespace

// Starts tracing on a single or multiple hosts and saves the result in the
// given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts.
Status Trace(const std::string& service_addr, const std::string& logdir,
             const std::string& workers_list, int duration_ms,
             int num_tracing_attempts, const ProfileOptions& opts) {
  // Use the current timestamp as the run name.
  std::string session_id = GetCurrentTimeStampAsString();
  std::vector<std::string> hostnames;
  if (!workers_list.empty()) {
    hostnames = absl::StrSplit(workers_list, ',');
  }
  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));
  std::string repository_root =
      profiler::GetTensorBoardProfilePluginDir(logdir);

  Status status = Status::OK();
  int remaining_attempts = num_tracing_attempts;
  while (true) {
    std::cout << "Starting to trace for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << --remaining_attempts << std::endl;
    if (hostnames.empty()) {
      status =
          Profile(service_addr, repository_root, duration_ms, session_id, opts);
    } else {
      status = NewSession(service_addr, repository_root, hostnames, duration_ms,
                          session_id, opts);
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

Status Monitor(const std::string& service_addr, int duration_ms,
               int monitoring_level, bool display_timestamp,
               std::string* result) {
  MonitorRequest request =
      PopulateMonitorRequest(duration_ms, monitoring_level, display_timestamp);
  MonitorResponse response;
  TF_RETURN_IF_ERROR(MonitorGrpc(service_addr, request, &response));
  *result = response.data();
  return Status::OK();
}

Status ExportToTensorBoard(const XSpace& xspace, const std::string& logdir) {
  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));

  ProfileResponse response;
  ProfileRequest request = PopulateProfileRequest(
      /*duration_ms=*/0, GetTensorBoardProfilePluginDir(logdir),
      GetCurrentTimeStampAsString(), port::Hostname(), /*opts=*/{});
  TF_RETURN_IF_ERROR(
      ConvertXSpaceToProfileResponse(xspace, request, &response));

  std::stringstream ss;  // Record LOG messages.
  TF_RETURN_IF_ERROR(SaveProfile(request.repository_root(),
                                 request.session_id(), request.host_name(),
                                 response, &ss));
  LOG(INFO) << ss.str();
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
