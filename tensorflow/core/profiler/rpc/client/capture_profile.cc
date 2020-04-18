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

#include <vector>

#include "grpcpp/grpcpp.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr uint64 kMaxEvents = 1000000;

ProfileRequest PopulateProfileRequest(int duration_ms,
                                      const string& repository_root,
                                      const string& session_id,
                                      const ProfileOptions& opts) {
  ProfileRequest request;
  request.set_duration_ms(duration_ms);
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root);
  request.set_session_id(session_id);
  request.add_tools("trace_viewer");
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("memory_viewer");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  request.add_tools("tensorflow_stats");
  *request.mutable_opts() = opts;
  return request;
}

inline Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? Status::OK()
                : Status(static_cast<error::Code>(s.error_code()),
                         s.error_message());
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

// Returns whether the returned trace is empty.
// Failure are handled by CHECK, i.e. abort()
Status Profile(const string& service_addr, const string& logdir,
               int duration_ms, const string& session_id,
               const ProfileOptions& opts) {
  ProfileRequest request =
      PopulateProfileRequest(duration_ms, logdir, session_id, opts);
  std::vector<string> parts = absl::StrSplit(service_addr, ':');
  request.set_host_name(parts[0]);

  ::grpc::ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(qiuminxu): use `NewHostPortGrpcChannel` instead once their
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      grpc::ProfilerService::NewStub(::grpc::CreateCustomChannel(
          "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
          channel_args));
  ProfileResponse response;
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Profile(&context, request, &response)));

  if (!response.empty_trace()) {
    TF_RETURN_IF_ERROR(SaveTensorboardProfile(
        logdir, session_id, request.host_name(), response, &std::cout));
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
Status NewSession(const string& service_addr, const string& repository_root,
                  const std::vector<string>& hostnames, int duration_ms,
                  const string& session_id, const ProfileOptions& opts) {
  NewProfileSessionRequest new_session_request;
  *new_session_request.mutable_request() =
      PopulateProfileRequest(duration_ms, repository_root, session_id, opts);
  new_session_request.set_repository_root(repository_root);
  new_session_request.set_session_id(session_id);
  for (const auto& hostname : hostnames) {
    new_session_request.add_hosts(hostname);
  }

  ::grpc::ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(qiuminxu): use `NewHostPortGrpcChannel` instead once their
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  // TODO(jiesun): GRPC support following relevant naming scheme:
  // 1. dns:///host:port
  // 2. ipv4:host:port or ipv6:[host]:port
  // We might need to change the prefix which depends on what cluster name
  // resolver will give us.
  std::unique_ptr<grpc::ProfileAnalysis::Stub> stub =
      grpc::ProfileAnalysis::NewStub(::grpc::CreateCustomChannel(
          "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
          channel_args));
  NewProfileSessionResponse new_session_response;
  TF_RETURN_IF_ERROR(FromGrpcStatus(
      stub->NewSession(&context, new_session_request, &new_session_response)));

  std::cout << "Profile session succeed for host(s):"
            << absl::StrJoin(hostnames, ",") << std::endl;
  if (new_session_response.empty_trace()) {
    return Status(error::Code::UNAVAILABLE, "No trace event is collected");
  }
  return Status::OK();
}

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level,
                                      bool timestamp) {
  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  request.set_timestamp(timestamp);
  return request;
}

}  // namespace

Status ValidateHostPortPair(const string& host_port) {
  uint32 port;
  std::vector<string> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      parts[0].find("/") != string::npos || parts[0].empty()) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}

// Starts tracing on a single or multiple hosts and saves the result in the
// given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts.
Status Trace(const string& service_addr, const string& logdir,
             const string& workers_list, int duration_ms,
             int num_tracing_attempts, const ProfileOptions& opts) {
  // Use the current timestamp as the run name.
  tensorflow::string session_id = GetCurrentTimeStampAsString();
  std::vector<string> hostnames;
  if (!workers_list.empty()) {
    hostnames = absl::StrSplit(workers_list, ',');
  }

  Status status = Status::OK();
  int remaining_attempts = num_tracing_attempts;
  while (true) {
    std::cout << "Starting to trace for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << --remaining_attempts << std::endl;
    if (hostnames.empty()) {
      status = Profile(service_addr, logdir, duration_ms, session_id, opts);
    } else {
      status = NewSession(service_addr, logdir, hostnames, duration_ms,
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

Status Monitor(const string& service_addr, int duration_ms,
               int monitoring_level, bool display_timestamp, string* result) {
  MonitorRequest request =
      PopulateMonitorRequest(duration_ms, monitoring_level, display_timestamp);

  ::grpc::ClientContext context;
  ::grpc::ChannelArguments channel_args;
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      grpc::ProfilerService::NewStub(::grpc::CreateCustomChannel(
          "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
          channel_args));
  MonitorResponse response;
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Monitor(&context, request, &response)));
  *result = response.data();
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
