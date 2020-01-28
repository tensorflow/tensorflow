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

#include <cstdio>
#include <ctime>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace profiler {
namespace client {

constexpr uint64 kMaxEvents = 1000000;

string GetCurrentTimeStampAsString() {
  char s[128];
  std::time_t t = std::time(nullptr);
  auto result = std::strftime(s, sizeof(s), "%F_%T", std::localtime(&t));
  DCHECK_NE(result, 0);
  return s;
}

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

ProfileRequest PopulateProfileRequest(int duration_ms,
                                      const string& repository_root,
                                      const string& session_id,
                                      const ProfileOptions& opts) {
  ProfileRequest request;
  request.set_duration_ms(duration_ms);
  request.set_max_events(kMaxEvents);
  if (absl::StartsWith(repository_root, "gs://")) {
    // For backward compatibilities, only generate tracetable etc when the
    // user provide a GCS path for model directory.
    request.set_repository_root(repository_root);
    request.set_session_id(session_id);
  }
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("memory_viewer");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  *request.mutable_opts() = opts;
  return request;
}

bool ShouldRetryTracing(Status status) {
  return status.code() == error::Code::UNAVAILABLE ||
         status.code() == error::Code::ALREADY_EXISTS;
}

// Returns whether the returned trace is empty.
// Failure are handled by CHECK, i.e. abort()
Status Profile(const string& service_addr, const string& logdir,
               int duration_ms, const string& repository_root,
               const string& session_id, const ProfileOptions& opts) {
  ProfileRequest request =
      PopulateProfileRequest(duration_ms, repository_root, session_id, opts);

  ::grpc::ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(qiuminxu): use `NewHostPortGrpcChannel` instead once their
  // `ValidateHostPortPair` checks for empty host string case.
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      grpc::ProfilerService::NewStub(::grpc::CreateCustomChannel(
          "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
          channel_args));
  ProfileResponse response;
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Profile(&context, request, &response)));

  if (!response.encoded_trace().empty()) {
    TF_CHECK_OK(
        SaveTensorboardProfile(logdir, session_id, "", response, &std::cout));
    // Print this at the end so that it's not buried in irrelevant LOG messages.
    std::cout
        << "NOTE: using the trace duration " << duration_ms << "ms.\n"
        << "Set an appropriate duration (with --duration_ms) if you "
           "don't see a full step in your trace or the captured trace is too "
           "large."
        << std::endl;
  }

  if (response.encoded_trace().empty()) {
    return Status(tensorflow::error::Code::UNAVAILABLE,
                  "No trace event is collected");
  }
  return Status::OK();
}

// Start a new profiling session that include all the hosts included in
// hostnames, for the time interval of duration_ms. Possibly save the profiling
// result in the directory specified by repository_root and session_id.
Status NewSession(const string& service_addr,
                  const std::vector<tensorflow::string>& hostnames,
                  int duration_ms, const string& repository_root,
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
  // `ValidateHostPortPair` checks for empty host string case.
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  // TODO(jiesun): GRPC support following relevant naming scheme:
  // 1. dns:///host:port
  // 2. ipv4:host:port or ipv6:[host]:port
  // We might need to change the prefix which depends on what TPU name resolver
  // will give us.
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
    return Status(tensorflow::error::Code::UNAVAILABLE,
                  "No trace event is collected");
  }
  return Status::OK();
}

// Creates an empty event file if not already exists, which indicates that we
// have a plugins/profile/ directory in the current logdir.
Status MaybeCreateEmptyEventFile(const tensorflow::string& logdir) {
  // Suffix for an empty event file.  it should be kept in sync with
  // _EVENT_FILE_SUFFIX in tensorflow/python/eager/profiler.py.
  constexpr char kProfileEmptySuffix[] = ".profile-empty";
  std::vector<string> children;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(logdir, &children));
  for (const string& child : children) {
    if (absl::EndsWith(child, kProfileEmptySuffix)) {
      return Status::OK();
    }
  }
  EventsWriter event_writer(io::JoinPath(logdir, "events"));
  return event_writer.InitWithSuffix(kProfileEmptySuffix);
}

// Starts tracing on a single or multiple TPU hosts and saves the result in the
// given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts.
Status StartTracing(const tensorflow::string& service_addr,
                    const tensorflow::string& logdir,
                    const tensorflow::string& workers_list,
                    bool include_dataset_ops, int duration_ms,
                    int num_tracing_attempts) {
  // Use the current timestamp as the run name.
  tensorflow::string session_id = GetCurrentTimeStampAsString();
  constexpr char kProfilePluginDirectory[] = "plugins/profile/";
  tensorflow::string repository_root =
      io::JoinPath(logdir, kProfilePluginDirectory);
  std::vector<tensorflow::string> hostnames;
  if (!workers_list.empty()) {
    hostnames = absl::StrSplit(workers_list, ',');
  }

  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));

  Status status = Status::OK();
  int remaining_attempts = num_tracing_attempts;
  tensorflow::ProfileOptions opts;
  opts.set_include_dataset_ops(include_dataset_ops);
  while (true) {
    std::cout << "Starting to profile TPU traces for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << --remaining_attempts << std::endl;
    if (hostnames.empty()) {
      status = Profile(service_addr, logdir, duration_ms, repository_root,
                       session_id, opts);
    } else {
      tensorflow::string tpu_master = service_addr;
      status = NewSession(tpu_master, hostnames, duration_ms, repository_root,
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

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level,
                                      bool timestamp) {
  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  request.set_timestamp(timestamp);
  return request;
}

Status Monitor(const tensorflow::string& service_addr, int duration_ms,
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

}  // namespace client
}  // namespace profiler
}  // namespace tensorflow
