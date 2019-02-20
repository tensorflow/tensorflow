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

#include "grpcpp/grpcpp.h"

#include <cstdio>
#include <ctime>
#include <vector>

#include "tensorflow/contrib/tpu/profiler/dump_tpu_profile.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/grpc_services.h"

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
  std::vector<string> parts = str_util::Split(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !strings::safe_strtou32(parts[1], &port) ||
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
  if (tensorflow::str_util::StartsWith(repository_root, "gs://")) {
    // For backward compatibilities, only generate tracetable etc when the
    // user provide a GCS path for model directory.
    request.set_repository_root(repository_root);
    request.set_session_id(session_id);
  }
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("memory_viewer");
  request.add_tools("overview_page");
  *request.mutable_opts() = opts;
  return request;
}

// Returns whether the returned trace is empty.
// Failure are handled by CHECK, i.e. abort()
bool Profile(const string& service_addr, const string& logdir, int duration_ms,
             const string& repository_root, const string& session_id,
             const ProfileOptions& opts) {
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
  TF_QCHECK_OK(FromGrpcStatus(stub->Profile(&context, request, &response)));

  if (!response.encoded_trace().empty()) {
    TF_CHECK_OK(tensorflow::tpu::WriteTensorboardTPUProfile(
        logdir, session_id, "", response, &std::cout));
    // Print this at the end so that it's not buried in irrelevant LOG messages.
    std::cout
        << "NOTE: using the trace duration " << duration_ms << "ms."
        << std::endl
        << "Set an appropriate duration (with --duration_ms) if you "
           "don't see a full step in your trace or the captured trace is too "
           "large."
        << std::endl;
  }

  return response.encoded_trace().empty();
}

// Start a new profiling session that include all the hosts included in
// hostnames, for the time interval of duration_ms. Possibly save the profiling
// result in the directory specified by repository_root and session_id.
bool NewSession(const string& service_addr,
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
  TF_QCHECK_OK(FromGrpcStatus(
      stub->NewSession(&context, new_session_request, &new_session_response)));

  std::cout << "Profile session succeed for host(s):"
            << str_util::Join(hostnames, ",") << std::endl;
  return new_session_response.empty_trace();
}

// Starts tracing on a single or multiple TPU hosts and saves the result in the
// given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts.
void StartTracing(const tensorflow::string& service_addr,
                  const tensorflow::string& logdir,
                  const tensorflow::string& workers_list,
                  bool include_dataset_ops, int duration_ms,
                  int num_tracing_attempts) {
  // Use the current timestamp as the run name.
  tensorflow::string session_id = GetCurrentTimeStampAsString();
  constexpr char kProfilePluginDirectory[] = "plugins/profile/";
  tensorflow::string repository_root =
      io::JoinPath(logdir, kProfilePluginDirectory);
  std::vector<tensorflow::string> hostnames =
      tensorflow::str_util::Split(workers_list, ",");

  bool empty_trace = false;
  int remaining_attempts = num_tracing_attempts;
  tensorflow::ProfileOptions opts;
  opts.set_include_dataset_ops(include_dataset_ops);
  while (true) {
    std::cout << "Starting to profile TPU traces for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << remaining_attempts-- << std::endl;
    if (hostnames.empty()) {
      empty_trace = Profile(service_addr, logdir, duration_ms, repository_root,
                            session_id, opts);
    } else {
      tensorflow::string tpu_master = service_addr;
      empty_trace = NewSession(tpu_master, hostnames, duration_ms,
                               repository_root, session_id, opts);
    }
    if (remaining_attempts <= 0 || !empty_trace) break;
    std::cout << "No trace event is collected. Automatically retrying."
              << std::endl
              << std::endl;
  }

  if (empty_trace) {
    std::cout << "No trace event is collected after " << num_tracing_attempts
              << " attempt(s). "
              << "Perhaps, you want to try again (with more attempts?)."
              << std::endl
              << "Tip: increase number of attempts with --num_tracing_attempts."
              << std::endl;
  }
}

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level) {
  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  return request;
}

// Repeatedly collects profiles and shows user-friendly metrics for
// 'num_queries' time(s).
void StartMonitoring(const tensorflow::string& service_addr, int duration_ms,
                     int monitoring_level, int num_queries) {
  for (int query = 0; query < num_queries; ++query) {
    MonitorRequest request =
        PopulateMonitorRequest(duration_ms, monitoring_level);

    ::grpc::ClientContext context;
    ::grpc::ChannelArguments channel_args;
    channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                        std::numeric_limits<int32>::max());
    std::unique_ptr<grpc::ProfilerService::Stub> stub =
        grpc::ProfilerService::NewStub(::grpc::CreateCustomChannel(
            "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
            channel_args));
    MonitorResponse response;
    TF_QCHECK_OK(FromGrpcStatus(stub->Monitor(&context, request, &response)));

    std::cout << "Cloud TPU Monitoring Results (Sample " << query + 1
              << "):\n\n"
              << response.data() << std::flush;
  }
}

}  // namespace client
}  // namespace profiler
}  // namespace tensorflow

