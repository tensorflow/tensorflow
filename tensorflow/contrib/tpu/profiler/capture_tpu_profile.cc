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

// Usage: capture_tpu_profile --service_addr="localhost:8466" --logdir=/tmp/log
//
// Initiates a TPU profiling on the TPUProfiler service at service_addr,
// receives and dumps the profile data to a tensorboard log directory.

#include "grpcpp/grpcpp.h"

#include <cstdio>
#include <ctime>
#include <vector>

#include "tensorflow/contrib/tpu/profiler/dump_tpu_profile.h"
#include "tensorflow/contrib/tpu/profiler/tpu_profiler.grpc.pb.h"
#include "tensorflow/contrib/tpu/profiler/tpu_profiler_analysis.grpc.pb.h"
#include "tensorflow/contrib/tpu/profiler/version.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::tensorflow::TPUProfileAnalysis;
using ::tensorflow::TPUProfiler;

constexpr uint64 kMaxEvents = 1000000;

string GetCurrentTimeStampAsString() {
  char s[128];
  std::time_t t = std::time(nullptr);
  CHECK_NE(std::strftime(s, sizeof(s), "%F_%T", std::localtime(&t)), 0);
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
  std::cout << "Limiting the number of trace events to " << kMaxEvents
            << std::endl;
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
  // TODO(ioeric): use `SetMaxReceiveMessageSize` instead once it's available.
  // TODO(qiuminxu): use `NewHostPortGrpcChannel` instead once their
  // `ValidateHostPortPair` checks for empty host string case.
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<TPUProfiler::Stub> stub =
      TPUProfiler::NewStub(::grpc::CreateCustomChannel(
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
  std::unique_ptr<TPUProfileAnalysis::Stub> stub =
      TPUProfileAnalysis::NewStub(::grpc::CreateCustomChannel(
          "dns:///" + service_addr, ::grpc::InsecureChannelCredentials(),
          channel_args));
  NewProfileSessionResponse new_session_response;
  TF_QCHECK_OK(FromGrpcStatus(
      stub->NewSession(&context, new_session_request, &new_session_response)));

  std::cout << "Profile session succeed for host(s):"
            << str_util::Join(hostnames, ",") << std::endl;
  return new_session_response.empty_trace();
}

}  // namespace
}  // namespace tpu
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::string FLAGS_service_addr;
  tensorflow::string FLAGS_logdir;
  tensorflow::string FLAGS_workers_list;
  int FLAGS_duration_ms = 2000;
  int FLAGS_num_tracing_attempts = 3;
  bool FLAGS_include_dataset_ops = true;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("service_addr", &FLAGS_service_addr,
                       "Address of TPU profiler service e.g. localhost:8466"),
      tensorflow::Flag("workers_list", &FLAGS_workers_list,
                       "The list of worker TPUs that we are about to profile "
                       "in the current session."),
      tensorflow::Flag("logdir", &FLAGS_logdir,
                       "Path of TensorBoard log directory e.g. /tmp/tb_log, "
                       "gs://tb_bucket"),
      tensorflow::Flag("duration_ms", &FLAGS_duration_ms,
                       "Duration of tracing in ms. Default is 2000ms."),
      tensorflow::Flag("num_tracing_attempts", &FLAGS_num_tracing_attempts,
                       "Automatically retry N times when no trace event "
                       "is collected. Default is 3."),
      tensorflow::Flag("include_dataset_ops", &FLAGS_include_dataset_ops,
                       "Set to false to profile longer TPU device traces."),
  };

  std::cout << "Welcome to the Cloud TPU Profiler v" << TPU_PROFILER_VERSION
            << std::endl;

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_service_addr.empty() || FLAGS_logdir.empty()) {
    std::cout << usage.c_str() << std::endl;
    return 2;
  }
  tensorflow::Status status =
      tensorflow::tpu::ValidateHostPortPair(FLAGS_service_addr);
  if (!status.ok()) {
    std::cout << status.error_message() << std::endl;
    std::cout << usage.c_str() << std::endl;
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // Sets the minimum duration_ms and tracing attempts to one.
  int duration_ms = std::max(FLAGS_duration_ms, 1);
  int remaining_attempts = std::max(FLAGS_num_tracing_attempts, 1);
  tensorflow::ProfileOptions opts;
  opts.set_include_dataset_ops(FLAGS_include_dataset_ops);
  tensorflow::ProfileResponse response;

  // Use the current timestamp as the run name.
  tensorflow::string session_id =
      tensorflow::tpu::GetCurrentTimeStampAsString();
  constexpr char kProfilePluginDirectory[] = "plugins/profile/";
  tensorflow::string repository_root =
      ::tensorflow::io::JoinPath(FLAGS_logdir, kProfilePluginDirectory);
  std::vector<tensorflow::string> hostnames =
      tensorflow::str_util::Split(FLAGS_workers_list, ",");

  bool empty_trace = false;
  while (true) {
    std::cout << "Starting to profile TPU traces for " << duration_ms << " ms. "
              << "Remaining attempt(s): " << remaining_attempts-- << std::endl;
    if (hostnames.empty()) {
      empty_trace = tensorflow::tpu::Profile(FLAGS_service_addr, FLAGS_logdir,
                                             duration_ms, repository_root,
                                             session_id, opts);
    } else {
      tensorflow::string tpu_master = FLAGS_service_addr;
      empty_trace =
          tensorflow::tpu::NewSession(tpu_master, hostnames, duration_ms,
                                      repository_root, session_id, opts);
    }
    if (remaining_attempts <= 0 || !empty_trace) break;
    std::cout << "No trace event is collected. Automatically retrying."
              << std::endl
              << std::endl;
  }

  if (empty_trace) {
    std::cout << "No trace event is collected after "
              << FLAGS_num_tracing_attempts << " attempt(s). "
              << "Perhaps, you want to try again (with more attempts?)."
              << std::endl
              << "Tip: increase number of attempts with --num_tracing_attempts."
              << std::endl;
    // Don't dump profile data if no trace is collected.
    return 0;
  }

  return 0;
}
