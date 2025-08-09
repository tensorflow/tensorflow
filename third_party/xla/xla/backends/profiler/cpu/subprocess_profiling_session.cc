/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/cpu/subprocess_profiling_session.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/backends/profiler/cpu/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace cpu {
namespace {

// Set a 2 hour duration as the default for programmatic and end-to-end
// profiling.
constexpr uint64_t kDefaultDurationMs =
    absl::ToInt64Milliseconds(absl::Hours(2));

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
}

// Returns the repository root to be used for the subprocess profiling session.
// If the repository path specified in the options is a valid directory, it is
// returned as the repository root. Otherwise, a local temp directory is
// returned. If no local temp directory is available, an error is returned.
absl::StatusOr<std::string> GetRepositoryRoot(
    const tensorflow::ProfileOptions& options) {
  if (tsl::Env::Default()->IsDirectory(options.repository_path()).ok()) {
    return options.repository_path();
  }
  LOG(ERROR) << "Repository path can not be used as repository root. "
                "Attempting to use a local temp directory as root.";
  std::vector<std::string> local_temp_directories;
  tsl::Env::Default()->GetLocalTempDirectories(&local_temp_directories);
  if (!local_temp_directories.empty()) {
    return tsl::profiler::ProfilerJoinPath(local_temp_directories[0],
                                           "subprocess_profiling");
  }
  return absl::InternalError(
      "Unable to find a repository root for subprocess profiling.");
}

absl::StatusOr<tensorflow::ProfileRequest> BuildProfileRequest(
    const SubprocessInfo& subprocess_info,
    const tensorflow::ProfileOptions& options) {
  tensorflow::ProfileRequest request;
  TF_ASSIGN_OR_RETURN(*request.mutable_repository_root(),
                      GetRepositoryRoot(options));
  *request.mutable_opts() = options;
  request.set_session_id(absl::StrCat("subprocess_", subprocess_info.pid));
  request.set_host_name(
      absl::StrReplaceAll(subprocess_info.address, {{":", "_"}}));
  // If an on-demand profiling request is made with a longer duration, use that
  // duration. Otherwise, use the default duration.
  uint64_t duration_ms = std::max(options.duration_ms(), kDefaultDurationMs);
  request.set_duration_ms(duration_ms);
  request.mutable_opts()->set_duration_ms(duration_ms);
  return request;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>>
SubprocessProfilingSession::Create(const SubprocessInfo& subprocess_info,
                                   const tensorflow::ProfileOptions& options) {
  TF_ASSIGN_OR_RETURN(tensorflow::ProfileRequest request,
                      BuildProfileRequest(subprocess_info, options));
  if (subprocess_info.profiler_stub == nullptr) {
    return absl::InvalidArgumentError("Profiler stub is null");
  }
  return absl::WrapUnique(
      new SubprocessProfilingSession(subprocess_info, request));
}

SubprocessProfilingSession::SubprocessProfilingSession(
    const SubprocessInfo& subprocess_info,
    const tensorflow::ProfileRequest& request)
    : subprocess_info_(subprocess_info), request_(request) {
  // Set the empty trace flag to true by default. This will be set to false
  // once the response is received from the subprocess.
  response_.set_empty_trace(true);
}

absl::Status SubprocessProfilingSession::Start() {
  absl::MutexLock lock(&mu_);
  if (rpc_) {
    return absl::FailedPreconditionError(
        "Another subprocess profiling session already started.");
  }
  context_.set_wait_for_ready(true);
  rpc_ =
      subprocess_info_.profiler_stub->AsyncProfile(&context_, request_, &cq_);
  if (!rpc_) {
    return absl::InternalError("Failed to start profiling session.");
  }
  rpc_->Finish(&response_, &grpc_status_, (void*)1);
  TF_RETURN_IF_ERROR(FromGrpcStatus(grpc_status_));
  return absl::OkStatus();
}

absl::Status SubprocessProfilingSession::Stop() {
  absl::MutexLock lock(&mu_);
  if (!rpc_) {
    return absl::FailedPreconditionError(
        "Subprocess profiling session not started.");
  }
  tensorflow::TerminateRequest request;
  request.set_session_id(request_.session_id());
  tensorflow::TerminateResponse response;
  grpc::ClientContext context;
  TF_RETURN_IF_ERROR(FromGrpcStatus(
      subprocess_info_.profiler_stub->Terminate(&context, request, &response)));

  void* got_tag;
  bool ok = false;
  bool success = cq_.Next(&got_tag, &ok);
  if (!success || !ok || got_tag != (void*)1) {
    return absl::InternalError("Failed to get response from profiler service");
  }
  return absl::OkStatus();
}

absl::Status SubprocessProfilingSession::CollectData(
    tensorflow::profiler::XSpace* space) {
  absl::MutexLock lock(&mu_);
  if (response_.empty_trace()) {
    return absl::InternalError("Response is empty");
  }

  std::string xspace_output_path = response_.output_path();
  if (xspace_output_path.empty()) {
    std::string run_dir = tsl::profiler::ProfilerJoinPath(
        request_.repository_root(), request_.session_id());
    std::vector<std::string> children;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(run_dir, &children));
    LOG_IF(ERROR, children.size() != 1)
        << "Found unexpected number of children: " << children.size();
    for (const auto& child : children) {
      if (absl::EndsWith(child, "xplane.pb")) {
        xspace_output_path = tsl::profiler::ProfilerJoinPath(run_dir, child);
        break;
      }
    }
  }

  if (xspace_output_path.empty()) {
    // If no file is found, the subprocess might not have generated a trace.
    LOG(ERROR) << "No xplane.pb file found in the repository root: "
               << request_.repository_root()
               << " and session id: " << request_.session_id();
    return absl::OkStatus();
  }
  tensorflow::profiler::XSpace child_space;
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(),
                                          xspace_output_path, &child_space));
  space->MergeFrom(child_space);
  return absl::OkStatus();
}

namespace {

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateSubprocessProfilers(
    const tensorflow::ProfileOptions& options) {
  std::vector<SubprocessInfo> subprocesses = GetRegisteredSubprocesses();
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>> sessions;
  sessions.reserve(subprocesses.size());
  for (const auto& subprocess_info : subprocesses) {
    absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>> session =
        SubprocessProfilingSession::Create(subprocess_info, options);
    if (!session.ok()) {
      LOG(ERROR) << "Failed to create subprocess profiling session: "
                 << session.status();
      continue;
    }
    sessions.push_back(std::move(*session));
  }
  return std::make_unique<tsl::profiler::ProfilerCollection>(
      std::move(sessions));
};

}  // namespace

auto register_subprocess_profiler_factory = [] {
  RegisterProfilerFactory(&CreateSubprocessProfilers);
  return 0;
}();

}  // namespace cpu
}  // namespace profiler
}  // namespace xla
