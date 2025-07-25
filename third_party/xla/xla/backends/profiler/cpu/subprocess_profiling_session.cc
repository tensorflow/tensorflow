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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/backends/profiler/cpu/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
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
inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
}
}  // namespace

absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>>
SubprocessProfilingSession::Create(const SubprocessInfo& subprocess_info,
                                   const tensorflow::ProfileOptions& options) {
  if (subprocess_info.profiler_stub == nullptr) {
    return absl::InvalidArgumentError("Profiler stub is null");
  }
  return absl::WrapUnique(
      new SubprocessProfilingSession(subprocess_info, options));
}

SubprocessProfilingSession::SubprocessProfilingSession(
    const SubprocessInfo& subprocess_info,
    const tensorflow::ProfileOptions& options)
    : subprocess_info_(subprocess_info),
      options_(options),
      session_id_(absl::StrCat("subprocess_", subprocess_info.pid)),
      repository_root_(options.repository_path()),
      response_(std::make_unique<tensorflow::ProfileResponse>()) {}

absl::Status SubprocessProfilingSession::Start() {
  absl::MutexLock lock(&mu_);
  if (rpc_) {
    return absl::FailedPreconditionError(
        "Another subprocess profiling session already started.");
  }
  tensorflow::ProfileRequest request;
  request.set_session_id(session_id_);
  request.set_repository_root(repository_root_);
  request.set_host_name(
      absl::StrReplaceAll(subprocess_info_.address, {{":", "_"}}));
  // Set a 2 hour duration as the default to enable a max programmatic trace
  // duration.
  uint64_t duration_ms = absl::Hours(2) / absl::Milliseconds(1);
  request.set_duration_ms(duration_ms);
  options_.set_duration_ms(duration_ms);
  *request.mutable_opts() = options_;

  context_.set_wait_for_ready(true);
  context_.set_deadline(absl::ToChronoTime(absl::Now() + absl::Minutes(1)));
  rpc_ = subprocess_info_.profiler_stub->AsyncProfile(&context_, request, &cq_);
  rpc_->Finish(response_.get(), &status_, (void*)1);
  return absl::OkStatus();
}

absl::Status SubprocessProfilingSession::Stop() {
  absl::MutexLock lock(&mu_);
  if (!rpc_) {
    return absl::FailedPreconditionError(
        "Subprocess profiling session not started.");
  }
  tensorflow::TerminateRequest request;
  request.set_session_id(session_id_);
  tensorflow::TerminateResponse response;
  grpc::ClientContext context;
  TF_RETURN_IF_ERROR(FromGrpcStatus(status_));
  TF_RETURN_IF_ERROR(FromGrpcStatus(
      subprocess_info_.profiler_stub->Terminate(&context, request, &response)));

  void* got_tag;
  bool ok = false;
  cq_.Next(&got_tag, &ok);
  if (!ok || got_tag != (void*)1) {
    return absl::InternalError("Failed to get response from profiler service");
  }
  return FromGrpcStatus(status_);
}

absl::Status SubprocessProfilingSession::CollectData(
    tensorflow::profiler::XSpace* space) {
  absl::MutexLock lock(&mu_);
  if (!response_) {
    return absl::InternalError("Response is null");
  }
  if (response_->empty_trace()) {
    return absl::InternalError("Response is empty");
  }

  std::string xspace_output_path = response_->output_path();
  if (xspace_output_path.empty()) {
    std::string run_dir =
        tsl::profiler::ProfilerJoinPath(repository_root_, session_id_);
    std::vector<std::string> children;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(run_dir, &children));
    LOG_IF(WARNING, children.size() != 1)
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
