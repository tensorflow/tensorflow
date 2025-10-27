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

#include "xla/backends/profiler/subprocess/subprocess_profiling_session.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timestamp_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {

namespace {

// Set an infinite default duration to support programmatic and end-to-end
// profiling.
constexpr uint64_t kInfiniteDurationMs = std::numeric_limits<uint64_t>::max();

// Builds a ProfileRequest for the given subprocess and options.
tensorflow::ProfileRequest BuildProfileRequest(
    const SubprocessInfo& subprocess_info,
    const tensorflow::ProfileOptions& options) {
  tensorflow::ProfileRequest request;
  *request.mutable_opts() = options;
  // Only support CPU profiling for now. To support other device types, they
  // will need to be updated to correctly handle subprocess traces.
  request.mutable_opts()->set_device_type(tensorflow::ProfileOptions::CPU);
  request.set_session_id(absl::StrCat("subprocess_", subprocess_info.pid, "_",
                                      absl::ToUnixMillis(absl::Now())));
  request.set_emit_xspace(true);
  // Use an infinite duration to support programmatic and end-to-end profiling.
  request.set_duration_ms(kInfiniteDurationMs);
  request.mutable_opts()->set_duration_ms(kInfiniteDurationMs);
  return request;
}

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
}

std::string WrapSubprocessMessage(absl::string_view message,
                                  const SubprocessInfo& subprocess_info) {
  return absl::StrCat("[Subprocess ", subprocess_info.pid, "] ", message);
}

}  // namespace

absl::Status SubprocessProfilingSession::Start() {
  if (rpc_) {
    return absl::FailedPreconditionError(
        "Another subprocess profiling session already started.");
  }
  context_.set_wait_for_ready(true);
  rpc_ = subprocess_info_.profiler_stub->AsyncProfile(&context_, request_,
                                                      &completion_queue_);
  if (!rpc_) {
    return absl::InternalError("Failed to start profiling session.");
  }
  // Register a memory location for the response with a tag of 1. This tag will
  // be used to verify the results from the CompletionQueue::Next call in
  // Stop().
  rpc_->Finish(&response_, &grpc_status_, (void*)1);
  return absl::OkStatus();
}

absl::Status SubprocessProfilingSession::Stop() {
  if (!rpc_) {
    return absl::FailedPreconditionError(
        "Subprocess profiling session not started.");
  }
  // If there is an error, make sure to cancel the context to avoid
  // heap-use-after-free inside the gRPC library.
  absl::Cleanup cleanup = [&]() { context_.TryCancel(); };
  tensorflow::TerminateRequest terminate_request;
  terminate_request.set_session_id(request_.session_id());
  tensorflow::TerminateResponse terminate_response;
  grpc::ClientContext context;
  TF_RETURN_IF_ERROR(FromGrpcStatus(subprocess_info_.profiler_stub->Terminate(
      &context, terminate_request, &terminate_response)));

  // Wait for the response from the AsyncProfile+Finish calls.
  void* got_tag;
  bool ok = false;
  bool success = completion_queue_.Next(&got_tag, &ok);
  // Verify the response is correct by checking for the tag we passed in the
  // call to Finish(). See
  // https://grpc.io/docs/languages/cpp/async/#async-client for more details.
  if (!success || !ok || got_tag != (void*)1) {
    return absl::InternalError("Failed to get response from profiler service");
  }
  TF_RETURN_IF_ERROR(FromGrpcStatus(grpc_status_));
  return absl::OkStatus();
}

absl::Status SubprocessProfilingSession::CollectData(
    tensorflow::profiler::XSpace* space) {
  if (space == nullptr) {
    return absl::InvalidArgumentError("space is null");
  }
  if (response_.empty_trace()) {
    space->add_warnings(
        absl::StrCat("No XSpace data returned from subprocess: ",
                     subprocess_info_.DebugString()));
  }
  if (auto timestamps = tsl::profiler::GetSessionTimestamps(response_.xspace());
      timestamps.has_value()) {
    tsl::profiler::DenormalizeTimestamps(response_.mutable_xspace(),
                                         timestamps->first);
  } else {
    LOG(WARNING) << "No session timestamps found. Skipping denormalizing "
                    "timestamps.";
  }
  std::optional<uint32_t> pid;
  for (const auto& plane : response_.xspace().planes()) {
    auto& copied_plane = *space->add_planes();
    copied_plane.CopyFrom(plane);
    if (!pid.has_value()) {
      tsl::profiler::XPlaneVisitor visitor =
          tsl::profiler::CreateTfXPlaneVisitor(&copied_plane);
      if (auto pid_stat = visitor.GetStat(tsl::profiler::StatType::kProcessId);
          pid_stat.has_value()) {
        pid = pid_stat->IntOrUintValue();
      } else {
        LOG(WARNING) << "No PID found in trace for subprocess: "
                     << subprocess_info_.DebugString();
        pid = subprocess_info_.pid;
      }
    }
    copied_plane.set_name(absl::StrCat(plane.name(), " [", *pid, "]"));
  }
  for (const auto& warning : response_.xspace().warnings()) {
    space->add_warnings(WrapSubprocessMessage(warning, subprocess_info_));
  }
  for (const auto& error : response_.xspace().errors()) {
    space->add_errors(WrapSubprocessMessage(error, subprocess_info_));
  }
  // Do not fail other profilers due to subprocess profiling failure.
  return absl::OkStatus();
}

SubprocessProfilingSession::SubprocessProfilingSession(
    const SubprocessInfo& subprocess_info,
    const tensorflow::ProfileRequest& request)
    : subprocess_info_(subprocess_info), request_(request) {
  // Set the empty trace flag to true by default. This will be set to false
  // once the response is received from the subprocess.
  response_.set_empty_trace(true);
}

absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>>
SubprocessProfilingSession::Create(const SubprocessInfo& subprocess_info,
                                   const tensorflow::ProfileOptions& options) {
  if (subprocess_info.profiler_stub == nullptr) {
    return absl::InvalidArgumentError("Profiler stub is null.");
  }
  return absl::WrapUnique(new SubprocessProfilingSession(
      subprocess_info, BuildProfileRequest(subprocess_info, options)));
}

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
