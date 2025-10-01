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
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
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
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

// Set an infinite default duration to support programmatic and end-to-end
// profiling.
constexpr uint64_t kInfiniteDurationMs = std::numeric_limits<uint64_t>::max();

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
}

absl::StatusOr<tensorflow::ProfileRequest> BuildProfileRequest(
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

class SubprocessProfilingSession : public tsl::profiler::ProfilerInterface {
 public:
  SubprocessProfilingSession(const SubprocessInfo& subprocess_info,
                             const tensorflow::ProfileRequest& request)
      : subprocess_info_(subprocess_info), request_(request) {
    // Set the empty trace flag to true by default. This will be set to false
    // once the response is received from the subprocess.
    response_.set_empty_trace(true);
  }
  // Not copyable or movable
  SubprocessProfilingSession(const SubprocessProfilingSession&) = delete;
  SubprocessProfilingSession& operator=(const SubprocessProfilingSession&) =
      delete;

  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  SubprocessInfo subprocess_info_;
  tensorflow::ProfileRequest request_;
  tensorflow::ProfileResponse response_;
  grpc::ClientContext context_;
  grpc::CompletionQueue completion_queue_;
  grpc::Status grpc_status_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<tensorflow::ProfileResponse>>
      rpc_;
};

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

std::string WrapSubprocessMessage(absl::string_view message,
                                  const SubprocessInfo& subprocess_info) {
  return absl::StrCat("[Subprocess ", subprocess_info.pid, "] ", message);
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
  for (const auto& plane : response_.xspace().planes()) {
    // TODO(b/416884677): Implement merging task env planes from subprocesses to
    // propagate metadata.
    if (plane.name() == tsl::profiler::kTaskEnvPlaneName) {
      // Throw away the task env plane from subprocesses for now.
      continue;
    }
    *space->add_planes() = plane;
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

}  // namespace

absl::StatusOr<std::unique_ptr<tsl::profiler::ProfilerInterface>>
CreateSubprocessProfilingSession(const SubprocessInfo& subprocess_info,
                                 const tensorflow::ProfileOptions& options) {
  TF_ASSIGN_OR_RETURN(tensorflow::ProfileRequest request,
                      BuildProfileRequest(subprocess_info, options));
  if (subprocess_info.profiler_stub == nullptr) {
    return absl::InvalidArgumentError("Profiler stub is null");
  }
  return std::make_unique<SubprocessProfilingSession>(subprocess_info, request);
}

namespace {

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateSubprocessProfilers(
    const tensorflow::ProfileOptions& options) {
  std::vector<SubprocessInfo> subprocesses = GetRegisteredSubprocesses();
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>>
      subprocess_profilers;
  subprocess_profilers.reserve(subprocesses.size());
  for (const auto& subprocess_info : subprocesses) {
    absl::StatusOr<std::unique_ptr<tsl::profiler::ProfilerInterface>>
        subprocess_profiler =
            CreateSubprocessProfilingSession(subprocess_info, options);
    if (!subprocess_profiler.ok()) {
      LOG(ERROR) << "Failed to create subprocess profiling session: "
                 << subprocess_profiler.status();
      continue;
    }
    subprocess_profilers.push_back(std::move(*subprocess_profiler));
  }
  return std::make_unique<tsl::profiler::ProfilerCollection>(
      std::move(subprocess_profilers));
};

// Register the subprocess profiler factory.
auto register_subprocess_profiler_factory = [] {
  RegisterProfilerFactory(&CreateSubprocessProfilers);
  return 0;
}();

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
