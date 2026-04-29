/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/rpc/profiler_service_impl.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::ContinuousProfilingResponse;
using tensorflow::GetSnapshotRequest;
using tensorflow::MonitorRequest;
using tensorflow::MonitorResponse;
using tensorflow::ProfileRequest;
using tensorflow::ProfileResponse;
using tensorflow::StopContinuousProfilingRequest;
using tensorflow::StopContinuousProfilingResponse;
using tensorflow::TerminateRequest;
using tensorflow::TerminateResponse;

std::string GetHostname(const ProfileRequest& request) {
  if (!request.opts().override_hostname().empty()) {
    return request.opts().override_hostname();
  }
  return request.host_name();
}

// Collects data in XSpace format. The data is saved to a repository
// unconditionally.
absl::Status CollectData(const ProfileRequest& request,
                         ProfilerSession* profiler, ProfileResponse* response) {
  response->set_empty_trace(true);
  // Read the profile data into xspace.
  tensorflow::profiler::XSpace xspace;
  tensorflow::profiler::XSpace* xspace_ptr =
      request.emit_xspace() ? response->mutable_xspace() : &xspace;
  TF_RETURN_IF_ERROR(profiler->CollectData(xspace_ptr));
  VLOG(3) << "Collected XSpace to "
          << (request.emit_xspace() ? "response" : "repository") << ".";
  response->set_empty_trace(IsEmpty(*xspace_ptr));

  if (request.emit_xspace()) {
    return absl::OkStatus();
  }

  return SaveXSpace(request.repository_root(), request.session_id(),
                    GetHostname(request), xspace);
}

class ProfilerServiceImpl : public tensorflow::grpc::ProfilerService::Service {
 public:
  struct ContinuousSession {
    tensorflow::ProfileRequest request;
    std::unique_ptr<ProfilerSession> profiler;
  };
  ::grpc::Status Monitor(::grpc::ServerContext* ctx, const MonitorRequest* req,
                         MonitorResponse* response) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "unimplemented.");
  }

  ::grpc::Status Profile(::grpc::ServerContext* ctx, const ProfileRequest* req,
                         ProfileResponse* response) override {
    VLOG(1) << "Received a profile request: " << req->DebugString();
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(req->opts());
    absl::Status status = profiler->Status();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to create profiler session: " << status;
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            std::string(status.message()));
    }

    Env* env = Env::Default();
    int64_t start_time_ns = GetCurrentTimeNanos();
    // TODO(b/416884677): Handle server shutdown gracefully by surfacing a
    // shutdown signal here and responding with what has been profiled so far.
    while (NanoToMilli(GetCurrentTimeNanos() - start_time_ns) <
           req->opts().duration_ms()) {
      env->SleepForMicroseconds(EnvTime::kMillisToMicros);
      if (ctx->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
      if (TF_PREDICT_FALSE(IsStopped(req->session_id()))) {
        absl::MutexLock lock(mutex_);
        stop_signals_per_session_.erase(req->session_id());
        break;
      }
    }

    status = CollectData(*req, profiler.get(), response);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to collect profile data: " << status;
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            std::string(status.message()));
    }

    return ::grpc::Status::OK;
  }

  // Note: This implementation only allows one continuous profiling session
  // to be active at a time.
  ::grpc::Status StartContinuousProfiling(
      ::grpc::ServerContext* ctx, const ProfileRequest* req,
      ContinuousProfilingResponse* response) override {
    absl::MutexLock lock(mutex_);
    if (continuous_profiling_session_.has_value()) {
      return ::grpc::Status(::grpc::StatusCode::ALREADY_EXISTS,
                            "A profiling session is already running.");
    }
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(req->opts());
    absl::Status status = profiler->Status();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to create profiler session: " << status;
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            "Failed to create profiler session: " +
                                std::string(status.message()));
    }
    tensorflow::ProfileRequest request = *req;
    request.set_emit_xspace(true);
    continuous_profiling_session_ = {request, std::move(profiler)};
    return ::grpc::Status::OK;
  }

  ::grpc::Status StopContinuousProfiling(
      ::grpc::ServerContext* ctx, const StopContinuousProfilingRequest* req,
      StopContinuousProfilingResponse* response) override {
    std::optional<ContinuousSession> session_to_destroy;
    {
      absl::MutexLock lock(mutex_);
      if (!continuous_profiling_session_.has_value()) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "No continuous profiling session found.");
      }
      // Move session to a local variable so that it is destroyed after mutex
      // is released, avoiding potentially expensive destruction of
      // ProfilerSession under lock.
      session_to_destroy.swap(continuous_profiling_session_);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetSnapshot(::grpc::ServerContext* ctx,
                             const GetSnapshotRequest* req,
                             ProfileResponse* response) override {
    absl::MutexLock lock(mutex_);
    if (!continuous_profiling_session_.has_value()) {
      return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                            "No continuous profiling session found.");
    }

    absl::Status status =
        CollectData(continuous_profiling_session_->request,
                    continuous_profiling_session_->profiler.get(), response);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to collect profile data: " << status;
      // If CollectData fails, stop continuous profiling.
      continuous_profiling_session_.reset();
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            std::string(status.message()));
    }

    // Restart profiling.
    // Generate new session_id and update request.
    std::string new_session_id = std::to_string(GetCurrentTimeNanos());
    continuous_profiling_session_->request.set_session_id(new_session_id);
    continuous_profiling_session_->request.mutable_opts()->set_session_id(
        new_session_id);
    std::unique_ptr<ProfilerSession> new_profiler =
        ProfilerSession::Create(continuous_profiling_session_->request.opts());
    absl::Status new_status = new_profiler->Status();
    if (!new_status.ok()) {
      LOG(ERROR) << "Failed to create profiler session: " << new_status;
      continuous_profiling_session_.reset();
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            std::string(new_status.message()));
    }
    continuous_profiling_session_->profiler = std::move(new_profiler);
    return ::grpc::Status::OK;
  }

  ::grpc::Status Terminate(::grpc::ServerContext* ctx,
                           const TerminateRequest* req,
                           TerminateResponse* response) override {
    absl::MutexLock lock(mutex_);
    stop_signals_per_session_[req->session_id()] = true;
    return ::grpc::Status::OK;
  }

 private:
  bool IsStopped(const std::string& session_id) {
    absl::MutexLock lock(mutex_);
    auto it = stop_signals_per_session_.find(session_id);
    return it != stop_signals_per_session_.end() && it->second;
  }

  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, bool> stop_signals_per_session_
      ABSL_GUARDED_BY(mutex_);
  std::optional<ContinuousSession> continuous_profiling_session_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace

std::unique_ptr<tensorflow::grpc::ProfilerService::Service>
CreateProfilerService() {
  return std::make_unique<ProfilerServiceImpl>();
}

}  // namespace profiler
}  // namespace tsl
