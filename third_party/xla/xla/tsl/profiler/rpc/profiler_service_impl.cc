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
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::MonitorRequest;
using tensorflow::MonitorResponse;
using tensorflow::ProfileRequest;
using tensorflow::ProfileResponse;
using tensorflow::TerminateRequest;
using tensorflow::TerminateResponse;

std::string GetHostname(const ProfileRequest& request) {
  std::optional<std::variant<std::string, bool, int64_t>> hostname_override =
      GetConfigValue(request.opts(), "override_hostname");
  if (!hostname_override.has_value()) {
    return request.host_name();
  }
  const std::string* hostname_str =
      std::get_if<std::string>(&*hostname_override);
  if (hostname_str != nullptr && !hostname_str->empty()) {
    return *hostname_str;
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
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            std::string(status.message()));
    }

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
};

}  // namespace

std::unique_ptr<tensorflow::grpc::ProfilerService::Service>
CreateProfilerService() {
  return std::make_unique<ProfilerServiceImpl>();
}

}  // namespace profiler
}  // namespace tsl
