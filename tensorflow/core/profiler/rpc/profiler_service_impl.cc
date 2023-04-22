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

#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"

#include <memory>

#include "grpcpp/support/status.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/file_system_utils.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kXPlanePb = "xplane.pb";

// Collects data in XSpace format. The data is saved to a repository
// unconditionally.
Status CollectDataToRepository(const ProfileRequest& request,
                               ProfilerSession* profiler,
                               ProfileResponse* response) {
  response->set_empty_trace(true);
  // Read the profile data into xspace.
  XSpace xspace;
  TF_RETURN_IF_ERROR(profiler->CollectData(&xspace));
  xspace.add_hostnames(request.host_name());
  VLOG(3) << "Collected XSpace to repository.";
  response->set_empty_trace(IsEmpty(xspace));

  std::string log_dir_path =
      ProfilerJoinPath(request.repository_root(), request.session_id());
  VLOG(1) << "Creating " << log_dir_path;
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(log_dir_path));

  std::string file_name = absl::StrCat(request.host_name(), ".", kXPlanePb);
  // Windows file names do not support colons.
  absl::StrReplaceAll({{":", "_"}}, &file_name);
  // Dumps profile data to <repository_root>/<run>/<host>_<port>.<kXPlanePb>
  std::string out_path = ProfilerJoinPath(log_dir_path, file_name);
  LOG(INFO) << "Collecting XSpace to repository: " << out_path;

  return WriteBinaryProto(Env::Default(), out_path, xspace);
}

class ProfilerServiceImpl : public grpc::ProfilerService::Service {
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
    Status status = profiler->Status();
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    Env* env = Env::Default();
    for (uint64 i = 0; i < req->opts().duration_ms(); ++i) {
      env->SleepForMicroseconds(EnvTime::kMillisToMicros);
      if (ctx->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
      if (TF_PREDICT_FALSE(IsStopped(req->session_id()))) {
        mutex_lock lock(mutex_);
        stop_signals_per_session_.erase(req->session_id());
        break;
      }
    }

    status = CollectDataToRepository(*req, profiler.get(), response);
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    return ::grpc::Status::OK;
  }

  ::grpc::Status Terminate(::grpc::ServerContext* ctx,
                           const TerminateRequest* req,
                           TerminateResponse* response) override {
    mutex_lock lock(mutex_);
    stop_signals_per_session_[req->session_id()] = true;
    return ::grpc::Status::OK;
  }

 private:
  bool IsStopped(const std::string& session_id) {
    mutex_lock lock(mutex_);
    auto it = stop_signals_per_session_.find(session_id);
    return it != stop_signals_per_session_.end() && it->second;
  }

  mutex mutex_;
  absl::flat_hash_map<std::string, bool> stop_signals_per_session_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace

std::unique_ptr<grpc::ProfilerService::Service> CreateProfilerService() {
  return absl::make_unique<ProfilerServiceImpl>();
}

}  // namespace profiler
}  // namespace tensorflow
