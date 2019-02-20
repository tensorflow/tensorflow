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
#include "grpcpp/support/status.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/grpc_services.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

class ProfilerServiceImpl : public grpc::ProfilerService::Service {
 public:
  explicit ProfilerServiceImpl(const ProfilerContext& profiler_context)
      : profiler_context_(profiler_context) {}
  ~ProfilerServiceImpl() override {}

  ::grpc::Status Monitor(::grpc::ServerContext* ctx, const MonitorRequest* req,
                         MonitorResponse* response) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "unimplemented.");
  }

  ::grpc::Status Profile(::grpc::ServerContext* ctx, const ProfileRequest* req,
                         ProfileResponse* response) override {
    LOG(INFO) << "Received a profile request.";
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(&profiler_context_);
    if (!profiler->Status().ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            profiler->Status().error_message());
    }

    Env* env = profiler_context_.eager_context != nullptr
                   ? profiler_context_.eager_context->TFEnv()
                   : Env::Default();
    for (size_t i = 0; i < req->duration_ms(); ++i) {
      env->SleepForMicroseconds(1000);
      if (ctx->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
    }

    Status s = profiler->SerializeToString(response->mutable_encoded_trace());
    if (!s.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, s.error_message());
    }

    return ::grpc::Status::OK;
  }

 private:
  ProfilerContext profiler_context_;
};
}  // namespace

std::unique_ptr<grpc::ProfilerService::Service> CreateProfilerService(
    const ProfilerContext& profiler_context) {
  return MakeUnique<ProfilerServiceImpl>(profiler_context);
}

}  // namespace tensorflow
