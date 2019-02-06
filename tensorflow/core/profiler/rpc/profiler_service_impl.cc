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
#include "tensorflow/contrib/tpu/profiler/tpu_profiler.grpc.pb.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// TODO(fishx): Rename TPUProfiler to something more generic.
class ProfilerServiceImpl : public TPUProfiler::Service {
 public:
  explicit ProfilerServiceImpl(EagerContext* const eager_context)
      : eager_context_(eager_context) {}
  ~ProfilerServiceImpl() override {}

  ::grpc::Status Monitor(::grpc::ServerContext* ctx, const MonitorRequest* req,
                         MonitorResponse* response) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "unimplemented.");
  }

  ::grpc::Status Profile(::grpc::ServerContext* ctx, const ProfileRequest* req,
                         ProfileResponse* response) override {
    LOG(INFO) << "Received a profile request.";
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(eager_context_);
    if (!profiler->Status().ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            profiler->Status().error_message());
    }

    Env* env = eager_context_->TFEnv();
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
  EagerContext* const eager_context_;
};
}  // namespace

std::unique_ptr<TPUProfiler::Service> CreateProfilerService(
    EagerContext* const eager_context) {
  return MakeUnique<ProfilerServiceImpl>(eager_context);
}

}  // namespace tensorflow
