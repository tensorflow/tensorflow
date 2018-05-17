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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_EAGER_GRPC_SERVER_LIB_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_EAGER_GRPC_SERVER_LIB_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"

namespace tensorflow {
namespace eager {

class EagerGrpcServer : public GrpcServer {
 public:
  static Status Create(const ServerDef& server_def,
                       std::unique_ptr<EagerGrpcServer>* server) {
    std::unique_ptr<EagerGrpcServer> ret(new EagerGrpcServer(server_def));

    TF_RETURN_IF_ERROR(ret->InitEager());

    *server = std::move(ret);

    return Status::OK();
  }

  Status Start() override {
    TF_RETURN_IF_ERROR(GrpcServer::Start());

    eager_service_->Start();

    return Status::OK();
  }

  Status Stop() override {
    TF_RETURN_IF_ERROR(GrpcServer::Stop());

    eager_service_->Stop();

    return Status::OK();
  }

  using GrpcServer::channel_cache;
  using GrpcServer::master_env;
  using GrpcServer::worker_env;

 private:
  EagerGrpcServer(const ServerDef& server_def)
      : GrpcServer(server_def, Env::Default()),
        worker_name_(
            strings::StrCat("/job:", server_def.job_name(),
                            "/replica:0/task:", server_def.task_index())) {}

  Status InitEager() {
    TF_RETURN_IF_ERROR(this->Init(
        [this](const WorkerEnv* worker_env,
               ::grpc::ServerBuilder* server_builder) {
          this->eager_service_.reset(
              new eager::GrpcEagerServiceImpl(worker_env, server_builder));
        },
        nullptr));

    worker_session_ = WorkerSession::CreateWithBorrowedDeviceMgr(
        "", worker_name_,
        std::unique_ptr<WorkerCacheInterface>(
            new WorkerCacheWrapper(master_env()->worker_cache)),
        worker_env()->device_mgr, {});

    auto* r = worker_env()->rendezvous_mgr->Find(0);
    return r->Initialize(worker_session_.get());
  }

  std::unique_ptr<GrpcEagerServiceImpl> eager_service_;
  std::shared_ptr<WorkerSession> worker_session_;
  const string worker_name_;
};  // namespace eager

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_EAGER_GRPC_SERVER_LIB_H_
