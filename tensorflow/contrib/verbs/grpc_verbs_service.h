/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_H_
#define TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_H_

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/grpc_verbs_service_impl.h"
#include "tensorflow/contrib/verbs/rdma_mgr.h"
#include "tensorflow/contrib/verbs/verbs_service.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace grpc {
class ServerBuilder;
class ServerCompletionQueue;
class Alarm;
}  // namespace grpc

namespace tensorflow {

class GrpcVerbsService : public AsyncServiceInterface {
 public:
  GrpcVerbsService(const WorkerEnv* worker_env, ::grpc::ServerBuilder* builder);
  ~GrpcVerbsService();
  void HandleRPCsLoop() override;
  void Shutdown() override;
  void SetRdmaMgr(RdmaMgr* rdma_mgr) { rdma_mgr_ = rdma_mgr; }

 private:
  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcVerbsService, grpc::VerbsService::AsyncService,
                          RequestMessage, ResponseMessage>;
  void GetRemoteAddressHandler(
      WorkerCall<GetRemoteAddressRequest, GetRemoteAddressResponse>* call);
  Status GetRemoteAddressSync(const GetRemoteAddressRequest* request,
                              GetRemoteAddressResponse* response);

  ::grpc::ServerCompletionQueue* cq_;
  grpc::VerbsService::AsyncService verbs_service_;
  mutex shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(shutdown_mu_);
  ::grpc::Alarm* shutdown_alarm_;
  // not owned
  RdmaMgr* rdma_mgr_;
  const WorkerEnv* const worker_env_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcVerbsService);
};

// Create a GrpcVerbsService, then assign it to a given handle.
void SetNewVerbsService(GrpcVerbsService** handle, const WorkerEnv* worker_env,
                        ::grpc::ServerBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // TENSORFLOW_CONTRIB_VERBS_GRPC_VERBS_SERVICE_H_
