/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_SERVICE_IMPL_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_SERVICE_IMPL_H_

#include <memory>

#include "grpcpp/alarm.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_rpc_handler.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_call.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/threadpool.h"
#include "tsl/protobuf/coordination_service.grpc.pb.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tsl {

class GrpcCoordinationServiceImpl : public AsyncServiceInterface {
 public:
  template <class RequestMessage, class ResponseMessage>
  using CoordCall = Call<GrpcCoordinationServiceImpl,
                         tensorflow::grpc::CoordinationService::AsyncService,
                         RequestMessage, ResponseMessage>;

  GrpcCoordinationServiceImpl(thread::ThreadPool* compute_pool,
                              ::grpc::ServerBuilder* server_builder);
  ~GrpcCoordinationServiceImpl() override {}

  void HandleRPCsLoop() override;
  void Shutdown() override;
  void SetCoordinationServiceAgentInstance(CoordinationServiceAgent* agent) {
    rpc_handler_.SetAgentInstance(agent);
  }
  void SetCoordinationServiceInstance(CoordinationServiceInterface* service) {
    rpc_handler_.SetServiceInstance(service);
  }
  CoordinationServiceRpcHandler* GetRpcHandler() { return &rpc_handler_; }

 private:
#define HANDLER(method)                                                       \
  void method##Handler(CoordCall<tensorflow::method##Request,                 \
                                 tensorflow::method##Response>* call) {       \
    tf_shared_lock l(shutdown_mu_);                                           \
    if (shutdown_) {                                                          \
      call->SendResponse(ToGrpcStatus(                                        \
          errors::Internal("Coordination service has been shut down.")));     \
      return;                                                                 \
    }                                                                         \
    compute_pool_.Schedule([this, call]() {                                   \
      rpc_handler_.method##Async(&call->request, &call->response,             \
                                 [call](const Status& s) {                    \
                                   call->ClearCancelCallback();               \
                                   call->SendResponse(ToGrpcStatus(s));       \
                                 });                                          \
    });                                                                       \
    Call<GrpcCoordinationServiceImpl,                                         \
         tensorflow::grpc::CoordinationService::AsyncService,                 \
         tensorflow::method##Request, tensorflow::method##Response>::         \
        EnqueueRequest(&service_, cq_.get(),                                  \
                       &tensorflow::grpc::CoordinationService::AsyncService:: \
                           Request##method,                                   \
                       &GrpcCoordinationServiceImpl::method##Handler,         \
                       /*supports_cancel=*/false);                            \
  }
  HANDLER(RegisterTask);
  HANDLER(WaitForAllTasks);
  HANDLER(ShutdownTask);
  HANDLER(ResetTask);
  HANDLER(Heartbeat);
  HANDLER(ReportErrorToTask);
  HANDLER(ReportErrorToService);
  HANDLER(GetTaskState);
  HANDLER(InsertKeyValue);
  HANDLER(GetKeyValue);
  HANDLER(TryGetKeyValue);
  HANDLER(GetKeyValueDir);
  HANDLER(DeleteKeyValue);
  HANDLER(Barrier);
  HANDLER(CancelBarrier);
#undef HANDLER

  thread::ThreadPool& compute_pool_;
  CoordinationServiceRpcHandler rpc_handler_;

  mutex shutdown_mu_;
  bool shutdown_ TF_GUARDED_BY(shutdown_mu_);
  std::unique_ptr<::grpc::Alarm> shutdown_alarm_;

  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  tensorflow::grpc::CoordinationService::AsyncService service_;

  GrpcCoordinationServiceImpl(const GrpcCoordinationServiceImpl&) = delete;
  void operator=(const GrpcCoordinationServiceImpl&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_RPC_COORDINATION_GRPC_COORDINATION_SERVICE_IMPL_H_
