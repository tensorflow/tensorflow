/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace tensorflow {

namespace grpc {

static const char* grpcWorkerService_method_names[] = {
    "/tensorflow.WorkerService/GetStatus",
    "/tensorflow.WorkerService/RegisterGraph",
    "/tensorflow.WorkerService/DeregisterGraph",
    "/tensorflow.WorkerService/RunGraph",
    "/tensorflow.WorkerService/CleanupGraph",
    "/tensorflow.WorkerService/CleanupAll",
    "/tensorflow.WorkerService/RecvTensor",
    "/tensorflow.WorkerService/Logging",
    "/tensorflow.WorkerService/Tracing",
};

std::unique_ptr<WorkerService::Stub> WorkerService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<WorkerService::Stub> stub(new WorkerService::Stub(channel));
  return stub;
}

WorkerService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_GetStatus_(grpcWorkerService_method_names[0],
                           ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RegisterGraph_(grpcWorkerService_method_names[1],
                               ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_DeregisterGraph_(grpcWorkerService_method_names[2],
                                 ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RunGraph_(grpcWorkerService_method_names[3],
                          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CleanupGraph_(grpcWorkerService_method_names[4],
                              ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CleanupAll_(grpcWorkerService_method_names[5],
                            ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RecvTensor_(grpcWorkerService_method_names[6],
                            ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Logging_(grpcWorkerService_method_names[7],
                         ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Tracing_(grpcWorkerService_method_names[8],
                         ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::ClientAsyncResponseReader<GetStatusResponse>*
WorkerService::Stub::AsyncGetStatusRaw(::grpc::ClientContext* context,
                                       const GetStatusRequest& request,
                                       ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<GetStatusResponse>(
      channel_.get(), cq, rpcmethod_GetStatus_, context, request);
}

::grpc::ClientAsyncResponseReader<RegisterGraphResponse>*
WorkerService::Stub::AsyncRegisterGraphRaw(::grpc::ClientContext* context,
                                           const RegisterGraphRequest& request,
                                           ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<RegisterGraphResponse>(
      channel_.get(), cq, rpcmethod_RegisterGraph_, context, request);
}

::grpc::ClientAsyncResponseReader<DeregisterGraphResponse>*
WorkerService::Stub::AsyncDeregisterGraphRaw(
    ::grpc::ClientContext* context, const DeregisterGraphRequest& request,
    ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<DeregisterGraphResponse>(
      channel_.get(), cq, rpcmethod_DeregisterGraph_, context, request);
}

::grpc::ClientAsyncResponseReader<RunGraphResponse>*
WorkerService::Stub::AsyncRunGraphRaw(::grpc::ClientContext* context,
                                      const RunGraphRequest& request,
                                      ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<RunGraphResponse>(
      channel_.get(), cq, rpcmethod_RunGraph_, context, request);
}

::grpc::ClientAsyncResponseReader<CleanupGraphResponse>*
WorkerService::Stub::AsyncCleanupGraphRaw(::grpc::ClientContext* context,
                                          const CleanupGraphRequest& request,
                                          ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<CleanupGraphResponse>(
      channel_.get(), cq, rpcmethod_CleanupGraph_, context, request);
}

::grpc::ClientAsyncResponseReader<CleanupAllResponse>*
WorkerService::Stub::AsyncCleanupAllRaw(::grpc::ClientContext* context,
                                        const CleanupAllRequest& request,
                                        ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<CleanupAllResponse>(
      channel_.get(), cq, rpcmethod_CleanupAll_, context, request);
}

::grpc::ClientAsyncResponseReader<RecvTensorResponse>*
WorkerService::Stub::AsyncRecvTensorRaw(::grpc::ClientContext* context,
                                        const RecvTensorRequest& request,
                                        ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<RecvTensorResponse>(
      channel_.get(), cq, rpcmethod_RecvTensor_, context, request);
}

::grpc::ClientAsyncResponseReader<LoggingResponse>*
WorkerService::Stub::AsyncLoggingRaw(::grpc::ClientContext* context,
                                     const LoggingRequest& request,
                                     ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<LoggingResponse>(
      channel_.get(), cq, rpcmethod_Logging_, context, request);
}

::grpc::ClientAsyncResponseReader<TracingResponse>*
WorkerService::Stub::AsyncTracingRaw(::grpc::ClientContext* context,
                                     const TracingRequest& request,
                                     ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader<TracingResponse>(
      channel_.get(), cq, rpcmethod_Tracing_, context, request);
}

WorkerService::AsyncService::AsyncService() {
  (void)grpcWorkerService_method_names;
  for (int i = 0; i < 9; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(grpcWorkerService_method_names[i],
                                           ::grpc::RpcMethod::NORMAL_RPC,
                                           nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
