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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler_impl.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {
namespace eager {

namespace grpc {

static const char* grpcEagerService_method_names[] = {
    "/tensorflow.eager.EagerService/CreateContext",
    "/tensorflow.eager.EagerService/Enqueue",
    "/tensorflow.eager.EagerService/WaitQueueDone",
    "/tensorflow.eager.EagerService/KeepAlive",
    "/tensorflow.eager.EagerService/CloseContext",
    "/tensorflow.eager.EagerService/RegisterFunction",
};

std::unique_ptr<EagerService::Stub> EagerService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<EagerService::Stub> stub(new EagerService::Stub(channel));
  return stub;
}

EagerService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_CreateContext_(grpcEagerService_method_names[0],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_Enqueue_(grpcEagerService_method_names[1],
                         ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WaitQueueDone_(grpcEagerService_method_names[2],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_KeepAlive_(grpcEagerService_method_names[3],
                           ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CloseContext_(grpcEagerService_method_names[4],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RegisterFunction_(grpcEagerService_method_names[5],
                                  ::grpc::internal::RpcMethod::NORMAL_RPC,
                                  channel) {}

::grpc::Status EagerService::Stub::CreateContext(
    ::grpc::ClientContext* context, const CreateContextRequest& request,
    CreateContextResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CreateContext_, context, request, response);
}

::grpc::Status EagerService::Stub::Enqueue(::grpc::ClientContext* context,
                                           const EnqueueRequest& request,
                                           EnqueueResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_Enqueue_,
                                             context, request, response);
}

::grpc::Status EagerService::Stub::WaitQueueDone(
    ::grpc::ClientContext* context, const WaitQueueDoneRequest& request,
    WaitQueueDoneResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_WaitQueueDone_, context, request, response);
}

::grpc::Status EagerService::Stub::KeepAlive(::grpc::ClientContext* context,
                                             const KeepAliveRequest& request,
                                             KeepAliveResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_KeepAlive_, context, request, response);
}

::grpc::Status EagerService::Stub::CloseContext(
    ::grpc::ClientContext* context, const CloseContextRequest& request,
    CloseContextResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CloseContext_, context, request, response);
}

::grpc::Status EagerService::Stub::RegisterFunction(
    ::grpc::ClientContext* context, const RegisterFunctionRequest& request,
    RegisterFunctionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_RegisterFunction_, context, request, response);
}

EagerService::AsyncService::AsyncService() {
  for (int i = 0; i < 6; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcEagerService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

EagerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace eager
}  // namespace tensorflow
