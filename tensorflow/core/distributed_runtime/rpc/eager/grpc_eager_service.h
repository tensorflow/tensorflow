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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"

#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

namespace grpc {

// GRPC stubs of `tensorflow.eager.EagerService`, based on the
// definition in "//tensorflow/core/protobuf/eager_service.proto",
// and the gRPC generated stub and service classes.
// See that file for the definition of methods and messages.
// Similar to the Master/Worker tensorflow GRPC services, this is not gen'ned
// via a rule, but included as an implementation directly.
class EagerService final {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status CreateContext(::grpc::ClientContext* context,
                                         const CreateContextRequest& request,
                                         CreateContextResponse* response) = 0;
    virtual ::grpc::Status Enqueue(::grpc::ClientContext* context,
                                   const EnqueueRequest& request,
                                   EnqueueResponse* response) = 0;
    virtual ::grpc::Status WaitQueueDone(::grpc::ClientContext* context,
                                         const WaitQueueDoneRequest& request,
                                         WaitQueueDoneResponse* response) = 0;
    virtual ::grpc::Status KeepAlive(::grpc::ClientContext* context,
                                     const KeepAliveRequest& request,
                                     KeepAliveResponse* response) = 0;
    virtual ::grpc::Status CloseContext(::grpc::ClientContext* context,
                                        const CloseContextRequest& request,
                                        CloseContextResponse* response) = 0;
    virtual ::grpc::Status RegisterFunction(
        ::grpc::ClientContext* context, const RegisterFunctionRequest& request,
        RegisterFunctionResponse* response) = 0;
    virtual ::grpc::Status SendTensor(::grpc::ClientContext* context,
                                      const SendTensorRequest& request,
                                      SendTensorResponse* response) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status CreateContext(::grpc::ClientContext* context,
                                 const CreateContextRequest& request,
                                 CreateContextResponse* response) override;
    ::grpc::Status Enqueue(::grpc::ClientContext* context,
                           const EnqueueRequest& request,
                           EnqueueResponse* response) override;
    ::grpc::Status WaitQueueDone(::grpc::ClientContext* context,
                                 const WaitQueueDoneRequest& request,
                                 WaitQueueDoneResponse* response) override;
    ::grpc::Status KeepAlive(::grpc::ClientContext* context,
                             const KeepAliveRequest& request,
                             KeepAliveResponse* response) override;
    ::grpc::Status CloseContext(::grpc::ClientContext* context,
                                const CloseContextRequest& request,
                                CloseContextResponse* response) override;
    ::grpc::Status RegisterFunction(
        ::grpc::ClientContext* context, const RegisterFunctionRequest& request,
        RegisterFunctionResponse* response) override;
    ::grpc::Status SendTensor(::grpc::ClientContext* context,
                              const SendTensorRequest& request,
                              SendTensorResponse* response) override;

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::internal::RpcMethod rpcmethod_CreateContext_;
    const ::grpc::internal::RpcMethod rpcmethod_Enqueue_;
    const ::grpc::internal::RpcMethod rpcmethod_WaitQueueDone_;
    const ::grpc::internal::RpcMethod rpcmethod_KeepAlive_;
    const ::grpc::internal::RpcMethod rpcmethod_CloseContext_;
    const ::grpc::internal::RpcMethod rpcmethod_RegisterFunction_;
    const ::grpc::internal::RpcMethod rpcmethod_SendTensor_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr< ::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestCreateContext(
        ::grpc::ServerContext* context, CreateContextRequest* request,
        ::grpc::ServerAsyncResponseWriter<CreateContextResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestEnqueue(
        ::grpc::ServerContext* context, EnqueueRequest* request,
        ::grpc::ServerAsyncResponseWriter<EnqueueResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestWaitQueueDone(
        ::grpc::ServerContext* context, WaitQueueDoneRequest* request,
        ::grpc::ServerAsyncResponseWriter<WaitQueueDoneResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestKeepAlive(
        ::grpc::ServerContext* context, KeepAliveRequest* request,
        ::grpc::ServerAsyncResponseWriter<KeepAliveResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(3, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCloseContext(
        ::grpc::ServerContext* context, CloseContextRequest* request,
        ::grpc::ServerAsyncResponseWriter<CloseContextResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(4, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRegisterFunction(
        ::grpc::ServerContext* context, RegisterFunctionRequest* request,
        ::grpc::ServerAsyncResponseWriter<RegisterFunctionResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(5, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestSendTensor(
        ::grpc::ServerContext* context, SendTensorRequest* request,
        ::grpc::ServerAsyncResponseWriter<SendTensorResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(6, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_H_
