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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/impl/codegen/rpc_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/status.h"
#include "grpc++/impl/codegen/stub_options.h"
#include "grpc++/impl/codegen/sync_stream.h"
#include "grpc++/support/byte_buffer.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_serialization_traits.h"
#include "tensorflow/core/protobuf/worker.pb.h"

// Contains potentially large GraphDef.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RegisterGraphRequest);
// Contains potentially large TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RunGraphRequest);
// Contains potentially large StepStats, TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RunGraphResponse);
// Contains potentially large TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RecvTensorResponse);

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace tensorflow {

namespace grpc {

// Implementation of `tensorflow.WorkerService`, based on the
// definition in "//tensorflow/core/protobuf/worker_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class WorkerService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::GetStatusResponse>>
    AsyncGetStatus(::grpc::ClientContext* context,
                   const ::tensorflow::GetStatusRequest& request,
                   ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::GetStatusResponse>>(
          AsyncGetStatusRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RegisterGraphResponse>>
    AsyncRegisterGraph(::grpc::ClientContext* context,
                       const ::tensorflow::RegisterGraphRequest& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::RegisterGraphResponse>>(
          AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::DeregisterGraphResponse>>
    AsyncDeregisterGraph(::grpc::ClientContext* context,
                         const ::tensorflow::DeregisterGraphRequest& request,
                         ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::DeregisterGraphResponse>>(
          AsyncDeregisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RunGraphResponse>>
    AsyncRunGraph(::grpc::ClientContext* context,
                  const ::tensorflow::RunGraphRequest& request,
                  ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::RunGraphResponse>>(
          AsyncRunGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::CleanupGraphResponse>>
    AsyncCleanupGraph(::grpc::ClientContext* context,
                      const ::tensorflow::CleanupGraphRequest& request,
                      ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::CleanupGraphResponse>>(
          AsyncCleanupGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::CleanupAllResponse>>
    AsyncCleanupAll(::grpc::ClientContext* context,
                    const ::tensorflow::CleanupAllRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::CleanupAllResponse>>(
          AsyncCleanupAllRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RecvTensorResponse>>
    AsyncRecvTensor(::grpc::ClientContext* context,
                    const ::tensorflow::RecvTensorRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::RecvTensorResponse>>(
          AsyncRecvTensorRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::LoggingResponse>>
    AsyncLogging(::grpc::ClientContext* context,
                 const ::tensorflow::LoggingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::LoggingResponse>>(
          AsyncLoggingRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::TracingResponse>>
    AsyncTracing(::grpc::ClientContext* context,
                 const ::tensorflow::TracingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::tensorflow::TracingResponse>>(
          AsyncTracingRaw(context, request, cq));
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::GetStatusResponse>*
    AsyncGetStatusRaw(::grpc::ClientContext* context,
                      const ::tensorflow::GetStatusRequest& request,
                      ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RegisterGraphResponse>*
    AsyncRegisterGraphRaw(::grpc::ClientContext* context,
                          const ::tensorflow::RegisterGraphRequest& request,
                          ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::DeregisterGraphResponse>*
    AsyncDeregisterGraphRaw(::grpc::ClientContext* context,
                            const ::tensorflow::DeregisterGraphRequest& request,
                            ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RunGraphResponse>*
    AsyncRunGraphRaw(::grpc::ClientContext* context,
                     const ::tensorflow::RunGraphRequest& request,
                     ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::CleanupGraphResponse>*
    AsyncCleanupGraphRaw(::grpc::ClientContext* context,
                         const ::tensorflow::CleanupGraphRequest& request,
                         ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::CleanupAllResponse>*
    AsyncCleanupAllRaw(::grpc::ClientContext* context,
                       const ::tensorflow::CleanupAllRequest& request,
                       ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::RecvTensorResponse>*
    AsyncRecvTensorRaw(::grpc::ClientContext* context,
                       const ::tensorflow::RecvTensorRequest& request,
                       ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::LoggingResponse>*
    AsyncLoggingRaw(::grpc::ClientContext* context,
                    const ::tensorflow::LoggingRequest& request,
                    ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::tensorflow::TracingResponse>*
    AsyncTracingRaw(::grpc::ClientContext* context,
                    const ::tensorflow::TracingRequest& request,
                    ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::GetStatusResponse>>
    AsyncGetStatus(::grpc::ClientContext* context,
                   const ::tensorflow::GetStatusRequest& request,
                   ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::GetStatusResponse>>(
          AsyncGetStatusRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::RegisterGraphResponse>>
    AsyncRegisterGraph(::grpc::ClientContext* context,
                       const ::tensorflow::RegisterGraphRequest& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::tensorflow::RegisterGraphResponse>>(
          AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReader<
        ::tensorflow::DeregisterGraphResponse>>
    AsyncDeregisterGraph(::grpc::ClientContext* context,
                         const ::tensorflow::DeregisterGraphRequest& request,
                         ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::tensorflow::DeregisterGraphResponse>>(
          AsyncDeregisterGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::RunGraphResponse>>
    AsyncRunGraph(::grpc::ClientContext* context,
                  const ::tensorflow::RunGraphRequest& request,
                  ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::RunGraphResponse>>(
          AsyncRunGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::CleanupGraphResponse>>
    AsyncCleanupGraph(::grpc::ClientContext* context,
                      const ::tensorflow::CleanupGraphRequest& request,
                      ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<
          ::tensorflow::CleanupGraphResponse>>(
          AsyncCleanupGraphRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::CleanupAllResponse>>
    AsyncCleanupAll(::grpc::ClientContext* context,
                    const ::tensorflow::CleanupAllRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::CleanupAllResponse>>(
          AsyncCleanupAllRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::RecvTensorResponse>>
    AsyncRecvTensor(::grpc::ClientContext* context,
                    const ::tensorflow::RecvTensorRequest& request,
                    ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::RecvTensorResponse>>(
          AsyncRecvTensorRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::LoggingResponse>>
    AsyncLogging(::grpc::ClientContext* context,
                 const ::tensorflow::LoggingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::LoggingResponse>>(
          AsyncLoggingRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::tensorflow::TracingResponse>>
    AsyncTracing(::grpc::ClientContext* context,
                 const ::tensorflow::TracingRequest& request,
                 ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::tensorflow::TracingResponse>>(
          AsyncTracingRaw(context, request, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader<::tensorflow::GetStatusResponse>*
    AsyncGetStatusRaw(::grpc::ClientContext* context,
                      const ::tensorflow::GetStatusRequest& request,
                      ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::RegisterGraphResponse>*
    AsyncRegisterGraphRaw(::grpc::ClientContext* context,
                          const ::tensorflow::RegisterGraphRequest& request,
                          ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::DeregisterGraphResponse>*
    AsyncDeregisterGraphRaw(::grpc::ClientContext* context,
                            const ::tensorflow::DeregisterGraphRequest& request,
                            ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::RunGraphResponse>*
    AsyncRunGraphRaw(::grpc::ClientContext* context,
                     const ::tensorflow::RunGraphRequest& request,
                     ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::CleanupGraphResponse>*
    AsyncCleanupGraphRaw(::grpc::ClientContext* context,
                         const ::tensorflow::CleanupGraphRequest& request,
                         ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::CleanupAllResponse>*
    AsyncCleanupAllRaw(::grpc::ClientContext* context,
                       const ::tensorflow::CleanupAllRequest& request,
                       ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::RecvTensorResponse>*
    AsyncRecvTensorRaw(::grpc::ClientContext* context,
                       const ::tensorflow::RecvTensorRequest& request,
                       ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::LoggingResponse>*
    AsyncLoggingRaw(::grpc::ClientContext* context,
                    const ::tensorflow::LoggingRequest& request,
                    ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    ::grpc::ClientAsyncResponseReader<::tensorflow::TracingResponse>*
    AsyncTracingRaw(::grpc::ClientContext* context,
                    const ::tensorflow::TracingRequest& request,
                    ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    const ::grpc::RpcMethod rpcmethod_GetStatus_;
    const ::grpc::RpcMethod rpcmethod_RegisterGraph_;
    const ::grpc::RpcMethod rpcmethod_DeregisterGraph_;
    const ::grpc::RpcMethod rpcmethod_RunGraph_;
    const ::grpc::RpcMethod rpcmethod_CleanupGraph_;
    const ::grpc::RpcMethod rpcmethod_CleanupAll_;
    const ::grpc::RpcMethod rpcmethod_RecvTensor_;
    const ::grpc::RpcMethod rpcmethod_Logging_;
    const ::grpc::RpcMethod rpcmethod_Tracing_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestGetStatus(
        ::grpc::ServerContext* context, ::tensorflow::GetStatusRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::GetStatusResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRegisterGraph(
        ::grpc::ServerContext* context,
        ::tensorflow::RegisterGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::RegisterGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestDeregisterGraph(
        ::grpc::ServerContext* context,
        ::tensorflow::DeregisterGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<
            ::tensorflow::DeregisterGraphResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRunGraph(
        ::grpc::ServerContext* context, ::tensorflow::RunGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::RunGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(3, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCleanupGraph(
        ::grpc::ServerContext* context,
        ::tensorflow::CleanupGraphRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::CleanupGraphResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(4, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestCleanupAll(
        ::grpc::ServerContext* context,
        ::tensorflow::CleanupAllRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::CleanupAllResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(5, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestRecvTensorRaw(
        ::grpc::ServerContext* context,
        ::tensorflow::RecvTensorRequest* request,
        ::grpc::ServerAsyncResponseWriter<::grpc::ByteBuffer>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(6, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestLogging(
        ::grpc::ServerContext* context, ::tensorflow::LoggingRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::LoggingResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(7, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
    void RequestTracing(
        ::grpc::ServerContext* context, ::tensorflow::TracingRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::TracingResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(8, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
