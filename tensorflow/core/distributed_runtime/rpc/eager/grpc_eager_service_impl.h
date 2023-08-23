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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_

#include "grpcpp/alarm.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/tsl/distributed_runtime/rpc/grpc_call.h"

namespace tensorflow {
namespace eager {

// This class is a wrapper that handles communication for gRPC.
class GrpcEagerServiceImpl : public tsl::AsyncServiceInterface {
 public:
  template <class RequestMessage, class ResponseMessage>
  using EagerCall =
      tsl::Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,
                RequestMessage, ResponseMessage>;
  template <class RequestMessage, class ResponseMessage>
  using StreamingCall =
      tsl::ServerBidirectionalStreamingCall<GrpcEagerServiceImpl,
                                            grpc::EagerService::AsyncService,
                                            RequestMessage, ResponseMessage>;

  GrpcEagerServiceImpl(WorkerEnv* env, ::grpc::ServerBuilder* server_builder);
  virtual ~GrpcEagerServiceImpl() {}

  // Create a master context in eager service.
  Status CreateMasterContext(const tensorflow::uint64 context_id,
                             EagerContext* context);

  void HandleRPCsLoop() override;
  void Shutdown() override;

 private:
#define HANDLER(method)                                                       \
  void method##Handler(EagerCall<method##Request, method##Response>* call) {  \
    env_->compute_pool->Schedule([this, call]() {                             \
      call->SendResponse(                                                     \
          ToGrpcStatus(local_impl_.method(&call->request, &call->response))); \
    });                                                                       \
    tsl::Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,         \
              method##Request, method##Response>::                            \
        EnqueueRequest(&service_, cq_.get(),                                  \
                       &grpc::EagerService::AsyncService::Request##method,    \
                       &GrpcEagerServiceImpl::method##Handler, false);        \
  }
  HANDLER(CreateContext);
  HANDLER(UpdateContext);
  HANDLER(WaitQueueDone);
  HANDLER(KeepAlive);
  HANDLER(CloseContext);
#undef HANDLER

  void EnqueueHandler(EagerCall<EnqueueRequest, EnqueueResponse>* call) {
    env_->compute_pool->Schedule([this, call]() {
      auto call_opts = std::make_shared<CallOptions>();
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      call->SendResponse(ToGrpcStatus(local_impl_.Enqueue(
          call_opts.get(), &call->request, &call->response)));
    });
    tsl::Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,
              EnqueueRequest, EnqueueResponse>::
        EnqueueRequest(&service_, cq_.get(),
                       &grpc::EagerService::AsyncService::RequestEnqueue,
                       &GrpcEagerServiceImpl::EnqueueHandler,
                       /*supports_cancel=*/true);
  }

  void RunComponentFunctionHandler(
      EagerCall<RunComponentFunctionRequest, RunComponentFunctionResponse>*
          call) {
    env_->compute_pool->Schedule([this, call]() {
      auto call_opts = std::make_shared<CallOptions>();
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      local_impl_.RunComponentFunction(call_opts.get(), &call->request,
                                       &call->response,
                                       [call, call_opts](const Status& s) {
                                         call->ClearCancelCallback();
                                         call->SendResponse(ToGrpcStatus(s));
                                       });
    });
    tsl::Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,
              RunComponentFunctionRequest, RunComponentFunctionResponse>::
        EnqueueRequest(
            &service_, cq_.get(),
            &grpc::EagerService::AsyncService::RequestRunComponentFunction,
            &GrpcEagerServiceImpl::RunComponentFunctionHandler,
            /*supports_cancel=*/true);
  }

  // Called when a new request has been received as part of a StreamingEnqueue
  // call.
  // StreamingEnqueueHandler gets the request from the `call` and fills the
  // response (also found in `call`) by invoking the local EagerServiceImpl.
  // The local EagerServiceImpl is invoked in a single-threaded thread pool. We
  // do this to preserve request order. The local service can parallelize based
  // on context_id in request if necessary. Remote contexts are created in async
  // mode by default, so the local service impl just puts the request on eager
  // executor queue.
  void StreamingEnqueueHandler(
      StreamingCall<EnqueueRequest, EnqueueResponse>* call) {
    call->Ref();
    enqueue_streaming_thread_.Schedule([this, call]() {
      if (call->RefCountIsOne()) {
        // This StreamingCall has already been shutdown. Don't need to anything.
        call->Unref();
        return;
      }
      // NOTE(fishx): Use the address of StreamingCall as the stream_id since we
      // reuse the same StreamingCall for multiple requests in the same
      // streaming connection.
      Status status = local_impl_.Enqueue(
          /*call_opts=*/nullptr, &call->request(), call->mutable_response(),
          reinterpret_cast<uint64>(static_cast<void*>(call)));

      if (status.ok()) {
        VLOG(1) << "local_impl_.Enqueue completed successfully";
        call->SendResponse();
      } else {
        VLOG(1) << "local_impl_.Enqueue failed with " << status.ToString()
                << " on request " << call->request().DebugString();
        call->Finish(ToGrpcStatus(status));
      }
      call->Unref();

      // We do not tell gRPC to accept a new StreamingEnqueue request because
      // this method can be called multiple times for a given streaming call.
      // The StreamingCall does this per call instead, after a call has been
      // opened.
    });
  }

  WorkerEnv* const env_;  // Not owned.
  EagerServiceImpl local_impl_;

  // A single-threaded thread pool to handle streaming enqueue rpc request.
  thread::ThreadPool enqueue_streaming_thread_;
  std::unique_ptr<::grpc::Alarm> shutdown_alarm_;

  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::EagerService::AsyncService service_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcEagerServiceImpl);
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_SERVICE_IMPL_H_
