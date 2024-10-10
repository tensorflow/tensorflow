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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"

#include <memory>

#include "net/grpc/public/include/grpc/grpc.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

namespace tensorflow {
namespace eager {

GrpcEagerServiceImpl::GrpcEagerServiceImpl(
    WorkerEnv* env, ::grpc::ServerBuilder* server_builder)
    : env_(env),
      local_impl_(env),
      enqueue_streaming_thread_(env_->env, "enqueue_streaming_thread", 1) {
  server_builder->RegisterService(&service_);
  // gRPC by default will cancel requests that sit in a completion queue for
  // more than 30s. See
  // https://github.com/grpc/grpc/blob/e52e48b7ef83feeff56ed0894ce39841ea8bd483/include/grpc/impl/channel_arg_names.h#L106-L111
  // Extending this to 1 hour for Tensorflow since some graphs may have periods
  // of heavy load which may cause the server to run into these cancellations.
  server_builder->AddChannelArgument(
      GRPC_ARG_SERVER_MAX_UNREQUESTED_TIME_IN_SERVER_SECONDS, 3600);
  cq_ = server_builder->AddCompletionQueue();
}

Status GrpcEagerServiceImpl::CreateMasterContext(
    const tensorflow::uint64 context_id, EagerContext* context) {
  return local_impl_.CreateMasterContext(context_id, context);
}

void GrpcEagerServiceImpl::HandleRPCsLoop() {
#define ENQUEUE_REQUEST(method)                                            \
  do {                                                                     \
    tsl::Call<GrpcEagerServiceImpl, grpc::EagerService::AsyncService,      \
              method##Request, method##Response>::                         \
        EnqueueRequest(&service_, cq_.get(),                               \
                       &grpc::EagerService::AsyncService::Request##method, \
                       &GrpcEagerServiceImpl::method##Handler, false);     \
  } while (0)
  ENQUEUE_REQUEST(CreateContext);
  ENQUEUE_REQUEST(UpdateContext);
  ENQUEUE_REQUEST(Enqueue);
  ENQUEUE_REQUEST(WaitQueueDone);
  ENQUEUE_REQUEST(RunComponentFunction);
  ENQUEUE_REQUEST(KeepAlive);
  ENQUEUE_REQUEST(CloseContext);
#undef ENQUEUE_REQUEST

  // Request a StreamingEnqueue call.
  tsl::ServerBidirectionalStreamingCall<GrpcEagerServiceImpl,
                                        grpc::EagerService::AsyncService,
                                        EnqueueRequest, EnqueueResponse>::
      EnqueueRequest(&service_, cq_.get(),
                     &grpc::EagerService::AsyncService::RequestStreamingEnqueue,
                     &GrpcEagerServiceImpl::StreamingEnqueueHandler);

  void* tag;  // Matches the operation started against this cq_.
  bool ok;

  while (true) {
    if (!cq_->Next(&tag, &ok)) {
      // The queue is shutting down.
      break;
    }
    tsl::GrpcCallTag<GrpcEagerServiceImpl>* callback_tag =
        static_cast<tsl::GrpcCallTag<GrpcEagerServiceImpl>*>(tag);

    if (callback_tag) {
      callback_tag->OnCompleted(this, ok);
    } else {
      cq_->Shutdown();
      break;
    }
  }
}

void GrpcEagerServiceImpl::Shutdown() {
  // This enqueues a special event (with a null tag)
  // that causes the completion queue to be shut down on the
  // polling thread.
  shutdown_alarm_ = std::make_unique<::grpc::Alarm>(
      cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
}

}  // namespace eager
}  // namespace tensorflow
