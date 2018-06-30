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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace eager {

GrpcEagerServiceImpl::GrpcEagerServiceImpl(
    const WorkerEnv* env, ::grpc::ServerBuilder* server_builder)
    : local_impl_(env) {
  request_handler_threadpool_ =
      MakeUnique<thread::ThreadPool>(env->env, "EagerServiceRequestHandler", 4);
  server_builder->RegisterService(&service_);
  cq_ = server_builder->AddCompletionQueue();
}

void GrpcEagerServiceImpl::HandleRPCsLoop() {
#define ENQUEUE_REQUEST(method)                                                \
  do {                                                                         \
    Call<GrpcEagerServiceImpl,                                                 \
         tensorflow::eager::grpc::EagerService::AsyncService, method##Request, \
         method##Response>::                                                   \
        EnqueueRequest(&service_, cq_.get(),                                   \
                       &grpc::EagerService::AsyncService::Request##method,     \
                       &GrpcEagerServiceImpl::method##Handler, false);         \
  } while (0)
  ENQUEUE_REQUEST(CreateContext);
  ENQUEUE_REQUEST(Enqueue);
  ENQUEUE_REQUEST(WaitQueueDone);
  ENQUEUE_REQUEST(KeepAlive);
  ENQUEUE_REQUEST(CloseContext);
  ENQUEUE_REQUEST(RegisterFunction);
#undef ENQUEUE_REQUEST

  void* tag;  // Matches the operation started against this cq_.
  bool ok;

  while (true) {
    if (!cq_->Next(&tag, &ok)) {
      // The queue is shutting down.
      break;
    }
    UntypedCall<GrpcEagerServiceImpl>::Tag* callback_tag =
        static_cast<UntypedCall<GrpcEagerServiceImpl>::Tag*>(tag);

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
  shutdown_alarm_ = MakeUnique<::grpc::Alarm>(
      cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
}

}  // namespace eager
}  // namespace tensorflow
