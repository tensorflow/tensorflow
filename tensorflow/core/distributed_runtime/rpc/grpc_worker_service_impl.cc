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

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler_impl.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {

const char* GrpcWorkerMethodName(GrpcWorkerMethod id) {
  switch (id) {
    case GrpcWorkerMethod::kGetStatus:
      return "/tensorflow.WorkerService/GetStatus";
    case GrpcWorkerMethod::kCreateWorkerSession:
      return "/tensorflow.WorkerService/CreateWorkerSession";
    case GrpcWorkerMethod::kDeleteWorkerSession:
      return "/tensorflow.WorkerService/DeleteWorkerSession";
    case GrpcWorkerMethod::kRegisterGraph:
      return "/tensorflow.WorkerService/RegisterGraph";
    case GrpcWorkerMethod::kDeregisterGraph:
      return "/tensorflow.WorkerService/DeregisterGraph";
    case GrpcWorkerMethod::kRunGraph:
      return "/tensorflow.WorkerService/RunGraph";
    case GrpcWorkerMethod::kCleanupGraph:
      return "/tensorflow.WorkerService/CleanupGraph";
    case GrpcWorkerMethod::kCleanupAll:
      return "/tensorflow.WorkerService/CleanupAll";
    case GrpcWorkerMethod::kRecvTensor:
      return "/tensorflow.WorkerService/RecvTensor";
    case GrpcWorkerMethod::kRecvBuf:
      return "/tensorflow.WorkerService/RecvBuf";
    case GrpcWorkerMethod::kLogging:
      return "/tensorflow.WorkerService/Logging";
    case GrpcWorkerMethod::kTracing:
      return "/tensorflow.WorkerService/Tracing";
    case GrpcWorkerMethod::kCompleteGroup:
      return "/tensorflow.WorkerService/CompleteGroup";
    case GrpcWorkerMethod::kCompleteInstance:
      return "/tensorflow.WorkerService/CompleteInstance";
    case GrpcWorkerMethod::kGetStepSequence:
      return "/tensorflow.WorkerService/GetStepSequence";
    case GrpcWorkerMethod::kMarkRecvFinished:
      return "/tensorflow.WorkerService/MarkRecvFinished";
  }
  // Shouldn't be reached.
  LOG(FATAL) << "Invalid id: this line shouldn't be reached.";
  return "invalid id";
}

namespace grpc {

WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
