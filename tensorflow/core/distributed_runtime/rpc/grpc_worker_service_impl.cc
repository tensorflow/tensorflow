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

const char* GrpcWorkerMethodName(GrpcWorkerMethod id) {
  switch (id) {
    case GrpcWorkerMethod::kGetStatus:
      return "/tensorflow.WorkerService/GetStatus";
    case GrpcWorkerMethod::kCreateWorkerSession:
      return "/tensorflow.WorkerService/CreateWorkerSession";
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
    case GrpcWorkerMethod::kLogging:
      return "/tensorflow.WorkerService/Logging";
    case GrpcWorkerMethod::kTracing:
      return "/tensorflow.WorkerService/Tracing";
  }
  // Shouldn't be reached.
  LOG(FATAL) << "Invalid id: this line shouldn't be reached.";
  return "invalid id";
}

namespace grpc {

WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(
        GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
        ::grpc::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
