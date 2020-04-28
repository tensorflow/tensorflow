/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/grpc_master_impl.h"

#include "grpcpp/server_context.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace tensorflow {
namespace data {

using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;

GrpcMasterImpl::GrpcMasterImpl(ServerBuilder* server_builder,
                               const std::string& protocol)
    : impl_(protocol) {
  server_builder->RegisterService(this);
  VLOG(1) << "Registered data service master";
}

#define HANDLER(method)                                         \
  Status GrpcMasterImpl::method(ServerContext* context,         \
                                const method##Request* request, \
                                method##Response* response) {   \
    return ToGrpcStatus(impl_.method(request, response));       \
  }
HANDLER(RegisterWorker);
HANDLER(WorkerUpdate);
HANDLER(GetOrRegisterDataset);
HANDLER(CreateJob);
HANDLER(GetTasks);
#undef HANDLER

}  // namespace data
}  // namespace tensorflow
