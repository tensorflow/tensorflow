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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRPC_MASTER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRPC_MASTER_IMPL_H_

#include "grpcpp/server_builder.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/master_impl.h"

namespace tensorflow {
namespace data {

// This class is a wrapper that handles communication for gRPC.
//
// Example usage:
//
// ::grpc::ServerBuilder builder;
// // configure builder
// GrpcMasterImpl data_service(&builder);
// builder.BuildAndStart()
//
class GrpcMasterImpl : public MasterService::Service {
 public:
  explicit GrpcMasterImpl(grpc::ServerBuilder* server_builder,
                          const std::string& protocol);
  ~GrpcMasterImpl() override {}

#define HANDLER(method)                               \
  grpc::Status method(grpc::ServerContext* context,   \
                      const method##Request* request, \
                      method##Response* response) override;
  HANDLER(RegisterWorker);
  HANDLER(WorkerUpdate);
  HANDLER(GetOrRegisterDataset);
  HANDLER(CreateJob);
  HANDLER(GetOrCreateJob);
  HANDLER(GetTasks);
#undef HANDLER

 private:
  DataServiceMasterImpl impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_MASTER_IMPL_H_
