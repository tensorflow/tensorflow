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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRPC_WORKER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRPC_WORKER_IMPL_H_

#include "grpcpp/server_builder.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"

namespace tensorflow {
namespace data {

// This class is a wrapper that handles communication for gRPC.
//
// Example usage:
//
// ::grpc::ServerBuilder builder;
// // configure builder
// GrpcWorkerImpl data_service(&builder);
// builder.BuildAndStart()
//
class GrpcWorkerImpl : public WorkerService::Service {
 public:
  explicit GrpcWorkerImpl(grpc::ServerBuilder* server_builder,
                          const std::string& master_address,
                          const std::string& protocol);
  ~GrpcWorkerImpl() override {}

  void Start(const std::string& worker_address);

#define HANDLER(method)                               \
  grpc::Status method(grpc::ServerContext* context,   \
                      const method##Request* request, \
                      method##Response* response) override;
  HANDLER(ProcessTask);
  HANDLER(GetElement);
#undef HANDLER

 private:
  DataServiceWorkerImpl impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_WORKER_IMPL_H_
