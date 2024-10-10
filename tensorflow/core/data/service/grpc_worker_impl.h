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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/server_builder.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// This class is a wrapper that handles communication for gRPC.
class GrpcWorkerImpl : public WorkerService::Service {
 public:
  // Constructs a GrpcWorkerImpl with the given config, and registers it with
  // `server_builder`.
  explicit GrpcWorkerImpl(const experimental::WorkerConfig& config,
                          ::grpc::ServerBuilder& server_builder);
  ~GrpcWorkerImpl() override { Stop(); }

  absl::Status Start(
      const std::string& worker_address,
      const std::vector<DataTransferServerInfo>& transfer_servers);
  void Stop();

  std::function<absl::Status(const GetElementRequest*, GetElementResult*)>
  get_element_getter() {
    return [this](const GetElementRequest* request, GetElementResult* result) {
      return impl_->GetElementResult(request, result);
    };
  }

  WorkerStateExport ExportState() const;

#define HANDLER(method)                                 \
  ::grpc::Status method(::grpc::ServerContext* context, \
                        const method##Request* request, \
                        method##Response* response) override;
  HANDLER(ProcessTask);
  HANDLER(GetElement);
  HANDLER(GetWorkerTasks);
  HANDLER(GetSnapshotTaskProgresses);
#undef HANDLER

 private:
  std::string worker_address_;
  // A std::shared_ptr allows clients to access local servers and directly call
  // the servers' methods to avoid RPC calls and data copy.
  std::shared_ptr<DataServiceWorkerImpl> impl_;

  GrpcWorkerImpl(const GrpcWorkerImpl&) = delete;
  void operator=(const GrpcWorkerImpl&) = delete;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_WORKER_IMPL_H_
