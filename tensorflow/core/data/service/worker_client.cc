/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/worker_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

Status DataServiceWorkerClient::GetElement(const GetElementRequest& req,
                                           GetElementResult& result) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  return client_->GetElement(req, result);
}

Status DataServiceWorkerClient::EnsureInitialized() {
  mutex_lock l(mu_);
  if (client_) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(DataTransferClient::Build(
      transfer_protocol_, {protocol_, address_}, &client_));
  return Status::OK();
}

void DataServiceWorkerClient::TryCancel() { client_->TryCancel(); }

Status CreateDataServiceWorkerClient(
    const std::string& address, const std::string& protocol,
    const std::string& transfer_protocol,
    std::unique_ptr<DataServiceWorkerClient>& out) {
  auto client = absl::make_unique<DataServiceWorkerClient>(address, protocol,
                                                           transfer_protocol);
  TF_RETURN_IF_ERROR(client->Initialize());
  out = std::move(client);
  return Status::OK();
}

class GrpcDataTransferClient : public DataTransferClient {
 public:
  GrpcDataTransferClient(std::shared_ptr<grpc::ChannelCredentials> credentials,
                         std::string address) {
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(-1);
    auto channel = grpc::CreateCustomChannel(address, credentials, args);
    stub_ = WorkerService::NewStub(channel);
  }

  Status GetElement(const GetElementRequest& req,
                    GetElementResult& result) override {
    {
      mutex_lock l(mu_);
      if (cancelled_) {
        return errors::Cancelled("Client was cancelled.");
      }
    }
    grpc::ClientContext ctx;
    {
      mutex_lock l(mu_);
      active_contexts_.insert(&ctx);
    }
    GetElementResponse resp;
    grpc::Status s = stub_->GetElement(&ctx, req, &resp);
    result.end_of_sequence = resp.end_of_sequence();
    result.skip = resp.skip_task();
    switch (resp.element_case()) {
      case GetElementResponse::kCompressed: {
        Tensor tensor(DT_VARIANT, TensorShape{});
        tensor.scalar<Variant>()() = std::move(resp.compressed());
        result.components.push_back(tensor);
        break;
      }
      case GetElementResponse::kUncompressed:
        for (const auto& component : resp.uncompressed().components()) {
          result.components.emplace_back();
          if (!result.components.back().FromProto(component)) {
            return errors::Internal("Failed to parse tensor.");
          }
        }
        break;
      case GetElementResponse::ELEMENT_NOT_SET:
        break;
    }
    {
      mutex_lock l(mu_);
      active_contexts_.erase(&ctx);
    }
    if (!s.ok()) {
      return grpc_util::WrapError("Failed to get element", s);
    }
    return Status::OK();
  }

  void TryCancel() override {
    mutex_lock l(mu_);
    cancelled_ = true;
    for (const auto& ctx : active_contexts_) {
      ctx->TryCancel();
    }
  }

 private:
  mutex mu_;
  std::unique_ptr<WorkerService::Stub> stub_;
  // Set of all currently active clients contexts. Used to support
  // cancellation.
  absl::flat_hash_set<::grpc::ClientContext*> active_contexts_
      TF_GUARDED_BY(mu_);
  // Indicates that the client has been cancelled, so no further requests should
  // be accepted.
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
};

class GrpcTransferClientRegistrar {
 public:
  GrpcTransferClientRegistrar() {
    DataTransferClient::Register(
        "grpc", [](DataTransferClient::Config config,
                   std::unique_ptr<DataTransferClient>* out) {
          std::shared_ptr<grpc::ChannelCredentials> credentials;
          TF_RETURN_IF_ERROR(CredentialsFactory::CreateClientCredentials(
              config.protocol, &credentials));
          *out = std::make_unique<GrpcDataTransferClient>(credentials,
                                                          config.address);
          return Status::OK();
        });
  }
};
static GrpcTransferClientRegistrar registrar;

}  // namespace data
}  // namespace tensorflow
