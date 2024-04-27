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

#include <cstdint>
#include <functional>
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
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace data {

StatusOr<std::unique_ptr<DataServiceWorkerClient>>
CreateDataServiceWorkerClient(const std::string& dispatcher_protocol,
                              const DataTransferServerInfo& info,
                              Allocator* allocator) {
  auto client = std::make_unique<DataServiceWorkerClient>(
      info.address(), dispatcher_protocol, info.protocol(), allocator);
  TF_RETURN_IF_ERROR(client->Initialize());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      client->CheckCompatibility(info.compatibility_info()),
      "for data transfer protocol '", client->GetDataTransferProtocol(),
      "', the compatibility check between the trainer worker and the ",
      "tf.data service worker at ", info.address(), "failed");
  return client;
}

Status DataServiceWorkerClient::GetElement(const GetElementRequest& req,
                                           GetElementResult& result) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  return client_->GetElement(req, result);
}

Status DataServiceWorkerClient::EnsureInitialized() {
  mutex_lock l(mu_);
  if (client_) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(DataTransferClient::Build(
      GetDataTransferProtocol(), {protocol_, address_, allocator_}, &client_));
  return absl::OkStatus();
}

std::string DataServiceWorkerClient::GetDataTransferProtocol() const {
  if (LocalWorkers::Get(address_) != nullptr) {
    return kLocalTransferProtocol;
  }
  return transfer_protocol_;
}

void DataServiceWorkerClient::TryCancel() { client_->TryCancel(); }

class GrpcDataTransferClient : public DataTransferClient {
 public:
  GrpcDataTransferClient(std::shared_ptr<grpc::ChannelCredentials> credentials,
                         std::string address) {
    VLOG(2) << "Create GrpcDataTransferClient for worker " << address << ".";
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(-1);
    auto channel = grpc::CreateCustomChannel(address, credentials, args);
    stub_ = WorkerService::NewStub(channel);
  }

  Status GetElement(const GetElementRequest& req,
                    GetElementResult& result) override {
    VLOG(3) << "GetElement for task " << req.task_id() << " from gRPC worker "
            << "server.";
    {
      mutex_lock l(mu_);
      if (cancelled_) {
        return errors::Cancelled("Client was cancelled.");
      }
    }
    grpc::ClientContext ctx;
    gtl::Cleanup<std::function<void()>> cleanup;
    {
      mutex_lock l(mu_);
      active_contexts_.insert(&ctx);
      cleanup = gtl::MakeCleanup([this, &ctx] {
        mutex_lock l(mu_);
        active_contexts_.erase(&ctx);
      });
    }
    GetElementResponse resp;
    int64_t start_time_us = env_->NowMicros();
    grpc::Status s = stub_->GetElement(&ctx, req, &resp);
    int64_t end_time_us = env_->NowMicros();
    if (!s.ok()) {
      return grpc_util::WrapError("Failed to get element", s);
    }
    metrics::RecordTFDataServiceGetElementDuration(kGrpcTransferProtocol,
                                                   end_time_us - start_time_us);
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
    return absl::OkStatus();
  }

  void TryCancel() override {
    VLOG(2) << "Cancel GrpcDataTransferClient.";
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
        kGrpcTransferProtocol, [](DataTransferClient::Config config,
                                  std::unique_ptr<DataTransferClient>* out) {
          std::shared_ptr<grpc::ChannelCredentials> credentials;
          TF_RETURN_IF_ERROR(CredentialsFactory::CreateClientCredentials(
              config.protocol, &credentials));
          *out = std::make_unique<GrpcDataTransferClient>(credentials,
                                                          config.address);
          return absl::OkStatus();
        });
  }
};
static GrpcTransferClientRegistrar gprc_client_registrar;

class LocalDataTransferClient : public DataTransferClient {
 public:
  explicit LocalDataTransferClient(absl::string_view worker_address)
      : worker_address_(worker_address) {
    VLOG(2) << "Create LocalDataTransferClient for worker " << worker_address_
            << ".";
  }

  Status GetElement(const GetElementRequest& req,
                    GetElementResult& result) override {
    VLOG(3) << "GetElement for task " << req.task_id() << " from local worker.";
    TF_RETURN_IF_ERROR(VerifyClientIsNotCancelled());
    TF_ASSIGN_OR_RETURN(std::shared_ptr<DataServiceWorkerImpl> worker,
                        GetWorker(req));
    int64_t start_time_us = env_->NowMicros();
    Status s = worker->GetElementResult(&req, &result);
    int64_t end_time_us = env_->NowMicros();
    TF_RETURN_IF_ERROR(s);
    metrics::RecordTFDataServiceGetElementDuration(kLocalTransferProtocol,
                                                   end_time_us - start_time_us);
    return s;
  }

  void TryCancel() override {
    VLOG(2) << "Cancel LocalDataTransferClient for worker " << worker_address_
            << ".";
    // Cancels incoming requests. Currently local reads assume the requests are
    // first-come-first-served. If we need to support coordinated reads, we need
    // to cancel in-flight requests since they may wait infinitely.
    mutex_lock l(mu_);
    cancelled_ = true;
  }

 private:
  Status VerifyClientIsNotCancelled() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (cancelled_) {
      return errors::Cancelled(absl::Substitute(
          "Client for worker $0 has been cancelled.", worker_address_));
    }
    return absl::OkStatus();
  }

  StatusOr<std::shared_ptr<DataServiceWorkerImpl>> GetWorker(
      const GetElementRequest& req) const {
    std::shared_ptr<DataServiceWorkerImpl> worker =
        LocalWorkers::Get(worker_address_);
    if (!worker) {
      return errors::Cancelled(absl::Substitute(
          "Local worker at address $0 is no longer available; cancel request "
          "for task $1.",
          worker_address_, req.task_id()));
    }
    return worker;
  }

  const std::string worker_address_;

  mutex mu_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
};

class LocalTransferClientRegistrar {
 public:
  LocalTransferClientRegistrar() {
    DataTransferClient::Register(
        kLocalTransferProtocol, [](DataTransferClient::Config config,
                                   std::unique_ptr<DataTransferClient>* out) {
          *out = std::make_unique<LocalDataTransferClient>(config.address);
          return absl::OkStatus();
        });
  }
};
static LocalTransferClientRegistrar local_client_registrar;

}  // namespace data
}  // namespace tensorflow
