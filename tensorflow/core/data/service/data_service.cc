/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/data_service.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

namespace {
constexpr const char kParallelEpochs[] = "parallel_epochs";
constexpr const char kDistributedEpoch[] = "distributed_epoch";

}  // namespace

Status ParseProcessingMode(const std::string& s, ProcessingMode& mode) {
  if (s == kParallelEpochs) {
    mode = ProcessingMode::PARALLEL_EPOCHS;
  } else if (s == kDistributedEpoch) {
    mode = ProcessingMode::DISTRIBUTED_EPOCH;
  } else {
    return errors::InvalidArgument("Unrecognized processing mode: ", s);
  }
  return Status::OK();
}

std::string ProcessingModeToString(ProcessingMode mode) {
  switch (mode) {
    case ProcessingMode::PARALLEL_EPOCHS:
      return kParallelEpochs;
    case ProcessingMode::DISTRIBUTED_EPOCH:
      return kDistributedEpoch;
    default:
      DCHECK(false);
      return "Unknown";
  }
}

Status DataServiceDispatcherClient::WorkerHeartbeat(
    const std::string& worker_address, const std::string& transfer_address,
    const std::vector<int64>& current_tasks, std::vector<TaskDef>& new_tasks,
    std::vector<int64>& tasks_to_delete) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  WorkerHeartbeatRequest req;
  req.set_worker_address(worker_address);
  req.set_transfer_address(transfer_address);
  for (int64 task : current_tasks) {
    req.add_current_tasks(task);
  }
  WorkerHeartbeatResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerHeartbeat(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to perform worker heartbeat", status);
  }
  for (const auto& task : resp.new_tasks()) {
    new_tasks.push_back(task);
  }
  for (int64 task_to_delete : resp.tasks_to_delete()) {
    tasks_to_delete.push_back(task_to_delete);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::WorkerUpdate(
    const std::string& worker_address,
    std::vector<TaskProgress>& task_progress) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  WorkerUpdateRequest req;
  req.set_worker_address(worker_address);
  for (const auto& update : task_progress) {
    *(req.add_updates()) = update;
  }
  WorkerUpdateResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerUpdate(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to send worker update", status);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::GetDatasetDef(int64 dataset_id,
                                                  DatasetDef& dataset_def) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDatasetDefRequest req;
  req.set_dataset_id(dataset_id);
  GetDatasetDefResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetDatasetDef(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get dataset def", status);
  }
  dataset_def = resp.dataset_def();
  return Status::OK();
}

Status DataServiceDispatcherClient::GetSplit(int64 job_id, int64 repetition,
                                             Tensor& split,
                                             bool& end_of_splits) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetSplitRequest req;
  req.set_job_id(job_id);
  req.set_repetition(repetition);
  GetSplitResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetSplit(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get split", status);
  }
  end_of_splits = resp.end_of_splits();
  if (!end_of_splits) {
    if (!split.FromProto(resp.split())) {
      return errors::Internal("Failed to parse split tensor proto");
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::RegisterDataset(GraphDef dataset,
                                                    int64& dataset_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset()->mutable_graph() = dataset;
  GetOrRegisterDatasetResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrRegisterDataset(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to register dataset", status);
  }
  dataset_id = resp.dataset_id();
  return Status::OK();
}

Status DataServiceDispatcherClient::GetOrCreateJob(
    int64 dataset_id, ProcessingMode processing_mode,
    const absl::optional<JobKey>& job_key, absl::optional<int64> num_consumers,
    int64& job_client_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrCreateJobRequest req;
  req.set_dataset_id(dataset_id);
  req.set_processing_mode(ProcessingModeDef(processing_mode));
  if (job_key.has_value()) {
    *req.mutable_job_key() = job_key.value();
  }
  if (num_consumers.has_value()) {
    req.set_num_consumers(num_consumers.value());
  }
  GetOrCreateJobResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrCreateJob(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get or create job for dataset with id ",
                     dataset_id),
        status);
  }
  job_client_id = resp.job_client_id();
  return Status::OK();
}

Status DataServiceDispatcherClient::ReleaseJobClient(int64 job_client_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  ReleaseJobClientRequest req;
  req.set_job_client_id(job_client_id);
  ReleaseJobClientResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->ReleaseJobClient(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to release job client with id ", job_client_id),
        status);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::MaybeRemoveTask(int64 task_id,
                                                    int64 consumer_index,
                                                    int64 round,
                                                    bool& removed) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  MaybeRemoveTaskRequest req;
  req.set_task_id(task_id);
  req.set_consumer_index(consumer_index);
  req.set_round(round);
  MaybeRemoveTaskResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->MaybeRemoveTask(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to call MaybeRemoveTask", status);
  }
  removed = resp.removed();
  return Status::OK();
}

Status DataServiceDispatcherClient::ClientHeartbeat(
    ClientHeartbeatRequest& req, ClientHeartbeatResponse& resp) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  grpc::ClientContext ctx;
  grpc::Status s = stub_->ClientHeartbeat(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::GetWorkers(
    std::vector<WorkerInfo>& workers) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetWorkersRequest req;
  GetWorkersResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetWorkers(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  workers.clear();
  for (auto& worker : resp.workers()) {
    workers.push_back(worker);
  }
  return Status::OK();
}

Status DataServiceDispatcherClient::EnsureInitialized() {
  mutex_lock l(mu_);
  if (stub_) {
    return Status::OK();
  }
  std::shared_ptr<grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
  auto channel = grpc::CreateCustomChannel(address_, credentials, args);
  stub_ = DispatcherService::NewStub(channel);
  GetVersionRequest req;
  GetVersionResponse resp;
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&] {
        grpc::ClientContext ctx;
        grpc::Status s = stub_->GetVersion(&ctx, req, &resp);
        if (!s.ok()) {
          return grpc_util::WrapError("Failed to get dispatcher version", s);
        }
        return Status::OK();
      },
      "checking service version",
      /*deadline_micros=*/kint64max));
  if (resp.version() != kDataServiceVersion) {
    return errors::FailedPrecondition(
        "Version mismatch with tf.data service server. The server is running "
        "version ",
        resp.version(), ", while the client is running version ",
        kDataServiceVersion,
        ". Please ensure that the client and server side are running the "
        "same version of TensorFlow.");
  }
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

Status CreateDataServiceDispatcherClient(
    const std::string& address, const std::string& protocol,
    std::unique_ptr<DataServiceDispatcherClient>& out) {
  auto client =
      absl::make_unique<DataServiceDispatcherClient>(address, protocol);
  TF_RETURN_IF_ERROR(client->Initialize());
  out = std::move(client);
  return Status::OK();
}

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
}  // namespace data
}  // namespace tensorflow
