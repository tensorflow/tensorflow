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
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

namespace {
constexpr const char kParallelEpochs[] = "parallel_epochs";
constexpr const char kOneEpoch[] = "one_epoch";
}  // namespace

Status ParseProcessingMode(const std::string& s, ProcessingMode* mode) {
  if (s == kParallelEpochs) {
    *mode = ProcessingMode::PARALLEL_EPOCHS;
  } else if (s == kOneEpoch) {
    *mode = ProcessingMode::ONE_EPOCH;
  } else {
    return errors::InvalidArgument("Unrecognized processing mode: ", s);
  }
  return Status::OK();
}

std::string ProcessingModeToString(ProcessingMode mode) {
  switch (mode) {
    case ProcessingMode::PARALLEL_EPOCHS:
      return kParallelEpochs;
    case ProcessingMode::ONE_EPOCH:
      return kOneEpoch;
    default:
      DCHECK(false);
      return "Unknown";
  }
}

Status DataServiceMasterClient::RegisterDataset(GraphDef dataset,
                                                int64* dataset_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset()->mutable_graph() = dataset;
  GetOrRegisterDatasetResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrRegisterDataset(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to register dataset", status);
  }
  *dataset_id = resp.dataset_id();
  return Status::OK();
}

Status DataServiceMasterClient::CreateJob(int64 dataset_id,
                                          ProcessingMode processing_mode,
                                          int64* job_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  CreateJobRequest req;
  req.set_dataset_id(dataset_id);
  req.set_processing_mode(ProcessingModeDef(processing_mode));
  CreateJobResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->CreateJob(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to create job for dataset with id ", dataset_id),
        status);
  }
  *job_id = resp.job_id();
  return Status::OK();
}

Status DataServiceMasterClient::GetOrCreateJob(int64 dataset_id,
                                               ProcessingMode processing_mode,
                                               const std::string& job_name,
                                               int job_name_index,
                                               int64* job_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrCreateJobRequest req;
  req.set_dataset_id(dataset_id);
  req.set_processing_mode(ProcessingModeDef(processing_mode));
  req.set_job_name(job_name);
  req.set_job_name_index(job_name_index);
  GetOrCreateJobResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrCreateJob(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get or create job for dataset with id ",
                     dataset_id),
        status);
  }
  *job_id = resp.job_id();
  return Status::OK();
}

Status DataServiceMasterClient::GetTasks(int64 job_id,
                                         std::vector<TaskInfo>* tasks,
                                         bool* job_finished) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetTasksRequest req;
  req.set_job_id(job_id);
  GetTasksResponse resp;
  grpc_impl::ClientContext ctx;
  grpc::Status s = stub_->GetTasks(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  tasks->clear();
  for (auto& task : resp.task_info()) {
    tasks->push_back(task);
  }
  *job_finished = resp.job_finished();
  return Status::OK();
}

Status DataServiceMasterClient::GetWorkers(std::vector<WorkerInfo>* workers) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetWorkersRequest req;
  GetWorkersResponse resp;
  grpc_impl::ClientContext ctx;
  grpc::Status s = stub_->GetWorkers(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  workers->clear();
  for (auto& worker : resp.workers()) {
    workers->push_back(worker);
  }
  return Status::OK();
}

Status DataServiceMasterClient::EnsureInitialized() {
  std::shared_ptr<grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  auto channel = grpc::CreateChannel(address_, credentials);
  stub_ = MasterService::NewStub(channel);
  return Status::OK();
}

Status DataServiceWorkerClient::GetElement(int64 task_id,
                                           CompressedElement* element,
                                           bool* end_of_sequence) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetElementRequest req;
  req.set_task_id(task_id);
  GetElementResponse resp;
  grpc_impl::ClientContext ctx;
  grpc::Status s = stub_->GetElement(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get element", s);
  }
  *end_of_sequence = resp.end_of_sequence();
  if (!*end_of_sequence) {
    *element = std::move(*resp.mutable_compressed_element());
  }
  return Status::OK();
}

Status DataServiceWorkerClient::EnsureInitialized() {
  std::shared_ptr<grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  auto channel = grpc::CreateCustomChannel(address_, credentials, args);
  stub_ = WorkerService::NewStub(channel);
  return Status::OK();
}

Status CreateDataServiceMasterClient(
    const std::string& address, const std::string& protocol,
    std::unique_ptr<DataServiceMasterClient>* out) {
  auto client = absl::make_unique<DataServiceMasterClient>(address, protocol);
  TF_RETURN_IF_ERROR(client->Initialize());
  *out = std::move(client);
  return Status::OK();
}

Status CreateDataServiceWorkerClient(
    const std::string& address, const std::string& protocol,
    std::unique_ptr<DataServiceWorkerClient>* out) {
  auto client = absl::make_unique<DataServiceWorkerClient>(address, protocol);
  TF_RETURN_IF_ERROR(client->Initialize());
  *out = std::move(client);
  return Status::OK();
}
}  // namespace data
}  // namespace tensorflow
