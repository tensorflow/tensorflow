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

#include "tensorflow/core/data/service/master_impl.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {

namespace {
Status CreateWorkerStub(const std::string& address,
                        const std::string& protocol_,
                        std::unique_ptr<WorkerService::Stub>* stub) {
  ::grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<::grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  auto channel = ::grpc::CreateCustomChannel(address, credentials, args);
  *stub = WorkerService::NewStub(channel);
  return Status::OK();
}
}  // namespace

DataServiceMasterImpl::DataServiceMasterImpl(const std::string protocol)
    : protocol_(protocol) {}

Status DataServiceMasterImpl::RegisterWorker(
    const RegisterWorkerRequest* request, RegisterWorkerResponse* response) {
  VLOG(3) << "Received register worker request";
  mutex_lock l(mu_);
  int64 worker_id = next_worker_id_++;
  workers_.emplace_back();
  workers_.back().address = request->worker_address();
  workers_.back().id = worker_id;
  response->set_worker_id(worker_id);

  // Allocate tasks to the worker.
  for (auto& entry : epochs_) {
    Epoch& epoch = entry.second;
    int64 task_id = next_task_id_++;
    DCHECK(!tasks_.contains(task_id));
    Task& task = tasks_[task_id];
    task.id = task_id;
    task.dataset_id = epoch.dataset_id;
    task.worker_address = request->worker_address();
    epoch.task_ids.push_back(task_id);

    TaskDef* task_def = response->add_tasks();
    *task_def->mutable_dataset() =
        datasets_by_id_[task.dataset_id]->dataset_def;
    task_def->set_dataset_id(task.dataset_id);
    task_def->set_epoch_id(epoch.id);
    task_def->set_task_id(task.id);
  }

  VLOG(1) << "Registered worker " << workers_.back().DebugString();
  return Status::OK();
}

Status DataServiceMasterImpl::GetOrRegisterDataset(
    const GetOrRegisterDatasetRequest* request,
    GetOrRegisterDatasetResponse* response) {
  uint64 fingerprint;
  TF_RETURN_IF_ERROR(HashGraph(request->dataset().graph(), &fingerprint));
  mutex_lock l(mu_);
  VLOG(3) << "Registering dataset graph: "
          << request->dataset().graph().DebugString();
  if (datasets_by_fingerprint_.contains(fingerprint)) {
    int64 id = datasets_by_fingerprint_[fingerprint]->id;
    VLOG(3) << "Received duplicate RegisterDataset request with fingerprint "
            << fingerprint << ". Returning id " << id;
    response->set_dataset_id(id);
    return Status::OK();
  }
  int64 id = RegisterDataset(fingerprint, request->dataset());

  response->set_dataset_id(id);
  VLOG(3) << "Registered new dataset with id " << id;
  return Status::OK();
}

int64 DataServiceMasterImpl::RegisterDataset(uint64 fingerprint,
                                             const DatasetDef& dataset)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  auto new_dataset = std::make_shared<Dataset>();
  int64 dataset_id = next_dataset_id_++;
  new_dataset->id = dataset_id;
  new_dataset->fingerprint = fingerprint;
  new_dataset->dataset_def = dataset;

  DCHECK(!datasets_by_id_.contains(dataset_id));
  datasets_by_id_[dataset_id] = new_dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[dataset_id] = new_dataset;
  return dataset_id;
}

Status DataServiceMasterImpl::BeginEpoch(const BeginEpochRequest* request,
                                         BeginEpochResponse* response) {
  VLOG(3) << "Received begin epoch request for dataset id "
          << request->dataset_id();
  mutex_lock l(mu_);
  if (!datasets_by_id_.contains(request->dataset_id())) {
    return errors::NotFound("BeginEpoch failed. Dataset id: <",
                            request->dataset_id(), "> not found.");
  }

  int64 epoch_id = next_epoch_id_++;
  DCHECK(!epochs_.contains(epoch_id));
  Epoch& epoch = epochs_[epoch_id];
  epoch.id = epoch_id;
  epoch.dataset_id = request->dataset_id();
  response->set_epoch_id(epoch_id);

  for (auto& worker : workers_) {
    int64 task_id = next_task_id_++;
    DCHECK(!tasks_.contains(task_id));
    Task& task = tasks_[task_id];
    task.id = task_id;
    task.dataset_id = request->dataset_id();
    task.worker_address = worker.address;
    epoch.task_ids.push_back(task_id);

    std::unique_ptr<WorkerService::Stub> stub;
    TF_RETURN_IF_ERROR(CreateWorkerStub(worker.address, protocol_, &stub));
    // TODO(aaudibert): perform these calls asynchronously.
    TF_RETURN_IF_ERROR(AllocateTaskToWorker(task, &worker));
  }

  VLOG(3) << "Beginning epoch " << epoch_id << " for dataset "
          << request->dataset_id();
  return Status::OK();
}

Status DataServiceMasterImpl::AllocateTaskToWorker(const Task& task,
                                                   WorkerInfo* worker)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!worker->stub) {
    TF_RETURN_IF_ERROR(
        CreateWorkerStub(worker->address, protocol_, &worker->stub));
  }
  grpc::ClientContext client_ctx;
  ProcessTaskRequest req;
  req.mutable_task()->set_dataset_id(task.dataset_id);
  DCHECK(datasets_by_id_.contains(task.dataset_id));
  *req.mutable_task()->mutable_dataset() =
      datasets_by_id_[task.dataset_id]->dataset_def;
  req.mutable_task()->set_task_id(task.id);
  ProcessTaskResponse resp;
  grpc::Status s = worker->stub->ProcessTask(&client_ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to submit task to worker ", worker->address), s);
  }
  return Status::OK();
}

Status DataServiceMasterImpl::GetTasks(const GetTasksRequest* request,
                                       GetTasksResponse* response) {
  mutex_lock l(mu_);
  VLOG(3) << "Looking up tasks for epoch id " << request->epoch_id();
  auto it = epochs_.find(request->epoch_id());
  if (it == epochs_.end()) {
    return errors::NotFound("GetTasks failed. Epoch id <", request->epoch_id(),
                            "> not found.");
  }
  Epoch& epoch = it->second;
  for (const auto& task_id : epoch.task_ids) {
    auto task_iter = tasks_.find(task_id);
    DCHECK(task_iter != tasks_.end());
    Task& task = task_iter->second;
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task.worker_address);
    task_info->set_id(task.id);
  }
  VLOG(3) << "Found " << response->task_info_size() << " tasks for epoch id "
          << request->epoch_id();
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
