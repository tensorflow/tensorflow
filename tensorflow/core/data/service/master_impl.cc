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

#include <memory>
#include <tuple>
#include <utility>

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
#include "tensorflow/core/platform/errors.h"
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
  workers_.emplace_back(worker_id, request->worker_address());
  response->set_worker_id(worker_id);

  // Allocate tasks to the worker.
  for (auto& entry : jobs_) {
    Job& job = entry.second;
    if (job.finished()) {
      continue;
    }
    int64 task_id = CreateTask(&job, request->worker_address());

    TaskDef* task_def = response->add_tasks();
    *task_def->mutable_dataset() =
        datasets_by_id_[job.dataset_id()]->dataset_def();
    task_def->set_dataset_id(job.dataset_id());
    task_def->set_job_id(job.job_id());
    task_def->set_task_id(task_id);
  }

  VLOG(1) << "Registered worker " << workers_.back().DebugString();
  return Status::OK();
}

Status DataServiceMasterImpl::WorkerUpdate(const WorkerUpdateRequest* request,
                                           WorkerUpdateResponse* response) {
  mutex_lock l(mu_);
  int64 worker_id = request->worker_id();
  for (auto& update : request->updates()) {
    int64 task_id = update.task_id();
    if (!tasks_.contains(task_id)) {
      return errors::NotFound("WorkerUpdate called for worker ", worker_id,
                              " with unknown task id ", task_id);
    }
    if (update.completed()) {
      int64 job_id = tasks_.at(task_id).job_id();
      DCHECK(jobs_.contains(job_id));
      jobs_.at(job_id).task_finished(task_id);
      VLOG(3) << "Task " << task_id << " from job " << job_id << " completed";
    }
  }
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
    int64 id = datasets_by_fingerprint_[fingerprint]->dataset_id();
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
  int64 dataset_id = next_dataset_id_++;
  auto new_dataset =
      std::make_shared<Dataset>(dataset_id, fingerprint, dataset);

  DCHECK(!datasets_by_id_.contains(dataset_id));
  datasets_by_id_[dataset_id] = new_dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[dataset_id] = new_dataset;
  return dataset_id;
}

Status DataServiceMasterImpl::CreateJob(const CreateJobRequest* request,
                                        CreateJobResponse* response) {
  VLOG(3) << "Received begin job request for dataset id "
          << request->dataset_id();
  switch (request->processing_mode()) {
    case PARALLEL_EPOCHS:
      break;
    case ONE_EPOCH:
      return errors::Unimplemented(
          "CreateJob only supports the PARALLEL_EPOCHS job mode. "
          "ONE_EPOCH is not currently supported.");
    default:
      return errors::Unimplemented(
          "ProcessingMode ", request->processing_mode(), " not recognized");
  }
  mutex_lock l(mu_);
  if (!datasets_by_id_.contains(request->dataset_id())) {
    return errors::NotFound("CreateJob failed. Dataset id: <",
                            request->dataset_id(), "> not found.");
  }

  int64 job_id = next_job_id_++;
  DCHECK(!jobs_.contains(job_id));
  auto result =
      jobs_.emplace(std::piecewise_construct, std::forward_as_tuple(job_id),
                    std::forward_as_tuple(job_id, request->dataset_id()));
  DCHECK(result.second);
  Job& job = result.first->second;
  response->set_job_id(job_id);

  for (auto& worker : workers_) {
    int64 task_id = CreateTask(&job, worker.address());

    // TODO(aaudibert): perform these calls asynchronously.
    TF_RETURN_IF_ERROR(AllocateTaskToWorker(tasks_.at(task_id), &worker));
  }

  VLOG(3) << "Beginning job " << job_id << " for dataset "
          << request->dataset_id();
  return Status::OK();
}

int64 DataServiceMasterImpl::CreateTask(Job* job,
                                        const std::string& worker_address)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64 task_id = next_task_id_++;
  DCHECK(!tasks_.contains(task_id));
  auto result =
      tasks_.emplace(std::piecewise_construct, std::forward_as_tuple(task_id),
                     std::forward_as_tuple(task_id, job->job_id(),
                                           job->dataset_id(), worker_address));
  job->add_task_id(task_id);
  DCHECK(result.second);
  return task_id;
}

Status DataServiceMasterImpl::EnsureWorkerStubInitialized(Worker* worker) {
  if (!worker->stub()) {
    std::unique_ptr<WorkerService::Stub> stub;
    TF_RETURN_IF_ERROR(CreateWorkerStub(worker->address(), protocol_, &stub));
    worker->set_stub(std::move(stub));
  }
  return Status::OK();
}

Status DataServiceMasterImpl::AllocateTaskToWorker(const Task& task,
                                                   Worker* worker)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(EnsureWorkerStubInitialized(worker));
  grpc::ClientContext client_ctx;
  ProcessTaskRequest req;
  req.mutable_task()->set_dataset_id(task.dataset_id());
  DCHECK(datasets_by_id_.contains(task.dataset_id()));
  *req.mutable_task()->mutable_dataset() =
      datasets_by_id_.at(task.dataset_id())->dataset_def();
  req.mutable_task()->set_task_id(task.task_id());
  ProcessTaskResponse resp;
  grpc::Status s = worker->stub()->ProcessTask(&client_ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to submit task to worker ", worker->address()), s);
  }
  return Status::OK();
}

Status DataServiceMasterImpl::GetTasks(const GetTasksRequest* request,
                                       GetTasksResponse* response) {
  mutex_lock l(mu_);
  VLOG(3) << "Looking up tasks for job id " << request->job_id();
  auto it = jobs_.find(request->job_id());
  if (it == jobs_.end()) {
    return errors::NotFound("GetTasks failed. Job id <", request->job_id(),
                            "> not found.");
  }
  Job& job = it->second;
  for (const auto& task_id : job.task_ids()) {
    auto task_iter = tasks_.find(task_id);
    DCHECK(task_iter != tasks_.end());
    Task& task = task_iter->second;
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task.worker_address());
    task_info->set_id(task.task_id());
  }
  response->set_job_finished(job.finished());
  VLOG(3) << "Found " << response->task_info_size() << " tasks for job id "
          << request->job_id();
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
