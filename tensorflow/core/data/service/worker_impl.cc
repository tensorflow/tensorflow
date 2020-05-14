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

#include "tensorflow/core/data/service/worker_impl.h"

#include "grpcpp/create_channel.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/data/service/compression_utils.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {

const constexpr uint64 kHeartbeatIntervalMicros = 5ull * 1000 * 1000;

namespace {
auto* tf_data_service_created =
    monitoring::Gauge<bool, 0>::New("/tensorflow/data/service/created",
                                    "Whether a tf.data service server "
                                    "has been created.");
}  // namespace

DataServiceWorkerImpl::DataServiceWorkerImpl(const std::string& master_address,
                                             const std::string& protocol)
    : master_address_(master_address), protocol_(protocol) {
  tf_data_service_created->GetCell()->Set(true);
}

DataServiceWorkerImpl::~DataServiceWorkerImpl() {
  mutex_lock l(mu_);
  cancelled_ = true;
  heartbeat_cv_.notify_one();
}

void DataServiceWorkerImpl::Start(const std::string& worker_address) {
  VLOG(3) << "Starting tf.data service worker at address " << worker_address;
  mutex_lock l(mu_);
  worker_address_ = worker_address;

  Thread* thread = Env::Default()->StartThread(
      {}, "data-service-worker-heartbeat", [this]() { HeartbeatThread(); });
  heartbeat_thread_.reset(thread);
  Status s = Register();
  while (!s.ok()) {
    LOG(WARNING) << "Failed to register with master at " << master_address_
                 << ": " << s;
    Env::Default()->SleepForMicroseconds(kHeartbeatIntervalMicros);
    s = Register();
  }
}


Status DataServiceWorkerImpl::ProcessTask(const ProcessTaskRequest* request,
                                          ProcessTaskResponse* response) {
  mutex_lock l(mu_);
  const TaskDef& task = request->task();
  VLOG(3) << "Received request to process task " << task.task_id();
  return ProcessTaskInternal(task);
}

Status DataServiceWorkerImpl::ProcessTaskInternal(const TaskDef& task_def)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(3) << "Received request to process task " << task_def.task_id();
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      params, task_def.dataset().graph(), &dataset));

  std::unique_ptr<standalone::Iterator> iterator;
  TF_RETURN_IF_ERROR(dataset->MakeIterator(&iterator));

  if (tasks_.contains(task_def.task_id())) {
    return errors::AlreadyExists("A task with id ", task_def.task_id(),
                                 " already exists.");
  }
  Task& task = tasks_[task_def.task_id()];
  task.id = task_def.task_id();
  task.dataset = std::move(dataset);
  task.iterator = std::move(iterator);
  VLOG(3) << "Began processing for task " << task_def.task_id();
  return Status::OK();
}

Status DataServiceWorkerImpl::GetElement(const GetElementRequest* request,
                                         GetElementResponse* response) {
  VLOG(3) << "Received GetElement request for task " << request->task_id();
  bool end_of_sequence = false;
  std::vector<tensorflow::Tensor> outputs;
  {
    mutex_lock l(mu_);
    auto it = tasks_.find(request->task_id());
    if (it == tasks_.end()) {
      return errors::NotFound("DataServiceWorkerImpl::GetElement failed. ",
                              "Task id ", request->task_id(), " not found");
    }
    std::unique_ptr<standalone::Iterator>& iter = it->second.iterator;
    if (iter == nullptr) {
      VLOG(3) << "Task " << request->task_id() << " is already finished";
      response->set_end_of_sequence(true);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(iter->GetNext(&outputs, &end_of_sequence));
    if (end_of_sequence) {
      VLOG(3) << "Reached end_of_sequence for task " << request->task_id();
      // Release iterator memory and leave a null entry as a tombstone.
      iter.reset();
      pending_completed_tasks_.push_back(request->task_id());
      heartbeat_cv_.notify_one();
    }
  }

  if (!end_of_sequence) {
    VLOG(3) << "Producing an element for task " << request->task_id();
    TF_RETURN_IF_ERROR(service_util::Compress(
        outputs, response->mutable_compressed_element()));
  }
  response->set_end_of_sequence(end_of_sequence);

  return Status::OK();
}

Status DataServiceWorkerImpl::EnsureMasterStubInitialized()
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!master_stub_) {
    ::grpc::ChannelArguments args;
    std::shared_ptr<::grpc::ChannelCredentials> credentials;
    TF_RETURN_IF_ERROR(
        CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
    auto channel =
        ::grpc::CreateCustomChannel(master_address_, credentials, args);
    master_stub_ = MasterService::NewStub(channel);
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::Register() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(3) << "Registering with master at " << master_address_;
  TF_RETURN_IF_ERROR(EnsureMasterStubInitialized());
  RegisterWorkerRequest req;
  req.set_worker_address(worker_address_);
  RegisterWorkerResponse resp;

  grpc::ClientContext ctx;
  grpc::Status s = master_stub_->RegisterWorker(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to register worker", s);
  }
  for (const TaskDef& task : resp.tasks()) {
    TF_RETURN_IF_ERROR(ProcessTaskInternal(task));
  }
  worker_id_ = resp.worker_id();
  VLOG(3) << "Registered worker with id " << resp.worker_id();
  return Status::OK();
}

Status DataServiceWorkerImpl::SendTaskUpdate() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(3) << "Sending " << pending_completed_tasks_.size()
          << " task updates to master";
  TF_RETURN_IF_ERROR(EnsureMasterStubInitialized());
  WorkerUpdateRequest req;
  req.set_worker_id(worker_id_);
  for (int task_id : pending_completed_tasks_) {
    TaskProgress* update = req.add_updates();
    update->set_task_id(task_id);
    update->set_completed(true);
  }

  WorkerUpdateResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = master_stub_->WorkerUpdate(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to send task updates", s);
  }
  pending_completed_tasks_.clear();
  VLOG(3) << "Sent " << req.updates().size() << " task updates ";
  return Status::OK();
}

void DataServiceWorkerImpl::HeartbeatThread() {
  while (true) {
    mutex_lock l(mu_);
    while (!cancelled_ && pending_completed_tasks_.empty()) {
      heartbeat_cv_.wait(l);
    }
    if (cancelled_) {
      VLOG(3) << "Heartbeat thread shutting down";
      return;
    }
    Status s = SendTaskUpdate();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send task updates to master: " << s;
    }
  }
}

}  // namespace data
}  // namespace tensorflow
