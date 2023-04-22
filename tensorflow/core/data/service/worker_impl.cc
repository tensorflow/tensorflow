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

#include <memory>
#include <string>
#include <utility>

#include "grpcpp/create_channel.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

const constexpr uint64 kRetryIntervalMicros = 5ull * 1000 * 1000;

// Moves the element into the response. If the tensor contains a single
// CompressedElement variant, the move will be zero-copy. Otherwise, the tensor
// data will be serialized as TensorProtos.
Status MoveElementToResponse(std::vector<Tensor>&& element,
                             GetElementResponse& resp) {
  if (element.size() != 1 || element[0].dtype() != DT_VARIANT ||
      !TensorShapeUtils::IsScalar(element[0].shape())) {
    for (const auto& component : element) {
      UncompressedElement* uncompressed = resp.mutable_uncompressed();
      component.AsProtoTensorContent(uncompressed->add_components());
    }
    return Status::OK();
  }
  Variant& variant = element[0].scalar<Variant>()();
  CompressedElement* compressed = variant.get<CompressedElement>();
  if (compressed == nullptr) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a CompressedElement variant tensor, but "
        "it produced ",
        variant.TypeName());
  }
  *resp.mutable_compressed() = *compressed;
  return Status::OK();
}
}  // namespace

mutex LocalWorkers::mu_(LINKER_INITIALIZED);
LocalWorkers::AddressToWorkerMap* LocalWorkers::local_workers_ =
    new AddressToWorkerMap();

DataServiceWorkerImpl::DataServiceWorkerImpl(
    const experimental::WorkerConfig& config)
    : config_(config) {
  metrics::RecordTFDataServiceWorkerCreated();
}

DataServiceWorkerImpl::~DataServiceWorkerImpl() {
  mutex_lock l(mu_);
  cancelled_ = true;
  task_completion_cv_.notify_one();
  heartbeat_cv_.notify_one();
}

Status DataServiceWorkerImpl::Start(const std::string& worker_address,
                                    const std::string& transfer_address) {
  VLOG(3) << "Starting tf.data service worker at address " << worker_address;
  worker_address_ = worker_address;
  transfer_address_ = transfer_address;

  dispatcher_ = absl::make_unique<DataServiceDispatcherClient>(
      config_.dispatcher_address(), config_.protocol());
  TF_RETURN_IF_ERROR(dispatcher_->Initialize());

  Status s = Heartbeat();
  while (!s.ok()) {
    if (!errors::IsUnavailable(s) && !errors::IsAborted(s) &&
        !errors::IsCancelled(s)) {
      return s;
    }
    LOG(WARNING) << "Failed to register with dispatcher at "
                 << config_.dispatcher_address() << ": " << s;
    Env::Default()->SleepForMicroseconds(kRetryIntervalMicros);
    s = Heartbeat();
  }
  LOG(INFO) << "Worker registered with dispatcher running at "
            << config_.dispatcher_address();
  task_completion_thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "data-service-worker-task-completion",
                                  [this]() { TaskCompletionThread(); }));
  heartbeat_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      {}, "data-service-worker-heartbeat", [this]() { HeartbeatThread(); }));
  mutex_lock l(mu_);
  registered_ = true;
  return Status::OK();
}

void DataServiceWorkerImpl::Stop() {
  std::vector<std::shared_ptr<Task>> tasks;
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    for (const auto& entry : tasks_) {
      tasks.push_back(entry.second);
    }
  }
  for (auto& task : tasks) {
    StopTask(*task);
  }
  // At this point there are no outstanding requests in this RPC handler.
  // However, requests successfully returned from this RPC handler may still be
  // in progress within the gRPC server. If we shut down the gRPC server
  // immediately, it could cause these requests to fail, e.g. with broken pipe.
  // To mitigate this, we sleep for some time to give the gRPC server time to
  // complete requests.
  Env::Default()->SleepForMicroseconds(config_.shutdown_quiet_period_ms() *
                                       1000);
}

Status DataServiceWorkerImpl::GetElementResult(
    const GetElementRequest* request, struct GetElementResult* result) {
  Task* task;
  {
    mutex_lock l(mu_);
    if (cancelled_) {
      return errors::Cancelled("Worker is shutting down");
    }
    if (!registered_) {
      // We need to reject requests until the worker has registered with the
      // dispatcher, so that we don't return NOT_FOUND for tasks that the worker
      // had before preemption.
      return errors::Unavailable(
          "Worker has not yet registered with dispatcher.");
    }
    auto it = tasks_.find(request->task_id());
    if (it == tasks_.end()) {
      if (finished_tasks_.contains(request->task_id())) {
        VLOG(3) << "Task is already finished";
        result->end_of_sequence = true;
        result->skip = false;
        return Status::OK();
      } else {
        // Perhaps the workers hasn't gotten the task from the dispatcher yet.
        // Return Unavailable so that the client knows to continue retrying.
        return errors::Unavailable("Task ", request->task_id(), " not found");
      }
    }
    task = it->second.get();
    TF_RETURN_IF_ERROR(EnsureTaskInitialized(*task));
    task->outstanding_requests++;
  }
  auto cleanup = gtl::MakeCleanup([&] {
    mutex_lock l(mu_);
    task->outstanding_requests--;
    cv_.notify_all();
  });
  TF_RETURN_IF_ERROR(task->task_runner->GetNext(*request, *result));

  if (result->end_of_sequence) {
    mutex_lock l(mu_);
    VLOG(3) << "Reached end_of_sequence for task " << request->task_id();
    pending_completed_tasks_.insert(request->task_id());
    task_completion_cv_.notify_one();
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::ProcessTask(const ProcessTaskRequest* request,
                                          ProcessTaskResponse* response) {
  mutex_lock l(mu_);
  const TaskDef& task = request->task();
  VLOG(3) << "Received request to process task " << task.task_id();
  return ProcessTaskInternal(task);
}

Status DataServiceWorkerImpl::ProcessTaskInternal(const TaskDef& task_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<Task>& task = tasks_[task_def.task_id()];
  if (task) {
    VLOG(1) << "Received request to process already-processed task "
            << task->task_def.task_id();
    return Status::OK();
  }
  task = absl::make_unique<Task>(task_def);
  VLOG(3) << "Began processing for task " << task_def.task_id()
          << " with processing mode " << task_def.processing_mode();
  return Status::OK();
}

Status DataServiceWorkerImpl::EnsureTaskInitialized(
    DataServiceWorkerImpl::Task& task) {
  mutex_lock l(task.mu);
  if (task.initialized) {
    return Status::OK();
  }
  TF_ASSIGN_OR_RETURN(DatasetDef dataset_def, GetDatasetDef(task.task_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Dataset> dataset,
                      MakeDataset(dataset_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Iterator> iterator,
                      MakeDatasetIterator(*dataset, task.task_def));
  auto task_iterator = absl::make_unique<StandaloneTaskIterator>(
      std::move(dataset), std::move(iterator));
  TF_RETURN_IF_ERROR(TaskRunner::Create(
      config_, task.task_def, std::move(task_iterator), task.task_runner));

  task.initialized = true;
  VLOG(3) << "Created iterator for task " << task.task_def.task_id();
  return Status::OK();
}

StatusOr<DatasetDef> DataServiceWorkerImpl::GetDatasetDef(
    const TaskDef& task_def) const {
  switch (task_def.dataset_case()) {
    case TaskDef::kDatasetDef:
      return task_def.dataset_def();
    case TaskDef::kPath: {
      DatasetDef def;
      Status s = ReadDatasetDef(task_def.path(), def);
      if (!s.ok()) {
        LOG(INFO) << "Failed to read dataset from " << task_def.path() << ": "
                  << s << ". Falling back to reading from dispatcher.";
        TF_RETURN_IF_ERROR(
            dispatcher_->GetDatasetDef(task_def.dataset_id(), def));
      }
      return def;
    }
    case TaskDef::DATASET_NOT_SET:
      return errors::Internal("Unrecognized dataset case: ",
                              task_def.dataset_case());
  }
}

StatusOr<std::unique_ptr<standalone::Dataset>>
DataServiceWorkerImpl::MakeDataset(const DatasetDef& dataset_def) const {
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), dataset_def.graph(), &dataset));
  return dataset;
}

StatusOr<std::unique_ptr<standalone::Iterator>>
DataServiceWorkerImpl::MakeDatasetIterator(standalone::Dataset& dataset,
                                           const TaskDef& task_def) const {
  std::unique_ptr<standalone::Iterator> iterator;
  switch (task_def.processing_mode()) {
    case DISTRIBUTED_EPOCH: {
      std::vector<std::unique_ptr<SplitProvider>> split_providers;
      split_providers.reserve(task_def.num_split_providers());
      for (int i = 0; i < task_def.num_split_providers(); ++i) {
        split_providers.push_back(absl::make_unique<DataServiceSplitProvider>(
            config_.dispatcher_address(), config_.protocol(), task_def.job_id(),
            i, config_.dispatcher_timeout_ms()));
      }
      TF_RETURN_IF_ERROR(
          dataset.MakeIterator(std::move(split_providers), &iterator));
      break;
    }
    case PARALLEL_EPOCHS:
      TF_RETURN_IF_ERROR(dataset.MakeIterator(&iterator));
      break;
    default:
      return errors::InvalidArgument("Unrecognized processing mode: ",
                                     task_def.processing_mode());
  }
  return iterator;
}

void DataServiceWorkerImpl::StopTask(Task& task) TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(task.mu);
    task.initialized = true;
  }
  if (task.task_runner) {
    task.task_runner->Cancel();
  }
  mutex_lock l(mu_);
  while (task.outstanding_requests > 0) {
    cv_.wait(l);
  }
}

Status DataServiceWorkerImpl::GetElement(const GetElementRequest* request,
                                         GetElementResponse* response) {
  VLOG(3) << "Received GetElement request for task " << request->task_id();
  struct GetElementResult result;
  TF_RETURN_IF_ERROR(GetElementResult(request, &result));
  response->set_end_of_sequence(result.end_of_sequence);
  response->set_skip_task(result.skip);
  if (!response->end_of_sequence() && !response->skip_task()) {
    TF_RETURN_IF_ERROR(
        MoveElementToResponse(std::move(result.components), *response));
    VLOG(3) << "Producing an element for task " << request->task_id();
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::GetWorkerTasks(
    const GetWorkerTasksRequest* request, GetWorkerTasksResponse* response) {
  mutex_lock l(mu_);
  for (const auto& it : tasks_) {
    Task* task = it.second.get();
    TaskInfo* task_info = response->add_tasks();
    task_info->set_worker_address(worker_address_);
    task_info->set_task_id(task->task_def.task_id());
    task_info->set_job_id(task->task_def.job_id());
  }
  return Status::OK();
}

void DataServiceWorkerImpl::TaskCompletionThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && pending_completed_tasks_.empty()) {
        task_completion_cv_.wait(l);
      }
      if (cancelled_) {
        VLOG(3) << "Task completion thread shutting down";
        return;
      }
    }
    Status s = SendTaskUpdates();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send task updates to dispatcher: " << s;
      mutex_lock l(mu_);
      if (!cancelled_) {
        task_completion_cv_.wait_for(
            l, std::chrono::microseconds(kRetryIntervalMicros));
      }
    }
  }
}

Status DataServiceWorkerImpl::SendTaskUpdates() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<TaskProgress> task_progress;
  {
    mutex_lock l(mu_);
    VLOG(3) << "Sending " << pending_completed_tasks_.size()
            << " task updates to dispatcher";
    task_progress.reserve(pending_completed_tasks_.size());
    for (int task_id : pending_completed_tasks_) {
      task_progress.emplace_back();
      task_progress.back().set_task_id(task_id);
      task_progress.back().set_completed(true);
    }
  }

  TF_RETURN_IF_ERROR(dispatcher_->WorkerUpdate(worker_address_, task_progress));
  mutex_lock l(mu_);
  for (const auto& update : task_progress) {
    pending_completed_tasks_.erase(update.task_id());
  }
  VLOG(3) << "Sent " << task_progress.size() << " task updates ";
  return Status::OK();
}

void DataServiceWorkerImpl::HeartbeatThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    int64 next_heartbeat_micros =
        Env::Default()->NowMicros() + (config_.heartbeat_interval_ms() * 1000);
    {
      mutex_lock l(mu_);
      while (!cancelled_ &&
             Env::Default()->NowMicros() < next_heartbeat_micros) {
        int64 time_to_wait_micros =
            next_heartbeat_micros - Env::Default()->NowMicros();
        heartbeat_cv_.wait_for(l,
                               std::chrono::microseconds(time_to_wait_micros));
      }
      if (cancelled_) {
        VLOG(3) << "Heartbeat thread shutting down";
        return;
      }
      if (!registered_) {
        VLOG(1) << "Not performing heartbeat; worker is not yet registered";
        continue;
      }
    }
    Status s = Heartbeat();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send heartbeat to dispatcher: " << s;
    }
  }
}

Status DataServiceWorkerImpl::Heartbeat() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<int64> current_tasks;
  {
    mutex_lock l(mu_);
    for (const auto& task : tasks_) {
      current_tasks.push_back(task.first);
    }
  }
  std::vector<TaskDef> new_tasks;
  std::vector<int64> task_ids_to_delete;
  TF_RETURN_IF_ERROR(dispatcher_->WorkerHeartbeat(
      worker_address_, transfer_address_, current_tasks, new_tasks,
      task_ids_to_delete));
  std::vector<std::shared_ptr<Task>> tasks_to_delete;
  {
    mutex_lock l(mu_);
    for (const auto& task : new_tasks) {
      VLOG(1) << "Received new task from dispatcher with id " << task.task_id();
      Status s = ProcessTaskInternal(task);
      if (!s.ok() && !errors::IsAlreadyExists(s)) {
        LOG(WARNING) << "Failed to start processing task " << task.task_id()
                     << ": " << s;
      }
    }
    tasks_to_delete.reserve(task_ids_to_delete.size());
    for (int64 task_id : task_ids_to_delete) {
      VLOG(3) << "Deleting task " << task_id
              << " at the request of the dispatcher";
      tasks_to_delete.push_back(std::move(tasks_[task_id]));
      tasks_.erase(task_id);
      finished_tasks_.insert(task_id);
    }
  }
  for (const auto& task : tasks_to_delete) {
    StopTask(*task);
  }
  return Status::OK();
}

void LocalWorkers::Add(absl::string_view worker_address,
                       std::shared_ptr<DataServiceWorkerImpl> worker) {
  DCHECK(worker != nullptr) << "Adding a nullptr local worker is disallowed.";
  VLOG(1) << "Register local worker at address " << worker_address;
  mutex_lock l(mu_);
  (*local_workers_)[worker_address] = worker;
}

std::shared_ptr<DataServiceWorkerImpl> LocalWorkers::Get(
    absl::string_view worker_address) {
  tf_shared_lock l(mu_);
  AddressToWorkerMap::const_iterator it = local_workers_->find(worker_address);
  if (it == local_workers_->end()) {
    return nullptr;
  }
  return it->second;
}

bool LocalWorkers::Empty() {
  tf_shared_lock l(mu_);
  return local_workers_->empty();
}

void LocalWorkers::Remove(absl::string_view worker_address) {
  VLOG(1) << "Remove local worker at address " << worker_address;
  mutex_lock l(mu_);
  local_workers_->erase(worker_address);
}

}  // namespace data
}  // namespace tensorflow
