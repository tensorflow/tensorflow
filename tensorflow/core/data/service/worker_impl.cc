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
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {

const constexpr uint64 kRetryIntervalMicros = 5ull * 1000 * 1000;

namespace {
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
  {
    mutex_lock l(mu_);
    cancellation_manager_.StartCancel();
    cancelled_ = true;
    while (outstanding_requests_ > 0) {
      cv_.wait(l);
    }
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
  auto cleanup = gtl::MakeCleanup([&] {
    mutex_lock l(mu_);
    outstanding_requests_--;
    cv_.notify_all();
  });
  Task* task;
  {
    mutex_lock l(mu_);
    outstanding_requests_++;
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
  }
  TF_RETURN_IF_ERROR(task->task_runner->GetNext(*request, *result));
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
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  std::unique_ptr<standalone::Iterator> iterator;

  switch (task.task_def.dataset_case()) {
    case TaskDef::kDatasetDef:
      TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
          params, task.task_def.dataset_def().graph(), &dataset));
      break;
    case TaskDef::kPath: {
      DatasetDef def;
      Status s = ReadDatasetDef(task.task_def.path(), def);
      if (!s.ok()) {
        LOG(INFO) << "Failed to read dataset from " << task.task_def.path()
                  << ": " << s << ". Falling back to reading from dispatcher.";
        TF_RETURN_IF_ERROR(
            dispatcher_->GetDatasetDef(task.task_def.dataset_id(), def));
      }
      TF_RETURN_IF_ERROR(
          standalone::Dataset::FromGraph(params, def.graph(), &dataset));
      break;
    }
    case TaskDef::DATASET_NOT_SET:
      return errors::Internal("Unrecognized dataset case: ",
                              task.task_def.dataset_case());
  }
  switch (task.task_def.processing_mode()) {
    case DISTRIBUTED_EPOCH: {
      auto split_provider = absl::make_unique<DataServiceSplitProvider>(
          config_.dispatcher_address(), config_.protocol(),
          task.task_def.job_id(), config_.dispatcher_timeout_ms());
      TF_RETURN_IF_ERROR(
          dataset->MakeIterator(std::move(split_provider), &iterator));
      break;
    }
    case PARALLEL_EPOCHS:
      TF_RETURN_IF_ERROR(dataset->MakeIterator(&iterator));
      break;
    default:
      return errors::InvalidArgument("Unrecognized processing mode: ",
                                     task.task_def.processing_mode());
  }
  auto task_iterator = absl::make_unique<StandaloneTaskIterator>(
      std::move(dataset), std::move(iterator));
  TF_RETURN_IF_ERROR(
      TaskRunner::Create(config_, task.task_def, cancellation_manager_,
                         std::move(task_iterator), task.task_runner));

  task.initialized = true;
  VLOG(3) << "Created iterator for task " << task.task_def.task_id();
  return Status::OK();
}

Status DataServiceWorkerImpl::GetElement(const GetElementRequest* request,
                                         GetElementResponse* response) {
  VLOG(3) << "Received GetElement request for task " << request->task_id();
  struct GetElementResult result;
  TF_RETURN_IF_ERROR(GetElementResult(request, &result));
  response->set_end_of_sequence(result.end_of_sequence);
  response->set_skip_task(result.skip);
  if (response->end_of_sequence()) {
    mutex_lock l(mu_);
    VLOG(3) << "Reached end_of_sequence for task " << request->task_id();
    pending_completed_tasks_.insert(request->task_id());
    task_completion_cv_.notify_one();
  } else if (!response->skip_task()) {
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
  std::vector<int64> tasks_to_delete;
  TF_RETURN_IF_ERROR(
      dispatcher_->WorkerHeartbeat(worker_address_, transfer_address_,
                                   current_tasks, new_tasks, tasks_to_delete));
  mutex_lock l(mu_);
  for (const auto& task : new_tasks) {
    VLOG(1) << "Received new task from dispatcher with id " << task.task_id();
    Status s = ProcessTaskInternal(task);
    if (!s.ok() && !errors::IsAlreadyExists(s)) {
      LOG(WARNING) << "Failed to start processing task " << task.task_id()
                   << ": " << s;
    }
  }
  for (int64 task_id : tasks_to_delete) {
    VLOG(3) << "Deleting task " << task_id
            << " at the request of the dispatcher";
    tasks_.erase(task_id);
    finished_tasks_.insert(task_id);
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
