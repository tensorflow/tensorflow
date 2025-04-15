/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/client/data_service_client.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/client/validate_utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_client.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tsl/platform/retrying_utils.h"

namespace tensorflow {
namespace data {
namespace {

bool IsColocatedTask(const TaskInfo& task) {
  return absl::c_any_of(task.worker_tags(), [](std::string_view worker_tag) {
    return absl::AsciiStrToUpper(worker_tag) == kColocatedWorkerTag;
  });
}

absl::StatusOr<DataTransferServerInfo> GetTransferServer(
    const std::string& protocol, const TaskInfo& task_info) {
  for (const auto& transfer_server : task_info.transfer_servers()) {
    if (transfer_server.protocol() == protocol) {
      return transfer_server;
    }
  }
  return absl::NotFoundError(absl::StrCat("Protocol '", protocol,
                                          "' is not available for worker '",
                                          task_info.worker_address(), "'."));
}

}  // namespace

DataServiceClient::DataServiceClient(const DataServiceParams& params)
    : params_(params),
      max_outstanding_requests_(params.max_outstanding_requests) {}

DataServiceClient::~DataServiceClient() {
  VLOG(2) << "Destroying data service client for iteration id "
          << iteration_client_id_;
  task_thread_manager_.reset();
  if (initialized_) {
    absl::Status s = dispatcher_->ReleaseIterationClient(iteration_client_id_);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to release iteration client id: " << s;
    }
  }
  for (auto& worker_thread : worker_threads_) {
    worker_thread.reset();
  }
  DeleteLocalWorkerTasks();
  VLOG(2) << "Destroyed data service dataset iterator for iteration id "
          << iteration_client_id_;
}

absl::Status DataServiceClient::Initialize(
    const DeviceBase::AcceleratorDeviceInfo* accelerator_device_info,
    Allocator* allocator) {
  accelerator_device_info_ = accelerator_device_info;
  allocator_ = allocator;
  TF_RETURN_IF_ERROR(ValidateDataServiceParams(params_));
  VLOG(3) << "Connecting to " << params_.address
          << " in tf.data service client.";
  dispatcher_ = std::make_unique<DataServiceDispatcherClient>(params_.address,
                                                              params_.protocol);
  int64_t deadline_micros = kint64max;
  std::optional<std::string> job_name;
  if (!params_.job_name.empty()) {
    job_name = params_.job_name;
  }
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&]() {
        return dispatcher_->GetOrCreateJob(
            params_.dataset_id, params_.processing_mode, job_name,
            params_.num_consumers,
            params_.cross_trainer_cache_options.has_value(),
            params_.target_workers, job_id_);
      },
      /*description=*/
      strings::StrCat("get or create job with dispatcher at ", params_.address),
      deadline_micros));
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&]() {
        return dispatcher_->GetOrCreateIteration(job_id_, params_.repetition,
                                                 iteration_client_id_);
      },
      /*description=*/
      strings::StrCat("get or create iteration with dispatcher at ",
                      params_.address),
      deadline_micros));
  initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<GetNextResult> DataServiceClient::GetNext(
    DataServiceContextFactory context_factory) TF_LOCKS_EXCLUDED(mu_) {
  VLOG(3) << "Getting the next element from tf.data service client.";
  mutex_lock l(mu_);
  if (ctx_ == nullptr) {
    ctx_ = context_factory();
  }
  EnsureThreadsStarted();
  std::shared_ptr<Result> result;
  do {
    while (!ResultReady() && !Finished() && !cancelled_ && status_.ok()) {
      VLOG(3) << "Blocking in GetNext: " << DebugString();
      get_next_cv_.wait(l);
    }
    if (cancelled_) {
      VLOG(3) << "Returning from GetNext due to cancellation";
      return errors::Cancelled("Data service iterator was cancelled");
    }
    if (!status_.ok()) {
      VLOG(3) << "Returning from GetNext with error " << status_;
      return status_;
    }
    if (results_.empty()) {
      VLOG(3) << "Returning from GetNext with end_of_sequence";
      return GetNextResult::EndOfSequence();
    }
    if (!ResultReady()) {
      VLOG(3) << "Returning from GetNext with internal error";
      return errors::Internal("Expected a result to be ready, but none were.");
    }
    result = PopNextResult();
    worker_thread_cv_.notify_one();
    if (result->skip) {
      VLOG(3) << "Skipping result from task " << result->task_id;
    }
  } while (result->skip);

  GetNextResult next;
  next.end_of_sequence = result->end_of_sequence;
  if (next.end_of_sequence) {
    VLOG(1) << "Returning end_of_sequence";
    return next;
  }
  VLOG(1) << "Returning the next element from data service dataset's "
          << "Iterator: task " << result->task_id << ", element "
          << result->element_index;
  if (IsCoordinatedRead()) {
    VLOG(1) << "Consumer " << *params_.consumer_index << ": Result "
            << get_next_index_++;
  }
  next.tensors.swap(result->element);
  return next;
}

void DataServiceClient::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  for (const auto& task : tasks_) {
    task->worker->TryCancel();
  }
  cancelled_ = true;
  worker_thread_cv_.notify_all();
  manager_thread_cv_.notify_all();
  get_next_cv_.notify_all();
}

TraceMeMetadata DataServiceClient::GetTraceMeMetadata() const {
  TraceMeMetadata result;
  int64_t num_tasks = -1;
  int64_t autotuned_max_outstanding_requests = model::kAutotune;
  if (mu_.try_lock()) {
    num_tasks = tasks_.size() - finished_tasks_;
    autotuned_max_outstanding_requests = max_outstanding_requests_;
    mu_.unlock();
  }
  result.push_back(std::make_pair(
      "num_tasks",
      num_tasks == -1
          ? kTraceInfoUnavailable
          : strings::Printf("%lld", static_cast<long long>(num_tasks))));
  result.push_back(std::make_pair("job_name", params_.job_name));
  result.push_back(std::make_pair(
      "max_outstanding_requests",
      strings::Printf(
          "%lld", static_cast<long long>(params_.max_outstanding_requests))));
  if (params_.max_outstanding_requests == model::kAutotune) {
    result.push_back(std::make_pair(
        "autotuned_max_outstanding_requests",
        strings::Printf("%lld", static_cast<long long>(
                                    autotuned_max_outstanding_requests))));
  }
  return result;
}

void DataServiceClient::EnsureThreadsStarted()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!task_thread_manager_ && !cancelled_) {
    task_thread_manager_ = ctx_->StartThread("task-thread-manager",
                                             [this]() { TaskThreadManager(); });
  }
}

bool DataServiceClient::Finished() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return num_running_worker_threads_ == 0 && !ShouldWaitForNext();
}

bool DataServiceClient::ShouldWaitForNext() const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (should_finish_iteration_) {
    return !iteration_finished_;
  }
  return tasks_.empty() || finished_tasks_ < tasks_.size();
}

void DataServiceClient::DeleteLocalWorkerTasks() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<std::shared_ptr<Task>> tasks;
  {
    mutex_lock l(mu_);
    tasks = tasks_;
  }

  for (const std::shared_ptr<Task>& task : tasks) {
    std::shared_ptr<DataServiceWorkerImpl> worker =
        LocalWorkers::Get(task->info.worker_address());
    if (worker && ShouldDeleteLocalTask(task->info)) {
      worker->DeleteLocalTask(task->info);
    }
  }
}

// Deletes the task if it is only read by the local client.
bool DataServiceClient::ShouldDeleteLocalTask(const TaskInfo& task) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (IsCoordinatedRead()) {
    return false;
  }

  if (params_.target_workers == TARGET_WORKERS_LOCAL) {
    return true;
  }

  return params_.target_workers == TARGET_WORKERS_AUTO && IsColocatedTask(task);
}

void DataServiceClient::TaskThreadManager() TF_LOCKS_EXCLUDED(mu_) {
  auto cleanup =
      gtl::MakeCleanup([] { VLOG(1) << "Task thread manager exiting"; });
  VLOG(1) << "Starting task thread manager";
  uint64 next_check = Env::Default()->NowMicros();
  while (true) {
    {
      mutex_lock l(mu_);
      // All units are microseconds.
      while (!cancelled_ && Env::Default()->NowMicros() < next_check) {
        int64_t remaining_time = next_check - Env::Default()->NowMicros();
        VLOG(4) << "Task thread manager waiting for " << remaining_time << "us";
        manager_thread_cv_.wait_for(l,
                                    std::chrono::microseconds(remaining_time));
      }
      if (cancelled_) {
        VLOG(3) << "Task thread manager finished";
        return;
      }
    }
    Heartbeat();
    UpdateBufferSize();
    UpdateWorkerThreads();
    next_check = Env::Default()->NowMicros() +
                 absl::ToInt64Microseconds(params_.task_refresh_interval);
  }
}

void DataServiceClient::TryBlockRound(int64_t round)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (round_robin_round_limit_.has_value() &&
      round_robin_round_limit_.value() == round) {
    return;
  }
  if (current_round_ >= round) {
    // In the next heartbeat, notify the dispatcher that we failed to add
    // the task.
    VLOG(1) << "Rejecting request to block round " << round
            << ", because processing has already begun for round "
            << current_round_;
    return;
  }
  VLOG(1) << "Accepting request to block round " << round;
  round_robin_round_limit_ = round;
}

void DataServiceClient::UpdateIterationFinished(bool iteration_finished)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!iteration_finished) {
    return;
  }
  iteration_finished_ = true;
  get_next_cv_.notify_all();
  worker_thread_cv_.notify_all();
}

absl::StatusOr<std::unique_ptr<DataServiceWorkerClient>>
DataServiceClient::CreateWorkerClient(const std::string& protocol,
                                      const TaskInfo& task_info) {
  TF_ASSIGN_OR_RETURN(DataTransferServerInfo transfer_server,
                      GetTransferServer(protocol, task_info));
  return CreateDataServiceWorkerClient(params_.protocol, transfer_server,
                                       accelerator_device_info_, allocator_);
}

absl::StatusOr<std::unique_ptr<DataServiceWorkerClient>>
DataServiceClient::CreateGrpcWorkerClient(const TaskInfo& task_info) {
  return CreateWorkerClient(kGrpcTransferProtocol, task_info);
}

absl::StatusOr<std::unique_ptr<DataServiceWorkerClient>>
DataServiceClient::CreateAlternativeWorkerClientMaybeWithGrpcFallback(
    const DataTransferServerInfo& transfer_server, const TaskInfo& task_info) {
  absl::StatusOr<std::unique_ptr<DataServiceWorkerClient>> worker =
      CreateDataServiceWorkerClient(params_.protocol, transfer_server,
                                    accelerator_device_info_, allocator_);
  if (worker.ok()) {
    LOG(INFO) << "Successfully started client for data transfer protocol '"
              << transfer_server.protocol() << "' for worker '"
              << task_info.worker_address() << "'.";
    return worker;
  }
  std::string client_creation_error_message =
      absl::StrCat("Failed to start client for data transfer protocol '",
                   transfer_server.protocol(), "' for worker '",
                   task_info.worker_address(), "'.");
  if (!transfer_server.fall_back_to_grpc_at_client_creation_time()) {
    return absl::InternalError(
        absl::StrCat(client_creation_error_message,
                     " Original error: ", worker.status().message()));
  }
  LOG(INFO) << client_creation_error_message
            << "; falling back to gRPC. Original error: " << worker.status();
  metrics::RecordTFDataServiceDataTransferProtocolFallback(
      transfer_server.protocol(),
      static_cast<error::Code>(worker.status().raw_code()),
      std::string(worker.status().message()));
  return CreateGrpcWorkerClient(task_info);
}

absl::StatusOr<std::unique_ptr<DataServiceWorkerClient>>
DataServiceClient::CreateWorkerClient(const TaskInfo& task_info) {
  if (params_.data_transfer_protocol == kLocalTransferProtocol ||
      ForceLocalProtocol(task_info.worker_address())) {
    DataTransferServerInfo info;
    info.set_protocol(kLocalTransferProtocol);
    info.set_address(task_info.worker_address());
    return CreateDataServiceWorkerClient(params_.protocol, info,
                                         accelerator_device_info_, allocator_);
  }
  if (!params_.data_transfer_protocol.empty()) {
    TF_ASSIGN_OR_RETURN(
        DataTransferServerInfo transfer_server,
        GetTransferServer(params_.data_transfer_protocol, task_info));
    return CreateAlternativeWorkerClientMaybeWithGrpcFallback(transfer_server,
                                                              task_info);
  }
  if (std::string default_protocol = DefaultDataTransferProtocol();
      default_protocol != kGrpcTransferProtocol) {
    absl::StatusOr<DataTransferServerInfo> transfer_server =
        GetTransferServer(default_protocol, task_info);
    if (transfer_server.ok()) {
      return CreateAlternativeWorkerClientMaybeWithGrpcFallback(
          *transfer_server, task_info);
    }
    VLOG(1) << "Failed to find transfer server for default data transfer "
               "protocol '"
            << default_protocol << "' for worker '"
            << task_info.worker_address()
            << "'; falling back to grpc. Original error: "
            << transfer_server.status();
    metrics::RecordTFDataServiceDataTransferProtocolFallback(
        default_protocol, error::Code::NOT_FOUND,
        "Failed to find transfer server for default protocol");
  }
  return CreateGrpcWorkerClient(task_info);
}

absl::Status DataServiceClient::AddTask(const TaskInfo& task_info)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<DataServiceWorkerClient> worker,
                      CreateWorkerClient(task_info));
  metrics::RecordTFDataServiceDataTransferProtocolUsed(
      worker->GetDataTransferProtocol(),
      /*user_specified=*/!params_.data_transfer_protocol.empty());
  tasks_.push_back(std::make_shared<Task>(task_info, std::move(worker)));
  worker_thread_cv_.notify_one();
  if (IsCoordinatedRead()) {
    VLOG(1) << "Consumer " << params_.consumer_index.value() << " adding task "
            << task_info.task_id() << " to read from worker "
            << task_info.worker_address()
            << ". Task starting round: " << task_info.starting_round();
    DCHECK_LE(current_round_, task_info.starting_round());
    if (current_round_ == task_info.starting_round()) {
      DCHECK_EQ(next_task_index_, 0);
    }
  }
  if (!IsCoordinatedRead()) {
    // Shuffle task order within each client to avoid thundering herd effect.
    std::mt19937 rng;
    std::shuffle(tasks_.begin(), tasks_.end(), rng);
  }
  return absl::OkStatus();
}

void DataServiceClient::Heartbeat() TF_LOCKS_EXCLUDED(mu_) {
  ClientHeartbeatRequest req;
  req.set_iteration_client_id(iteration_client_id_);
  if (IsCoordinatedRead()) {
    mutex_lock l(mu_);
    req.set_current_round(current_round_);
    if (round_robin_round_limit_.has_value()) {
      req.set_blocked_round(round_robin_round_limit_.value());
    }
  }
  {
    mutex_lock l(mu_);
    double target_processing_time_nsec = ctx_->GetTargetProcessingTimeNsec();
    req.set_target_processing_time_nsec(target_processing_time_nsec);
  }
  ClientHeartbeatResponse resp;
  absl::Status s = dispatcher_->ClientHeartbeat(req, resp);
  if (!s.ok()) {
    if (IsPreemptedError(s)) {
      LOG(WARNING)
          << "Failed to heartbeat to dispatcher from iteration client id "
          << iteration_client_id_ << ". Dispatcher address: " << params_.address
          << ". Error: " << s;
      return;
    }
    mutex_lock l(mu_);
    status_ = s;
    get_next_cv_.notify_all();
  }
  mutex_lock l(mu_);
  UpdateIterationFinished(resp.iteration_finished());
  if (resp.optional_block_round_case() ==
      ClientHeartbeatResponse::kBlockRound) {
    TryBlockRound(resp.block_round());
  } else {
    round_robin_round_limit_ = std::nullopt;
    worker_thread_cv_.notify_all();
  }
  UpdateTasks(resp);
  RecordTFMetrics(resp);
}

void DataServiceClient::UpdateTasks(const ClientHeartbeatResponse& resp)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  absl::flat_hash_map<int64_t, TaskInfo> task_id_to_task;
  for (auto& task : resp.task_info()) {
    task_id_to_task[task.task_id()] = task;
  }
  if (iteration_finished_) {
    return;
  }

  int index = 0;
  while (index < tasks_.size()) {
    std::shared_ptr<Task> task = tasks_[index];
    if (task_id_to_task.contains(task->info.task_id())) {
      // Remove already-known tasks from `task_id_to_task`, so that at the
      // end of the loop, only new tasks remain.
      task_id_to_task.erase(task->info.task_id());
      ++index;
    } else {
      // Task has been removed.
      if (task->end_of_sequence) {
        finished_tasks_--;
      }
      tasks_.erase(tasks_.begin() + index);
      if (index < next_task_index_) {
        next_task_index_--;
      }
      if (!tasks_.empty() && next_task_index_ >= tasks_.size()) {
        AdvanceTaskIndex();
      }
    }
  }
  for (auto& task : resp.task_info()) {
    auto it = task_id_to_task.find(task.task_id());
    if (it == task_id_to_task.end()) {
      continue;
    }
    if (!ShouldReadFromTask(task)) {
      VLOG(3) << "Skipping untargeted worker task " << task.task_id();
      should_finish_iteration_ = false;
      continue;
    }
    absl::Status s = AddTask(it->second);
    if (!s.ok()) {
      status_ = s;
      get_next_cv_.notify_all();
      break;
    }
  }
}

bool DataServiceClient::ShouldReadFromTask(const TaskInfo& task) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (IsCoordinatedRead()) {
    return true;
  }

  const bool is_local_task =
      (LocalWorkers::Get(task.worker_address()) != nullptr);
  if (params_.target_workers == TARGET_WORKERS_LOCAL && !is_local_task) {
    return false;
  }

  // Cross-TF/TPU host reads may cause resource contention on the TF/TPU
  // hosts. tf.data service avoids reading from non-local TF-hosted workers.
  const bool is_cross_tf_host_read = !is_local_task && IsColocatedTask(task);
  if (params_.target_workers == TARGET_WORKERS_AUTO && is_cross_tf_host_read) {
    return false;
  }
  return true;
}

void DataServiceClient::RecordTFMetrics(const ClientHeartbeatResponse& resp)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  for (const auto& task : resp.task_info()) {
    if (worker_uids_.contains(task.worker_uid())) {
      continue;
    }
    metrics::RecordTFDataServiceClientIterators(
        task.worker_uid(), resp.deployment_mode(), params_.processing_mode,
        IsCoordinatedRead());
    worker_uids_.insert(task.worker_uid());
  }
}

void DataServiceClient::UpdateBufferSize() TF_LOCKS_EXCLUDED(mu_) {
  if (params_.max_outstanding_requests == model::kAutotune) {
    // Adjust `max_outstanding_requests_` to account for newly added tasks.
    // `tasks_` includes the local tasks, so we subtract one from the
    // configured local task buffer size.
    mutex_lock l(mu_);
    int64_t max_outstanding_requests = ctx_->UpdateMaxOutstandingRequests(
        max_outstanding_requests_, tasks_.size());
    if (max_outstanding_requests > max_outstanding_requests_) {
      worker_thread_cv_.notify_all();
    }
    VLOG(3) << "Updated `max_outstanding_requests` from "
            << max_outstanding_requests_ << " to " << max_outstanding_requests
            << " with " << tasks_.size() << " tasks.";
    max_outstanding_requests_ = max_outstanding_requests;
  }
}

void DataServiceClient::UpdateWorkerThreads() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  const int64_t max_num_threads =
      std::min<int64_t>(tasks_.size(), max_outstanding_requests_);
  while (num_running_worker_threads_ < max_num_threads && !cancelled_ &&
         status_.ok()) {
    num_running_worker_threads_++;
    auto done = [this]() {
      mutex_lock l(mu_);
      num_running_worker_threads_--;
      get_next_cv_.notify_all();
    };
    worker_threads_.push_back(ctx_->StartThread(
        "tf-data-service-task_thread", [this, done = std::move(done)]() {
          RunWorkerThread(std::move(done));
        }));
  }
}

void DataServiceClient::RunWorkerThread(std::function<void()> done)
    TF_LOCKS_EXCLUDED(mu_) {
  auto cleanup = gtl::MakeCleanup([done = std::move(done)]() {
    done();
    VLOG(1) << "Worker thread exiting";
  });
  VLOG(1) << "Starting worker thread";
  std::shared_ptr<Task> task_to_process;
  int64_t num_consecutive_skipped = 0;
  constexpr int64_t MAX_ROUND_FALLBACK_TO_BLOCKING = 5;
  bool allow_skip = true;

  while (true) {
    std::shared_ptr<Result> result;
    {
      mutex_lock l(mu_);
      if (task_to_process) {
        task_to_process->in_use = false;
        --outstanding_requests_;
        task_to_process = nullptr;
        worker_thread_cv_.notify_one();
      }
      while (true) {
        if (cancelled_ || !ShouldWaitForNext()) {
          return;
        }
        task_to_process = GetTaskToProcess();
        if (task_to_process) {
          VLOG(3) << "Selected a task to process: "
                  << task_to_process->info.ShortDebugString();
          break;
        }
        worker_thread_cv_.wait(l);
      }
      DCHECK(task_to_process != nullptr);
      task_to_process->in_use = true;
      ++outstanding_requests_;
      if (IsCoordinatedRead()) {
        // Reserve a spot in the results_ queue.
        results_.push(std::make_shared<Result>());
        ctx_->RecordBufferEnqueue(results_.back()->element);
        result = results_.back();
      } else {
        // The result will be added to results_ when it's ready.
        result = std::make_shared<Result>();
      }
      VLOG(3) << "Processing task " << task_to_process->info.task_id();
    }
    int64_t deadline_micros = kint64max;
    absl::Status s = GetElementTraced(task_to_process.get(), deadline_micros,
                                      /*enqueue_result=*/!IsCoordinatedRead(),
                                      allow_skip, result);
    if (!s.ok()) {
      mutex_lock l(mu_);
      VLOG(1) << "Failed to get element from worker "
              << task_to_process->info.worker_address() << ": " << s;
      task_to_process->in_use = false;
      --outstanding_requests_;
      status_ = errors::CreateWithUpdatedMessage(
          s, absl::StrCat("Failed to get element from worker ",
                          task_to_process->info.worker_address(), ": ",
                          s.message()));
      get_next_cv_.notify_all();
      return;
    }

    if (!IsCoordinatedRead()) {
      if (mutex_lock l(mu_); result->skip) {
        num_consecutive_skipped++;
        if (num_consecutive_skipped >=
            MAX_ROUND_FALLBACK_TO_BLOCKING * tasks_.size()) {
          // Switches to blocking call when we already skip
          // all workers enough rounds.
          // This is to ensures we do not spam the network traffic.
          allow_skip = false;
          VLOG(1) << "`allow_skip` is turned off. Switching to blocking "
                     "get element calls to the workers.";
        }
      } else {
        num_consecutive_skipped = 0;
        allow_skip = true;
      }
    }
  }
}

// Reports whether we can request another element without violating
// `max_outstanding_requests_`.
bool DataServiceClient::ShouldProcessTask() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // When doing round-robin reads, outstanding requests pre-allocate a
  // result in `results_`, so we only need to check the size of `results_`.
  if (IsCoordinatedRead()) {
    return results_.size() < max_outstanding_requests_;
  }
  // Otherwise, results aren't added to `results_` until the data has been
  // successfully retrieved. We need to count requests already added to
  // `results_` as well as in-progress requests.
  return results_.size() + outstanding_requests_ < max_outstanding_requests_;
}

// Searches for a task to process, visiting tasks in-order and giving every
// task a chance to proceed.
std::shared_ptr<DataServiceClient::Task> DataServiceClient::GetTaskToProcess()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!ShouldProcessTask()) {
    return nullptr;
  }

  for (int i = 0; i < tasks_.size(); ++i) {
    std::shared_ptr<Task>& task = tasks_[next_task_index_];
    if (IsCoordinatedRead() &&
        (task->in_use ||
         current_round_ >= round_robin_round_limit_.value_or(
                               std::numeric_limits<int64_t>::max()))) {
      VLOG(4) << "No round robin task found. in_use: " << task->in_use
              << ". current_round: " << current_round_
              << ". round_robin_round_limit: "
              << round_robin_round_limit_.value_or(-1);
      return nullptr;
    }
    if (current_round_ < task->info.starting_round() || task->in_use ||
        task->end_of_sequence || task->removed) {
      VLOG(3) << "Skipping task " << next_task_index_
              << ". starting round: " << task->info.starting_round()
              << ". current round: " << current_round_
              << ". task->in_use: " << task->in_use
              << ". end_of_sequence: " << task->end_of_sequence
              << ". task->removed: " << task->removed;
      AdvanceTaskIndex();
      continue;
    }
    task->round = current_round_;
    AdvanceTaskIndex();
    return task;
  }
  return nullptr;
}

// Increments the next task index, starting over if all tasks have been
// processed.
void DataServiceClient::AdvanceTaskIndex() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  next_task_index_++;
  if (next_task_index_ >= tasks_.size()) {
    current_round_++;
    next_task_index_ = 0;
  }
}

absl::Status DataServiceClient::TryGetElement(const Task& task, bool allow_skip,
                                              GetElementResult& result) {
  GetElementRequest req;
  req.set_task_id(task.info.task_id());
  req.set_skipped_previous_round(task.skipped_previous_round);
  if (IsCoordinatedRead()) {
    req.set_consumer_index(params_.consumer_index.value());
    req.set_round_index(task.round);
    req.set_allow_skip(true);
  } else {
    req.set_allow_skip(allow_skip);
  }
  if (params_.cross_trainer_cache_options) {
    req.set_trainer_id(params_.cross_trainer_cache_options->trainer_id());
  }
  return task.worker->GetElement(req, result);
}

void DataServiceClient::ProcessGetElementResponse(
    bool enqueue_result, GetElementResult& get_element_result,
    std::shared_ptr<Result> result, Task& task) TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  result->ready = true;
  result->end_of_sequence = get_element_result.end_of_sequence;
  result->skip = get_element_result.skip;
  if (!get_element_result.end_of_sequence && !get_element_result.skip) {
    task.skipped_previous_round = false;
    result->element = std::move(get_element_result.components);
    result->element_index = get_element_result.element_index;
    result->task_id = task.info.task_id();
  } else if (get_element_result.skip) {
    task.skipped_previous_round = true;
  } else {
    task.end_of_sequence = true;
    finished_tasks_++;
  }
  if (enqueue_result && !result->end_of_sequence && !result->skip) {
    ctx_->RecordBufferEnqueue(result->element);
    results_.push(std::move(result));
  }
  get_next_cv_.notify_all();
}

absl::Status DataServiceClient::GetElementTraced(
    Task* task, int64_t deadline_micros, bool enqueue_result, bool allow_skip,
    std::shared_ptr<Result> result) TF_LOCKS_EXCLUDED(mu_) {
  VLOG(3) << "Getting an element for task id " << task->info.task_id();
  tsl::profiler::TraceMe activity("GetDataServiceElement",
                                  tsl::profiler::TraceMeLevel::kInfo);
  activity.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode(
        {{"address", task->info.worker_address()}});
  });
  if (IsCoordinatedRead()) {
    VLOG(3) << "Requesting element from consumer index "
            << params_.consumer_index.value() << ", round " << task->round;
    activity.AppendMetadata([&]() {
      return tsl::profiler::TraceMeEncode(
          {{"consumer_index", params_.consumer_index.value()},
           {"round_index", task->round}});
    });
  }
  absl::Status s =
      GetElement(task, deadline_micros, enqueue_result, allow_skip, result);
  mutex_lock l(mu_);
  VLOG(3) << "Got an element for task id " << task->info.task_id();
  return s;
}

absl::Status DataServiceClient::MaybeRemoveTask(Task& task,
                                                int64_t deadline_micros,
                                                Result& result)
    TF_LOCKS_EXCLUDED(mu_) {
  bool removed;
  VLOG(1) << "Requesting task removal for worker " << task.info.worker_address()
          << " in round " << task.round;
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&] {
        return dispatcher_->MaybeRemoveTask(task.info.task_id(),
                                            params_.consumer_index.value(),
                                            task.round, removed);
      },
      /*should_retry=*/
      [&] {
        mutex_lock l(mu_);
        return !cancelled_;
      },
      /*description=*/"request task removal ", deadline_micros));
  if (removed) {
    mutex_lock l(mu_);
    task.removed = true;
    result.ready = true;
    result.skip = true;
    get_next_cv_.notify_all();
    return absl::OkStatus();
  }
  VLOG(1) << "Failed to remove task for worker " << task.info.worker_address();
  return absl::OkStatus();
}

absl::Status DataServiceClient::GetElement(Task* task, int64_t deadline_micros,
                                           bool enqueue_result, bool allow_skip,
                                           std::shared_ptr<Result> result)
    TF_LOCKS_EXCLUDED(mu_) {
  GetElementResult get_element_result;
  while (true) {
    absl::Status s = TryGetElement(*task, allow_skip, get_element_result);
    if (s.ok()) {
      task->num_retries = 0;
      break;
    }
    if (!IsPreemptedError(s)) {
      if (task->worker->GetDataTransferProtocol() == kGrpcTransferProtocol ||
          task->worker->GetDataTransferProtocol() == kLocalTransferProtocol) {
        return absl::Status(
            s.code(),
            absl::StrCat(
                "Failed to get an element, with a nonretryable error: ",
                s.message()));
      }
      if (!task->worker->FallBackToGrpcAtGetElementTime()) {
        return absl::Status(
            s.code(),
            absl::StrCat("Failed to get an element over data "
                         "transfer protocol '",
                         task->worker->GetDataTransferProtocol(),
                         "', with a nonretryable error: ", s.message()));
      }
      LOG(ERROR) << "Failed to get an element over data transfer protocol '"
                 << task->worker->GetDataTransferProtocol()
                 << "', with a nonretryable error; falling back to grpc. "
                    "Original error: "
                 << s;
      metrics::RecordTFDataServiceDataTransferProtocolError(
          task->worker->GetDataTransferProtocol(),
          static_cast<error::Code>(s.raw_code()), std::string(s.message()));
      mutex_lock l(mu_);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<DataServiceWorkerClient> worker,
                          CreateGrpcWorkerClient(task->info));
      task->worker = std::move(worker);
      continue;
    }
    {
      mutex_lock l(mu_);
      if (cancelled_) {
        return errors::Cancelled("DataServiceDataset iterator cancelled");
      }
    }
    int64_t now_micros = Env::Default()->NowMicros();
    if (now_micros > deadline_micros) {
      return s;
    }
    if (IsCoordinatedRead() && task->num_retries > 0) {
      TF_RETURN_IF_ERROR(MaybeRemoveTask(*task, deadline_micros, *result));
      mutex_lock l(mu_);
      if (result->skip) {
        return absl::OkStatus();
      }
    }
    int64_t backoff_until = std::min(
        deadline_micros,
        now_micros + absl::ToInt64Microseconds(
                         tsl::ComputeRetryBackoff(task->num_retries++)));
    VLOG(1) << "Failed to get an element from worker "
            << task->info.worker_address() << ": " << s << ". Will retry in "
            << (backoff_until - now_micros) << " microseconds";
    Env::Default()->SleepForMicroseconds(backoff_until - now_micros);
    if (!IsCoordinatedRead()) {
      mutex_lock l(mu_);
      // Mark the result as skipped so that we try reading from a different
      // task before returning to this one.
      result->ready = true;
      result->skip = true;
      return absl::OkStatus();
    }
  }
  ProcessGetElementResponse(enqueue_result, get_element_result, result, *task);
  return absl::OkStatus();
}

bool DataServiceClient::ResultReady() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return !results_.empty() && results_.front()->ready;
}

std::shared_ptr<DataServiceClient::Result> DataServiceClient::PopNextResult()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<Result> result = results_.front();
  results_.pop();
  ctx_->RecordBufferDequeue(result->element);
  return result;
}

bool DataServiceClient::IsCoordinatedRead() const {
  return params_.num_consumers.has_value();
}

std::string DataServiceClient::DebugString() const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return absl::Substitute(
      "results_ { size: $0 front.ready: $1 } iteration_finished_: $2 "
      "tasks { size: $3 } finished_tasks_: $4 "
      "num_running_worker_threads_: $5",
      results_.size(), !results_.empty() && results_.front()->ready,
      iteration_finished_, tasks_.size(), finished_tasks_,
      num_running_worker_threads_);
}

}  // namespace data
}  // namespace tensorflow
