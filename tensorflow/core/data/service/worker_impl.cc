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

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/create_channel.h"
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/auto_shard_rewriter.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_split_provider.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/tsl/platform/status_to_from_proto.h"
#include "tensorflow/tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr absl::Duration kRetryInterval = absl::Seconds(5);
constexpr absl::Duration kDefaultHeartBeatInterval = absl::Seconds(30);
constexpr absl::Duration kDefaultDispatcherTimeout = absl::Hours(1);

using WorkerConfig = experimental::WorkerConfig;

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
    return OkStatus();
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
  return OkStatus();
}

WorkerConfig ApplyWorkerDefaults(const WorkerConfig& config) {
  WorkerConfig new_config(config);
  if (new_config.heartbeat_interval_ms() == 0) {
    new_config.set_heartbeat_interval_ms(
        absl::ToInt64Milliseconds(kDefaultHeartBeatInterval));
  }
  if (new_config.dispatcher_timeout_ms() == 0) {
    new_config.set_dispatcher_timeout_ms(
        absl::ToInt64Milliseconds(kDefaultDispatcherTimeout));
  }
  return new_config;
}

TaskDef Export(const TaskDef& task) {
  TaskDef result;
  switch (task.dataset_case()) {
    case TaskDef::kDatasetDef:
      result.set_path(
          "In-memory dataset graphs are omitted for brevity. To view datasets "
          "stored on the dispatcher, configure a `work_dir`.");
      break;
    case TaskDef::kPath:
      result.set_path(task.path());
      break;
    default:
      break;
  }
  result.set_dataset_id(task.dataset_id());
  result.set_task_id(task.task_id());
  result.set_iteration_id(task.iteration_id());
  result.set_num_split_providers(task.num_split_providers());
  result.set_worker_address(task.worker_address());
  *result.mutable_processing_mode_def() = task.processing_mode_def();
  switch (task.optional_num_consumers_case()) {
    case TaskDef::kNumConsumers:
      result.set_num_consumers(task.num_consumers());
      break;
    default:
      break;
  }
  result.set_num_workers(task.num_workers());
  result.set_worker_index(task.worker_index());
  return result;
}
}  // namespace

mutex LocalWorkers::mu_(LINKER_INITIALIZED);
LocalWorkers::AddressToWorkerMap* LocalWorkers::local_workers_ =
    new AddressToWorkerMap();

DataServiceWorkerImpl::DataServiceWorkerImpl(const WorkerConfig& config)
    : config_(ApplyWorkerDefaults(config)), worker_uid_(port::JobUid()) {
  metrics::RecordTFDataServiceWorkerCreated();
}

DataServiceWorkerImpl::~DataServiceWorkerImpl() {
  mutex_lock l(mu_);
  cancelled_ = true;
  task_completion_cv_.notify_one();
  heartbeat_cv_.notify_one();
}

Status DataServiceWorkerImpl::Start(
    const std::string& worker_address,
    const std::vector<DataTransferServerInfo>& transfer_servers) {
  VLOG(3) << "Starting tf.data service worker at address " << worker_address;
  TF_RETURN_IF_ERROR(ValidateWorkerConfig());
  worker_address_ = worker_address;
  transfer_servers_ = transfer_servers;

  TF_ASSIGN_OR_RETURN(dispatcher_, CreateDispatcherClient());
  auto should_retry = [this]() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return !cancelled_;
  };
  TF_RETURN_IF_ERROR(grpc_util::Retry([this]() { return Heartbeat(); },
                                      should_retry, "Worker heartbeat.",
                                      /*deadline_micros=*/kint64max));
  LOG(INFO) << "Worker registered with dispatcher running at "
            << config_.dispatcher_address();
  task_completion_thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "data-service-worker-task-completion",
                                  [this]() { TaskCompletionThread(); }));
  heartbeat_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      {}, "data-service-worker-heartbeat", [this]() { HeartbeatThread(); }));
  mutex_lock l(mu_);
  registered_ = true;
  return OkStatus();
}

void DataServiceWorkerImpl::Stop() {
  absl::flat_hash_map<int64_t, std::shared_ptr<Task>> tasks;
  absl::flat_hash_map<SnapshotTask, std::unique_ptr<SnapshotStreamWriter>,
                      absl::Hash<SnapshotTask>>
      snapshot_writers;
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    tasks.swap(tasks_);
    snapshot_writers.swap(snapshot_writers_);
  }
  for (const auto& [task_id, task] : tasks) {
    StopTask(*task);
  }
  for (const auto& [unused, snapshot_writer] : snapshot_writers) {
    snapshot_writer->Cancel();
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

Status DataServiceWorkerImpl::ValidateWorkerConfig() const {
  const bool any_tag_is_empty = absl::c_any_of(
      config_.worker_tags(),
      [](const std::string& worker_tag) { return worker_tag.empty(); });
  if (any_tag_is_empty) {
    return errors::FailedPrecondition(
        "Worker tags cannot be empty. Got tags {",
        absl::StrJoin(config_.worker_tags().begin(),
                      config_.worker_tags().end(), ", "),
        "}");
  }
  return OkStatus();
}

StatusOr<std::unique_ptr<DataServiceDispatcherClient>>
DataServiceWorkerImpl::CreateDispatcherClient() const TF_LOCKS_EXCLUDED(mu_) {
  auto dispatcher = std::make_unique<DataServiceDispatcherClient>(
      config_.dispatcher_address(), config_.protocol());
  auto should_retry = [this]() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return !cancelled_;
  };
  TF_RETURN_IF_ERROR(
      grpc_util::Retry([&dispatcher]() { return dispatcher->Initialize(); },
                       should_retry, "Initialize dispatcher client.",
                       /*deadline_micros=*/kint64max));
  return dispatcher;
}

Status DataServiceWorkerImpl::GetElementResult(
    const GetElementRequest* request, struct GetElementResult* result) {
  Task* task = nullptr;
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
      if (deleted_tasks_.contains(request->task_id())) {
        return errors::FailedPrecondition(
            "Got request for local task ", request->task_id(), " of worker ",
            worker_address_, ", which has been deleted. You may be creating ",
            "a duplicate iteration which has already finished. To fix this, "
            "make sure to create your dataset only once, as opposed to "
            "re-creating it repeatedly inside a loop.");
      }
      if (finished_tasks_.contains(request->task_id())) {
        VLOG(3) << "Task is already finished";
        result->end_of_sequence = true;
        result->skip = false;
        return OkStatus();
      }
      // Perhaps the worker hasn't gotten the task from the dispatcher yet.
      // Return Unavailable so that the client knows to continue retrying.
      return errors::Unavailable("Task ", request->task_id(), " not found");
    }
    task = it->second.get();
    task->outstanding_requests++;
  }
  auto cleanup = gtl::MakeCleanup([&] {
    mutex_lock l(mu_);
    task->outstanding_requests--;
    cv_.notify_all();
  });
  TF_RETURN_IF_ERROR(EnsureTaskInitialized(*task));
  TF_RETURN_IF_ERROR(task->task_runner->GetNext(*request, *result));

  if (result->end_of_sequence) {
    mutex_lock l(mu_);
    VLOG(3) << "Reached end_of_sequence for task " << request->task_id();
    pending_completed_tasks_.insert(request->task_id());
    task_completion_cv_.notify_one();
  }
  return OkStatus();
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
    return OkStatus();
  }
  task = std::make_unique<Task>(task_def);
  VLOG(3) << "Began processing for task " << task_def.task_id()
          << " with processing mode "
          << task_def.processing_mode_def().DebugString();
  return OkStatus();
}

Status DataServiceWorkerImpl::EnsureTaskInitialized(
    DataServiceWorkerImpl::Task& task) {
  if (task.task_def.worker_address() != worker_address_) {
    return errors::Internal(absl::Substitute(
        "Dispatcher's worker address $0 does not match worker's address $1.",
        task.task_def.worker_address(), worker_address_));
  }

  mutex_lock l(task.mu);
  if (task.initialized) {
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(DatasetDef dataset_def, GetDatasetDef(task.task_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Dataset> dataset,
                      MakeDataset(dataset_def, task.task_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Iterator> iterator,
                      MakeDatasetIterator(*dataset, task.task_def));
  auto task_iterator = std::make_unique<StandaloneTaskIterator>(
      std::move(dataset), std::move(iterator));
  TF_RETURN_IF_ERROR(TaskRunner::Create(
      config_, task.task_def, std::move(task_iterator), task.task_runner));

  task.initialized = true;
  VLOG(3) << "Created iterator for task " << task.task_def.task_id();
  return OkStatus();
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
DataServiceWorkerImpl::MakeDataset(const DatasetDef& dataset_def,
                                   const TaskDef& task_def) const {
  TF_ASSIGN_OR_RETURN(AutoShardRewriter auto_shard_rewriter,
                      AutoShardRewriter::Create(task_def));
  // `ApplyAutoShardRewrite` does nothing if auto-sharding is disabled.
  TF_ASSIGN_OR_RETURN(
      GraphDef rewritten_graph,
      auto_shard_rewriter.ApplyAutoShardRewrite(dataset_def.graph()));
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), rewritten_graph, &dataset));
  return dataset;
}

StatusOr<std::unique_ptr<standalone::Iterator>>
DataServiceWorkerImpl::MakeDatasetIterator(standalone::Dataset& dataset,
                                           const TaskDef& task_def) const {
  std::unique_ptr<standalone::Iterator> iterator;
  if (IsNoShard(task_def.processing_mode_def()) ||
      IsStaticShard(task_def.processing_mode_def())) {
    TF_RETURN_IF_ERROR(dataset.MakeIterator(&iterator));
    return iterator;
  }

  if (IsDynamicShard(task_def.processing_mode_def())) {
    std::vector<std::unique_ptr<SplitProvider>> split_providers;
    split_providers.reserve(task_def.num_split_providers());
    for (int i = 0; i < task_def.num_split_providers(); ++i) {
      split_providers.push_back(std::make_unique<DataServiceSplitProvider>(
          config_.dispatcher_address(), config_.protocol(),
          task_def.iteration_id(), i, config_.dispatcher_timeout_ms()));
    }
    TF_RETURN_IF_ERROR(
        dataset.MakeIterator(std::move(split_providers), &iterator));
    return iterator;
  }

  return errors::InvalidArgument("Unrecognized processing mode: ",
                                 task_def.processing_mode_def().DebugString());
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
  return OkStatus();
}

Status DataServiceWorkerImpl::GetWorkerTasks(
    const GetWorkerTasksRequest* request, GetWorkerTasksResponse* response) {
  mutex_lock l(mu_);
  for (const auto& it : tasks_) {
    Task* task = it.second.get();
    TaskInfo* task_info = response->add_tasks();
    task_info->set_worker_address(worker_address_);
    task_info->set_task_id(task->task_def.task_id());
    task_info->set_iteration_id(task->task_def.iteration_id());
  }
  return OkStatus();
}

Status DataServiceWorkerImpl::GetSnapshotTaskProgresses(
    const GetSnapshotTaskProgressesRequest* request,
    GetSnapshotTaskProgressesResponse* response) {
  for (const auto& snapshot_task_progress : GetSnapshotTaskProgress()) {
    *response->add_snapshot_task_progresses() = snapshot_task_progress;
  }
  return OkStatus();
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
            l, absl::ToChronoMicroseconds(kRetryInterval));
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
  return OkStatus();
}

void DataServiceWorkerImpl::HeartbeatThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    int64_t next_heartbeat_micros =
        Env::Default()->NowMicros() + (config_.heartbeat_interval_ms() * 1000);
    {
      mutex_lock l(mu_);
      while (!cancelled_ &&
             Env::Default()->NowMicros() < next_heartbeat_micros) {
        int64_t time_to_wait_micros =
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

Status DataServiceWorkerImpl::Heartbeat() {
  WorkerHeartbeatRequest request = BuildWorkerHeartbeatRequest();
  TF_ASSIGN_OR_RETURN(WorkerHeartbeatResponse response,
                      dispatcher_->WorkerHeartbeat(request));
  UpdateTasks(response);
  return UpdateSnapshotWriters(response);
}

WorkerHeartbeatRequest DataServiceWorkerImpl::BuildWorkerHeartbeatRequest()
    const TF_LOCKS_EXCLUDED(mu_) {
  std::vector<int64_t> current_tasks;
  {
    mutex_lock l(mu_);
    for (const auto& task : tasks_) {
      current_tasks.push_back(task.first);
    }
  }

  WorkerHeartbeatRequest request;
  request.set_worker_address(worker_address_);
  *request.mutable_transfer_servers() = {transfer_servers_.begin(),
                                         transfer_servers_.end()};
  *request.mutable_worker_tags() = config_.worker_tags();
  request.set_worker_uid(worker_uid_);
  *request.mutable_current_tasks() = {current_tasks.begin(),
                                      current_tasks.end()};
  for (const auto& snapshot_task_progress : GetSnapshotTaskProgress()) {
    request.mutable_snapshot_task_progress()->insert(
        {snapshot_task_progress.snapshot_task().base_path(),
         snapshot_task_progress});
  }
  return request;
}

std::vector<SnapshotTaskProgress>
DataServiceWorkerImpl::GetSnapshotTaskProgress() const {
  mutex_lock l(mu_);
  std::vector<SnapshotTaskProgress> snapshot_task_progress;
  for (const auto& [snapshot_task, stream_writer] : snapshot_writers_) {
    SnapshotTaskProgress progress;
    progress.mutable_snapshot_task()->set_base_path(snapshot_task.base_path);
    progress.mutable_snapshot_task()->set_stream_index(
        snapshot_task.stream_index);
    StatusOr<bool> completed = stream_writer->Completed();
    if (completed.ok()) {
      progress.set_completed(*completed);
    } else {
      *progress.mutable_status() = tsl::StatusToProto(completed.status());
    }
    snapshot_task_progress.push_back(std::move(progress));
  }
  return snapshot_task_progress;
}

void DataServiceWorkerImpl::UpdateTasks(const WorkerHeartbeatResponse& response)
    TF_LOCKS_EXCLUDED(mu_) {
  std::vector<std::shared_ptr<Task>> tasks_to_delete;
  {
    mutex_lock l(mu_);
    for (const auto& task : response.new_tasks()) {
      VLOG(1) << "Received new task from dispatcher with id " << task.task_id();
      if (deleted_tasks_.contains(task.task_id())) {
        continue;
      }
      Status s = ProcessTaskInternal(task);
      if (!s.ok() && !errors::IsAlreadyExists(s)) {
        LOG(WARNING) << "Failed to start processing task " << task.task_id()
                     << ": " << s;
      }
    }
    tasks_to_delete.reserve(response.tasks_to_delete_size());
    for (int64_t task_id : response.tasks_to_delete()) {
      VLOG(3) << "Deleting task " << task_id
              << " at the request of the dispatcher";
      if (!tasks_.contains(task_id)) {
        continue;
      }
      tasks_to_delete.push_back(std::move(tasks_[task_id]));
      tasks_.erase(task_id);
      finished_tasks_.insert(task_id);
    }
  }
  for (const auto& task : tasks_to_delete) {
    StopTask(*task);
  }
}

Status DataServiceWorkerImpl::UpdateSnapshotWriters(
    const WorkerHeartbeatResponse& response) {
  mutex_lock l(mu_);
  absl::flat_hash_set<SnapshotTask> assigned_snapshot_task_keys;
  for (const SnapshotTaskDef& snapshot_task : response.snapshot_tasks()) {
    SnapshotTask snapshot_task_key{snapshot_task.base_path(),
                                   snapshot_task.stream_index()};
    assigned_snapshot_task_keys.insert(snapshot_task_key);
    if (snapshot_writers_.contains(snapshot_task_key)) {
      continue;
    }

    DatasetDef dataset_def;
    TF_RETURN_IF_ERROR(ReadBinaryProto(
        Env::Default(), DatasetDefFilePath(snapshot_task.base_path()),
        &dataset_def));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<StandaloneTaskIterator> iterator,
                        MakeSnapshotTaskIterator(snapshot_task, dataset_def));
    snapshot_writers_.emplace(
        snapshot_task_key,
        std::make_unique<SnapshotStreamWriter>(
            SnapshotWriterParams{
                snapshot_task.base_path(), snapshot_task.stream_index(),
                snapshot_task.metadata().compression(), Env::Default()},
            std::move(iterator)));
  }

  // Cancel writers for snapshots that are no longer assigned by the dispatcher.
  for (auto it = snapshot_writers_.begin(); it != snapshot_writers_.end();) {
    if (!assigned_snapshot_task_keys.contains(it->first)) {
      it->second->Cancel();
      snapshot_writers_.erase(it++);
    } else {
      ++it;
    }
  }

  return OkStatus();
}

StatusOr<std::unique_ptr<StandaloneTaskIterator>>
DataServiceWorkerImpl::MakeSnapshotTaskIterator(
    const SnapshotTaskDef& snapshot_task, const DatasetDef& dataset_def) const {
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), dataset_def.graph(), &dataset));

  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  split_providers.reserve(snapshot_task.num_sources());
  for (int i = 0; i < snapshot_task.num_sources(); ++i) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<DataServiceDispatcherClient> dispatcher,
                        CreateDispatcherClient());
    split_providers.push_back(std::make_unique<SnapshotSplitProvider>(
        worker_address_, snapshot_task,
        /*source_index=*/i, absl::Milliseconds(config_.dispatcher_timeout_ms()),
        std::move(dispatcher), Env::Default()));
  }
  std::unique_ptr<standalone::Iterator> iterator;
  TF_RETURN_IF_ERROR(
      dataset->MakeIterator(std::move(split_providers), &iterator));
  return std::make_unique<StandaloneTaskIterator>(std::move(dataset),
                                                  std::move(iterator));
}

void DataServiceWorkerImpl::DeleteLocalTask(const TaskInfo& task_info)
    TF_LOCKS_EXCLUDED(mu_) {
  std::shared_ptr<Task> task;
  {
    mutex_lock l(mu_);
    auto it = tasks_.find(task_info.task_id());
    if (it == tasks_.end() || !it->second) {
      return;
    }
    task = std::move(it->second);
    tasks_.erase(task_info.task_id());
    pending_completed_tasks_.insert(task_info.task_id());
    deleted_tasks_.insert(task_info.task_id());
  }

  VLOG(2) << "Delete local task " << task_info.task_id() << " from worker "
          << worker_address_ << " at the request of the client.";
  StopTask(*task);
}

WorkerStateExport DataServiceWorkerImpl::ExportState() const {
  WorkerStateExport worker_state_export;
  *worker_state_export.mutable_worker_config() = config_;
  mutex_lock l(mu_);
  if (!registered_) {
    return worker_state_export;
  }
  for (const auto& task : tasks_) {
    *worker_state_export.add_tasks() = Export(task.second->task_def);
  }
  for (int64_t finished_task : finished_tasks_) {
    worker_state_export.add_finished_task_ids(finished_task);
  }
  for (int64_t deleted_task : deleted_tasks_) {
    worker_state_export.add_deleted_task_ids(deleted_task);
  }
  return worker_state_export;
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
