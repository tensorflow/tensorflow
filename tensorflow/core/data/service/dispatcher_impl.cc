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

#include "tensorflow/core/data/service/dispatcher_impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/tsl/platform/errors.h"

#ifdef PLATFORM_GOOGLE
#include "file/logging/log_lines.h"
#endif
#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/service/validate_utils.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::protobuf::util::MessageDifferencer;

// The name of the journal directory inside the dispatcher's working directory.
// This name is load-bearing; do not change.
constexpr char kJournalDir[] = "tf_data_dispatcher_journal";
// The name of the datasets directory inside the dispatcher's working directory.
constexpr char kDatasetsDir[] = "datasets";

constexpr absl::Duration kDefaultIterationGcCheckInterval = absl::Minutes(10);
constexpr absl::Duration kDefaultIterationGcTimeout = absl::Minutes(5);
constexpr absl::Duration kDefaultClientTimeout = absl::Minutes(2);
constexpr absl::Duration kDefaultWorkerTimeout = absl::Minutes(1);

constexpr std::array<const char*, 8> kNodeNameSharingOps = {
    "HashTable",
    "HashTableV2",
    "MutableHashTable",
    "MutableHashTableV2",
    "MutableDenseHashTable",
    "MutableDenseHashTableV2",
    "MutableHashTableOfTensors",
    "MutableHashTableOfTensorsV2",
};

using DispatcherConfig = experimental::DispatcherConfig;
using Dataset = DispatcherState::Dataset;
using Worker = DispatcherState::Worker;
using Job = DispatcherState::Job;
using IterationKey = DispatcherState::IterationKey;
using Iteration = DispatcherState::Iteration;
using Task = DispatcherState::Task;

std::string JournalDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kJournalDir);
}

std::string DatasetsDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kDatasetsDir);
}

Status CreateWorkerStub(const std::string& address, const std::string& protocol,
                        std::unique_ptr<WorkerService::Stub>& stub) {
  ::grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<::grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol, &credentials));
  auto channel = ::grpc::CreateCustomChannel(address, credentials, args);
  stub = WorkerService::NewStub(channel);
  return OkStatus();
}

void PrepareGraph(GraphDef* graph) {
  for (NodeDef& node : *graph->mutable_node()) {
    for (const auto& op : kNodeNameSharingOps) {
      // Set `use_node_name_sharing` to `true` so that resources aren't deleted
      // prematurely. Otherwise, resources may be deleted when their ops are
      // deleted at the end of the GraphRunner::Run used by standalone::Dataset.
      if (node.op() == op) {
        (*node.mutable_attr())["use_node_name_sharing"].set_b(true);
      }
      if (!node.device().empty()) {
        *node.mutable_device() = "";
      }
    }
  }
  StripDevicePlacement(graph->mutable_library());
}

DispatcherConfig ApplyConfigDefaults(const DispatcherConfig& config) {
  DispatcherConfig new_config(config);
  if (new_config.job_gc_check_interval_ms() == 0) {
    new_config.set_job_gc_check_interval_ms(
        absl::ToInt64Milliseconds(kDefaultIterationGcCheckInterval));
  }
  if (new_config.job_gc_timeout_ms() == 0) {
    new_config.set_job_gc_timeout_ms(
        absl::ToInt64Milliseconds(kDefaultIterationGcTimeout));
  }
  if (new_config.client_timeout_ms() == 0) {
    new_config.set_client_timeout_ms(
        absl::ToInt64Milliseconds(kDefaultClientTimeout));
  }
  if (new_config.worker_timeout_ms() == 0) {
    new_config.set_worker_timeout_ms(
        absl::ToInt64Milliseconds(kDefaultWorkerTimeout));
  }
  return new_config;
}
}  // namespace

DataServiceDispatcherImpl::DataServiceDispatcherImpl(
    const DispatcherConfig& config)
    : config_(ApplyConfigDefaults(config)),
      env_(Env::Default()),
      state_(config_) {
  if (config_.work_dir().empty()) {
    dataset_store_ = std::make_unique<MemoryDatasetStore>();
  } else {
    dataset_store_ = std::make_unique<FileSystemDatasetStore>(
        DatasetsDir(config_.work_dir()));
  }
}

DataServiceDispatcherImpl::~DataServiceDispatcherImpl() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    maintenance_thread_cv_.notify_all();
  }
  maintenance_thread_.reset();
}

Status DataServiceDispatcherImpl::Start() {
  mutex_lock l(mu_);
  if (config_.job_gc_timeout_ms() >= 0) {
    maintenance_thread_ = absl::WrapUnique(env_->StartThread(
        {}, "maintenance-thread", [&] { MaintenanceThread(); }));
  }
  if (config_.work_dir().empty()) {
    if (config_.fault_tolerant_mode()) {
      return errors::InvalidArgument(
          "fault_tolerant_mode is True, but no work_dir is configured.");
    }
  } else {
    TF_RETURN_IF_ERROR(
        env_->RecursivelyCreateDir(DatasetsDir(config_.work_dir())));
  }
  if (!config_.fault_tolerant_mode()) {
    LOG(INFO) << "Running with fault_tolerant_mode=False. The dispatcher will "
                 "not be able to recover its state on restart.";
    started_ = true;
    return OkStatus();
  }
  journal_writer_ =
      std::make_unique<FileJournalWriter>(env_, JournalDir(config_.work_dir()));
  LOG(INFO) << "Attempting to restore dispatcher state from journal in "
            << JournalDir(config_.work_dir());
  Update update;
  bool end_of_journal = false;
  FileJournalReader reader(env_, JournalDir(config_.work_dir()));
  Status s = reader.Read(update, end_of_journal);
  if (errors::IsNotFound(s)) {
    LOG(INFO) << "No journal found. Starting dispatcher from new state.";
  } else if (!s.ok()) {
    return s;
  } else {
    while (!end_of_journal) {
      TF_RETURN_IF_ERROR(ApplyWithoutJournaling(update));
      TF_RETURN_IF_ERROR(reader.Read(update, end_of_journal));
    }
  }
  for (const auto& iteration : state_.ListIterations()) {
    if (IsDynamicShard(iteration->job->processing_mode)) {
      TF_RETURN_IF_ERROR(RestoreSplitProviders(
          *iteration, split_providers_[iteration->iteration_id]));
    }
  }
  for (const auto& client_id : state_.ListActiveClientIds()) {
    // Conservatively pretend we just received a heartbeat from all clients, so
    // that we don't garbage collect iterations too early.
    latest_client_heartbeats_time_[client_id] =
        absl::FromUnixMicros(env_->NowMicros());
  }
  // Initialize the journal writer in `Start` so that we fail fast in case it
  // can't be initialized.
  TF_RETURN_IF_ERROR(journal_writer_.value()->EnsureInitialized());

  for (const auto& path : state_.ListSnapshotPaths()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<SnapshotManager> snapshot_manager,
                        SnapshotManager::Resume(path, env_));
    snapshots_.insert({path, std::move(snapshot_manager)});
  }

  started_ = true;
  return OkStatus();
}

size_t DataServiceDispatcherImpl::NumActiveIterations() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  size_t count = 0;
  for (const auto& iteration : state_.ListIterations()) {
    if (!iteration->finished) {
      count++;
    }
  }
  return count;
}

Status DataServiceDispatcherImpl::RestoreSplitProviders(
    const Iteration& iteration,
    std::vector<std::unique_ptr<SplitProvider>>& restored)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  const std::vector<int64_t>& indices =
      iteration.distributed_epoch_state.value().indices;
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(
      MakeSplitProviders(iteration.job->dataset_id, split_providers));
  for (int provider_index = 0; provider_index < indices.size();
       ++provider_index) {
    int index = indices[provider_index];
    VLOG(1) << "Restoring split provider " << provider_index
            << " for iteration " << iteration.iteration_id << " to index "
            << index;
    Tensor unused_tensor;
    bool unused_end_of_splits;
    for (int i = 0; i < index; ++i) {
      TF_RETURN_IF_ERROR(split_providers[provider_index]->GetNext(
          &unused_tensor, &unused_end_of_splits));
    }
  }
  restored = std::move(split_providers);
  return OkStatus();
}

Status DataServiceDispatcherImpl::FindTasksToDelete(
    const absl::flat_hash_set<int64_t>& current_tasks,
    const std::vector<std::shared_ptr<const Task>>& assigned_tasks,
    WorkerHeartbeatResponse* response) {
  absl::flat_hash_set<int64_t> assigned_ids;
  for (const auto& assigned : assigned_tasks) {
    assigned_ids.insert(assigned->task_id);
  }
  for (int64_t current_task : current_tasks) {
    if (!assigned_ids.contains(current_task)) {
      response->add_tasks_to_delete(current_task);
    }
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::FindNewTasks(
    const std::string& worker_address,
    const absl::flat_hash_set<int64_t>& current_tasks,
    std::vector<std::shared_ptr<const Task>>& assigned_tasks,
    WorkerHeartbeatResponse* response) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // Check for round-robin iterations that had tasks on the worker removed. Now
  // that the worker is back, we create a new pending task for the worker.
  absl::flat_hash_set<int64_t> assigned_iteration_ids;
  for (const auto& task : assigned_tasks) {
    assigned_iteration_ids.insert(task->iteration->iteration_id);
  }
  for (const auto& iteration : state_.ListIterations()) {
    if (!assigned_iteration_ids.contains(iteration->iteration_id) &&
        iteration->IsRoundRobin() && !iteration->finished) {
      VLOG(1) << "Creating pending task for reconnected worker "
              << worker_address;
      TF_RETURN_IF_ERROR(CreatePendingTask(iteration, worker_address));
    }
  }
  // Refresh assigned_tasks to include newly added pending tasks.
  TF_RETURN_IF_ERROR(state_.TasksForWorker(worker_address, assigned_tasks));
  for (const auto& task : assigned_tasks) {
    if (current_tasks.contains(task->task_id)) {
      continue;
    }
    TaskDef* task_def = response->add_new_tasks();
    TF_RETURN_IF_ERROR(PopulateTaskDef(task, task_def));
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::WorkerHeartbeat(
    const WorkerHeartbeatRequest* request, WorkerHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(4) << "Received worker heartbeat request from worker "
          << request->worker_address();
  mutex_lock l(mu_);
  const std::string& worker_address = request->worker_address();
  latest_worker_heartbeats_time_[worker_address] =
      absl::FromUnixMicros(env_->NowMicros());
  // Assigned tasks from the perspective of the dispatcher.
  std::vector<std::shared_ptr<const Task>> assigned_tasks;
  Status s = state_.TasksForWorker(worker_address, assigned_tasks);
  if (!s.ok()) {
    if (!errors::IsNotFound(s)) {
      return s;
    }
    VLOG(1) << "Registering new worker at address " << worker_address;
    TF_RETURN_IF_ERROR(state_.ValidateWorker(worker_address));
    Update update;
    update.mutable_register_worker()->set_worker_address(worker_address);
    *update.mutable_register_worker()->mutable_transfer_servers() =
        request->transfer_servers();
    *update.mutable_register_worker()->mutable_worker_tags() =
        request->worker_tags();
    update.mutable_register_worker()->set_worker_uid(request->worker_uid());
    TF_RETURN_IF_ERROR(Apply(update));
    TF_RETURN_IF_ERROR(CreateTasksForWorker(worker_address));
    TF_RETURN_IF_ERROR(state_.TasksForWorker(worker_address, assigned_tasks));
  }
  absl::flat_hash_set<int64_t> current_tasks;
  current_tasks.insert(request->current_tasks().cbegin(),
                       request->current_tasks().cend());
  TF_RETURN_IF_ERROR(
      FindTasksToDelete(current_tasks, assigned_tasks, response));
  TF_RETURN_IF_ERROR(
      FindNewTasks(worker_address, current_tasks, assigned_tasks, response));

  for (const auto& [path, snapshot_manager] : snapshots_) {
    TF_RETURN_IF_ERROR(snapshot_manager->WorkerHeartbeat(*request, *response));
  }

  VLOG(4) << "Finished worker heartbeat for worker at address "
          << request->worker_address();
  return OkStatus();
}

Status DataServiceDispatcherImpl::WorkerUpdate(
    const WorkerUpdateRequest* request, WorkerUpdateResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  for (auto& update : request->updates()) {
    int64_t task_id = update.task_id();
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, task));
    if (update.completed()) {
      if (task->finished) {
        VLOG(1) << "Received completion update for already-finished task "
                << task->task_id << " on worker " << task->worker_address;
        continue;
      }
      Update update;
      update.mutable_finish_task()->set_task_id(task_id);
      TF_RETURN_IF_ERROR(Apply(update));
      VLOG(3) << "Task " << task_id << " from iteration "
              << task->iteration->iteration_id << " completed";
    }
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const GetDatasetDefRequest* request, GetDatasetDefResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(request->dataset_id(), dataset));
  std::shared_ptr<const DatasetDef> dataset_def;
  TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, dataset_def));
  *response->mutable_dataset_def() = *dataset_def;
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetSplit(const GetSplitRequest* request,
                                           GetSplitResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  int64_t iteration_id = request->iteration_id();
  int64_t repetition = request->repetition();
  int64_t provider_index = request->split_provider_index();
  VLOG(3) << "Received GetSplit request for iteration " << iteration_id
          << ", repetition " << repetition << ", split provider index "
          << provider_index;
  std::shared_ptr<const Iteration> iteration;
  TF_RETURN_IF_ERROR(state_.IterationFromId(iteration_id, iteration));
  if (!iteration->distributed_epoch_state.has_value()) {
    return errors::FailedPrecondition(
        "Cannot get split for iteration ", iteration_id,
        ", since it is not a distributed_epoch iteration.");
  }
  int64_t current_repetition =
      iteration->distributed_epoch_state.value().repetitions[provider_index];
  if (repetition < current_repetition) {
    response->set_end_of_splits(true);
    VLOG(3) << "Returning end_of_splits since current repetition "
            << current_repetition
            << " is greater than the requested repetition " << repetition;
    return OkStatus();
  }
  if (repetition > current_repetition) {
    // This could happen if an iterator is repeated before reaching end of
    // input, e.g. for the longer input to `Dataset.zip`. In this case we mark
    // the previous repetitions as completed and advance to the requested
    // repetition.
    TF_RETURN_IF_ERROR(split_providers_[iteration_id][provider_index]->Reset());
  }
  SplitProvider* split_provider =
      split_providers_[iteration_id][provider_index].get();
  DCHECK(split_provider != nullptr);
  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider->GetNext(&split, &end_of_splits));
  TF_RETURN_IF_ERROR(RecordSplitProduced(iteration_id, repetition,
                                         request->split_provider_index(),
                                         end_of_splits));
  response->set_end_of_splits(end_of_splits);
  if (end_of_splits) {
    // Reset the split provider to prepare for the next iteration.
    TF_RETURN_IF_ERROR(split_providers_[iteration_id][provider_index]->Reset());
  } else {
    split.AsProtoTensorContent(response->mutable_split());
  }
  VLOG(3) << "Returning from GetSplit, end_of_splits=" << end_of_splits;
  return OkStatus();
}

Status DataServiceDispatcherImpl::MakeSplitProviders(
    const std::string& dataset_id,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  std::shared_ptr<const DatasetDef> dataset_def;
  TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, dataset_def));
  TF_RETURN_IF_ERROR(CreateSplitProviders(*dataset_def, split_providers));
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetVersion(const GetVersionRequest* request,
                                             GetVersionResponse* response) {
  response->set_version(kDataServiceVersion);
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetOrRegisterDataset(
    const GetOrRegisterDatasetRequest* request,
    GetOrRegisterDatasetResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  DatasetDef dataset_def = request->dataset();
  GraphDef* graph = dataset_def.mutable_graph();
  PrepareGraph(graph);

  mutex_lock l(mu_);
  TF_ASSIGN_OR_RETURN(std::optional<std::string> dataset_id,
                      FindDataset(*request));
  if (dataset_id.has_value()) {
    VLOG(3) << "RegisterDataset returns an existing dataset with ID = "
            << *dataset_id;
    response->set_dataset_id(*dataset_id);
    return OkStatus();
  }

  std::string new_dataset_id;
  TF_RETURN_IF_ERROR(RegisterDataset(dataset_def, request->metadata(),
                                     request->dataset_id(), new_dataset_id));
  response->set_dataset_id(new_dataset_id);
  VLOG(3) << "Registered new dataset with id " << new_dataset_id;
  return OkStatus();
}

StatusOr<std::optional<std::string>> DataServiceDispatcherImpl::FindDataset(
    const GetOrRegisterDatasetRequest& request)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> existing_dataset;
  Status status = state_.DatasetFromId(request.dataset_id(), existing_dataset);

  if (errors::IsNotFound(status)) {
    return std::optional<std::string>();
  }
  TF_RETURN_IF_ERROR(status);
  if (!request.dataset_id().empty()) {
    TF_RETURN_IF_ERROR(ValidateMatchingDataset(
        request.dataset_id(), request.metadata(), existing_dataset->metadata));
  }
  return std::optional<std::string>(existing_dataset->dataset_id);
}

Status DataServiceDispatcherImpl::RegisterDataset(
    const DatasetDef& dataset, const DataServiceMetadata& metadata,
    const std::string& requested_dataset_id, std::string& dataset_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  dataset_id = requested_dataset_id;
  if (dataset_id.empty()) {
    dataset_id = state_.NextAvailableDatasetId();
  }
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  *register_dataset->mutable_metadata() = metadata;
  TF_RETURN_IF_ERROR(dataset_store_->Put(dataset_id, dataset));
  return Apply(update);
}

Status DataServiceDispatcherImpl::GetDataServiceMetadata(
    const GetDataServiceMetadataRequest* request,
    GetDataServiceMetadataResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  std::string dataset_id = request->dataset_id();
  std::shared_ptr<const Dataset> dataset;

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  VLOG(3) << "Get the data service metadata for dataset id: " << dataset_id
          << ".";
  *response->mutable_metadata() = dataset->metadata;
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetDataServiceConfig(
    const GetDataServiceConfigRequest* request,
    GetDataServiceConfigResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  response->mutable_config()->set_deployment_mode(config_.deployment_mode());
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetOrCreateJob(
    const GetOrCreateJobRequest* request, GetOrCreateJobResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(3) << "GetOrCreateJob(" << request->DebugString() << ")";
  std::shared_ptr<const Job> job;
  {
    mutex_lock l(mu_);
    std::string job_name;
    if (request->optional_job_name_case() == GetOrCreateJobRequest::kJobName) {
      job_name = request->job_name();
    } else {
      job_name = absl::StrCat("anonymous_job_", state_.NextAvailableJobId(),
                              "_", random::New64());
    }
    Status s = state_.JobByName(job_name, job);
    if (s.ok()) {
      TF_RETURN_IF_ERROR(ValidateMatchingJob(job, *request));
    } else if (errors::IsNotFound(s)) {
      TF_RETURN_IF_ERROR(CreateJob(job_name, *request, job));
    } else {
      return s;
    }
    response->set_job_id(job->id);
  }
  VLOG(3) << "Received job id " << job->id << " for CreateJob("
          << request->DebugString() << ")";
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetOrCreateIteration(
    const GetOrCreateIterationRequest* request,
    GetOrCreateIterationResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(3) << "GetOrCreateIteration(" << request->DebugString() << ")";
  std::shared_ptr<const Iteration> iteration;
  std::vector<std::shared_ptr<const Task>> tasks;
  {
    mutex_lock l(mu_);
    std::shared_ptr<const Job> job;
    TF_RETURN_IF_ERROR(state_.JobFromId(request->job_id(), job));
    IterationKey key(job->job_name, request->repetition());
    Status s = state_.IterationByKey(key, iteration);
    if (!s.ok() && !errors::IsNotFound(s)) {
      return s;
    }
    if (errors::IsNotFound(s) || iteration->garbage_collected) {
      TF_RETURN_IF_ERROR(CreateIteration(*request, iteration));
      TF_RETURN_IF_ERROR(CreateTasksForIteration(iteration, tasks));
    }
    int64_t iteration_client_id;
    TF_RETURN_IF_ERROR(
        AcquireIterationClientId(iteration, iteration_client_id));
    response->set_iteration_client_id(iteration_client_id);
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));
  VLOG(3) << "Created iteration " << iteration->iteration_id
          << " for CreateIteration(" << request->DebugString() << ")";
  return OkStatus();
}

Status DataServiceDispatcherImpl::MaybeRemoveTask(
    const MaybeRemoveTaskRequest* request, MaybeRemoveTaskResponse* response) {
  VLOG(1) << "Attempting to remove task. Request: " << request->DebugString();
  std::shared_ptr<TaskRemover> remover;
  std::shared_ptr<const Task> task;
  {
    mutex_lock l(mu_);
    Status s = state_.TaskFromId(request->task_id(), task);
    if (errors::IsNotFound(s)) {
      // Task is already removed.
      response->set_removed(true);
      return OkStatus();
    }
    TF_RETURN_IF_ERROR(s);
    auto& remover_ref = remove_task_requests_[task->task_id];
    if (remover_ref == nullptr) {
      if (!task->iteration->IsRoundRobin()) {
        return errors::FailedPrecondition(
            "MaybeRemoveTask called on a non-round-robin task.");
      }
      remover_ref = std::make_shared<TaskRemover>(
          task->iteration->job->num_consumers.value());
    }
    remover = remover_ref;
  }
  bool removed =
      remover->RequestRemoval(request->consumer_index(), request->round());
  response->set_removed(removed);
  if (!removed) {
    VLOG(1) << "Failed to remove task " << task->task_id;
    return OkStatus();
  }
  mutex_lock l(mu_);
  if (!task->removed) {
    Update update;
    RemoveTaskUpdate* remove_task = update.mutable_remove_task();
    remove_task->set_task_id(request->task_id());
    TF_RETURN_IF_ERROR(Apply(update));
  }
  VLOG(1) << "Task " << task->task_id << " successfully removed";
  return OkStatus();
}

Status DataServiceDispatcherImpl::ReleaseIterationClient(
    const ReleaseIterationClientRequest* request,
    ReleaseIterationClientResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  int64_t iteration_client_id = request->iteration_client_id();
  std::shared_ptr<const Iteration> iteration;
  TF_RETURN_IF_ERROR(
      state_.IterationForIterationClientId(iteration_client_id, iteration));
  Update update;
  ReleaseIterationClientUpdate* release_iteration_client =
      update.mutable_release_iteration_client();
  release_iteration_client->set_iteration_client_id(iteration_client_id);
  release_iteration_client->set_time_micros(env_->NowMicros());
  TF_RETURN_IF_ERROR(Apply(update));
  return OkStatus();
}

// Validates that the job matches the requested processing mode.
Status DataServiceDispatcherImpl::ValidateMatchingJob(
    std::shared_ptr<const Job> job, const GetOrCreateJobRequest& request)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string diff;
  if (!MessageDifferencer::Equals(job->processing_mode,
                                  request.processing_mode_def())) {
    strings::StrAppend(&diff, "Existing processing mode: <",
                       job->processing_mode.ShortDebugString(), ">; got <",
                       request.processing_mode_def().ShortDebugString(), ">. ");
  }

  if (job->use_cross_trainer_cache != request.use_cross_trainer_cache()) {
    strings::StrAppend(
        &diff, "Existing cross-trainer cache: <",
        (job->use_cross_trainer_cache ? "enabled" : "disabled"), ">; got <",
        (request.use_cross_trainer_cache() ? "enabled" : "disabled"), ">. ");
  }

  if (job->target_workers != request.target_workers()) {
    strings::StrAppend(&diff, "Existing target workers: <",
                       TargetWorkersToString(job->target_workers), ">; got <",
                       TargetWorkersToString(request.target_workers()), ">. ");
  }

  if (!diff.empty()) {
    return errors::InvalidArgument(
        "Tried to create job with name ", job->job_name,
        ", but found an existing job with different parameters: ", diff);
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreateJob(
    const std::string& job_name, const GetOrCreateJobRequest& request,
    std::shared_ptr<const Job>& job) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(ValidateProcessingMode(request.processing_mode_def()));
  int64_t job_id = state_.NextAvailableJobId();
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_job_name(job_name);
  create_job->set_dataset_id(request.dataset_id());
  *create_job->mutable_processing_mode_def() = request.processing_mode_def();
  const bool is_coordinated_read = (request.optional_num_consumers_case() ==
                                    GetOrCreateJobRequest::kNumConsumers);
  if (is_coordinated_read) {
    create_job->set_num_consumers(request.num_consumers());
  }
  create_job->set_target_workers(request.target_workers());
  create_job->set_use_cross_trainer_cache(request.use_cross_trainer_cache());
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, job));
  tensorflow::metrics::RecordTFDataServiceJobsCreated(
      request.processing_mode_def(), is_coordinated_read);
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreateIteration(
    const GetOrCreateIterationRequest& request,
    std::shared_ptr<const Iteration>& iteration)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t iteration_id = state_.NextAvailableIterationId();
  int64_t num_split_providers = 0;
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobFromId(request.job_id(), job));
  if (IsDynamicShard(job->processing_mode)) {
    TF_RETURN_IF_ERROR(
        MakeSplitProviders(job->dataset_id, split_providers_[iteration_id]));
    num_split_providers = split_providers_[iteration_id].size();
  }
  Update update;
  CreateIterationUpdate* create_iteration = update.mutable_create_iteration();
  create_iteration->set_iteration_id(iteration_id);
  create_iteration->set_repetition(request.repetition());
  create_iteration->set_job_id(request.job_id());
  create_iteration->set_num_split_providers(num_split_providers);
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.IterationFromId(iteration_id, iteration));
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreateTasksForWorker(
    const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Iteration>> iterations =
      state_.ListIterations();
  for (const auto& iteration : iterations) {
    if (iteration->finished) {
      continue;
    }
    if (iteration->job->num_consumers.has_value()) {
      TF_RETURN_IF_ERROR(CreatePendingTask(iteration, worker_address));
      continue;
    }
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(iteration, worker_address, task));
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::AcquireIterationClientId(
    const std::shared_ptr<const Iteration>& iteration,
    int64_t& iteration_client_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  iteration_client_id = state_.NextAvailableIterationClientId();
  Update update;
  AcquireIterationClientUpdate* acquire_iteration_client =
      update.mutable_acquire_iteration_client();
  acquire_iteration_client->set_iteration_client_id(iteration_client_id);
  acquire_iteration_client->set_iteration_id(iteration->iteration_id);
  TF_RETURN_IF_ERROR(Apply(update));
  // Does not release clients before they start to read from the dataset.
  latest_client_heartbeats_time_[iteration_client_id] = absl::InfiniteFuture();
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreateTasksForIteration(
    std::shared_ptr<const Iteration> iteration,
    std::vector<std::shared_ptr<const Task>>& tasks)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Worker>> workers = state_.ListWorkers();
  tasks.clear();
  tasks.reserve(workers.size());
  for (const auto& worker : workers) {
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(iteration, worker->address, task));
    tasks.push_back(task);
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreatePendingTask(
    std::shared_ptr<const Iteration> iteration,
    const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t task_id = state_.NextAvailableTaskId();
  Update update;
  CreatePendingTaskUpdate* create_task = update.mutable_create_pending_task();
  create_task->set_task_id(task_id);
  create_task->set_iteration_id(iteration->iteration_id);
  create_task->set_worker_address(worker_address);
  create_task->set_starting_round(round_robin_rounds_[iteration->iteration_id] +
                                  1);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  *create_task->mutable_transfer_servers() = {worker->transfer_servers.begin(),
                                              worker->transfer_servers.end()};
  *create_task->mutable_worker_tags() = {worker->tags.begin(),
                                         worker->tags.end()};
  create_task->set_worker_uid(worker->uid);
  TF_RETURN_IF_ERROR(Apply(update));
  return OkStatus();
}

Status DataServiceDispatcherImpl::CreateTask(
    std::shared_ptr<const Iteration> iteration,
    const std::string& worker_address, std::shared_ptr<const Task>& task)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t task_id = state_.NextAvailableTaskId();
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_iteration_id(iteration->iteration_id);
  create_task->set_worker_address(worker_address);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  *create_task->mutable_transfer_servers() = {worker->transfer_servers.begin(),
                                              worker->transfer_servers.end()};
  *create_task->mutable_worker_tags() = {worker->tags.begin(),
                                         worker->tags.end()};
  create_task->set_worker_uid(worker->uid);
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, task));
  return OkStatus();
}

Status DataServiceDispatcherImpl::AssignTasks(
    std::vector<std::shared_ptr<const Task>> tasks) TF_LOCKS_EXCLUDED(mu_) {
  for (const auto& task : tasks) {
    TF_RETURN_IF_ERROR(AssignTask(task));
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetOrCreateWorkerStub(
    const std::string& worker_address, WorkerService::Stub*& out_stub)
    TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(mu_);
    auto it = worker_stubs_.find(worker_address);
    if (it != worker_stubs_.end()) {
      out_stub = it->second.get();
      return OkStatus();
    }
  }
  std::unique_ptr<WorkerService::Stub> stub;
  TF_RETURN_IF_ERROR(
      CreateWorkerStub(worker_address, config_.protocol(), stub));
  {
    mutex_lock l(mu_);
    // A concurrent call could have already created the stub.
    auto& worker = worker_stubs_[worker_address];
    if (worker == nullptr) {
      worker = std::move(stub);
    }
    out_stub = worker.get();
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::AssignTask(std::shared_ptr<const Task> task)
    TF_LOCKS_EXCLUDED(mu_) {
  VLOG(2) << "Started assigning task " << task->task_id << " to worker "
          << task->worker_address;
  grpc::ClientContext client_ctx;
  ProcessTaskRequest req;
  TaskDef* task_def = req.mutable_task();
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(PopulateTaskDef(task, task_def));
  }
  ProcessTaskResponse resp;
  WorkerService::Stub* stub;
  TF_RETURN_IF_ERROR(GetOrCreateWorkerStub(task->worker_address, stub));
  grpc::Status s = stub->ProcessTask(&client_ctx, req, &resp);
  if (!s.ok()) {
    if (s.error_code() == grpc::StatusCode::UNAVAILABLE ||
        s.error_code() == grpc::StatusCode::ABORTED ||
        s.error_code() == grpc::StatusCode::CANCELLED) {
      // Worker is presumably preempted. We will assign the task to the worker
      // when it reconnects.
      return OkStatus();
    }
    return grpc_util::WrapError(
        absl::StrCat("Failed to submit task to worker ", task->worker_address),
        s);
  }
  VLOG(2) << "Finished assigning task " << task->task_id << " to worker "
          << task->worker_address;
  return OkStatus();
}

Status DataServiceDispatcherImpl::ClientHeartbeat(
    const ClientHeartbeatRequest* request, ClientHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(4) << "Received heartbeat from client id "
          << request->iteration_client_id();
  latest_client_heartbeats_time_[request->iteration_client_id()] =
      absl::FromUnixMicros(env_->NowMicros());
  std::shared_ptr<const Iteration> iteration;
  Status s = state_.IterationForIterationClientId(
      request->iteration_client_id(), iteration);
  if (errors::IsNotFound(s) && !config_.fault_tolerant_mode()) {
    return errors::NotFound(
        "Unknown iteration client id ", request->iteration_client_id(),
        ". The dispatcher is not configured to be fault tolerant, so this "
        "could be caused by a dispatcher restart.");
  }
  TF_RETURN_IF_ERROR(s);
  if (iteration->garbage_collected) {
    return errors::FailedPrecondition(
        "The requested iteration has been garbage collected due to inactivity. "
        "Consider configuring the dispatcher with a higher "
        "`iteration_gc_timeout_ms`.");
  }
  if (request->optional_current_round_case() ==
      ClientHeartbeatRequest::kCurrentRound) {
    round_robin_rounds_[request->iteration_client_id()] =
        std::max(round_robin_rounds_[request->iteration_client_id()],
                 request->current_round());
  }
  if (!iteration->pending_tasks.empty()) {
    const auto& task = iteration->pending_tasks.front();
    Update update;
    ClientHeartbeatUpdate* client_heartbeat = update.mutable_client_heartbeat();
    bool apply_update = false;
    client_heartbeat->set_iteration_client_id(request->iteration_client_id());
    std::optional<int64_t> blocked_round;
    if (request->optional_blocked_round_case() ==
        ClientHeartbeatRequest::kBlockedRound) {
      blocked_round = request->blocked_round();
    }
    VLOG(1) << "Handling pending task in iteration client heartbeat. "
               "iteration_client_id: "
            << request->iteration_client_id()
            << ". current_round: " << request->current_round()
            << ". blocked_round: " << blocked_round.value_or(-1)
            << ". target_round: " << task.target_round;
    if (request->current_round() >= task.target_round) {
      TaskRejected* rejected = client_heartbeat->mutable_task_rejected();
      // Exponentially try later and later rounds until consumers all agree.
      int64_t round_offset = 2;
      for (int i = 0; i < task.failures; ++i) {
        round_offset *= 2;
      }
      rejected->set_new_target_round(
          round_robin_rounds_[request->iteration_client_id()] + round_offset);
      apply_update = true;
    }
    if (blocked_round.has_value() &&
        blocked_round.value() <= task.target_round &&
        !task.ready_consumers.contains(request->iteration_client_id())) {
      client_heartbeat->set_task_accepted(true);
      apply_update = true;
    }
    if (apply_update) {
      TF_RETURN_IF_ERROR(Apply(update));
    }
  }
  if (!iteration->pending_tasks.empty()) {
    response->set_block_round(iteration->pending_tasks.front().target_round);
  }

  std::vector<std::shared_ptr<const Task>> tasks;
  TF_RETURN_IF_ERROR(state_.TasksForIteration(iteration->iteration_id, tasks));
  for (const auto& task : tasks) {
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task->worker_address);
    *task_info->mutable_transfer_servers() = {task->transfer_servers.begin(),
                                              task->transfer_servers.end()};
    *task_info->mutable_worker_tags() = {task->worker_tags.begin(),
                                         task->worker_tags.end()};
    task_info->set_task_id(task->task_id);
    task_info->set_iteration_id(iteration->iteration_id);
    task_info->set_worker_uid(task->worker_uid);
    task_info->set_starting_round(task->starting_round);
  }
  response->set_iteration_finished(iteration->finished);
  response->set_deployment_mode(config_.deployment_mode());
  VLOG(4) << "Found " << response->task_info_size()
          << " tasks for iteration client id "
          << request->iteration_client_id();
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetWorkers(const GetWorkersRequest* request,
                                             GetWorkersResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(3) << "Enter GetWorkers";
  std::vector<std::shared_ptr<const Worker>> workers = state_.ListWorkers();
  for (const auto& worker : workers) {
    WorkerInfo* info = response->add_workers();
    info->set_address(worker->address);
  }
  VLOG(3) << "Returning list of " << response->workers_size()
          << " workers from GetWorkers";
  return OkStatus();
}

Status DataServiceDispatcherImpl::Snapshot(const SnapshotRequest* request,
                                           SnapshotResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);

  if (snapshots_.contains(request->path())) {
    return errors::InvalidArgument("a snapshot at ", request->path(),
                                   " is already started or completed");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<SnapshotManager> snapshot_manager,
                      SnapshotManager::Start(*request, env_));
  snapshots_.insert({request->path(), std::move(snapshot_manager)});

  Update update;
  SnapshotUpdate* snapshot = update.mutable_snapshot();
  snapshot->set_path(request->path());
  TF_RETURN_IF_ERROR(Apply(update));

  return OkStatus();
}

Status DataServiceDispatcherImpl::GetSnapshotStreams(
    const GetSnapshotStreamsRequest* request,
    GetSnapshotStreamsResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);

  auto it = snapshots_.find(request->path());
  if (it == snapshots_.end()) {
    return errors::InvalidArgument(
        "the dispatcher does not know of a snapshot at ", request->path());
  }
  TF_RETURN_IF_ERROR(it->second->GetSnapshotStreams(*response));
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetSnapshotSplit(
    const GetSnapshotSplitRequest* request,
    GetSnapshotSplitResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);

  auto it = snapshots_.find(request->base_path());
  if (it == snapshots_.end()) {
    return errors::InvalidArgument(
        "the dispatcher does not know of a snapshot at ", request->base_path());
  }

  TF_RETURN_IF_ERROR(it->second->GetSnapshotSplit(*request, *response));

  return OkStatus();
}

Status DataServiceDispatcherImpl::PopulateTaskDef(
    std::shared_ptr<const Task> task, TaskDef* task_def) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  task_def->set_dataset_id(task->iteration->job->dataset_id);
  task_def->set_iteration_id(task->iteration->iteration_id);
  task_def->set_worker_address(task->worker_address);
  task_def->set_task_id(task->task_id);
  *task_def->mutable_processing_mode_def() =
      task->iteration->job->processing_mode;
  if (IsStaticShard(task->iteration->job->processing_mode)) {
    task_def->set_num_workers(config_.worker_addresses_size());
    TF_ASSIGN_OR_RETURN(int64_t worker_index,
                        state_.GetWorkerIndex(task->worker_address));
    task_def->set_worker_index(worker_index);
  }
  if (task->iteration->distributed_epoch_state.has_value()) {
    task_def->set_num_split_providers(
        task->iteration->distributed_epoch_state.value().indices.size());
  }
  if (task->iteration->job->num_consumers.has_value()) {
    task_def->set_num_consumers(task->iteration->job->num_consumers.value());
  }
  task_def->set_use_cross_trainer_cache(
      task->iteration->job->use_cross_trainer_cache);
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(
      state_.DatasetFromId(task->iteration->job->dataset_id, dataset));
  if (config_.work_dir().empty()) {
    std::shared_ptr<const DatasetDef> dataset_def;
    TF_RETURN_IF_ERROR(dataset_store_->Get(dataset->dataset_id, dataset_def));
    *task_def->mutable_dataset_def() = *dataset_def;
  } else {
    std::string path =
        io::JoinPath(DatasetsDir(config_.work_dir()), dataset->dataset_id);
    task_def->set_path(path);
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::CheckStarted() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (!started_) {
    return errors::Unavailable("Dispatcher has not started yet.");
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::RecordSplitProduced(
    int64_t iteration_id, int64_t repetition, int64_t split_provider_index,
    bool finished) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Update update;
  ProduceSplitUpdate* produce_split = update.mutable_produce_split();
  produce_split->set_iteration_id(iteration_id);
  produce_split->set_repetition(repetition);
  produce_split->set_split_provider_index(split_provider_index);
  produce_split->set_finished(finished);
  return Apply(update);
}

Status DataServiceDispatcherImpl::ApplyWithoutJournaling(const Update& update)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return state_.Apply(update);
}

Status DataServiceDispatcherImpl::Apply(const Update& update)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (journal_writer_.has_value()) {
    TF_RETURN_IF_ERROR(journal_writer_.value()->Write(update));
  }
  return state_.Apply(update);
}

void DataServiceDispatcherImpl::MaintenanceThread() {
  int64_t next_check_micros = 0;
  while (true) {
    mutex_lock l(mu_);
    while (!cancelled_ && env_->NowMicros() < next_check_micros) {
      int64_t remaining_micros = next_check_micros - env_->NowMicros();
      maintenance_thread_cv_.wait_for(
          l, std::chrono::microseconds(remaining_micros));
    }
    if (cancelled_) {
      return;
    }
    {
      Status s = ReleaseMissingClients();
      if (!s.ok()) {
        LOG(WARNING) << "Error releasing missing clients: " << s;
      }
    }
    {
      Status s = GcOldIterations();
      if (!s.ok()) {
        LOG(WARNING) << "Error garbage collecting old iterations: " << s;
      }
    }
    // TODO(b/250921378): Once leases are supported, periodically handle failed
    // or missing workers by calling MaintainSnapshotWorkers().
    next_check_micros =
        env_->NowMicros() + (config_.job_gc_check_interval_ms() * 1000);
  }
}

Status DataServiceDispatcherImpl::ReleaseMissingClients()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t now = env_->NowMicros();
  for (const auto& client_id : state_.ListActiveClientIds()) {
    if (absl::FromUnixMicros(now) >
        latest_client_heartbeats_time_[client_id] +
            absl::Milliseconds(config_.client_timeout_ms())) {
      LOG(INFO) << "Releasing timed-out client with id " << client_id;
      Update update;
      ReleaseIterationClientUpdate* release_client =
          update.mutable_release_iteration_client();
      release_client->set_iteration_client_id(client_id);
      release_client->set_time_micros(now);
      TF_RETURN_IF_ERROR(Apply(update));
    }
  }
  return OkStatus();
}

void DataServiceDispatcherImpl::MaintainSnapshotWorkers()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  for (const auto& [ignore, snapshot_manager] : snapshots_) {
    snapshot_manager->UpdateStreams();
  }
  DetectMissingWorkers();
}

void DataServiceDispatcherImpl::DetectMissingWorkers()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t now = env_->NowMicros();
  for (auto it = latest_worker_heartbeats_time_.begin();
       it != latest_worker_heartbeats_time_.end();) {
    if (absl::FromUnixMicros(now) >
        it->second + absl::Milliseconds(config_.worker_timeout_ms())) {
      for (const auto& [ignore, snapshot_manager] : snapshots_) {
        snapshot_manager->HandleMissingWorker(it->first);
      }
      LOG(INFO) << "Lost worker " << it->first << " due to timeout";
      latest_worker_heartbeats_time_.erase(it++);
    } else {
      ++it;
    }
  }
}

Status DataServiceDispatcherImpl::GcOldIterations()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Iteration>> iterations =
      state_.ListIterations();
  int64_t now = env_->NowMicros();
  for (const auto& iteration : iterations) {
    if (iteration->finished || iteration->num_clients > 0 ||
        iteration->last_client_released_micros < 0 ||
        now < iteration->last_client_released_micros +
                  (config_.job_gc_timeout_ms() * 1000)) {
      continue;
    }
    Update update;
    update.mutable_garbage_collect_iteration()->set_iteration_id(
        iteration->iteration_id);
    TF_RETURN_IF_ERROR(state_.Apply(update));
    LOG(INFO) << "Garbage collected iteration " << iteration->DebugString();
  }
  return OkStatus();
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const std::string& dataset_id,
    std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  return GetDatasetDef(*dataset, dataset_def);
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const Dataset& dataset, std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return dataset_store_->Get(dataset.dataset_id, dataset_def);
}

DispatcherStateExport DataServiceDispatcherImpl::ExportState() const
    TF_LOCKS_EXCLUDED(mu_) {
  DispatcherStateExport dispatcher_state_export;
  *dispatcher_state_export.mutable_dispatcher_config() = config_;
  mutex_lock l(mu_);
  if (!started_) {
    return dispatcher_state_export;
  }

  std::vector<std::shared_ptr<const Worker>> workers = state_.ListWorkers();
  for (const auto& worker : workers) {
    dispatcher_state_export.add_worker_addresses(worker->address);
  }

  std::vector<std::shared_ptr<const Iteration>> iterations =
      state_.ListIterations();
  for (const auto& iteration : iterations) {
    DispatcherStateExport::Iteration* iteration_export =
        dispatcher_state_export.add_iterations();
    iteration_export->set_dataset_id(iteration->job->dataset_id);
    iteration_export->set_iteration_id(iteration->iteration_id);
    iteration_export->mutable_iteration_key()->set_name(
        iteration->iteration_key.name);
    iteration_export->mutable_iteration_key()->set_iteration(
        iteration->iteration_key.repetition);
    *iteration_export->mutable_processing_mode() =
        iteration->job->processing_mode;
    if (iteration->job->num_consumers) {
      iteration_export->set_num_consumers(*iteration->job->num_consumers);
    }
    iteration_export->set_num_clients(iteration->num_clients);
    iteration_export->set_finished(iteration->finished);
    iteration_export->set_garbage_collected(iteration->garbage_collected);
  }
  return dispatcher_state_export;
}

}  // namespace data
}  // namespace tensorflow
