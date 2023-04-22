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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef PLATFORM_GOOGLE
#include "file/logging/log_lines.h"
#endif
#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {

namespace {
// The name of the journal directory inside the dispatcher's working directory.
// This name is load-bearing; do not change.
constexpr char kJournalDir[] = "tf_data_dispatcher_journal";
// The name of the datasets directory inside the dispatcher's working directory.
constexpr char kDatasetsDir[] = "datasets";

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

using Dataset = DispatcherState::Dataset;
using Worker = DispatcherState::Worker;
using NamedJobKey = DispatcherState::NamedJobKey;
using Job = DispatcherState::Job;
using Task = DispatcherState::Task;

std::string JournalDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kJournalDir);
}

std::string DatasetsDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kDatasetsDir);
}

std::string DatasetKey(int64 id, uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint);
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
  return Status::OK();
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
}  // namespace

DataServiceDispatcherImpl::DataServiceDispatcherImpl(
    const experimental::DispatcherConfig& config)
    : config_(config), env_(Env::Default()) {
  if (config_.work_dir().empty()) {
    dataset_store_ = absl::make_unique<MemoryDatasetStore>();
  } else {
    dataset_store_ = absl::make_unique<FileSystemDatasetStore>(
        DatasetsDir(config_.work_dir()));
  }
}

DataServiceDispatcherImpl::~DataServiceDispatcherImpl() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    job_gc_thread_cv_.notify_all();
  }
  job_gc_thread_.reset();
}

Status DataServiceDispatcherImpl::Start() {
  mutex_lock l(mu_);
  if (config_.job_gc_timeout_ms() >= 0) {
    job_gc_thread_ = absl::WrapUnique(
        env_->StartThread({}, "job-gc-thread", [&] { JobGcThread(); }));
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
    return Status::OK();
  }
  journal_writer_ = absl::make_unique<FileJournalWriter>(
      env_, JournalDir(config_.work_dir()));
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
  for (const auto& job : state_.ListJobs()) {
    if (job->processing_mode == ProcessingMode::DISTRIBUTED_EPOCH) {
      TF_RETURN_IF_ERROR(
          RestoreSplitProviders(*job, split_providers_[job->job_id]));
    }
  }
  // Initialize the journal writer in `Start` so that we fail fast in case it
  // can't be initialized.
  TF_RETURN_IF_ERROR(journal_writer_.value()->EnsureInitialized());
  started_ = true;
  return Status::OK();
}

Status DataServiceDispatcherImpl::RestoreSplitProviders(
    const Job& job, std::vector<std::unique_ptr<SplitProvider>>& restored)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  const std::vector<int64>& indices =
      job.distributed_epoch_state.value().indices;
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(MakeSplitProviders(job.dataset_id, split_providers));
  for (int provider_index = 0; provider_index < indices.size();
       ++provider_index) {
    int index = indices[provider_index];
    VLOG(1) << "Restoring split provider " << provider_index << " for job "
            << job.job_id << " to index " << index;
    Tensor unused_tensor;
    bool unused_end_of_splits;
    for (int i = 0; i < index; ++i) {
      TF_RETURN_IF_ERROR(split_providers[provider_index]->GetNext(
          &unused_tensor, &unused_end_of_splits));
    }
  }
  restored = std::move(split_providers);
  return Status::OK();
}

Status DataServiceDispatcherImpl::FindTasksToDelete(
    const absl::flat_hash_set<int64>& current_tasks,
    const std::vector<std::shared_ptr<const Task>> assigned_tasks,
    WorkerHeartbeatResponse* response) {
  absl::flat_hash_set<int64> assigned_ids;
  for (const auto& assigned : assigned_tasks) {
    assigned_ids.insert(assigned->task_id);
  }
  for (int64 current_task : current_tasks) {
    if (!assigned_ids.contains(current_task)) {
      response->add_tasks_to_delete(current_task);
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::FindNewTasks(
    const std::string& worker_address,
    const absl::flat_hash_set<int64>& current_tasks,
    std::vector<std::shared_ptr<const Task>>& assigned_tasks,
    WorkerHeartbeatResponse* response) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // Check for round-robin jobs that had tasks on the worker removed. Now that
  // the worker is back, we create a new pending task for the worker.
  absl::flat_hash_set<int64> assigned_job_ids;
  for (const auto& task : assigned_tasks) {
    assigned_job_ids.insert(task->job->job_id);
  }
  for (const auto& job : state_.ListJobs()) {
    if (!assigned_job_ids.contains(job->job_id) && job->IsRoundRobin() &&
        !job->finished) {
      VLOG(1) << "Creating pending task for reconnected worker "
              << worker_address;
      TF_RETURN_IF_ERROR(CreatePendingTask(job, worker_address));
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
  return Status::OK();
}

Status DataServiceDispatcherImpl::WorkerHeartbeat(
    const WorkerHeartbeatRequest* request, WorkerHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(4) << "Received worker heartbeat request from worker "
          << request->worker_address();
  mutex_lock l(mu_);
  const std::string& worker_address = request->worker_address();
  // Assigned tasks from the perspective of the dispatcher.
  std::vector<std::shared_ptr<const Task>> assigned_tasks;
  Status s = state_.TasksForWorker(worker_address, assigned_tasks);
  if (!s.ok()) {
    if (!errors::IsNotFound(s)) {
      return s;
    }
    VLOG(1) << "Registering new worker at address " << worker_address;
    Update update;
    update.mutable_register_worker()->set_worker_address(worker_address);
    update.mutable_register_worker()->set_transfer_address(
        request->transfer_address());
    TF_RETURN_IF_ERROR(Apply(update));
    TF_RETURN_IF_ERROR(CreateTasksForWorker(worker_address));
    TF_RETURN_IF_ERROR(state_.TasksForWorker(worker_address, assigned_tasks));
  }
  absl::flat_hash_set<int64> current_tasks;
  current_tasks.insert(request->current_tasks().cbegin(),
                       request->current_tasks().cend());
  TF_RETURN_IF_ERROR(
      FindTasksToDelete(current_tasks, assigned_tasks, response));
  TF_RETURN_IF_ERROR(
      FindNewTasks(worker_address, current_tasks, assigned_tasks, response));

  VLOG(4) << "Finished worker heartbeat for worker at address "
          << request->worker_address();
  return Status::OK();
}

Status DataServiceDispatcherImpl::WorkerUpdate(
    const WorkerUpdateRequest* request, WorkerUpdateResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  for (auto& update : request->updates()) {
    int64 task_id = update.task_id();
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
      VLOG(3) << "Task " << task_id << " from job " << task->job->job_id
              << " completed";
    }
  }
  return Status::OK();
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
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetSplit(const GetSplitRequest* request,
                                           GetSplitResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  int64 job_id = request->job_id();
  int64 repetition = request->repetition();
  int64 provider_index = request->split_provider_index();
  VLOG(3) << "Received GetSplit request for job " << job_id << ", repetition "
          << repetition << ", split provider index " << provider_index;
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, job));
  if (!job->distributed_epoch_state.has_value()) {
    return errors::FailedPrecondition(
        "Cannot get split for job ", job_id,
        ", since it is not a distributed_epoch job.");
  }
  int64 current_repetition =
      job->distributed_epoch_state.value().repetitions[provider_index];
  if (repetition < current_repetition) {
    response->set_end_of_splits(true);
    VLOG(3) << "Returning end_of_splits since current reptition "
            << current_repetition << " is greater than the requested reptition "
            << repetition;
    return Status::OK();
  }
  SplitProvider* split_provider =
      split_providers_[job_id][provider_index].get();
  DCHECK(split_provider != nullptr);
  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider->GetNext(&split, &end_of_splits));
  TF_RETURN_IF_ERROR(RecordSplitProduced(
      job_id, repetition, request->split_provider_index(), end_of_splits));
  response->set_end_of_splits(end_of_splits);
  if (end_of_splits) {
    // Reset the split provider to prepare for the next repetition.
    TF_RETURN_IF_ERROR(split_providers_[job_id][provider_index]->Reset());
  } else {
    split.AsProtoTensorContent(response->mutable_split());
  }
  VLOG(3) << "Returning from GetSplit, end_of_splits=" << end_of_splits;
  return Status::OK();
}

Status DataServiceDispatcherImpl::MakeSplitProviders(
    int64 dataset_id,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  std::shared_ptr<const DatasetDef> dataset_def;
  TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, dataset_def));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> standalone_dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      params, dataset_def->graph(), &standalone_dataset));
  TF_RETURN_IF_ERROR(standalone_dataset->MakeSplitProviders(&split_providers));
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetVersion(const GetVersionRequest* request,
                                             GetVersionResponse* response) {
  response->set_version(kDataServiceVersion);
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrRegisterDataset(
    const GetOrRegisterDatasetRequest* request,
    GetOrRegisterDatasetResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  uint64 fingerprint;
  DatasetDef dataset_def = request->dataset();
  GraphDef* graph = dataset_def.mutable_graph();
  PrepareGraph(graph);
  TF_RETURN_IF_ERROR(HashGraph(*graph, &fingerprint));

  mutex_lock l(mu_);
#if defined(PLATFORM_GOOGLE)
  VLOG_LINES(4,
             absl::StrCat("Registering dataset graph: ", graph->DebugString()));
#else
  VLOG(4) << "Registering dataset graph: " << graph->DebugString();
#endif
  std::shared_ptr<const Dataset> dataset;
  Status s = state_.DatasetFromFingerprint(fingerprint, dataset);
  if (s.ok()) {
    int64 id = dataset->dataset_id;
    VLOG(3) << "Received duplicate RegisterDataset request with fingerprint "
            << fingerprint << ". Returning id " << id;
    response->set_dataset_id(id);
    return Status::OK();
  } else if (!errors::IsNotFound(s)) {
    return s;
  }

  int64 id;
  TF_RETURN_IF_ERROR(RegisterDataset(fingerprint, dataset_def, id));
  if (!request->element_spec().empty()) {
    TF_RETURN_IF_ERROR(SetElementSpec(id, request->element_spec()));
  }

  response->set_dataset_id(id);
  VLOG(3) << "Registered new dataset with id " << id;
  return Status::OK();
}

Status DataServiceDispatcherImpl::RegisterDataset(uint64 fingerprint,
                                                  const DatasetDef& dataset,
                                                  int64& dataset_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  dataset_id = state_.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(fingerprint);
  TF_RETURN_IF_ERROR(
      dataset_store_->Put(DatasetKey(dataset_id, fingerprint), dataset));
  return Apply(update);
}

Status DataServiceDispatcherImpl::SetElementSpec(
    int64 dataset_id, const std::string& element_spec)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Update update;
  SetElementSpecUpdate* set_element_spec = update.mutable_set_element_spec();
  set_element_spec->set_dataset_id(dataset_id);
  set_element_spec->set_element_spec(element_spec);
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetElementSpec(
    const GetElementSpecRequest* request, GetElementSpecResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(4) << "Read the element spec.";
  int64 dataset_id = request->dataset_id();

  std::string element_spec;
  TF_RETURN_IF_ERROR(state_.GetElementSpec(dataset_id, element_spec));
  VLOG(3) << "Get the `element_spec` for registered dataset with dataset id: "
          << dataset_id << ".";
  *response->mutable_element_spec() = element_spec;
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateJob(
    const GetOrCreateJobRequest* request, GetOrCreateJobResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(3) << "GetOrCreateJob(" << request->DebugString() << ")";
  absl::optional<NamedJobKey> key;
  if (request->has_job_key()) {
    key.emplace(request->job_key().job_name(),
                request->job_key().job_name_index());
  }
  ProcessingMode requested_processing_mode =
      ProcessingMode(request->processing_mode());
  std::shared_ptr<const Job> job;
  std::vector<std::shared_ptr<const Task>> tasks;
  {
    mutex_lock l(mu_);
    if (key.has_value()) {
      Status s = state_.NamedJobByKey(key.value(), job);
      if (s.ok()) {
        TF_RETURN_IF_ERROR(ValidateMatchingJob(job, requested_processing_mode,
                                               request->dataset_id()));
        // If the matching job was already garbage-collected, we fall through to
        // re-create the job.
        if (!job->garbage_collected) {
          int64 job_client_id;
          TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
          response->set_job_client_id(job_client_id);
          VLOG(3) << "Found existing job for name=" << key.value().name
                  << ", index=" << key.value().index
                  << ". job_id: " << job->job_id;
          return Status::OK();
        }
      } else if (!errors::IsNotFound(s)) {
        return s;
      }
    }
    absl::optional<int64> num_consumers;
    if (request->optional_num_consumers_case() ==
        GetOrCreateJobRequest::kNumConsumers) {
      num_consumers = request->num_consumers();
    }
    TF_RETURN_IF_ERROR(CreateJob(request->dataset_id(),
                                 requested_processing_mode, key, num_consumers,
                                 job));
    int64 job_client_id;
    TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
    response->set_job_client_id(job_client_id);
    TF_RETURN_IF_ERROR(CreateTasksForJob(job, tasks));
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));
  VLOG(3) << "Created job " << job->job_id << " for CreateJob("
          << request->DebugString() << ")";
  return Status::OK();
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
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(s);
    auto& remover_ref = remove_task_requests_[task->task_id];
    if (remover_ref == nullptr) {
      if (!task->job->IsRoundRobin()) {
        return errors::FailedPrecondition(
            "MaybeRemoveTask called on a non-round-robin task.");
      }
      remover_ref =
          std::make_shared<TaskRemover>(task->job->num_consumers.value());
    }
    remover = remover_ref;
  }
  bool removed =
      remover->RequestRemoval(request->consumer_index(), request->round());
  response->set_removed(removed);
  if (!removed) {
    VLOG(1) << "Failed to remove task " << task->task_id;
    return Status::OK();
  }
  mutex_lock l(mu_);
  if (!task->removed) {
    Update update;
    RemoveTaskUpdate* remove_task = update.mutable_remove_task();
    remove_task->set_task_id(request->task_id());
    TF_RETURN_IF_ERROR(Apply(update));
  }
  VLOG(1) << "Task " << task->task_id << " successfully removed";
  return Status::OK();
}

Status DataServiceDispatcherImpl::ReleaseJobClient(
    const ReleaseJobClientRequest* request,
    ReleaseJobClientResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  int64 job_client_id = request->job_client_id();
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobForJobClientId(job_client_id, job));
  Update update;
  ReleaseJobClientUpdate* release_job_client =
      update.mutable_release_job_client();
  release_job_client->set_job_client_id(job_client_id);
  release_job_client->set_time_micros(env_->NowMicros());
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

// Validates that the job matches the given processing_mode and dataset_id.
Status DataServiceDispatcherImpl::ValidateMatchingJob(
    std::shared_ptr<const Job> job, ProcessingMode processing_mode,
    int64 dataset_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  DCHECK(job->named_job_key.has_value());
  std::string job_name = job->named_job_key->name;
  if (job->processing_mode != processing_mode) {
    std::string requested = ProcessingModeToString(processing_mode);
    std::string actual = ProcessingModeToString(job->processing_mode);
    return errors::FailedPrecondition(
        "Tried to create a job with name ", job_name, " and processing_mode <",
        requested,
        "> but there is already an existing job with that name using "
        "processing mode <",
        actual, ">");
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateJob(
    int64 dataset_id, ProcessingMode processing_mode,
    absl::optional<NamedJobKey> named_job_key,
    absl::optional<int64> num_consumers, std::shared_ptr<const Job>& job)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  switch (processing_mode) {
    case ProcessingMode::PARALLEL_EPOCHS:
    case ProcessingMode::DISTRIBUTED_EPOCH:
      break;
    default:
      return errors::Internal(
          absl::StrCat("ProcessingMode ", processing_mode, " not recognized"));
  }
  int64 job_id = state_.NextAvailableJobId();
  int64 num_split_providers = 0;
  if (processing_mode == ProcessingMode::DISTRIBUTED_EPOCH) {
    TF_RETURN_IF_ERROR(
        MakeSplitProviders(dataset_id, split_providers_[job_id]));
    num_split_providers = split_providers_[job_id].size();
  }
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef(processing_mode));
  create_job->set_num_split_providers(num_split_providers);
  if (named_job_key.has_value()) {
    NamedJobKeyDef* key = create_job->mutable_named_job_key();
    key->set_name(named_job_key->name);
    key->set_index(named_job_key->index);
  }
  if (num_consumers.has_value()) {
    create_job->set_num_consumers(num_consumers.value());
  }
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, job));
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTasksForWorker(
    const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Job>> jobs = state_.ListJobs();
  for (const auto& job : jobs) {
    if (job->finished) {
      continue;
    }
    if (job->num_consumers.has_value()) {
      TF_RETURN_IF_ERROR(CreatePendingTask(job, worker_address));
      continue;
    }
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(job, worker_address, task));
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::AcquireJobClientId(
    const std::shared_ptr<const Job>& job, int64& job_client_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  job_client_id = state_.NextAvailableJobClientId();
  Update update;
  AcquireJobClientUpdate* acquire_job_client =
      update.mutable_acquire_job_client();
  acquire_job_client->set_job_client_id(job_client_id);
  acquire_job_client->set_job_id(job->job_id);
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTasksForJob(
    std::shared_ptr<const Job> job,
    std::vector<std::shared_ptr<const Task>>& tasks)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Worker>> workers = state_.ListWorkers();
  tasks.clear();
  tasks.reserve(workers.size());
  for (const auto& worker : workers) {
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(job, worker->address, task));
    tasks.push_back(task);
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreatePendingTask(
    std::shared_ptr<const Job> job, const std::string& worker_address)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64 task_id = state_.NextAvailableTaskId();
  Update update;
  CreatePendingTaskUpdate* create_task = update.mutable_create_pending_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job->job_id);
  create_task->set_worker_address(worker_address);
  create_task->set_starting_round(round_robin_rounds_[job->job_id] + 1);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  create_task->set_transfer_address(worker->transfer_address);
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTask(std::shared_ptr<const Job> job,
                                             const std::string& worker_address,
                                             std::shared_ptr<const Task>& task)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64 task_id = state_.NextAvailableTaskId();
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job->job_id);
  create_task->set_worker_address(worker_address);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  create_task->set_transfer_address(worker->transfer_address);
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, task));
  return Status::OK();
}

Status DataServiceDispatcherImpl::AssignTasks(
    std::vector<std::shared_ptr<const Task>> tasks) TF_LOCKS_EXCLUDED(mu_) {
  for (const auto& task : tasks) {
    TF_RETURN_IF_ERROR(AssignTask(task));
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateWorkerStub(
    const std::string& worker_address, WorkerService::Stub*& out_stub)
    TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(mu_);
    auto it = worker_stubs_.find(worker_address);
    if (it != worker_stubs_.end()) {
      out_stub = it->second.get();
      return Status::OK();
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
  return Status::OK();
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
    return grpc_util::WrapError(
        absl::StrCat("Failed to submit task to worker ", task->worker_address),
        s);
  }
  VLOG(2) << "Finished assigning task " << task->task_id << " to worker "
          << task->worker_address;
  return Status::OK();
}

Status DataServiceDispatcherImpl::ClientHeartbeat(
    const ClientHeartbeatRequest* request, ClientHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(4) << "Received heartbeat from client id " << request->job_client_id();
  std::shared_ptr<const Job> job;
  Status s = state_.JobForJobClientId(request->job_client_id(), job);
  if (errors::IsNotFound(s) && !config_.fault_tolerant_mode()) {
    return errors::NotFound(
        "Unknown job client id ", request->job_client_id(),
        ". The dispatcher is not configured to be fault tolerant, so this "
        "could be caused by a dispatcher restart.");
  }
  TF_RETURN_IF_ERROR(s);
  if (job->garbage_collected) {
    return errors::FailedPrecondition(
        "The requested job has been garbage collected due to inactivity. "
        "Consider configuring the dispatcher with a higher "
        "`job_gc_timeout_ms`.");
  }
  if (request->optional_current_round_case() ==
      ClientHeartbeatRequest::kCurrentRound) {
    round_robin_rounds_[request->job_client_id()] =
        std::max(round_robin_rounds_[request->job_client_id()],
                 request->current_round());
  }
  if (!job->pending_tasks.empty()) {
    const auto& task = job->pending_tasks.front();
    Update update;
    ClientHeartbeatUpdate* client_heartbeat = update.mutable_client_heartbeat();
    bool apply_update = false;
    client_heartbeat->set_job_client_id(request->job_client_id());
    absl::optional<int64> blocked_round;
    if (request->optional_blocked_round_case() ==
        ClientHeartbeatRequest::kBlockedRound) {
      blocked_round = request->blocked_round();
    }
    VLOG(1) << "Handling pending task in job client heartbeat. job_client_id: "
            << request->job_client_id()
            << ". current_round: " << request->current_round()
            << ". blocked_round: " << blocked_round.value_or(-1)
            << ". target_round: " << task.target_round;
    if (request->current_round() >= task.target_round) {
      TaskRejected* rejected = client_heartbeat->mutable_task_rejected();
      // Exponentially try later and later rounds until consumers all agree.
      int64 round_offset = 2;
      for (int i = 0; i < task.failures; ++i) {
        round_offset *= 2;
      }
      rejected->set_new_target_round(
          round_robin_rounds_[request->job_client_id()] + round_offset);
      apply_update = true;
    }
    if (blocked_round.has_value() &&
        blocked_round.value() <= task.target_round &&
        !task.ready_consumers.contains(request->job_client_id())) {
      client_heartbeat->set_task_accepted(true);
      apply_update = true;
    }
    if (apply_update) {
      TF_RETURN_IF_ERROR(Apply(update));
    }
  }
  if (!job->pending_tasks.empty()) {
    response->set_block_round(job->pending_tasks.front().target_round);
  }

  std::vector<std::shared_ptr<const Task>> tasks;
  TF_RETURN_IF_ERROR(state_.TasksForJob(job->job_id, tasks));
  for (const auto& task : tasks) {
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task->worker_address);
    task_info->set_transfer_address(task->transfer_address);
    task_info->set_task_id(task->task_id);
    task_info->set_job_id(job->job_id);
    task_info->set_starting_round(task->starting_round);
  }
  response->set_job_finished(job->finished);
  VLOG(4) << "Found " << response->task_info_size()
          << " tasks for job client id " << request->job_client_id();
  return Status::OK();
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
  return Status::OK();
}

Status DataServiceDispatcherImpl::PopulateTaskDef(
    std::shared_ptr<const Task> task, TaskDef* task_def) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  task_def->set_dataset_id(task->job->dataset_id);
  task_def->set_job_id(task->job->job_id);
  task_def->set_worker_address(task->worker_address);
  task_def->set_task_id(task->task_id);
  task_def->set_processing_mode(ProcessingModeDef(task->job->processing_mode));
  if (task->job->distributed_epoch_state.has_value()) {
    task_def->set_num_split_providers(
        task->job->distributed_epoch_state.value().indices.size());
  }
  if (task->job->num_consumers.has_value()) {
    task_def->set_num_consumers(task->job->num_consumers.value());
  }
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(task->job->dataset_id, dataset));
  std::string dataset_key =
      DatasetKey(dataset->dataset_id, dataset->fingerprint);
  if (config_.work_dir().empty()) {
    std::shared_ptr<const DatasetDef> dataset_def;
    TF_RETURN_IF_ERROR(dataset_store_->Get(dataset_key, dataset_def));
    *task_def->mutable_dataset_def() = *dataset_def;
  } else {
    std::string path =
        io::JoinPath(DatasetsDir(config_.work_dir()), dataset_key);
    task_def->set_path(path);
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CheckStarted() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (!started_) {
    return errors::Unavailable("Dispatcher has not started yet.");
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::RecordSplitProduced(
    int64 job_id, int64 repetition, int64 split_provider_index, bool finished)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Update update;
  ProduceSplitUpdate* produce_split = update.mutable_produce_split();
  produce_split->set_job_id(job_id);
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

void DataServiceDispatcherImpl::JobGcThread() {
  int64 next_check_micros = 0;
  while (true) {
    mutex_lock l(mu_);
    while (!cancelled_ && env_->NowMicros() < next_check_micros) {
      int64 remaining_micros = next_check_micros - env_->NowMicros();
      job_gc_thread_cv_.wait_for(l,
                                 std::chrono::microseconds(remaining_micros));
    }
    if (cancelled_) {
      return;
    }
    Status s = GcOldJobs();
    if (!s.ok()) {
      LOG(WARNING) << "Error garbage collecting old jobs: " << s;
    }
    next_check_micros =
        env_->NowMicros() + (config_.job_gc_check_interval_ms() * 1000);
  }
}

Status DataServiceDispatcherImpl::GcOldJobs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Job>> jobs = state_.ListJobs();
  int64 now = env_->NowMicros();
  for (const auto& job : jobs) {
    if (job->finished || job->num_clients > 0 ||
        job->last_client_released_micros < 0 ||
        now < job->last_client_released_micros +
                  (config_.job_gc_timeout_ms() * 1000)) {
      continue;
    }
    Update update;
    update.mutable_garbage_collect_job()->set_job_id(job->job_id);
    TF_RETURN_IF_ERROR(state_.Apply(update));
    LOG(INFO) << "Garbage collected job " << job->DebugString();
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    int64 dataset_id, std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  return GetDatasetDef(*dataset, dataset_def);
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const Dataset& dataset, std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string key = DatasetKey(dataset.dataset_id, dataset.fingerprint);
  return dataset_store_->Get(key, dataset_def);
}

}  // namespace data
}  // namespace tensorflow
