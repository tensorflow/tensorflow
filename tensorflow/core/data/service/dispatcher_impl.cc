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
#include <tuple>
#include <utility>

#ifdef PLATFORM_GOOGLE
#include "file/logging/log_lines.h"
#endif
#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/data/experimental/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {

namespace {
// The name of the journal directory inside the dispatcher's working directory.
constexpr char kJournalDir[] = "tf_data_dispatcher_journal";
// The name of the datasets directory inside the dispatcher's working directory.
constexpr char kDatasetsDir[] = "datasets";

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
}  // namespace

DataServiceDispatcherImpl::DataServiceDispatcherImpl(
    const experimental::DispatcherConfig& config)
    : config_(config) {
  if (config_.work_dir().empty()) {
    dataset_store_ = absl::make_unique<MemoryDatasetStore>();
  } else {
    dataset_store_ = absl::make_unique<FileSystemDatasetStore>(
        DatasetsDir(config_.work_dir()));
  }
}

Status DataServiceDispatcherImpl::Start() {
  mutex_lock l(mu_);
  if (config_.work_dir().empty()) {
    if (config_.fault_tolerant_mode()) {
      return errors::InvalidArgument(
          "fault_tolerant_mode is True, but no work_dir is configured.");
    }
  } else {
    TF_RETURN_IF_ERROR(
        Env::Default()->RecursivelyCreateDir(DatasetsDir(config_.work_dir())));
  }
  if (!config_.fault_tolerant_mode()) {
    LOG(INFO) << "Running with fault_tolerant_mode=False. The dispatcher will "
                 "not be able to recover its state on restart.";
    return Status::OK();
  }
  journal_writer_ = absl::make_unique<FileJournalWriter>(
      Env::Default(), JournalDir(config_.work_dir()));
  LOG(INFO) << "Restoring dispatcher state from journal in "
            << JournalDir(config_.work_dir());
  Update update;
  bool end_of_journal = false;
  FileJournalReader reader(Env::Default(), JournalDir(config_.work_dir()));
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
  // Initialize the journal writer in `Start` so that we fail fast in case it
  // can't be initialized.
  TF_RETURN_IF_ERROR(journal_writer_.value()->EnsureInitialized());
  return Status::OK();
}

Status DataServiceDispatcherImpl::RegisterWorker(
    const RegisterWorkerRequest* request, RegisterWorkerResponse* response) {
  VLOG(3) << "Received register worker request";
  mutex_lock l(mu_);
  std::string worker_address = request->worker_address();
  std::vector<std::shared_ptr<const Task>> tasks;
  Status s = state_.TasksForWorker(worker_address, tasks);
  if (errors::IsNotFound(s)) {
    Update update;
    update.mutable_register_worker()->set_worker_address(worker_address);
    TF_RETURN_IF_ERROR(Apply(update));
  } else if (!s.ok()) {
    return s;
  }

  absl::flat_hash_map<int64, std::shared_ptr<const Task>> tasks_by_job;
  for (const auto& task : tasks) {
    // Should never have multiple tasks on the same worker for the same job.
    auto& task_for_job = tasks_by_job[task->job_id];
    DCHECK(task_for_job == nullptr);
    task_for_job = task;
  }

  std::vector<std::shared_ptr<const Job>> jobs = state_.ListJobs();
  // Allocate tasks to the worker.
  for (const auto& job : jobs) {
    if (job->finished) {
      continue;
    }
    std::shared_ptr<const Task> task;
    auto it = tasks_by_job.find(job->job_id);
    if (it != tasks_by_job.end()) {
      task = it->second;
    } else {
      TF_RETURN_IF_ERROR(CreateTask(job, worker_address, task));
    }
    TaskDef* task_def = response->add_tasks();
    std::shared_ptr<const Dataset> dataset;
    TF_RETURN_IF_ERROR(state_.DatasetFromId(job->dataset_id, &dataset));
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
    task_def->set_dataset_id(job->dataset_id);
    task_def->set_job_id(job->job_id);
    task_def->set_task_id(task->task_id);
  }

  VLOG(1) << "Registered worker at address " << request->worker_address();
  return Status::OK();
}

Status DataServiceDispatcherImpl::WorkerUpdate(
    const WorkerUpdateRequest* request, WorkerUpdateResponse* response) {
  mutex_lock l(mu_);
  for (auto& update : request->updates()) {
    int64 task_id = update.task_id();
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, &task));
    if (update.completed()) {
      if (task->finished) {
        VLOG(1) << "Received completion update for already-finished task "
                << task->task_id << " on worker " << task->worker_address;
        continue;
      }
      Update update;
      update.mutable_finish_task()->set_task_id(task_id);
      TF_RETURN_IF_ERROR(Apply(update));
      VLOG(3) << "Task " << task_id << " from job " << task->job_id
              << " completed";
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const GetDatasetDefRequest* request, GetDatasetDefResponse* response) {
  mutex_lock l(mu_);
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(request->dataset_id(), &dataset));
  std::string key = DatasetKey(dataset->dataset_id, dataset->fingerprint);
  std::shared_ptr<const DatasetDef> dataset_def;
  TF_RETURN_IF_ERROR(dataset_store_->Get(key, dataset_def));
  *response->mutable_dataset_def() = *dataset_def;
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrRegisterDataset(
    const GetOrRegisterDatasetRequest* request,
    GetOrRegisterDatasetResponse* response) {
  uint64 fingerprint;
  const GraphDef& graph = request->dataset().graph();
  TF_RETURN_IF_ERROR(HashGraph(graph, &fingerprint));
  mutex_lock l(mu_);
#if defined(PLATFORM_GOOGLE)
  VLOG_LINES(4,
             absl::StrCat("Registering dataset graph: ", graph.DebugString()));
#else
  VLOG(4) << "Registering dataset graph: " << graph.DebugString();
#endif
  std::shared_ptr<const Dataset> dataset;
  Status s = state_.DatasetFromFingerprint(fingerprint, &dataset);
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
  TF_RETURN_IF_ERROR(RegisterDataset(fingerprint, request->dataset(), id));
  response->set_dataset_id(id);
  VLOG(3) << "Registered new dataset with id " << id;
  return Status::OK();
}

Status DataServiceDispatcherImpl::RegisterDataset(uint64 fingerprint,
                                                  const DatasetDef& dataset,
                                                  int64& dataset_id)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  dataset_id = state_.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(fingerprint);
  TF_RETURN_IF_ERROR(
      dataset_store_->Put(DatasetKey(dataset_id, fingerprint), dataset));
  return Apply(update);
}

Status DataServiceDispatcherImpl::CreateJob(const CreateJobRequest* request,
                                            CreateJobResponse* response) {
  VLOG(3) << "Received create job request for dataset id "
          << request->dataset_id();
  ProcessingMode processing_mode = ProcessingMode(request->processing_mode());
  std::shared_ptr<const Job> job;
  std::vector<std::shared_ptr<const Task>> tasks;
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CreateJob(request->dataset_id(), processing_mode,
                                 absl::optional<NamedJobKey>(), job));
    int64 job_client_id;
    TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
    response->set_job_client_id(job_client_id);
    TF_RETURN_IF_ERROR(CreateTasksForJob(job, tasks));
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));

  VLOG(3) << "Creating job " << job->job_id << " for dataset "
          << request->dataset_id();
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateJob(
    const GetOrCreateJobRequest* request, GetOrCreateJobResponse* response) {
  VLOG(3) << "Received get or create job request for dataset id "
          << request->dataset_id() << " with name " << request->job_name()
          << " and index " << request->job_name_index();
  NamedJobKey key(request->job_name(), request->job_name_index());
  ProcessingMode requested_processing_mode =
      ProcessingMode(request->processing_mode());
  std::shared_ptr<const Job> job;
  std::vector<std::shared_ptr<const Task>> tasks;
  {
    mutex_lock l(mu_);
    Status s = state_.NamedJobByKey(key, &job);
    if (s.ok()) {
      TF_RETURN_IF_ERROR(ValidateMatchingJob(job, requested_processing_mode,
                                             request->dataset_id()));
      int64 job_client_id;
      TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
      response->set_job_client_id(job_client_id);
      VLOG(3) << "Found existing job for name=" << key.name
              << ", index=" << key.index << ". job_id: " << job->job_id;
      return Status::OK();
    } else if (!errors::IsNotFound(s)) {
      return s;
    }
    TF_RETURN_IF_ERROR(
        CreateJob(request->dataset_id(), requested_processing_mode, key, job));
    int64 job_client_id;
    TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
    response->set_job_client_id(job_client_id);
    TF_RETURN_IF_ERROR(CreateTasksForJob(job, tasks));
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));
  VLOG(3) << "Created job " << job->job_id << " for dataset "
          << request->dataset_id() << " and name " << request->job_name();
  return Status::OK();
}

Status DataServiceDispatcherImpl::ReleaseJobClient(
    const ReleaseJobClientRequest* request,
    ReleaseJobClientResponse* response) {
  mutex_lock l(mu_);
  int64 job_client_id = request->job_client_id();
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobForJobClientId(job_client_id, job));
  Update update;
  ReleaseJobClientUpdate* release_job_client =
      update.mutable_release_job_client();
  release_job_client->set_job_client_id(job_client_id);
  release_job_client->set_time_micros(Env::Default()->NowMicros());
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

// Validates that the job matches the given processing_mode and dataset_id.
Status DataServiceDispatcherImpl::ValidateMatchingJob(
    std::shared_ptr<const Job> job, ProcessingMode processing_mode,
    int64 dataset_id) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  DCHECK(job->named_job_key.has_value());
  std::string job_name = job->named_job_key->name;
  if (job->processing_mode != processing_mode) {
    std::string requested = ProcessingModeToString(processing_mode);
    std::string actual = ProcessingModeToString(job->processing_mode);
    return errors::FailedPrecondition(
        "Found a job with name ", job_name, ", but the processing mode <",
        actual, "> doesn't match the requested processing mode <", requested,
        ">.");
  }
  if (job->dataset_id != dataset_id) {
    return errors::FailedPrecondition(
        "Found a job with name ", job_name, ", but the dataset id <",
        job->dataset_id, "> doesn't match the requested dataset id <",
        dataset_id, ">.");
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateJob(
    int64 dataset_id, ProcessingMode processing_mode,
    absl::optional<NamedJobKey> named_job_key, std::shared_ptr<const Job>& job)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  switch (processing_mode) {
    case ProcessingMode::PARALLEL_EPOCHS:
      break;
    case ProcessingMode::ONE_EPOCH:
      return errors::Unimplemented(
          "CreateJob only supports the PARALLEL_EPOCHS job mode. "
          "ONE_EPOCH is not currently supported.");
    default:
      return errors::Unimplemented("ProcessingMode ",
                                   ProcessingModeToString(processing_mode),
                                   " not recognized");
  }
  int64 job_id = state_.NextAvailableJobId();
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef(processing_mode));
  if (named_job_key.has_value()) {
    NamedJobKeyDef* key = create_job->mutable_named_job_key();
    key->set_name(named_job_key->name);
    key->set_index(named_job_key->index);
  }
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, &job));
  return Status::OK();
}

Status DataServiceDispatcherImpl::AcquireJobClientId(
    const std::shared_ptr<const Job>& job, int64& job_client_id)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

Status DataServiceDispatcherImpl::CreateTask(std::shared_ptr<const Job> job,
                                             const std::string& worker_address,
                                             std::shared_ptr<const Task>& task)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64 task_id = state_.NextAvailableTaskId();
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job->job_id);
  create_task->set_dataset_id(job->dataset_id);
  create_task->set_worker_address(worker_address);
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, &task));
  return Status::OK();
}

Status DataServiceDispatcherImpl::AssignTasks(
    std::vector<std::shared_ptr<const Task>> tasks) LOCKS_EXCLUDED(mu_) {
  for (const auto& task : tasks) {
    TF_RETURN_IF_ERROR(AssignTask(task));
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateWorkerStub(
    const std::string& worker_address, WorkerService::Stub*& out_stub)
    LOCKS_EXCLUDED(mu_) {
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
    LOCKS_EXCLUDED(mu_) {
  VLOG(2) << "Started assigning task " << task->task_id << " to worker "
          << task->worker_address;
  grpc::ClientContext client_ctx;
  ProcessTaskRequest req;
  TaskDef* task_def = req.mutable_task();
  task_def->set_dataset_id(task->dataset_id);
  {
    mutex_lock l(mu_);
    std::shared_ptr<const Dataset> dataset;
    TF_RETURN_IF_ERROR(state_.DatasetFromId(task->dataset_id, &dataset));
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
  }
  task_def->set_task_id(task->task_id);
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

Status DataServiceDispatcherImpl::GetTasks(const GetTasksRequest* request,
                                           GetTasksResponse* response) {
  mutex_lock l(mu_);
  VLOG(3) << "Looking up tasks for job client id " << request->job_client_id();
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobForJobClientId(request->job_client_id(), job));
  std::vector<std::shared_ptr<const Task>> tasks;
  TF_RETURN_IF_ERROR(state_.TasksForJob(job->job_id, &tasks));
  for (const auto& task : tasks) {
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task->worker_address);
    task_info->set_task_id(task->task_id);
    task_info->set_job_id(job->job_id);
  }
  response->set_job_finished(job->finished);
  VLOG(3) << "Found " << response->task_info_size()
          << " tasks for job client id " << request->job_client_id();
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetWorkers(const GetWorkersRequest* request,
                                             GetWorkersResponse* response) {
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

Status DataServiceDispatcherImpl::ApplyWithoutJournaling(const Update& update)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return state_.Apply(update);
}

Status DataServiceDispatcherImpl::Apply(const Update& update)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (journal_writer_.has_value()) {
    TF_RETURN_IF_ERROR(journal_writer_.value()->Write(update));
  }
  return state_.Apply(update);
}

}  // namespace data
}  // namespace tensorflow
