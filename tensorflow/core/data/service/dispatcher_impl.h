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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_IMPL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace data {

// A service which coordinates a pool of workers to serve dataset elements over
// RPC.
//
// Glossary:
// * Dataset: A definition of how to generate a potentially large collection of
//   elements.
// * Job: A coordinated phase of reading from the tf.data service. A job
//   produces some amount of data, and (potentially multiple) consumers consume
//   the data from the job until there is no data left. Each job has a
//   ProcessingModeDef which determines what data it produces.
// * Task: A job is broken into multiple tasks, which each represent
//   iterating over all of or part of the dataset. Workers process tasks.
class DataServiceDispatcherImpl {
 public:
  explicit DataServiceDispatcherImpl(
      const experimental::DispatcherConfig& config);

  ~DataServiceDispatcherImpl();

  // Starts the dispatcher. If there is a journal, this will read from the
  // journal to restore the dispatcher's state.
  Status Start();

  // See dispatcher.proto for API documentation.

  /// Worker-facing API.
  Status WorkerHeartbeat(const WorkerHeartbeatRequest* request,
                         WorkerHeartbeatResponse* response);
  Status WorkerUpdate(const WorkerUpdateRequest* request,
                      WorkerUpdateResponse* response);
  Status GetDatasetDef(const GetDatasetDefRequest* request,
                       GetDatasetDefResponse* response);
  Status GetSplit(const GetSplitRequest* request, GetSplitResponse* response);

  /// Client-facing API.
  Status GetOrRegisterDataset(const GetOrRegisterDatasetRequest* request,
                              GetOrRegisterDatasetResponse* response);
  Status CreateJob(const CreateJobRequest* request,
                   CreateJobResponse* response);
  Status GetOrCreateJob(const GetOrCreateJobRequest* request,
                        GetOrCreateJobResponse* response);
  Status ReleaseJobClient(const ReleaseJobClientRequest* request,
                          ReleaseJobClientResponse* response);
  Status GetTasks(const GetTasksRequest* request, GetTasksResponse* response);
  Status GetWorkers(const GetWorkersRequest* request,
                    GetWorkersResponse* response);

 private:
  // Restores a `SplitProvider` from the state in `job` and stores it in
  // `restored`.
  Status RestoreSplitProvider(const DispatcherState::Job& job,
                              std::unique_ptr<SplitProvider>& restored)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Makes a split provider for the specified `dataset_id`, and stores it in
  // `split_provider`.
  Status MakeSplitProvider(int64 dataset_id,
                           std::unique_ptr<SplitProvider>& split_provider)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Registers a dataset with the given fingerprint, storing the new dataset's
  // id in `dataset_id`.
  Status RegisterDataset(uint64 fingerprint, const DatasetDef& dataset,
                         int64& dataset_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Gets a worker's stub from `worker_stubs_`, or if none exists, creates a
  // stub and stores it in `worker_stubs_`. A borrowed pointer to the stub is
  // stored in `out_stub`.
  Status GetOrCreateWorkerStub(const std::string& worker_address,
                               WorkerService::Stub*& out_stub)
      TF_LOCKS_EXCLUDED(mu_);
  // Creates a job and stores it in `job`. This method updates the
  // dispatcher state with the new job, but does not assign tasks to workers.
  Status CreateJob(int64 dataset_id, ProcessingMode processing_mode,
                   absl::optional<DispatcherState::NamedJobKey> named_job_key,
                   std::shared_ptr<const DispatcherState::Job>& job)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates tasks for the specified worker, one task for every unfinished job.
  Status CreateTasksForWorker(const std::string& worker_address);
  // Acquires a job client id to read from the given job and sets
  // `job_client_id`.
  Status AcquireJobClientId(
      const std::shared_ptr<const DispatcherState::Job>& job,
      int64& job_client_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates one task for each worker, for the given job. The created tasks are
  // stored in `tasks`. This method only updates dispatcher metadata with the
  // new tasks, but doesn't assign the tasks to the workers.
  Status CreateTasksForJob(
      std::shared_ptr<const DispatcherState::Job> job,
      std::vector<std::shared_ptr<const DispatcherState::Task>>& tasks)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new task for a job, storing the created task in `task`.
  Status CreateTask(std::shared_ptr<const DispatcherState::Job> job,
                    const std::string& worker_address,
                    std::shared_ptr<const DispatcherState::Task>& task);
  // Assigns the list of tasks to the workers indicated by their
  // `worker_address` fields.
  Status AssignTasks(
      std::vector<std::shared_ptr<const DispatcherState::Task>> tasks)
      TF_LOCKS_EXCLUDED(mu_);
  // Assigns a task to the worker indicated by its `worker_address` field.
  Status AssignTask(std::shared_ptr<const DispatcherState::Task> task)
      TF_LOCKS_EXCLUDED(mu_);
  // Validates that an existing job matches the given processing_mode and
  // dataset_id, returning an error status describing any difference.
  Status ValidateMatchingJob(std::shared_ptr<const DispatcherState::Job> job,
                             ProcessingMode processing_mode, int64 dataset_id)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Checks that the dispatcher has started, returning UNAVAILABLE if it hasn't.
  Status CheckStarted() TF_LOCKS_EXCLUDED(mu_);
  // Records that a split was produced by a call to `GetSplit`.
  Status RecordSplitProduced(int64 job_id, int64 repetition, bool finished)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Applies a state update, updating both the journal and the in-memory state.
  Status Apply(const Update& update) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Applies a state update, but doesn't update the journal. Only meant to be
  // used when recovering state when the dispatcher starts.
  Status ApplyWithoutJournaling(const Update& update)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // A thread which periodically checks for jobs to clean up.
  void JobGcThread();
  // Scans for old jobs and marks them as finished.
  Status GcOldJobs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Gets a `DatasetDef` from `dataset_store_` for the given dataset id, and
  // stores it in `dataset_def`.
  Status GetDatasetDef(int64 dataset_id,
                       std::shared_ptr<const DatasetDef>& dataset_def)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Gets a `DatasetDef` from `dataset_store_` for the given dataset, and
  // stores it in `dataset_def`.
  Status GetDatasetDef(const DispatcherState::Dataset& dataset,
                       std::shared_ptr<const DatasetDef>& dataset_def)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const experimental::DispatcherConfig& config_;
  Env* env_;

  mutex mu_;
  bool started_ TF_GUARDED_BY(mu_) = false;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;

  // Cached worker stubs for communicating with workers.
  absl::flat_hash_map<std::string, std::unique_ptr<WorkerService::Stub>>
      worker_stubs_ TF_GUARDED_BY(mu_);
  // Store of dataset definitions.
  std::unique_ptr<DatasetStore> dataset_store_ TF_GUARDED_BY(mu_);
  // Mapping from job id to `SplitProvider`s for jobs with processing mode
  // DISTRIBUTED_EPOCH.
  absl::flat_hash_map<int64, std::unique_ptr<SplitProvider>> split_providers_
      TF_GUARDED_BY(mu_);

  absl::optional<std::unique_ptr<JournalWriter>> journal_writer_
      TF_GUARDED_BY(mu_);
  DispatcherState state_ TF_GUARDED_BY(mu_);
  // Condition variable for waking up the job gc thread.
  condition_variable job_gc_thread_cv_;
  std::unique_ptr<Thread> job_gc_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceDispatcherImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_IMPL_H_
