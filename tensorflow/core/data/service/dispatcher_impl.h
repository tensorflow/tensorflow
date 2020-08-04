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
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/data/experimental/service_config.pb.h"
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

  // Starts the dispatcher. If there is a journal, this will read from the
  // journal to restore the dispatcher's state.
  Status Start();

  // See dispatcher.proto for API documentation.

  /// Worker-facing API.
  Status RegisterWorker(const RegisterWorkerRequest* request,
                        RegisterWorkerResponse* response);
  Status WorkerUpdate(const WorkerUpdateRequest* request,
                      WorkerUpdateResponse* response);

  /// Client-facing API.
  Status GetOrRegisterDataset(const GetOrRegisterDatasetRequest* request,
                              GetOrRegisterDatasetResponse* response);
  Status CreateJob(const CreateJobRequest* request,
                   CreateJobResponse* response);
  Status GetOrCreateJob(const GetOrCreateJobRequest* request,
                        GetOrCreateJobResponse* response);
  Status GetTasks(const GetTasksRequest* request, GetTasksResponse* response);
  Status GetWorkers(const GetWorkersRequest* request,
                    GetWorkersResponse* response);

 private:
  struct Worker {
    Worker(int64 worker_id, const std::string& address)
        : worker_id(worker_id), address(address) {}

    const int64 worker_id;
    const std::string address;
    std::unique_ptr<WorkerService::Stub> stub;
  };

  // Registers a dataset with the given fingerprint, storing the new dataset's
  // id in `*dataset-id`.
  Status RegisterDataset(uint64 fingerprint, const DatasetDef& dataset,
                         int64* dataset_id) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Initializes a workers stub, if it hasn't been initialized already.
  Status EnsureWorkerStubInitialized(Worker* worker);
  // Creates a job and stores it in `*job`. This method updates the
  // dispatcher state with the new job, but does not assign tasks to workers.
  Status CreateJob(int64 dataset_id, ProcessingMode processing_mode,
                   absl::optional<DispatcherState::NamedJobKey> named_job_key,
                   std::shared_ptr<const DispatcherState::Job>* job)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates one task for each worker, for the given job. The created tasks are
  // stored in `*tasks`. This method only updates dispatcher metadata with the
  // new tasks, but doesn't assign the tasks to the workers.
  Status CreateTasksForJob(
      std::shared_ptr<const DispatcherState::Job> job,
      std::vector<std::shared_ptr<const DispatcherState::Task>>* tasks)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new task for a job, storing the created task in `*task`.
  Status CreateTask(std::shared_ptr<const DispatcherState::Job> job,
                    const std::string& worker_address,
                    std::shared_ptr<const DispatcherState::Task>* task);
  // Assigns the list of tasks to the workers indicated by their
  // `worker_address` fields.
  Status AssignTasks(
      std::vector<std::shared_ptr<const DispatcherState::Task>> tasks)
      LOCKS_EXCLUDED(mu_);
  // Assigns a task to the worker indicated by its `worker_address` field.
  Status AssignTask(std::shared_ptr<const DispatcherState::Task> task)
      LOCKS_EXCLUDED(mu_);
  // Validates that an existing job matches the given processing_mode and
  // dataset_id, returning an error status describing any difference.
  Status ValidateMatchingJob(std::shared_ptr<const DispatcherState::Job> job,
                             ProcessingMode processing_mode, int64 dataset_id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Applies a state update, updating both the journal and the in-memory state.
  Status Apply(const Update& update) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Applies a state update, but doesn't update the journal. Only meant to be
  // used when recovering state when the dispatcher starts.
  Status ApplyWithoutJournaling(const Update& update)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const experimental::DispatcherConfig& config_;

  mutex mu_;

  int64 next_worker_id_ TF_GUARDED_BY(mu_) = 0;
  int64 next_task_id_ TF_GUARDED_BY(mu_) = 0;

  // Registered workers, keyed by their addresses.
  absl::flat_hash_map<std::string, std::shared_ptr<Worker>> workers_
      TF_GUARDED_BY(mu_);

  absl::optional<std::unique_ptr<JournalWriter>> journal_writer_
      TF_GUARDED_BY(mu_);
  DispatcherState state_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceDispatcherImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_IMPL_H_
