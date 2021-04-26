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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// A class encapsulating the journaled state of the dispatcher. All state
// modifications must be done via `Apply`. This helps to ensure that
// replaying the journal will allow us to restore the exact same state.
//
// The following usage pattern will keep the journal in sync with the state of
// the dispatcher:
// {
//   mutex_lock l(mu_);
//   Update update = ...  // create an update
//   dispatcher_state.Apply(update);
//   journal_writer.write(Update);
//   // Unlock mu_
// }
//
// The division of functionality between DispatcherImpl and DispatcherState is
// as follows:
//   - DispatcherImpl is responsible for handling RPC requests, reading from
//     DispatcherState, and deciding what updates to apply to DispatcherState.
//     DispatcherImpl handles all synchronization.
//   - DispatcherState is responsible for making the state changes requested by
//     DispatcherImpl and for providing DispatcherImpl with read-only access to
//     the state.
//
// DispatcherState is thread-compatible but not thread-safe.
class DispatcherState {
 public:
  DispatcherState();
  DispatcherState(const DispatcherState&) = delete;
  DispatcherState& operator=(const DispatcherState&) = delete;

  // Applies the given update to the dispatcher's state.
  Status Apply(const Update& update);

  // A dataset registered with the dispatcher.
  struct Dataset {
    explicit Dataset(int64 dataset_id, int64 fingerprint)
        : dataset_id(dataset_id), fingerprint(fingerprint) {}

    const int64 dataset_id;
    const int64 fingerprint;
  };

  // A worker registered with the dispatcher.
  struct Worker {
    explicit Worker(const std::string& address,
                    const std::string& transfer_address)
        : address(address), transfer_address(transfer_address) {}

    const std::string address;
    const std::string transfer_address;
  };

  // A key for identifying a named job. The key contains a user-specified name,
  // as well as an index describing which iteration of the job we are on.
  struct NamedJobKey {
    explicit NamedJobKey(absl::string_view name, int64 index)
        : name(name), index(index) {}

    friend bool operator==(const NamedJobKey& lhs, const NamedJobKey& rhs) {
      return lhs.name == rhs.name && lhs.index == rhs.index;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NamedJobKey& k) {
      return H::combine(std::move(h), k.name, k.index);
    }

    const std::string name;
    const int64 index;
  };

  struct DistributedEpochState {
    // The current repetition.
    int64 repetition = 0;
    // Number of splits produced so far by the current split provider.
    int64 split_provider_index = 0;
  };

  struct Task;

  struct PendingTask {
    explicit PendingTask(std::shared_ptr<Task> task, int64 target_round)
        : task(std::move(task)), target_round(target_round) {}

    std::shared_ptr<Task> task;
    // The target round where we want to insert the task.
    int64 target_round;
    // Which consumers have responded that they have successfully blocked
    // before the target round.
    absl::flat_hash_set<int64> ready_consumers;
    // How many times we have failed to add the task.
    int64 failures = 0;
  };

  // A job for processing a dataset.
  struct Job {
    explicit Job(int64 job_id, int64 dataset_id, ProcessingMode processing_mode,
                 absl::optional<NamedJobKey> named_job_key,
                 absl::optional<int64> num_consumers)
        : job_id(job_id),
          dataset_id(dataset_id),
          processing_mode(processing_mode),
          named_job_key(named_job_key),
          num_consumers(num_consumers) {
      if (processing_mode == ProcessingMode::DISTRIBUTED_EPOCH) {
        distributed_epoch_state = DistributedEpochState();
      }
    }

    bool IsRoundRobin() const { return num_consumers.has_value(); }

    const int64 job_id;
    const int64 dataset_id;
    const ProcessingMode processing_mode;
    const absl::optional<NamedJobKey> named_job_key;
    absl::optional<DistributedEpochState> distributed_epoch_state;
    absl::optional<int64> num_consumers;
    std::queue<PendingTask> pending_tasks;
    int64 num_clients = 0;
    int64 last_client_released_micros = -1;
    bool finished = false;
  };

  struct Task {
    explicit Task(int64 task_id, const std::shared_ptr<Job>& job,
                  const std::string& worker_address,
                  const std::string& transfer_address)
        : task_id(task_id),
          job(job),
          worker_address(worker_address),
          transfer_address(transfer_address) {}

    const int64 task_id;
    const std::shared_ptr<Job> job;
    const std::string worker_address;
    const std::string transfer_address;
    int64 starting_round = 0;
    bool finished = false;
    bool removed = false;
  };

  using TasksById = absl::flat_hash_map<int64, std::shared_ptr<Task>>;

  // Returns the next available dataset id.
  int64 NextAvailableDatasetId() const;
  // Gets a dataset by id. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromId(int64 id, std::shared_ptr<const Dataset>& dataset) const;
  // Gets a dataset by fingerprint. Returns NOT_FOUND if there is no such
  // dataset.
  Status DatasetFromFingerprint(uint64 fingerprint,
                                std::shared_ptr<const Dataset>& dataset) const;

  // Gets a worker by address. Returns NOT_FOUND if there is no such worker.
  Status WorkerFromAddress(const std::string& address,
                           std::shared_ptr<const Worker>& worker) const;
  // Lists all workers registered with the dispatcher.
  std::vector<std::shared_ptr<const Worker>> ListWorkers() const;

  // Returns the next available job id.
  int64 NextAvailableJobId() const;
  // Returns a list of all jobs.
  std::vector<std::shared_ptr<const Job>> ListJobs();
  // Gets a job by id. Returns NOT_FOUND if there is no such job.
  Status JobFromId(int64 id, std::shared_ptr<const Job>& job) const;
  // Gets a named job by key. Returns NOT_FOUND if there is no such job.
  Status NamedJobByKey(NamedJobKey key, std::shared_ptr<const Job>& job) const;

  // Returns the job associated with the given job client id. Returns NOT_FOUND
  // if the job_client_id is unknown or has been released.
  Status JobForJobClientId(int64 job_client_id,
                           std::shared_ptr<const Job>& job);
  // Returns the next available job client id.
  int64 NextAvailableJobClientId() const;

  // Returns the next available task id.
  int64 NextAvailableTaskId() const;
  // Gets a task by id. Returns NOT_FOUND if there is no such task.
  Status TaskFromId(int64 id, std::shared_ptr<const Task>& task) const;
  // Stores a list of all tasks for the given job to `tasks`. Returns NOT_FOUND
  // if there is no such job.
  Status TasksForJob(int64 job_id,
                     std::vector<std::shared_ptr<const Task>>& tasks) const;
  // Stores a list of all tasks for the given worker to `tasks`. Returns
  // NOT_FOUND if there is no such worker.
  Status TasksForWorker(const absl::string_view worker_address,
                        std::vector<std::shared_ptr<const Task>>& tasks) const;

 private:
  void RegisterDataset(const RegisterDatasetUpdate& register_dataset);
  void RegisterWorker(const RegisterWorkerUpdate& register_worker);
  void CreateJob(const CreateJobUpdate& create_job);
  void ProduceSplit(const ProduceSplitUpdate& produce_split);
  void AcquireJobClient(const AcquireJobClientUpdate& acquire_job_client);
  void ReleaseJobClient(const ReleaseJobClientUpdate& release_job_client);
  void RemoveTask(const RemoveTaskUpdate& remove_task);
  void CreatePendingTask(const CreatePendingTaskUpdate& create_pending_task);
  void ClientHeartbeat(const ClientHeartbeatUpdate& client_heartbeat);
  void CreateTask(const CreateTaskUpdate& create_task);
  void FinishTask(const FinishTaskUpdate& finish_task);

  int64 next_available_dataset_id_ = 1000;
  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64, std::shared_ptr<Dataset>> datasets_by_id_;
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>>
      datasets_by_fingerprint_;

  // Registered workers, keyed by address.
  absl::flat_hash_map<std::string, std::shared_ptr<Worker>> workers_;

  int64 next_available_job_id_ = 2000;
  // Jobs, keyed by job ids.
  absl::flat_hash_map<int64, std::shared_ptr<Job>> jobs_;
  // Named jobs, keyed by their names and indices. Not all jobs have names, so
  // this is a subset of the jobs stored in `jobs_`.
  absl::flat_hash_map<NamedJobKey, std::shared_ptr<Job>> named_jobs_;

  int64 next_available_job_client_id_ = 3000;
  // Mapping from client ids to the jobs they are associated with.
  absl::flat_hash_map<int64, std::shared_ptr<Job>> jobs_for_client_ids_;

  int64 next_available_task_id_ = 4000;
  // Tasks, keyed by task ids.
  TasksById tasks_;
  // List of tasks associated with each job.
  absl::flat_hash_map<int64, std::vector<std::shared_ptr<Task>>> tasks_by_job_;
  // Tasks, keyed by worker addresses. The values are a map from task id to
  // task.
  absl::flat_hash_map<std::string, TasksById> tasks_by_worker_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
