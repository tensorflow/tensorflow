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

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/auto_shard_rewriter.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

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
  explicit DispatcherState(
      const experimental::DispatcherConfig& dispatcher_config);
  DispatcherState(const DispatcherState&) = delete;
  DispatcherState& operator=(const DispatcherState&) = delete;

  // Applies the given update to the dispatcher's state.
  Status Apply(const Update& update);

  // A dataset registered with the dispatcher.
  struct Dataset {
    explicit Dataset(int64_t dataset_id, int64_t fingerprint,
                     const std::string& name,
                     const DataServiceMetadata& metadata)
        : dataset_id(dataset_id),
          fingerprint(fingerprint),
          name(name),
          metadata(metadata) {}

    const int64_t dataset_id;
    const int64_t fingerprint;
    const std::string name;
    const DataServiceMetadata metadata;
  };

  // A worker registered with the dispatcher.
  struct Worker {
    explicit Worker(const RegisterWorkerUpdate& register_worker)
        : address(register_worker.worker_address()),
          transfer_address(register_worker.transfer_address()),
          tags(register_worker.worker_tags().begin(),
               register_worker.worker_tags().end()),
          uid(register_worker.worker_uid()) {}

    const std::string address;
    const std::string transfer_address;
    const std::vector<std::string> tags;
    const int64_t uid;
  };

  // A key for identifying a job. The key contains a job name,
  // as well as a iteration number describing which iteration of the job we are
  // on.
  struct JobKey {
    explicit JobKey(absl::string_view name, int64_t iteration)
        : name(name), iteration(iteration) {}

    friend bool operator==(const JobKey& lhs, const JobKey& rhs) {
      return lhs.name == rhs.name && lhs.iteration == rhs.iteration;
    }

    template <typename H>
    friend H AbslHashValue(H h, const JobKey& k) {
      return H::combine(std::move(h), k.name, k.iteration);
    }

    std::string DebugString() const {
      return absl::StrCat(name, "/", iteration);
    }

    const std::string name;
    const int64_t iteration;
  };

  struct DistributedEpochState {
    explicit DistributedEpochState(int64_t num_split_providers)
        : iterations(num_split_providers), indices(num_split_providers) {}

    // The current iteration for each split provider.
    std::vector<int64_t> iterations;
    // Number of splits produced so far by each split provider.
    std::vector<int64_t> indices;
  };

  struct Task;

  struct PendingTask {
    explicit PendingTask(std::shared_ptr<Task> task, int64_t target_round)
        : task(std::move(task)), target_round(target_round) {}

    std::shared_ptr<Task> task;
    // The target round where we want to insert the task.
    int64_t target_round;
    // Which consumers have responded that they have successfully blocked
    // before the target round.
    absl::flat_hash_set<int64_t> ready_consumers;
    // How many times we have failed to add the task.
    int64_t failures = 0;
  };

  // A job for processing a dataset.
  struct Job {
    explicit Job(int64_t job_id, int64_t dataset_id,
                 const ProcessingModeDef& processing_mode,
                 int64_t num_split_providers, JobKey job_key,
                 absl::optional<int64_t> num_consumers,
                 TargetWorkers target_workers)
        : job_id(job_id),
          dataset_id(dataset_id),
          processing_mode(processing_mode),
          job_key(job_key),
          num_consumers(num_consumers),
          target_workers(target_workers) {
      if (IsDynamicShard(processing_mode)) {
        distributed_epoch_state = DistributedEpochState(num_split_providers);
      }
    }

    bool IsRoundRobin() const { return num_consumers.has_value(); }

    std::string DebugString() const {
      return absl::StrCat(job_key.name, "_", job_key.iteration);
    }

    const int64_t job_id;
    const int64_t dataset_id;
    const ProcessingModeDef processing_mode;
    const JobKey job_key;
    absl::optional<DistributedEpochState> distributed_epoch_state;
    const absl::optional<int64_t> num_consumers;
    const TargetWorkers target_workers;
    std::queue<PendingTask> pending_tasks;
    int64_t num_clients = 0;
    int64_t last_client_released_micros = -1;
    bool finished = false;
    // Indicates whether the job was garbage collected.
    bool garbage_collected = false;
  };

  struct Task {
    template <class T>
    explicit Task(const T& create_task_update, const std::shared_ptr<Job>& job)
        : task_id(create_task_update.task_id()),
          job(job),
          worker_address(create_task_update.worker_address()),
          transfer_address(create_task_update.transfer_address()),
          worker_tags(create_task_update.worker_tags().begin(),
                      create_task_update.worker_tags().end()),
          worker_uid(create_task_update.worker_uid()) {}

    const int64_t task_id;
    const std::shared_ptr<Job> job;
    const std::string worker_address;
    const std::string transfer_address;
    const std::vector<std::string> worker_tags;
    const int64_t worker_uid;
    int64_t starting_round = 0;
    bool finished = false;
    bool removed = false;
  };

  using TasksById = absl::flat_hash_map<int64_t, std::shared_ptr<Task>>;

  // Returns the next available dataset id.
  int64_t NextAvailableDatasetId() const;
  // Gets a dataset by id. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromId(int64_t id,
                       std::shared_ptr<const Dataset>& dataset) const;
  // Gets a dataset by fingerprint. Returns NOT_FOUND if there is no such
  // dataset.
  Status DatasetFromFingerprint(uint64 fingerprint,
                                std::shared_ptr<const Dataset>& dataset) const;
  // Gets a dataset by name. Returns NOT_FOUND if there is no such dataset.
  Status DatasetFromName(const std::string& name,
                         std::shared_ptr<const Dataset>& dataset) const;
  // Gets a worker by address. Returns NOT_FOUND if there is no such worker.
  Status WorkerFromAddress(const std::string& address,
                           std::shared_ptr<const Worker>& worker) const;
  // Lists all workers registered with the dispatcher.
  std::vector<std::shared_ptr<const Worker>> ListWorkers() const;

  // Returns the next available job id.
  int64_t NextAvailableJobId() const;
  // Returns a list of all jobs.
  std::vector<std::shared_ptr<const Job>> ListJobs();
  // Gets a job by id. Returns NOT_FOUND if there is no such job.
  Status JobFromId(int64_t id, std::shared_ptr<const Job>& job) const;
  // Gets a job by key. Returns NOT_FOUND if there is no such job.
  Status JobByKey(JobKey key, std::shared_ptr<const Job>& job) const;

  // Returns the job associated with the given job client id. Returns NOT_FOUND
  // if the job_client_id is unknown or has been released.
  Status JobForJobClientId(int64_t job_client_id,
                           std::shared_ptr<const Job>& job);
  // Returns a list of all active client ids.
  std::vector<int64_t> ListActiveClientIds();
  // Returns the next available job client id.
  int64_t NextAvailableJobClientId() const;

  // Returns the next available task id.
  int64_t NextAvailableTaskId() const;
  // Gets a task by id. Returns NOT_FOUND if there is no such task.
  Status TaskFromId(int64_t id, std::shared_ptr<const Task>& task) const;
  // Stores a list of all tasks for the given job to `tasks`. Returns NOT_FOUND
  // if there is no such job.
  Status TasksForJob(int64_t job_id,
                     std::vector<std::shared_ptr<const Task>>& tasks) const;
  // Stores a list of all tasks for the given worker to `tasks`. Returns
  // NOT_FOUND if there is no such worker.
  Status TasksForWorker(const absl::string_view worker_address,
                        std::vector<std::shared_ptr<const Task>>& tasks) const;

  // If the dispatcher config explicitly specifies a list of workers, validates
  // `worker_address` is in the list.
  Status ValidateWorker(absl::string_view worker_address) const;

  // If the dispatcher config specifies worker addresses, `GetWorkerIndex`
  // returns the worker index according to the list. This is useful for
  // deterministically sharding a dataset among a fixed set of workers.
  StatusOr<int64_t> GetWorkerIndex(absl::string_view worker_address) const;

 private:
  void RegisterDataset(const RegisterDatasetUpdate& register_dataset);
  void RegisterWorker(const RegisterWorkerUpdate& register_worker);
  void CreateJob(const CreateJobUpdate& create_job);
  void ProduceSplit(const ProduceSplitUpdate& produce_split);
  void AcquireJobClient(const AcquireJobClientUpdate& acquire_job_client);
  void ReleaseJobClient(const ReleaseJobClientUpdate& release_job_client);
  void GarbageCollectJob(const GarbageCollectJobUpdate& garbage_collect_job);
  void RemoveTask(const RemoveTaskUpdate& remove_task);
  void CreatePendingTask(const CreatePendingTaskUpdate& create_pending_task);
  void ClientHeartbeat(const ClientHeartbeatUpdate& client_heartbeat);
  void CreateTask(const CreateTaskUpdate& create_task);
  void FinishTask(const FinishTaskUpdate& finish_task);

  int64_t next_available_dataset_id_ = 1000;
  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Dataset>> datasets_by_id_;
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>>
      datasets_by_fingerprint_;
  // Registered datasets, keyed by dataset names.
  absl::flat_hash_map<std::string, std::shared_ptr<Dataset>> datasets_by_name_;

  // Registered workers, keyed by address.
  absl::flat_hash_map<std::string, std::shared_ptr<Worker>> workers_;

  // Assigns an index to each worker according to worker addresses list
  // specified in the dispatcher config.
  WorkerIndexResolver worker_index_resolver_;

  int64_t next_available_job_id_ = 2000;
  // Jobs, keyed by job ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Job>> jobs_;
  // Jobs, keyed by their job keys.
  absl::flat_hash_map<JobKey, std::shared_ptr<Job>> jobs_by_key_;

  int64_t next_available_job_client_id_ = 3000;
  // Mapping from client ids to the jobs they are associated with.
  absl::flat_hash_map<int64_t, std::shared_ptr<Job>> jobs_for_client_ids_;

  int64_t next_available_task_id_ = 4000;
  // Tasks, keyed by task ids.
  TasksById tasks_;
  // List of tasks associated with each job.
  absl::flat_hash_map<int64_t, std::vector<std::shared_ptr<Task>>>
      tasks_by_job_;
  // Tasks, keyed by worker addresses. The values are a map from task id to
  // task.
  absl::flat_hash_map<std::string, TasksById> tasks_by_worker_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
