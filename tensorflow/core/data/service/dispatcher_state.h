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
#include <optional>
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
                     const DataServiceMetadata& metadata)
        : dataset_id(dataset_id),
          fingerprint(fingerprint),
          metadata(metadata) {}

    const int64_t dataset_id;
    const int64_t fingerprint;
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

  // A key for identifying an iteration. The key contains a job name,
  // as well as a repetition number describing which repetition of the job
  // we are on.
  struct IterationKey {
    explicit IterationKey(absl::string_view name, int64_t repetition)
        : name(name), repetition(repetition) {}

    friend bool operator==(const IterationKey& lhs, const IterationKey& rhs) {
      return lhs.name == rhs.name && lhs.repetition == rhs.repetition;
    }

    template <typename H>
    friend H AbslHashValue(H h, const IterationKey& k) {
      return H::combine(std::move(h), k.name, k.repetition);
    }

    std::string DebugString() const {
      return absl::StrCat(name, "/", repetition);
    }

    const std::string name;
    const int64_t repetition;
  };

  struct DistributedEpochState {
    explicit DistributedEpochState(int64_t num_split_providers)
        : repetitions(num_split_providers), indices(num_split_providers) {}

    // The current repetition for each split provider.
    std::vector<int64_t> repetitions;
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

  struct Job {
    explicit Job(int64_t id, int64_t dataset_id,
                 const ProcessingModeDef& processing_mode, std::string job_name,
                 std::optional<int64_t> num_consumers,
                 bool use_cross_trainer_cache, TargetWorkers target_workers)
        : id(id),
          dataset_id(dataset_id),
          processing_mode(processing_mode),
          job_name(job_name),
          num_consumers(num_consumers),
          use_cross_trainer_cache(use_cross_trainer_cache),
          target_workers(target_workers) {}

    const int64_t id;
    const int64_t dataset_id;
    const ProcessingModeDef processing_mode;
    const std::string job_name;
    const absl::optional<int64_t> num_consumers;
    const bool use_cross_trainer_cache;
    const TargetWorkers target_workers;
  };

  // An iteration for processing a dataset.
  struct Iteration {
    explicit Iteration(int64_t iteration_id, IterationKey iteration_key,
                       int64_t num_split_providers, std::shared_ptr<Job> job)
        : iteration_id(iteration_id), iteration_key(iteration_key), job(job) {
      if (IsDynamicShard(job->processing_mode)) {
        distributed_epoch_state = DistributedEpochState(num_split_providers);
      }
    }

    bool IsRoundRobin() const { return job->num_consumers.has_value(); }

    std::string DebugString() const {
      return absl::StrCat(iteration_key.name, "_", iteration_key.repetition);
    }

    const int64_t iteration_id;
    const IterationKey iteration_key;
    const std::shared_ptr<Job> job;
    absl::optional<DistributedEpochState> distributed_epoch_state;
    std::queue<PendingTask> pending_tasks;
    int64_t num_clients = 0;
    int64_t last_client_released_micros = -1;
    bool finished = false;
    // Indicates whether the iteration was garbage collected.
    bool garbage_collected = false;
  };

  struct Task {
    template <class T>
    explicit Task(const T& create_task_update,
                  const std::shared_ptr<Iteration>& iteration)
        : task_id(create_task_update.task_id()),
          iteration(iteration),
          worker_address(create_task_update.worker_address()),
          transfer_address(create_task_update.transfer_address()),
          worker_tags(create_task_update.worker_tags().begin(),
                      create_task_update.worker_tags().end()),
          worker_uid(create_task_update.worker_uid()) {}

    const int64_t task_id;
    const std::shared_ptr<Iteration> iteration;
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

  // Gets a worker by address. Returns NOT_FOUND if there is no such worker.
  Status WorkerFromAddress(const std::string& address,
                           std::shared_ptr<const Worker>& worker) const;
  // Lists all workers registered with the dispatcher.
  std::vector<std::shared_ptr<const Worker>> ListWorkers() const;

  // Returns the next available job id.
  int64_t NextAvailableJobId() const;
  // Gets a job by id. Returns NOT_FOUND if there is no such job.
  Status JobFromId(int64_t job_id, std::shared_ptr<const Job>& job) const;
  // Gets a job by name. Returns NOT_FOUND if there is no such job.
  Status JobByName(const std::string& job_name,
                   std::shared_ptr<const Job>& job) const;

  // Returns the next available iteration id.
  int64_t NextAvailableIterationId() const;
  // Returns a list of all iterations.
  std::vector<std::shared_ptr<const Iteration>> ListIterations() const;
  // Gets an iteration by id. Returns NOT_FOUND if there is no such iteration.
  Status IterationFromId(int64_t id,
                         std::shared_ptr<const Iteration>& iteration) const;
  // Gets an iteration by key. Returns NOT_FOUND if there is no such iteration.
  Status IterationByKey(IterationKey key,
                        std::shared_ptr<const Iteration>& iteration) const;

  // Returns the iteration associated with the given iteration client id.
  // Returns NOT_FOUND if the iteration_client_id is unknown or has been
  // released.
  Status IterationForIterationClientId(
      int64_t iteration_client_id, std::shared_ptr<const Iteration>& iteration);
  // Returns a list of all active client ids.
  std::vector<int64_t> ListActiveClientIds();
  // Returns the next available iteration client id.
  int64_t NextAvailableIterationClientId() const;

  // Returns the next available task id.
  int64_t NextAvailableTaskId() const;
  // Gets a task by id. Returns NOT_FOUND if there is no such task.
  Status TaskFromId(int64_t id, std::shared_ptr<const Task>& task) const;
  // Stores a list of all tasks for the given iteration to `tasks`. Returns
  // NOT_FOUND if there is no such iteration.
  Status TasksForIteration(
      int64_t iteration_id,
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
  void CreateIteration(const CreateIterationUpdate& create_iteration);
  void ProduceSplit(const ProduceSplitUpdate& produce_split);
  void AcquireIterationClient(
      const AcquireIterationClientUpdate& acquire_iteration_client);
  void ReleaseIterationClient(
      const ReleaseIterationClientUpdate& release_iteration_client);
  void GarbageCollectIteration(
      const GarbageCollectIterationUpdate& garbage_collect_iteration);
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

  // Registered workers, keyed by address.
  absl::flat_hash_map<std::string, std::shared_ptr<Worker>> workers_;

  // Assigns an index to each worker according to worker addresses list
  // specified in the dispatcher config.
  WorkerIndexResolver worker_index_resolver_;

  int64_t next_available_job_id_ = 5000;
  // Jobs, keyed by job ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Job>> jobs_by_id_;
  // Jobs, keyed by job names.
  absl::flat_hash_map<std::string, std::shared_ptr<Job>> jobs_by_name_;

  int64_t next_available_iteration_id_ = 2000;
  // Iterations, keyed by iteration ids.
  absl::flat_hash_map<int64_t, std::shared_ptr<Iteration>> iterations_;
  // Iterations, keyed by their iteration keys.
  absl::flat_hash_map<IterationKey, std::shared_ptr<Iteration>>
      iterations_by_key_;

  int64_t next_available_iteration_client_id_ = 3000;
  // Mapping from client ids to the iterations they are associated with.
  absl::flat_hash_map<int64_t, std::shared_ptr<Iteration>>
      iterations_for_client_ids_;

  int64_t next_available_task_id_ = 4000;
  // Tasks, keyed by task ids.
  TasksById tasks_;
  // List of tasks associated with each iteration.
  absl::flat_hash_map<int64_t, std::vector<std::shared_ptr<Task>>>
      tasks_by_iteration_;
  // Tasks, keyed by worker addresses. The values are a map from task id to
  // task.
  absl::flat_hash_map<std::string, TasksById> tasks_by_worker_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
