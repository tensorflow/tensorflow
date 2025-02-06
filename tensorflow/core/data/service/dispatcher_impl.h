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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/auto_scaler.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/snapshot/snapshot_manager.h"
#include "tensorflow/core/data/service/task_remover.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// A service which coordinates a pool of workers to serve dataset elements over
// RPC.
//
// Glossary:
// * Dataset: A definition of how to generate a potentially large collection of
//   elements.
// * Iteration: A coordinated phase of reading from the tf.data service. An
//   iteration produces some amount of data, and (potentially multiple)
//   consumers consume the data from the iteration until there is no data left.
//   Each iteration has a ProcessingModeDef which determines what data it
//   produces.
// * Task: An iteration is broken into multiple tasks, which each represent
//   iterating over all of or part of the dataset. Workers process tasks.
// * Consumer: A process reading from the tf.data service.
//
// **Adding workers**
//
// tf.data service supports adding workers mid-iteration. When a new worker
// connects to the dispatcher, the dispatcher creates a new task for the worker,
// one task for each outstanding iteration. Consumers periodically heartbeat to
// the dispatcher to learn about new tasks.
//
// For non-round-robin-reads, there is no coordination among consumers. Each
// consumer will start reading from the new task as soon as it learns about the
// task from its heartbeat. Round robin reads, on the other hand, require
// consumers to read from the same task at each step. This requires coordination
// to ensure that all consumers start reading from the new task in the same
// round.
//
// The protocol for adding round robin tasks works as follows:
//
// - The dispatcher keeps track of which round each round-robin iteration is on.
// This
//   information is reported by consumers in their heartbeats.
// - When a new worker joins and there is an outstanding round-robin iteration,
//   we create a new task for the iteration and assign it to the worker.
//   However, we don't yet report the task in consumer heartbeats.
//   We call the task a "pending task" and add it to its iteration's "pending
//   tasks" queue.
// - When we create a pending task, we choose a "target round" to try adding
//   the task to. The target round is chosen by adding a "target round delta" to
//   the latest reported round for the iteration.
// - When a consumer heartbeats for an iteration and there is a pending task for
//   that iteration, the dispatcher sends a heartbeat response telling the
//   consumer to block before reading from the target round.
// - When a consumer receives a heartbeat response telling it to block
//   (before reading) a round, the consumer try to block the round. If the
//   consumer has already started the round, it will too late to block the
//   round.
// - When consumers heartbeat, they tell the dispatcher their current round and
//   whether they have blocked themselves from reading past a certain round. If
//   a consumer reports a current round exceeding the target round, the target
//   round has failed and needs to be increased. We choose a new target round by
//   doubling the previous target round delta. If the consumer reports that it
//   has blocked before the target round, we record that the consumer is ready
//   to add the new task. Once all consumers are ready to add the new task, we
//   remove the task from the pending tasks list and begin reporting the task to
//   consumers. We set the "starting_round" field of the task to indicate the
//   target round where all consumers should start reading from the task.
// - If a new worker joins while there are already pending tasks, a pending
//   task for the new worker is created and queued behind the existing tasks.
//   The new task won't be considered until all previous pending tasks have been
//   successfully added.
//
// An example of executing this protocol with two consumers could go as follows:
// 1. Consumers read up to round 50 and heartbeat that they are on round 50.
// 2. A new worker joins. Dispatcher chooses round 51 as the target round.
// 3. Consumer 1 heartbeats that its current round is 50. Dispatcher tells it to
//    block round 51.
// 4. Consumer 2 heartbeats that its current round is 51. Dispatcher realizes
//    that it is too late to block round 51 and chooses round 53 as the new
//    target round. Dispatcher tells consumer 2 to block round 53.
// 5. Consumer 1 heartbeats that its current round is 50 and that it has blocked
//    round 51. Dispatcher tells it to block round 53 instead. Dispatcher
//    records that consumer 1 is ready to add a task in round 53.
// 6. Consumer 2 heartbeats that its current round is 52 and it has blocked
//    round 53. Dispatcher realizes that all consumers are blocked on round 53
//    or earlier and promotes the task from pending to regular. Dispatcher sends
//    consumer 2 a task list containing the new task, and tells consumer 2 that
//    it no longer needs to block.
// 7. Consumer 1 heartbeats. Dispatcher sends consumer 1 the task list
//    containing the new task, and tells it that it no longer needs to block.
//
class DataServiceDispatcherImpl {
 public:
  explicit DataServiceDispatcherImpl(
      const experimental::DispatcherConfig& config);

  ~DataServiceDispatcherImpl();

  // Starts the dispatcher. If there is a journal, this will read from the
  // journal to restore the dispatcher's state.
  absl::Status Start();

  // Stops the dispatcher. After stopping, RPCs should return without blocking.
  void Stop();

  // Returns the number of active iterations.
  size_t NumActiveIterations() TF_LOCKS_EXCLUDED(mu_);

  // See dispatcher.proto for API documentation.

  /// Worker-facing API.
  absl::Status WorkerHeartbeat(const WorkerHeartbeatRequest* request,
                               WorkerHeartbeatResponse* response);
  absl::Status WorkerUpdate(const WorkerUpdateRequest* request,
                            WorkerUpdateResponse* response);
  absl::Status GetDatasetDef(const GetDatasetDefRequest* request,
                             GetDatasetDefResponse* response);
  absl::Status GetSplit(const GetSplitRequest* request,
                        GetSplitResponse* response);

  /// Client-facing API.
  absl::Status GetVersion(const GetVersionRequest* request,
                          GetVersionResponse* response);
  absl::Status GetOrRegisterDataset(const GetOrRegisterDatasetRequest* request,
                                    GetOrRegisterDatasetResponse* response);
  absl::Status GetDataServiceMetadata(
      const GetDataServiceMetadataRequest* request,
      GetDataServiceMetadataResponse* response);
  absl::Status GetDataServiceConfig(const GetDataServiceConfigRequest* request,
                                    GetDataServiceConfigResponse* response);
  absl::Status GetOrCreateJob(const GetOrCreateJobRequest* request,
                              GetOrCreateJobResponse* response);
  absl::Status GetOrCreateIteration(const GetOrCreateIterationRequest* request,
                                    GetOrCreateIterationResponse* response);
  absl::Status ReleaseIterationClient(
      const ReleaseIterationClientRequest* request,
      ReleaseIterationClientResponse* response);
  absl::Status MaybeRemoveTask(const MaybeRemoveTaskRequest* request,
                               MaybeRemoveTaskResponse* response);
  absl::Status ClientHeartbeat(const ClientHeartbeatRequest* request,
                               ClientHeartbeatResponse* response);
  absl::Status GetWorkers(const GetWorkersRequest* request,
                          GetWorkersResponse* response);
  absl::Status Snapshot(const SnapshotRequest* request,
                        SnapshotResponse* response);
  absl::Status GetSnapshotSplit(const GetSnapshotSplitRequest* request,
                                GetSnapshotSplitResponse* response);
  absl::Status GetSnapshotStreams(const GetSnapshotStreamsRequest* request,
                                  GetSnapshotStreamsResponse* response);
  absl::Status DisableCompressionAtRuntime(
      const DisableCompressionAtRuntimeRequest* request,
      DisableCompressionAtRuntimeResponse* response);

  // Exports the dispatcher state for debugging.
  DispatcherStateExport ExportState() const;

 private:
  // A thread which periodically checks for iterations to clean up, clients to
  // release, workers to consider missing, and snapshot streams to reassign.
  void MaintenanceThread();

  // Restores split providers from the state in `iteration` and stores them in
  // `restored`.
  absl::Status RestoreSplitProviders(
      const DispatcherState::Iteration& iteration,
      std::vector<std::unique_ptr<SplitProvider>>& restored)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Makes split providers for the specified `dataset_id`, and stores them in
  // `split_providers`.
  absl::Status MakeSplitProviders(
      const std::string& dataset_id,
      std::vector<std::unique_ptr<SplitProvider>>& split_providers)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Registers a dataset, storing the new dataset's id in `dataset_id`.
  absl::Status RegisterDataset(const DatasetDef& dataset,
                               const DataServiceMetadata& metadata,
                               const std::string& requested_dataset_id,
                               std::string& dataset_id)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Finds the dataset ID with the requested dataset ID.
  // Returns nullptr if no such dataset exists.
  absl::StatusOr<std::optional<std::string>> FindDataset(
      const GetOrRegisterDatasetRequest& request);
  // Gets a worker's stub from `worker_stubs_`, or if none exists, creates a
  // stub and stores it in `worker_stubs_`. A borrowed pointer to the stub is
  // stored in `out_stub`.
  absl::Status GetOrCreateWorkerStub(const std::string& worker_address,
                                     WorkerService::Stub*& out_stub)
      TF_LOCKS_EXCLUDED(mu_);
  // Creates a job and stores it in `job`.
  absl::Status CreateJob(const std::string& job_name,
                         const GetOrCreateJobRequest& request,
                         std::shared_ptr<const DispatcherState::Job>& job)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates an iteration and stores it in `iteration`. This method updates the
  // dispatcher state with the new iteration, but does not assign tasks to
  // workers.
  absl::Status CreateIteration(
      const GetOrCreateIterationRequest& request,
      std::shared_ptr<const DispatcherState::Iteration>& iteration)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates tasks for the specified worker, one task for every unfinished
  // iteration.
  absl::Status CreateTasksForWorker(const std::string& worker_address);
  // Finds tasks that should be deleted from a worker, updating the heartbeat
  // response.
  absl::Status FindTasksToDelete(
      const absl::flat_hash_set<int64_t>& current_tasks,
      const std::vector<std::shared_ptr<const DispatcherState::Task>>&
          assigned_tasks,
      WorkerHeartbeatResponse* response);
  // Finds new tasks that should be assigned to a worker and adds them to
  // the heartbeat response.
  absl::Status FindNewTasks(
      const std::string& worker_address,
      const absl::flat_hash_set<int64_t>& current_tasks,
      std::vector<std::shared_ptr<const DispatcherState::Task>>& assigned_tasks,
      WorkerHeartbeatResponse* response);
  // Reports the processing time of each active task to `auto_scaler_`.
  void ReportProcessingTimesFromActiveTasks(
      const std::vector<ActiveTask>& active_tasks,
      const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Acquires an iteration client id to read from the given iteration and sets
  // `iteration_client_id`.
  absl::Status AcquireIterationClientId(
      const std::shared_ptr<const DispatcherState::Iteration>& iteration,
      int64_t& iteration_client_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates one task for each worker, for the given iteration. The created
  // tasks are stored in `tasks`. This method only updates dispatcher metadata
  // with the new tasks, but doesn't assign the tasks to the workers.
  absl::Status CreateTasksForIteration(
      std::shared_ptr<const DispatcherState::Iteration> iteration,
      std::vector<std::shared_ptr<const DispatcherState::Task>>& tasks)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new task for an iteration. The created task may be either
  // pending or active.
  absl::Status CreateTask(
      std::shared_ptr<const DispatcherState::Iteration> iteration,
      const std::string& worker_address,
      std::shared_ptr<const DispatcherState::Task>& task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates a pending task for a round robin iteration. All consumers need to
  // agree on which round to add the task in before the pending task can be
  // promoted to a regular task.
  absl::Status CreatePendingTask(
      std::shared_ptr<const DispatcherState::Iteration> iteration,
      const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Creates a new active task for an iteration, storing the created task in
  // `task`.
  absl::Status CreateActiveTask(
      std::shared_ptr<const DispatcherState::Iteration> iteration,
      const std::string& worker_address,
      std::shared_ptr<const DispatcherState::Task>& task);
  // Assigns the list of tasks to the workers indicated by their
  // `worker_address` fields.
  absl::Status AssignTasks(
      std::vector<std::shared_ptr<const DispatcherState::Task>> tasks)
      TF_LOCKS_EXCLUDED(mu_);
  // Assigns a task to the worker indicated by its `worker_address` field.
  absl::Status AssignTask(std::shared_ptr<const DispatcherState::Task> task)
      TF_LOCKS_EXCLUDED(mu_);
  // Validates that an existing job matches a given request.
  // Returns an error status describing any difference.
  absl::Status ValidateMatchingJob(
      std::shared_ptr<const DispatcherState::Job> job,
      const GetOrCreateJobRequest& request) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Fills out a TaskDef with information about a task.
  absl::Status PopulateTaskDef(
      std::shared_ptr<const DispatcherState::Task> task,
      TaskDef* task_def) const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Checks that the dispatcher has started, returning UNAVAILABLE if it hasn't.
  absl::Status CheckStarted() TF_LOCKS_EXCLUDED(mu_);
  // Restores ongoing tf.data snapshots.
  absl::Status RestoreSnapshots();
  // Records that a split was produced by a call to `GetSplit`.
  absl::Status RecordSplitProduced(int64_t iteration_id, int64_t repetition,
                                   int64_t split_provider_index, bool finished)
      TF_LOCKS_EXCLUDED(mu_);
  // Applies a state update, updating both the journal and the in-memory state.
  absl::Status Apply(const Update& update) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Applies a state update, but doesn't update the journal. Only meant to be
  // used when recovering state when the dispatcher starts.
  absl::Status ApplyWithoutJournaling(const Update& update)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Removes the client with `client_id` from `auto_scaler_`
  void RemoveClientFromAutoScaler(int64_t client_id)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Releases iteration clients that haven't heartbeated recently.
  absl::Status ReleaseMissingClients() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Removes the worker with `worker_address` from `auto_scaler_`, which is
  // potentially associated with multiple iterations.
  void RemoveWorkerFromAutoScaler(const std::string& worker_address)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Checks for workers that haven't heartbeated recently and alerts the
  // snapshot managers.
  void DetectMissingWorkers() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Scans for old iterations and marks them as finished.
  absl::Status GcOldIterations() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns true if an iteration should be garbage collected.
  bool ShouldGcIteration(const DispatcherState::Iteration& iteration,
                         int64_t now_us) const;
  // Gets a `DatasetDef` from `dataset_store_` for the given dataset id, and
  // stores it in `dataset_def`.
  absl::Status GetDatasetDef(const std::string& dataset_id,
                             std::shared_ptr<const DatasetDef>& dataset_def)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Gets a `DatasetDef` from `dataset_store_` for the given dataset, and
  // stores it in `dataset_def`.
  absl::Status GetDatasetDef(const DispatcherState::Dataset& dataset,
                             std::shared_ptr<const DatasetDef>& dataset_def)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const experimental::DispatcherConfig config_;
  Env* env_;

  mutable mutex mu_;
  // Uses a separate mutex for `GetSplit` requests. `GetSplit` may be blocking.
  // Locking `mu_` in `GetSplit` could block all other RPCs.
  mutable mutex get_split_mu_;
  bool started_ TF_GUARDED_BY(mu_) = false;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;

  // Cached worker stubs for communicating with workers.
  absl::flat_hash_map<std::string, std::unique_ptr<WorkerService::Stub>>
      worker_stubs_ TF_GUARDED_BY(mu_);
  // Store of dataset definitions.
  std::unique_ptr<DatasetStore> dataset_store_ TF_GUARDED_BY(mu_);
  // Mapping from iteration id to the split providers for the iteration.
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<SplitProvider>>>
      split_providers_ TF_GUARDED_BY(mu_);
  // Mapping from round robin iteration id to the round the iteration is
  // currently on. This is based on the data provided by client heartbeats,
  // and may be stale.
  absl::flat_hash_map<int64_t, int64_t> round_robin_rounds_ TF_GUARDED_BY(mu_);
  // Map from task id to a TaskRemover which determines when to remove the task.
  absl::flat_hash_map<int64_t, std::shared_ptr<TaskRemover>>
      remove_task_requests_ TF_GUARDED_BY(mu_);
  // Map from client id to the time of the client's last heartbeat.
  absl::flat_hash_map<int64_t, absl::Time> latest_client_heartbeats_time_
      TF_GUARDED_BY(mu_);
  // Map from worker address to the time of the worker's last heartbeat.
  absl::flat_hash_map<std::string, absl::Time> latest_worker_heartbeats_time_
      TF_GUARDED_BY(mu_);

  // A manager for each snapshot resumed or started during the lifetime of this
  // dispatcher instance.  Note that these are *not* garbage collected; managers
  // for completed snapshots will remain here for the lifetime of the dispatcher
  // instance.  They will even be recovered if the dispatcher is restarted.
  absl::flat_hash_map<std::string, std::unique_ptr<SnapshotManager>> snapshots_
      TF_GUARDED_BY(mu_);
  // A single stream assignment manager shared by all managers in `snapshots_`.
  SnapshotAssignmentManager snapshot_assignment_manager_;

  std::optional<std::unique_ptr<JournalWriter>> journal_writer_
      TF_GUARDED_BY(mu_);
  DispatcherState state_ TF_GUARDED_BY(mu_);
  // Condition variable for waking up the gc thread.
  condition_variable maintenance_thread_cv_;
  std::unique_ptr<Thread> maintenance_thread_;
  MultipleIterationsAutoScaler auto_scaler_;

  DataServiceDispatcherImpl(const DataServiceDispatcherImpl&) = delete;
  void operator=(const DataServiceDispatcherImpl&) = delete;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_IMPL_H_
