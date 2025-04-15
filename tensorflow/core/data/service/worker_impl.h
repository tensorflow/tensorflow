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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// A TensorFlow DataService serves dataset elements over RPC.
class DataServiceWorkerImpl {
 public:
  explicit DataServiceWorkerImpl(const experimental::WorkerConfig& config);
  ~DataServiceWorkerImpl();

  // Starts the worker. The worker needs to know its own address so that it can
  // register with the dispatcher. This is set in `Start` instead of in the
  // constructor because the worker may be binding to port `0`, in which case
  // the address isn't known until the worker has started and decided which port
  // to bind to.
  absl::Status Start(
      const std::string& worker_address,
      const std::vector<DataTransferServerInfo>& transfer_servers);
  // Stops the worker, attempting a clean shutdown by rejecting new requests
  // and waiting for outstanding requests to complete.
  void Stop();

  // Serves a GetElement request, storing the result in `*result`. See
  // worker.proto for GetElement API documentation.
  absl::Status GetElementResult(const GetElementRequest* request,
                                GetElementResult* result);

  // Deletes the local task and iterator. Only called by local clients to delete
  // unused task iterators assuming the task is not read by remote clients. This
  // method is not visible to gRPC clients.
  void DeleteLocalTask(const TaskInfo& task_info);

  // See worker.proto for API documentation.

  /// Dispatcher-facing API.
  absl::Status ProcessTask(const ProcessTaskRequest* request,
                           ProcessTaskResponse* response);

  /// Client-facing API.
  absl::Status GetElement(const GetElementRequest* request,
                          GetElementResponse* response);
  absl::Status GetWorkerTasks(const GetWorkerTasksRequest* request,
                              GetWorkerTasksResponse* response);
  absl::Status GetSnapshotTaskProgresses(
      const GetSnapshotTaskProgressesRequest* request,
      GetSnapshotTaskProgressesResponse* response);

  // Exports the worker state for debugging.
  WorkerStateExport ExportState() const;

 private:
  struct Task {
    explicit Task(TaskDef task_def) : task_def(std::move(task_def)) {}

    TaskDef task_def;
    mutex mu;
    bool initialized TF_GUARDED_BY(mu) = false;
    int64_t outstanding_requests TF_GUARDED_BY(&DataServiceWorkerImpl::mu_) = 0;
    std::unique_ptr<TaskRunner> task_runner;
  };

  struct SnapshotTask {
    // Base directory of the snapshot.
    std::string base_path;

    // Index of the snapshot stream written by this worker.
    int64_t stream_index = 0;

    // This is required to use it as a `flat_hash_map` key.
    template <typename H>
    friend H AbslHashValue(H h, const SnapshotTask& task) {
      return H::combine(std::move(h), task.base_path, task.stream_index);
    }

    friend bool operator==(const SnapshotTask& task1,
                           const SnapshotTask& task2) {
      return task1.base_path == task2.base_path &&
             task1.stream_index == task2.stream_index;
    }
  };

  // Validates the worker config.
  absl::Status ValidateWorkerConfig() const;
  // Creates and initializes a dispatcher client.
  absl::StatusOr<std::unique_ptr<DataServiceDispatcherClient>>
  CreateDispatcherClient() const TF_LOCKS_EXCLUDED(mu_);
  // Sends task status to the dispatcher and checks for dispatcher commands.
  absl::Status SendTaskUpdates() TF_LOCKS_EXCLUDED(mu_);
  // Creates an iterator to process a task.
  absl::Status ProcessTaskInternal(const TaskDef& task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsureTaskInitialized(Task& task);
  // Stops a task, cancelling the task's outstanding requests and waiting for
  // them to finish.
  void StopTask(Task& task) TF_LOCKS_EXCLUDED(mu_);
  // A thread for notifying the dispatcher when tasks complete.
  void TaskCompletionThread() TF_LOCKS_EXCLUDED(mu_);
  // A thread for doing periodic heartbeats to the dispatcher.
  void HeartbeatThread() TF_LOCKS_EXCLUDED(mu_);
  // Performs a heartbeat to the dispatcher.
  absl::Status Heartbeat();
  // Check with the dispatcher to see whether or not to disable compression.
  absl::StatusOr<bool> DisableCompressionAtRuntime(
      const std::string& dataset_id) const;
  // Returns the active tasks of this worker.
  std::vector<ActiveTask> GetActiveTasks() const TF_LOCKS_EXCLUDED(mu_);
  // Returns the task IDs of `active_tasks`.
  std::vector<int64_t> GetTaskIds(
      const std::vector<ActiveTask>& active_tasks) const;
  // Builds a heartbeat request.
  WorkerHeartbeatRequest BuildWorkerHeartbeatRequest() const
      TF_LOCKS_EXCLUDED(mu_);
  // Updates the tasks according to the heartbeat response.
  void UpdateTasks(const WorkerHeartbeatResponse& response)
      TF_LOCKS_EXCLUDED(mu_);
  // Updates the distributed snapshot tasks according to the heartbeat response.
  absl::Status UpdateSnapshotWriters(const WorkerHeartbeatResponse& response)
      TF_LOCKS_EXCLUDED(mu_);
  // Creates an dataset iterator for snapshot writers.
  absl::StatusOr<std::unique_ptr<StandaloneTaskIterator>>
  MakeSnapshotTaskIterator(const SnapshotTaskDef& snapshot_task,
                           const DatasetDef& dataset_def) const;
  // Gets the snapshot task progress from the snapshot writers.
  std::vector<SnapshotTaskProgress> GetSnapshotTaskProgress() const;
  // Gets the DatasetDef for `task_def`.
  absl::StatusOr<DatasetDef> GetDatasetDef(const TaskDef& task_def) const;
  // Creates a dataset from `dataset_def`.
  absl::StatusOr<std::unique_ptr<standalone::Dataset>> MakeDataset(
      const DatasetDef& dataset_def, const TaskDef& task_def) const;
  // Creates an iterator for `dataset`.
  absl::StatusOr<std::unique_ptr<standalone::Iterator>> MakeDatasetIterator(
      standalone::Dataset& dataset, const TaskDef& task_def) const;

  const experimental::WorkerConfig config_;
  // Worker Borg job UID for telemetry. -1 if not supported.
  const int64_t worker_uid_;

  // The worker's own address.
  std::string worker_address_;
  // The data transfer servers available to worker clients.
  std::vector<DataTransferServerInfo> transfer_servers_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_;

  mutable mutex mu_;
  condition_variable cv_;
  // Information about tasks, keyed by task ids. The tasks are updated based on
  // the heartbeat responses from the dispatcher.
  absl::flat_hash_map<int64_t, std::shared_ptr<Task>> tasks_ TF_GUARDED_BY(mu_);
  // Ids of tasks that have finished.
  absl::flat_hash_set<int64_t> finished_tasks_ TF_GUARDED_BY(mu_);
  // Completed tasks which haven't yet been communicated to the dispatcher.
  absl::flat_hash_set<int64_t> pending_completed_tasks_ TF_GUARDED_BY(mu_);
  // Tasks deleted by the local client. If the client tries to read from them
  // again, the worker will return a non-retriable FailedPrecondition error.
  absl::flat_hash_set<int64_t> deleted_tasks_ TF_GUARDED_BY(mu_);
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Whether the worker has registered with the dispatcher yet.
  bool registered_ TF_GUARDED_BY(mu_) = false;
  condition_variable task_completion_cv_ TF_GUARDED_BY(mu_);
  condition_variable heartbeat_cv_ TF_GUARDED_BY(mu_);
  CancellationManager cancellation_manager_;

  absl::flat_hash_map<SnapshotTask, std::unique_ptr<SnapshotStreamWriter>,
                      absl::Hash<SnapshotTask>>
      snapshot_writers_ TF_GUARDED_BY(mu_);

  // A thread for notifying the dispatcher when tasks complete.
  std::unique_ptr<Thread> task_completion_thread_;
  // A thread for performing regular heartbeats to the dispatcher.
  std::unique_ptr<Thread> heartbeat_thread_;

  DataServiceWorkerImpl(const DataServiceWorkerImpl&) = delete;
  void operator=(const DataServiceWorkerImpl&) = delete;
};

// Local in-process workers shared among clients and servers. If clients and
// workers colocate in the same process, clients can read from local workers to
// reduce RPC calls and data copy.
class LocalWorkers {
 public:
  // Adds a `worker` at `worker_address`. If a worker already exists at the
  // address, it will be updated to the new `worker`.
  // REQUIRES: worker != nullptr.
  static void Add(absl::string_view worker_address,
                  std::shared_ptr<DataServiceWorkerImpl> worker);

  // Gets a local worker at `worker_address`. Returns nullptr if a worker is not
  // found.
  static std::shared_ptr<DataServiceWorkerImpl> Get(
      absl::string_view worker_address);

  // Returns if there are any local workers in the process.
  static bool Empty();

  // Removes a worker at `worker_address`. It is no-op if a worker is not found
  // at the address.
  static void Remove(absl::string_view worker_address);

 private:
  using AddressToWorkerMap =
      absl::flat_hash_map<std::string, std::shared_ptr<DataServiceWorkerImpl>>;
  static mutex mu_;
  static AddressToWorkerMap* local_workers_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
