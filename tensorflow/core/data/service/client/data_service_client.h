/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/worker_client.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Interface for interacting with the tf.data service iterator context.
class DataServiceContext {
 public:
  virtual ~DataServiceContext() = default;
  virtual std::unique_ptr<Thread> StartThread(const string& name,
                                              std::function<void()> fn) = 0;
  virtual void RecordBufferEnqueue(const std::vector<Tensor>& element) = 0;
  virtual void RecordBufferDequeue(const std::vector<Tensor>& element) = 0;
};

using DataServiceContextFactory =
    std::function<std::unique_ptr<DataServiceContext>()>;

// API for reading data from tf.data service.
//
// The client works by reading from tf.data workers in parallel and interleaving
// the dataset elements. It periodically queries the dispatcher to decide which
// workers to read from (in case workers are added or removed). The data reading
// is non-deterministic. This class is thread-safe.
class DataServiceClient {
 public:
  explicit DataServiceClient(const DataServiceParams& params);
  virtual ~DataServiceClient();
  DataServiceClient(const DataServiceClient&) = delete;
  DataServiceClient& operator=(const DataServiceClient&) = delete;

  // Initializes the client.
  Status Initialize();

  // Reads the next element from tf.data workers. Blocks if the next element is
  // not ready.
  virtual StatusOr<GetNextResult> GetNext(
      DataServiceContextFactory context_factory);

  // Cancels the client.
  void Cancel();

  TraceMeMetadata GetTraceMeMetadata() const;

 private:
  struct Task {
    Task(const TaskInfo& info, std::unique_ptr<DataServiceWorkerClient> worker)
        : info(info), worker(std::move(worker)) {}

    const TaskInfo info;
    // Client for fetching task elements from the tf.data service worker.
    const std::unique_ptr<DataServiceWorkerClient> worker;
    // The next round to read from the task.
    int64_t round = 0;
    // Whether the task has been removed. The task will eventually be
    // deleted from `tasks_` on the next dispatcher heartbeat.
    bool removed = false;
    bool skipped_previous_round = false;
    // Indicates whether a worker thread is currently processing the task.
    bool in_use TF_GUARDED_BY(&DataServiceClient::mu_) = false;
    // Indicates whether the worker has returned end_of_sequence for the task.
    bool end_of_sequence TF_GUARDED_BY(&DataServiceClient::mu_) = false;
  };

  struct Result {
    Result() = default;
    Result(Result&&) = default;
    Result& operator=(Result&&) = default;
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    // Whether the result has been computed yet. GetNext needs to block
    // until the next result is ready.
    bool ready TF_GUARDED_BY(&DataServiceClient::mu_) = false;
    std::vector<Tensor> element TF_GUARDED_BY(&DataServiceClient::mu_);
    // The element's index within the tf.data worker it came from. Used for
    // debugging.
    int64_t element_index TF_GUARDED_BY(&DataServiceClient::mu_) = -1;
    // The id of the task that generated the result.
    int64_t task_id TF_GUARDED_BY(&DataServiceClient::mu_) = -1;
    bool end_of_sequence TF_GUARDED_BY(&DataServiceClient::mu_) = false;
    bool skip TF_GUARDED_BY(&DataServiceClient::mu_) = false;
  };

  void EnsureThreadsStarted();
  void CancelThreads();
  // Returns whether the client has finished and should return.
  bool Finished() const;
  // Returns whether the job has more data.
  bool ShouldWaitForNext() const;
  void DeleteLocalWorkerTasks();
  bool ShouldDeleteLocalTask(const TaskInfo& task) const;
  // Periodically refresh the task list.
  // Maintain one thread fetching elements for each task.
  // TODO(aaudibert): Instead of polling, have dispatcher send updates when
  // the list of tasks changes.
  void TaskThreadManager();
  void TryBlockRound(int64_t round) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void UpdateIterationFinished(bool iteration_finished);
  Status AddTask(const TaskInfo& task_info);
  void Heartbeat();
  void UpdateTasks(const ClientHeartbeatResponse& resp);
  bool ShouldReadFromTask(const TaskInfo& task) const;
  void RecordTFMetrics(const ClientHeartbeatResponse& resp);
  void UpdateBufferSize();
  void UpdateWorkerThreads();
  void RunWorkerThread(std::function<void()> done);
  // Reports whether we can request another element without violating
  // `max_outstanding_requests_`.
  bool ShouldProcessTask();
  // Searches for a task to process, visiting tasks in-order and giving every
  // task a chance to proceed.
  std::shared_ptr<Task> GetTaskToProcess();
  void AdvanceTaskIndex();
  Status TryGetElement(const Task& task, GetElementResult& result);
  void ProcessGetElementResponse(bool enqueue_result,
                                 GetElementResult& get_element_result,
                                 std::shared_ptr<Result> result, Task& task);
  Status GetElementTraced(Task* task, int64_t deadline_micros,
                          bool enqueue_result, std::shared_ptr<Result> result);
  Status MaybeRemoveTask(Task& task, int64_t deadline_micros, Result& result);
  Status GetElement(Task* task, int64_t deadline_micros, bool enqueue_result,
                    std::shared_ptr<Result> result);
  bool ResultReady() const;
  std::shared_ptr<Result> PopNextResult();
  bool IsCoordinatedRead() const;
  std::string DebugString() const;

  const DataServiceParams params_;

  mutable mutex mu_;
  condition_variable get_next_cv_ TF_GUARDED_BY(mu_);
  condition_variable worker_thread_cv_ TF_GUARDED_BY(mu_);
  condition_variable manager_thread_cv_ TF_GUARDED_BY(mu_);

  bool cancelled_ TF_GUARDED_BY(mu_) = false;

  // Number of outstanding requests.
  int64_t outstanding_requests_ TF_GUARDED_BY(mu_) = 0;

  // max_outstanding_requests controls how many elements may be held in memory
  // at the same time. This count includes both in-progress requests for
  // elements as well as completed requests which haven't yet been produced.
  int64_t max_outstanding_requests_ TF_GUARDED_BY(mu_);

  // The number of threads in `worker_threads_` which are still running.
  int64_t num_running_worker_threads_ TF_GUARDED_BY(mu_) = 0;

  // The index of the next task in `tasks_` to read from.
  int64_t next_task_index_ TF_GUARDED_BY(mu_) = 0;

  // The number tasks in the `tasks_` list that have reached end_of_sequence.
  int64_t finished_tasks_ TF_GUARDED_BY(mu_) = 0;

  // List of tasks to read from.
  std::vector<std::shared_ptr<Task>> tasks_ TF_GUARDED_BY(mu_);

  // The current round robin round we are engaged in. A round involves reading
  // from each task once.
  int64_t current_round_ TF_GUARDED_BY(mu_) = 0;

  // Maximum round robin round to read up to before blocking, not inclusive.
  // INVARIANT: current_round_ <= round_robin_round_limit_.
  //            If current_round_ == round_robin_round_limit_,
  //            next_task_index_ must be 0.
  std::optional<int64_t> round_robin_round_limit_ TF_GUARDED_BY(mu_);

  // A status to be returned from the next call to `GetNext`. This is set by
  // asynchronous threads when they encounter errors.
  Status status_ TF_GUARDED_BY(mu_) = OkStatus();
  // A queue of results for `GetElement` requests to read from. When doing
  // strict round robin reads, the queue will contain placeholder results with
  // their `Result::ready` field false until their data has been retrieved
  // from a worker. When not doing round-robin reads, results are only added
  // to the queue after they are ready, to avoid head-of-line blocking.
  std::queue<std::shared_ptr<Result>> results_ TF_GUARDED_BY(mu_);

  bool initialized_ = false;
  std::unique_ptr<DataServiceContext> ctx_ TF_GUARDED_BY(mu_);

  // Set once in Initialize().
  int64_t job_id_;
  int64_t iteration_client_id_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_;

  int64_t get_next_index_ TF_GUARDED_BY(mu_) = 0;

  bool iteration_finished_ TF_GUARDED_BY(mu_) = false;
  bool should_finish_iteration_ TF_GUARDED_BY(mu_) = true;

  // The set of worker UIDs that we have already recorded metrics for.
  absl::flat_hash_set<int64_t> worker_uids_ TF_GUARDED_BY(mu_);

  std::vector<std::unique_ptr<Thread>> worker_threads_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Thread> task_thread_manager_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_
