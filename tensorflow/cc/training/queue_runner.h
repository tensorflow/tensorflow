/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
#define TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {

/// QueueRunner class imitates the behavior of the python version of QueueRunner
/// which creates a thread for each enqueue op, runs close op on completion.
class QueueRunner : public RunnerInterface {
 public:
  /// Creates a new QueueRunner from proto.
  // TODO(yuefengz): we may want to initialize from queues and ops in the
  // future.
  static absl::Status New(const QueueRunnerDef& queue_runner_def,
                          std::unique_ptr<QueueRunner>* result);

  /// Creates a new QueueRunner with a coordinator, see coordinator.h for usage.
  static absl::Status New(const QueueRunnerDef& queue_runner_def,
                          Coordinator* coord,
                          std::unique_ptr<QueueRunner>* result);

  /// Adds a callback that the queue runner will call when it detects an error.
  void AddErrorCallback(const std::function<void(absl::Status)>& cb);

  /// Delete the previously registered callbacks.
  void ClearErrorCallbacks();

  /// The destructor would join all the threads.
  ~QueueRunner();

  /// Starts the queue runner with the given session.
  absl::Status Start(Session* sess);

  /// Starts the queue runner with the given session and sets the run arguments
  /// for sess->Run. It also collects and stores the cost model.
  absl::Status StartAndCollectCostGraph(
      Session* sess, const RunOptions& run_options = RunOptions());

  /// Starts the queue runner with the given session, and wait for up to the
  /// specified time (in milliseconds) for the queues to start to fill up.
  absl::Status Start(Session* sess, int wait_for_ms);
  absl::Status StartAndCollectCostGraph(
      Session* session, int wait_for_ms,
      const RunOptions& run_options = RunOptions());

  /// Requests to stop and runs the cancel op. It would be called in a separate
  /// thread when coordinator is set. If there is no coordinator it should be
  /// called before calling Join.
  void Stop(Session* sess);

  /// Joins all the threads. Returns okay if all threads run successfully;
  /// otherwise returns the first captured failure status.
  absl::Status Join() final;

  /// Returns the latest status.
  absl::Status GetStatus();

  // Returns the stored cost model.
  absl::Status ExportCostGraph(CostGraphDef* cost_graph) const override;

 private:
  QueueRunner() : coord_(nullptr), stopped_(false), cg_mu_(nullptr) {}

  // Initializes the instance with the QueueRunnerDef proto.
  absl::Status Init(const QueueRunnerDef& queue_runner_def);

  // The Run function for each thread.
  void Run(Session* sess, const string& enqueue_op);

  // Updates the internal status; it only keeps OK or the first unexpected error
  // status.
  void UpdateStatus(const absl::Status& status);

  bool IsQueueClosed(absl::Status status) const {
    return queue_closed_exception_types_.count(
               static_cast<int>(status.code())) > 0;
  }

  bool IsRunning() const override { return !stopped_; }

  void SetRunArgumentsAndCostGraph(const RunOptions& run_options);

  absl::Status RealRun(Session* sess, const string& op, bool update_costs);

  string queue_name_;
  std::vector<string> enqueue_op_names_;
  string close_op_name_;
  string cancel_op_name_;
  // code::Code casted to int to avoid a hash function.
  std::unordered_set<int> queue_closed_exception_types_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  mutex mu_;
  int runs_ = 0;
  absl::Status status_ TF_GUARDED_BY(mu_);
  absl::Status enqueue_status_ TF_GUARDED_BY(mu_);
  std::unique_ptr<BlockingCounter> counter_;

  Coordinator* coord_;

  std::atomic<bool> stopped_;

  mutex cb_mu_;
  std::vector<std::function<void(absl::Status)>> callbacks_;

  mutable std::unique_ptr<mutex> cg_mu_;
  std::unique_ptr<CostGraphDef> cost_graph_ TF_GUARDED_BY(cg_mu_);
  RunOptions run_options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
