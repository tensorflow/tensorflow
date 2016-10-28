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

#ifndef THIRD_PARTY_TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
#define THIRD_PARTY_TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

// QueueRunner class imitates the behavior of the python version of QueueRunner
// which creates a thread for each enqueue op, runs close op on completion.
class QueueRunner {
 public:
  QueueRunner();

  // The constructor initializes the class from the proto.
  // TODO(yuefengz): we may want to initialize from queues and ops in the
  // future.
  explicit QueueRunner(const QueueRunnerDef& queue_runner_def);

  // The destructor would join all the threads.
  ~QueueRunner();

  // Initializes the instance with the QueueRunnerDef proto.
  Status Init(const QueueRunnerDef& queue_runner_def);

  // Starts the queue runner with the given session.
  Status Start(Session* sess);

  // Requests to stop and runs the cancel op.
  Status Stop(Session* sess);

  // Joins all the threads. Returns okay if all threads run successfully;
  // otherwise returns the first captured failure status.
  Status Join();

 private:
  // The Run function for each thread.
  void Run(Session* sess, const string& enqueue_op);

  string queue_name_;
  std::vector<string> enqueue_op_names_;
  string close_op_name_;
  string cancel_op_name_;
  // code::Code casted to int to avoid a hash function.
  std::unordered_set<int> queue_closed_exception_types_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> started_;
  condition_variable wait_to_close_;
  mutex mu_;
  // TODO(yuefengz): implement c++ coordinator.
  int runs_ = 0;
  Status status_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
