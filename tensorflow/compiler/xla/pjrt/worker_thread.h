/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_WORKER_THREAD_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_WORKER_THREAD_H_

#include <functional>
#include <memory>
#include <queue>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

// A worker thread that runs a sequence of closures. Equivalent to a thread
// pool of size 1.
class WorkerThread {
 public:
  // 'name' is a name for the thread for debugging purposes.
  WorkerThread(tensorflow::Env* env, const std::string& name);

  // Blocks until all enqueued closures have completed.
  ~WorkerThread();

  // Adds 'fn' to the queue of closures to be executed by the worker thread.
  void Schedule(std::function<void()> fn);

 private:
  bool WorkAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WorkLoop();

  absl::Mutex mu_;
  std::queue<std::function<void()>> work_queue_ TF_GUARDED_BY(mu_);

  std::unique_ptr<tensorflow::Thread> thread_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_WORKER_THREAD_H_
