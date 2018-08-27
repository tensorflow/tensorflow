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

// Class method definitions for HostStream, the Stream implementation for
// the HostExecutor implementation.
#include "tensorflow/stream_executor/host/host_stream.h"

namespace stream_executor {
namespace host {

HostStream::HostStream()
    : host_executor_(new port::ThreadPool(port::Env::Default(),
                                          port::ThreadOptions(),
                                          "host_executor", kExecutorThreads)) {}

HostStream::~HostStream() {}

bool HostStream::EnqueueTask(std::function<void()> task) {
  struct NotifiedTask {
    HostStream* stream;
    std::function<void()> task;

    void operator()() {
      task();
      // Destroy the task before unblocking its waiters, as BlockHostUntilDone()
      // should guarantee that all tasks are destroyed.
      task = std::function<void()>();
      {
        mutex_lock lock(stream->mu_);
        --stream->pending_tasks_;
      }
      stream->completion_condition_.notify_all();
    }
  };

  {
    mutex_lock lock(mu_);
    ++pending_tasks_;
  }
  host_executor_->Schedule(NotifiedTask{this, std::move(task)});
  return true;
}

void HostStream::BlockUntilDone() {
  mutex_lock lock(mu_);
  while (pending_tasks_ != 0) {
    completion_condition_.wait(lock);
  }
}

}  // namespace host

}  // namespace stream_executor
