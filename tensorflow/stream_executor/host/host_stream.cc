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

#include "absl/synchronization/notification.h"

namespace stream_executor {
namespace host {

HostStream::HostStream()
    : thread_(port::Env::Default()->StartThread(
          port::ThreadOptions(), "host_executor", [this]() { WorkLoop(); })) {}

HostStream::~HostStream() {
  {
    absl::MutexLock lock(&mu_);
    work_queue_.push(nullptr);
  }
  // thread_'s destructor blocks until the thread finishes running.
  thread_.reset();
}

bool HostStream::EnqueueTask(std::function<void()> fn) {
  CHECK(fn != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(std::move(fn));
  return true;
}

bool HostStream::WorkAvailable() { return !work_queue_.empty(); }

void HostStream::WorkLoop() {
  while (true) {
    std::function<void()> fn;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &HostStream::WorkAvailable));
      fn = std::move(work_queue_.front());
      work_queue_.pop();
    }
    if (!fn) {
      return;
    }
    fn();
  }
}

void HostStream::BlockUntilDone() {
  absl::Notification done;
  EnqueueTask([&done]() { done.Notify(); });
  done.WaitForNotification();
}

}  // namespace host

}  // namespace stream_executor
