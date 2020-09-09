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

// Class declaration for Stream type that enqueues tasks onto a host/CPU-based
// execution context (as opposed to a GPU device), HostExecutor.
#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_STREAM_H_

#include <functional>
#include <memory>
#include <queue>

#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace host {

class HostStream : public internal::StreamInterface {
 public:
  // stack_size_in_bytes may be '0', meaning "use the default thread stack
  // size".
  explicit HostStream(size_t stack_size_in_bytes);
  ~HostStream() override;

  bool EnqueueTask(std::function<void()> task);

  void *GpuStreamHack() override { return nullptr; }
  void **GpuStreamMemberHack() override { return nullptr; }

  void BlockUntilDone();

 private:
  bool WorkAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WorkLoop();

  absl::Mutex mu_;
  std::queue<std::function<void()>> work_queue_ TF_GUARDED_BY(mu_);
  std::unique_ptr<port::Thread> thread_;
};

}  // namespace host
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_STREAM_H_
