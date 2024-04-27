/* Copyright 2016 The OpenXLA Authors.

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
#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_H_

#include <cstddef>
#include <memory>
#include <queue>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream_interface.h"
#include "tsl/platform/env.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {
namespace host {

class HostStream : public StreamInterface {
 public:
  HostStream();
  ~HostStream() override;

  // Enqueue a task that reports a status when finished. Tasks that fail do not
  // stop the stream or block any other tasks from executing; rather, the stream
  // will remember the first error encountered and return it from
  // 'BlockUntilDone'.
  bool EnqueueTaskWithStatus(absl::AnyInvocable<absl::Status() &&> task);
  // Enqueue a task that doesn't report any status.
  bool EnqueueTask(absl::AnyInvocable<void() &&> task);

  // Blocks until all tasks are done, returns the first error reported by a task
  // (if any) and clears the error status.
  absl::Status BlockUntilDone();

 private:
  bool WorkAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WorkLoop();

  absl::Mutex mu_;
  std::queue<absl::AnyInvocable<absl::Status() &&>> work_queue_
      ABSL_GUARDED_BY(mu_);
  std::unique_ptr<tsl::Thread> thread_;
  absl::Status status_;
};

}  // namespace host
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_H_
