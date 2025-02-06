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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_H_

#include <cstdint>
#include <memory>
#include <queue>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {
namespace host {

// Class declaration for Stream type that enqueues tasks onto a host/CPU-based
// execution context (as opposed to a GPU device), HostExecutor.
class HostStream : public StreamCommon {
 public:
  explicit HostStream(StreamExecutor* executor);
  ~HostStream() override;

  // Enqueue a task that reports a status when finished. Tasks that fail do not
  // stop the stream or block any other tasks from executing; rather, the stream
  // will remember the first error encountered and return it from
  // 'BlockUntilDone'.
  virtual bool EnqueueTaskWithStatus(
      absl::AnyInvocable<absl::Status() &&> task);
  // Enqueue a task that doesn't report any status.
  bool EnqueueTask(absl::AnyInvocable<void() &&> task);

  // Blocks until all tasks are done, returns the first error reported by a task
  // (if any) and clears the error status.
  absl::Status BlockUntilDone();

  absl::Status BlockHostUntilDone() override { return BlockUntilDone(); }

  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override;
  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                      uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

 protected:
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
