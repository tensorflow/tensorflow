/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_interface.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace stream_executor {

class Stream;

// A StreamExecutor manages a single device, in terms of executing work (kernel
// launches) and memory management (allocation/deallocation, memory copies to
// and from the device). It is conceptually the "handle" for a device -- Stream
// objects, which are used to enqueue work to run on the
// coprocessor have a StreamExecutor instance as their "parent" object.
//
// StreamExecutor objects have an underlying platform that is specified up
// front;
// e.g. either it is a CUDA or OpenCL executor.
//
// Thread-safe after initialization.
// StreamExecutor interface should not be invoked from a signal handler.
class StreamExecutor : public StreamExecutorInterface {
 public:
  explicit StreamExecutor(const Platform* platform);

  ~StreamExecutor() = default;

  const Platform* GetPlatform() const override { return platform_; }
  const DeviceDescription& GetDeviceDescription() const override;
  int64_t GetMemoryLimitBytes() const override { return memory_limit_bytes_; }

 private:
  // Reader/writer lock for mutable data structures on this StreamExecutor.
  //
  // Mutable so that caching functions (like DeviceDescription, AsBlas, etc.)
  // can acquire the lock on their first (mutating) call as well.
  mutable absl::Mutex mu_;

  // Reference to the platform that created this executor.
  const Platform* platform_;

  // Slot to cache the owned DeviceDescription for the underlying device
  // once it has been queried from DeviceDescription().
  mutable std::unique_ptr<DeviceDescription> device_description_
      ABSL_GUARDED_BY(mu_);

  // Memory limit in bytes. Value less or equal to 0 indicates there is no
  // limit.
  int64_t memory_limit_bytes_;

  StreamExecutor(const StreamExecutor&) = delete;
  void operator=(const StreamExecutor&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
