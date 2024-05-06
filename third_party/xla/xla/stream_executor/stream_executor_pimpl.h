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

  // Synchronously allocates an array on the device of type T with element_count
  // elements.
  template <typename T>
  DeviceMemory<T> AllocateArray(uint64_t element_count,
                                int64_t memory_space = 0);

  // Convenience wrapper that allocates space for a single element of type T in
  // device memory.
  template <typename T>
  DeviceMemory<T> AllocateScalar() {
    return AllocateArray<T>(1);
  }

  // An untyped version of GetSymbol.
  absl::StatusOr<DeviceMemoryBase> GetUntypedSymbol(
      const std::string& symbol_name, ModuleHandle module_handle);

  absl::Status SynchronousMemcpyH2D(const void* host_src, int64_t size,
                                    DeviceMemoryBase* device_dst);

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <class T>
  absl::Status SynchronousMemcpyH2D(absl::Span<const T> host_src,
                                    DeviceMemoryBase* device_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(device_dst->size() == 0 || device_dst->size() >= host_size);
    return SynchronousMemcpyH2D(host_src.begin(), host_size, device_dst);
  }

  // Same as SynchronousMemcpy(void*, ...) above.
  absl::Status SynchronousMemcpyD2H(const DeviceMemoryBase& device_src,
                                    int64_t size, void* host_dst);

  // Obtains metadata about the underlying device.
  // The value is cached on first use.
  const DeviceDescription& GetDeviceDescription() const;

  // Creates and initializes a Stream.
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority = std::nullopt);

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

  // Only one worker thread is needed; little work will be done by the
  // executor.
  static constexpr int kNumBackgroundThreads = 1;

  // Memory limit in bytes. Value less or equal to 0 indicates there is no
  // limit.
  int64_t memory_limit_bytes_;

  StreamExecutor(const StreamExecutor&) = delete;
  void operator=(const StreamExecutor&) = delete;
};

////////////
// Inlines

template <typename T>
inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64_t element_count,
                                                     int64_t memory_space) {
  uint64_t bytes = sizeof(T) * element_count;
  if (memory_limit_bytes_ > 0 &&
      static_cast<int64_t>(bytes) > memory_limit_bytes_) {
    LOG(WARNING) << "Not enough memory to allocate " << bytes << " on device "
                 << device_ordinal()
                 << " within provided limit.  limit=" << memory_limit_bytes_
                 << "]";
    return DeviceMemory<T>();
  }
  return DeviceMemory<T>(Allocate(bytes, memory_space));
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
