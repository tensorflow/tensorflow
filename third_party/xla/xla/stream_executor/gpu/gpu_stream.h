/* Copyright 2019 The OpenXLA Authors.

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

// Defines the GpuStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_

#include <variant>

#include "absl/log/check.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;

// Wraps a GpuStreamHandle in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class GpuStream : public StreamInterface {
 public:
  explicit GpuStream(GpuExecutor* parent)
      : parent_(parent), gpu_stream_(nullptr), completed_event_(nullptr) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~GpuStream() override = default;

  void* platform_specific_stream() override { return gpu_stream_; }

  // Explicitly initialize the CUDA resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();

  void SetPriority(StreamPriority priority) override {
    stream_priority_ = priority;
  }

  void SetPriority(int priority) override { stream_priority_ = priority; }

  std::variant<StreamPriority, int> priority() const override {
    return stream_priority_;
  }

  // Explicitly destroy the CUDA resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  GpuEventHandle* completed_event() { return &completed_event_; }

  // Returns the GpuStreamHandle value for passing to the CUDA API.
  //
  // Precond: this GpuStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  GpuStreamHandle gpu_stream() const {
    DCHECK(gpu_stream_ != nullptr);
    return const_cast<GpuStreamHandle>(gpu_stream_);
  }

  // TODO(timshen): Migrate away and remove this function.
  GpuStreamHandle cuda_stream() const { return gpu_stream(); }

  GpuExecutor* parent() const { return parent_; }

 private:
  GpuExecutor* parent_;         // Executor that spawned this stream.
  GpuStreamHandle gpu_stream_;  // Wrapped CUDA stream handle.
  std::variant<StreamPriority, int> stream_priority_;

  // Event that indicates this stream has completed.
  GpuEventHandle completed_event_ = nullptr;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying GpuStream implementation.
GpuStream* AsGpuStream(Stream* stream);

// Extracts a GpuStreamHandle from a GpuStream-backed Stream object.
GpuStreamHandle AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
