/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

class CommandBufferThunk : public Thunk {
 public:
  CommandBufferThunk(CommandBufferCmdSequence commands, ThunkInfo thunk_info,
                     std::unique_ptr<SequentialThunk> thunks = nullptr);

  const std::unique_ptr<SequentialThunk>& thunks() const { return thunks_; }

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // Return the allocation address that was lazilly allocated inside command
  // buffer. This API is required when the buffers are allocated inside command
  // buffer but will be consumed by non-command buffer operations.
  absl::StatusOr<se::DeviceMemoryBase> GetCommandBufferAllocationAddress(
      const ExecuteParams& params, int64_t index);

 private:
  // Command buffer instantiated on a `se::StreamExecutor` instance, and
  // auxiliary state required for efficient command buffer updates.
  struct ExecutorCommandBuffer {
    explicit ExecutorCommandBuffer(
        std::unique_ptr<se::CommandBuffer> command_buffer);

    // Returns true if `commands` cmd sequence has to be recorded into
    // `command_buffer` to update it (see `recorded_allocs` below).
    bool ShouldUpdateCommandBuffer(const CommandBufferCmdSequence& commands,
                                   const Thunk::ExecuteParams& params)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex);

    // se::CommandBuffer is not thread safe, and we guard it with a mutex to
    // guarantee that we do not mutate it concurrently.
    absl::Mutex mutex;
    std::unique_ptr<se::CommandBuffer> command_buffer ABSL_GUARDED_BY(mutex);

    // A manager for an external state attached by commands in a command
    // sequence to a command buffer.
    CommandBufferCmd::StateManager state ABSL_GUARDED_BY(mutex);

    // Mapping from buffer allocation index to the device memory passed at
    // that index to the last call of `commands_.Record(...)` for
    // `command_buffer`. We can just use a vector instead of map because
    // `BufferAllocation::Index` is a unique identifier assigned
    // contiguously and thus can be used as array index.
    //
    // If no device memory addresses changed from a previous call to
    // `Record`, we can skip command buffer update and simply submit it for
    // execution on a stream. All other pieces of information (like thread
    // and block sizes) captured by commands at construction time and do not
    // change.
    std::vector<se::DeviceMemoryBase> recorded_allocs ABSL_GUARDED_BY(mutex);

    // Number of command buffer executions since last update.
    int64_t num_executions ABSL_GUARDED_BY(mutex) = 0;
  };

  // Command buffer thunk owns commands buffers instantiated on all executors.
  struct State {
    absl::Mutex mutex;
    absl::flat_hash_map<se::StreamExecutor*,
                        std::shared_ptr<ExecutorCommandBuffer>>
        command_buffers ABSL_GUARDED_BY(mutex);
  };

  // Returns a command buffer instantiated for `executor` or creates new one.
  absl::StatusOr<std::shared_ptr<ExecutorCommandBuffer>>
  GetOrCreateCommandBuffer(se::StreamExecutor* executor);

  // Each individual command buffer allocates state on device (CUDA graph) and
  // it adds up pretty quickly. To prevent OOM errors we proactively evict
  // command buffers from device by clearing command buffer thunk state. We use
  // global state to track all command buffer thunks in a process and coordinate
  // command buffer eviction.
  struct GlobalState;

  // Returns a global state of tracked command buffers thunks.
  static GlobalState* GetGlobalState();

  // Adds command buffer thunk state for tracking.
  static void TrackCommandBuffers(std::weak_ptr<State> state);

  // Evicts all previously instantiated command buffers.
  static void EvictCommandBuffers();

  // Command sequence that initializes command buffers on each executor.
  CommandBufferCmdSequence commands_;

  // Thunk sequence that executes the same commands as in `commands_` but using
  // thunk mechanism. We use it as a fallback mechanism to work around CUPTI
  // bugs that lead to memory corruption when CUPTI traces CUDA graph execution.
  std::unique_ptr<SequentialThunk> thunks_;

  // Command buffer thunk state allocated in heap to allow global (per-process)
  // management of instantiated command buffers.
  std::shared_ptr<State> state_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_
