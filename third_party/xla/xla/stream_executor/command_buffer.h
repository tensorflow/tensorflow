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

#ifndef XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

class Stream;
class StreamExecutor;

//===----------------------------------------------------------------------===//
// CommandBuffer
//===----------------------------------------------------------------------===//

// Command buffer represent a "bundle of work items" for StreamExecutor device
// that can be submitted with one API call, e.g. command buffer might have
// multiple device kernels and synchronization barriers between them. Command
// buffers allow to amortize the cost of launching "work" on device by building
// it on the host ahead of time without expensive interaction with underlying
// device.
class CommandBuffer {
 public:
  // Builder constructs nested command buffers owned by a parent command buffer.
  using Builder = std::function<absl::Status(CommandBuffer*)>;

  CommandBuffer() = default;
  virtual ~CommandBuffer() = default;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;

  // Command buffer state:
  //
  //   (1) kCreate:    a new command buffer under construction
  //   (2) kUpdate:    updating a previously finalized command buffer
  //   (3) kFinalized: command buffer ready for execution
  //
  // Supported state transitions:
  //
  //   (1) Finalize: (kCreate|kUpdate) -> kFinalized
  //   (2) Update:   kFinalized -> kUpdate
  //
  enum class State { kCreate, kUpdate, kFinalized };

  // Command buffers have two modes of execution:
  //
  //   (1) kPrimary: command buffer can be submitted for execution via
  //                 StreamExecutor APIs
  //   (2) kNested:  command buffer can be executed only within a primary
  //                 command buffer
  //
  enum class Mode { kPrimary, kNested };

  //===--------------------------------------------------------------------===//
  // Command buffer constructors
  //===--------------------------------------------------------------------===//

  // TODO(b/323534971): Command buffer constructors should be moved to
  // StreamExecutor or a dedicated CommandBufferFactory accessible via
  // StreamExecutor.

  // Creates a new empty command buffer on the given executor.
  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Create(
      StreamExecutor* executor, Mode mode = Mode::kPrimary);

  // Creates a new command buffer on the given executor by tracing `function`
  // invocation. All StreamExecutor operations on a Stream argument will be
  // recorded into the command buffer. Returned command buffer is finalized, and
  // can't be updated.
  //
  // Command buffer tracing should be used only when it is impossible to use
  // explicit construction APIs, e.g. when calling external libraries. By
  // default we construct traced command buffers in nested mode because the
  // primary use case for traced command buffers is to be inserted into primary
  // command buffers constructed with explicit APIs.
  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Trace(
      StreamExecutor* executor,
      absl::AnyInvocable<absl::Status(Stream*)> function,
      Mode mode = Mode::kNested);

  // Creates a new command buffer on the given executor by tracing `function`
  // invocation using a user provided stream that will be passed to `function`.
  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Trace(
      StreamExecutor* executor, Stream* stream,
      absl::AnyInvocable<absl::Status(Stream*)> function,
      Mode mode = Mode::kNested);

  //===--------------------------------------------------------------------===//
  // Command buffer properties
  //===--------------------------------------------------------------------===//

  // Returns true if command buffer on a given platform supports conditional
  // commands (If, IfThen, While).
  static bool SupportsConditionalCommands(const Platform* platform);

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  // Adds an execution barrier to a command buffer: all commands added before a
  // barrier will complete before any of the commands added after a barrier.
  virtual absl::Status Barrier(StreamExecutor* executor) = 0;

  // Adds a kernel launch command to the command buffer.
  virtual absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                              const Kernel& kernel, const KernelArgs& args) = 0;

  // Type-safe wrapper for launching typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::Status Launch(const TypedKernel<Params...>& kernel,
                      const ThreadDim& threads, const BlockDim& blocks,
                      Args... args);

  // Adds a nested command buffer to the command buffer.
  virtual absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) = 0;

  // Adds a device-to-device memory copy to the command buffer.
  virtual absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                            const DeviceMemoryBase& src,
                                            uint64_t size) = 0;

  // Adds a memset node to the command buffer.
  using BitPattern = std::variant<uint8_t, uint16_t, uint32_t>;
  virtual absl::Status Memset(DeviceMemoryBase* dst, BitPattern bit_pattern,
                              size_t num_elements) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer memory allocation API
  //--------------------------------------------------------------------------//

  // Adds a device memory allocation command to the command buffer.
  virtual absl::StatusOr<DeviceMemoryBase> Allocate(size_t bytes) = 0;

  // This API free buffer that is allocated by Allocate command
  virtual absl::Status Free(DeviceMemoryBase dst) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`.
  virtual absl::Status If(StreamExecutor* executor, DeviceMemory<bool> pred,
                          Builder then_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`, or a command buffer
  // constructed by `else_builder` if `pred` is `false`.
  virtual absl::Status IfElse(StreamExecutor* executor, DeviceMemory<bool> pred,
                              Builder then_builder, Builder else_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `branches` builder at `index`. If `index` is out of range, then it
  // will run a conditional command buffer constructed by the last builder.
  //
  // See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
  virtual absl::Status Case(StreamExecutor* executor,
                            DeviceMemory<int32_t> index,
                            std::vector<Builder> branches) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `body_builder` exactly `num_iteration` times. This means the
  // condition is known at compile time (`num_iteration` < `loop_counter`), and
  // does not require a `cond_builder`.
  virtual absl::Status For(StreamExecutor* executor, int32_t num_iteration,
                           DeviceMemory<int32_t> loop_counter,
                           Builder body_builder) = 0;

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `cond_builder` that must update `pred` value, and then depending on
  // the value might execute command buffer constructed by `body_builder` and
  // `cond_builder`. Will continue while `pred` value (which is continuously
  // updated by `cond_builder`) is `true`.
  //
  // In pseudocode:
  //
  //   cond_builder()
  //   while(pred):
  //     body_builder()
  //     cond_builder()
  //
  virtual absl::Status While(StreamExecutor* executor, DeviceMemory<bool> pred,
                             Builder cond_builder, Builder body_builder) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer state management API
  //--------------------------------------------------------------------------//

  // Finalizes command buffer and makes it executable. Once command buffer is
  // finalized no commands can be added to it.
  virtual absl::Status Finalize() = 0;

  // Begins command buffer update. Command buffer update should be finalized
  // before it can be executed.
  virtual absl::Status Update() = 0;

  // Returns command buffer execution mode.
  virtual Mode mode() const = 0;

  // Returns command buffer state.
  virtual State state() const = 0;

  //--------------------------------------------------------------------------//
  // Command buffer tracing API
  //--------------------------------------------------------------------------//
 private:
  // Tracing APIs are private because they do not compose with command buffer
  // updates. Instead of tracing directly into the command buffer users should
  // create traced command buffers using factory methods and add them to primary
  // command buffers as nested operations.

  // Traces `function` invocation by recording all operations on the `stream`
  // into the command buffer. Command buffer must be empty.
  virtual absl::Status Trace(Stream* stream,
                             absl::AnyInvocable<absl::Status()> function) = 0;
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
inline absl::Status CommandBuffer::Launch(const TypedKernel<Params...>& kernel,
                                          const ThreadDim& threads,
                                          const BlockDim& blocks,
                                          Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(Launch(threads, blocks, *kernel, *kernel_args));
  return absl::OkStatus();
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
