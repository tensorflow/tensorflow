/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

class Stream;
class StreamExecutor;

namespace internal {
class CommandBufferInterface;
}

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
  using Builder = std::function<tsl::Status(CommandBuffer*)>;

  ~CommandBuffer();
  CommandBuffer(CommandBuffer&&);
  CommandBuffer& operator=(CommandBuffer&&);

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

  // Creates a new empty command buffer on the given executor.
  static tsl::StatusOr<CommandBuffer> Create(StreamExecutor* executor,
                                             Mode mode = Mode::kPrimary);

  // Creates a new command buffer on the given executor by tracing `function`
  // invocation. All StreamExecutor operations on a Stream argument will be
  // recorded into the command buffer. Returned command buffer is finalized, and
  // can't be updated.
  //
  // Command buffer tracing should be used only when it is impossible to use
  // explicit construction APIs, e.g. when calling external libraries.
  static tsl::StatusOr<CommandBuffer> Trace(
      StreamExecutor* executor,
      absl::AnyInvocable<tsl::Status(Stream*)> function,
      Mode mode = Mode::kPrimary);

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  // Adds a kernel launch command to the command buffer.
  tsl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                     const Kernel& kernel, const KernelArgs& args);

  // Adds a nested command buffer to the command buffer.
  tsl::Status AddNestedCommandBuffer(const CommandBuffer& nested);

  // Adds a device-to-device memory copy to the command buffer.
  tsl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                   const DeviceMemoryBase& src, uint64_t size);

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if predicate is true. Builder should not call `Update` or
  // `Finalize` on command buffer argument, parent command buffer is responsible
  // for updating and finalizing conditional command buffers.
  tsl::Status If(StreamExecutor* executor, DeviceMemory<bool> pred,
                 Builder then_builder);

  // Finalizes command buffer and makes it executable. Once command buffer is
  // finalized no commands can be added to it.
  tsl::Status Finalize();

  // Begins command buffer update. Command buffer update should be finalized
  // before it can be executed.
  tsl::Status Update();

  // Type-safe wrapper for launching typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  tsl::Status Launch(const TypedKernel<Params...>& kernel,
                     const ThreadDim& threads, const BlockDim& blocks,
                     Args... args);

  // Returns command buffer execution mode.
  Mode mode() const;

  // Returns command buffer state.
  State state() const;

  //===--------------------------------------------------------------------===//
  // Semi-internal APIs
  //===--------------------------------------------------------------------===//

  // Following APIs are public, but considered to be implementation detail and
  // discouraged from uses outside of StreamExecutor package.
  const internal::CommandBufferInterface* implementation() const;
  internal::CommandBufferInterface* implementation();

  // Wraps platform-specific command buffer implementation into a top-level
  // StreamExecutor command buffer.
  static CommandBuffer Wrap(
      std::unique_ptr<internal::CommandBufferInterface> implementation);

 private:
  explicit CommandBuffer(
      std::unique_ptr<internal::CommandBufferInterface> implementation);

  std::unique_ptr<internal::CommandBufferInterface> implementation_;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
inline tsl::Status CommandBuffer::Launch(const TypedKernel<Params...>& kernel,
                                         const ThreadDim& threads,
                                         const BlockDim& blocks, Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(Launch(threads, blocks, kernel, *kernel_args));
  return tsl::OkStatus();
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
