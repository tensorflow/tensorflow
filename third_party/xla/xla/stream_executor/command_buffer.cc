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

#include "xla/stream_executor/command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

CommandBuffer::~CommandBuffer() = default;
CommandBuffer::CommandBuffer(CommandBuffer&&) = default;
CommandBuffer& CommandBuffer::operator=(CommandBuffer&&) = default;

void CommandBuffer::Deleter::operator()(
    internal::CommandBufferInterface* impl) {
  if (owned) delete impl;
}

/*static*/ absl::StatusOr<CommandBuffer> CommandBuffer::Create(
    StreamExecutor* executor, Mode mode) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<internal::CommandBufferInterface> command_buffer,
      executor->implementation()->GetCommandBufferImplementation(mode));

  CommandBuffer cmd(std::move(command_buffer));
  return cmd;
}

/*static*/ absl::StatusOr<CommandBuffer> CommandBuffer::Trace(
    StreamExecutor* executor,
    absl::AnyInvocable<absl::Status(Stream*)> function, Mode mode) {
  Stream stream(executor);
  if (stream.Init(); !stream.ok())
    return absl::InternalError(
        "Failed to initialize stream for command buffer tracing");

  return Trace(executor, &stream, std::move(function), mode);
}

/*static*/ absl::StatusOr<CommandBuffer> CommandBuffer::Trace(
    StreamExecutor* executor, Stream* stream,
    absl::AnyInvocable<absl::Status(Stream*)> function, Mode mode) {
  if (stream == nullptr)
    return absl::InvalidArgumentError(
        "Can't trace command buffer on a null stream");

  // Prepare an empty command buffer instance.
  TF_ASSIGN_OR_RETURN(CommandBuffer command_buffer,
                      CommandBuffer::Create(executor, mode));

  // Trace and finalize the command buffer.
  TF_RETURN_IF_ERROR(command_buffer.implementation()->Trace(
      stream, [&]() { return function(stream); }));
  TF_RETURN_IF_ERROR(command_buffer.implementation()->Finalize());

  return command_buffer;
}

/*static*/ bool CommandBuffer::SupportsConditionalCommands(
    const Platform* platform) {
  // TODO(ezhulenev): We should extend a Platform with a way to query
  // implemented StreamExecutor features, for now we know that only CUDA
  // platform supports conditional commands in command buffers.
#if defined(STREAM_EXECUTOR_CUDA_ENABLE_GRAPH_CONDITIONAL)
  return platform->Name() == "CUDA";
#endif
  return false;
}

const internal::CommandBufferInterface* CommandBuffer::implementation() const {
  return implementation_.get();
}

internal::CommandBufferInterface* CommandBuffer::implementation() {
  return implementation_.get();
}

/*static*/ CommandBuffer CommandBuffer::Create(
    std::unique_ptr<internal::CommandBufferInterface> implementation) {
  return CommandBuffer(std::move(implementation));
}

/*static*/ absl::Status CommandBuffer::Build(
    internal::CommandBufferInterface* implementation,
    const CommandBuffer::Builder& builder) {
  CommandBuffer command_buffer(implementation);
  return builder(&command_buffer);
}

CommandBuffer::CommandBuffer(
    std::unique_ptr<internal::CommandBufferInterface> implementation)
    : implementation_(implementation.release(), {/*owned=*/true}) {}

CommandBuffer::CommandBuffer(internal::CommandBufferInterface* implementation)
    : implementation_(implementation, {/*owned=*/false}) {}

absl::Status CommandBuffer::Barrier(StreamExecutor* executor) {
  return implementation_->Barrier(executor);
}

absl::Status CommandBuffer::Launch(const ThreadDim& threads,
                                   const BlockDim& blocks, const Kernel& kernel,
                                   const KernelArgs& args) {
  return implementation_->Launch(threads, blocks, kernel, args);
}

absl::Status CommandBuffer::AddNestedCommandBuffer(
    const CommandBuffer& nested) {
  return implementation_->AddNestedCommandBuffer(nested);
}

absl::Status CommandBuffer::MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                                 const DeviceMemoryBase& src,
                                                 uint64_t size) {
  return implementation_->MemcpyDeviceToDevice(dst, src, size);
}

absl::Status CommandBuffer::Memset(DeviceMemoryBase* dst,
                                   BitPattern bit_pattern,
                                   size_t num_elements) {
  return implementation_->Memset(dst, bit_pattern, num_elements);
}

absl::StatusOr<DeviceMemoryBase> CommandBuffer::Allocate(size_t bytes) {
  return implementation_->Allocate(bytes);
}

absl::Status CommandBuffer::If(StreamExecutor* executor,
                               DeviceMemory<bool> pred, Builder then_builder) {
  return implementation_->If(executor, pred, std::move(then_builder));
}

absl::Status CommandBuffer::IfElse(StreamExecutor* executor,
                                   DeviceMemory<bool> pred,
                                   Builder then_builder, Builder else_builder) {
  return implementation_->IfElse(executor, pred, std::move(then_builder),
                                 std::move(else_builder));
}

absl::Status CommandBuffer::Case(StreamExecutor* executor,
                                 DeviceMemory<int32_t> index,
                                 std::vector<Builder> branches) {
  return implementation_->Case(executor, index, std::move(branches));
}

absl::Status CommandBuffer::For(StreamExecutor* executor, int32_t num_iteration,
                                DeviceMemory<int32_t> loop_counter,
                                Builder body_builder) {
  return implementation_->For(executor, num_iteration, loop_counter,
                              std::move(body_builder));
}

absl::Status CommandBuffer::While(StreamExecutor* executor,
                                  DeviceMemory<bool> pred, Builder cond_builder,
                                  Builder body_builder) {
  return implementation_->While(executor, pred, std::move(cond_builder),
                                std::move(body_builder));
}

absl::Status CommandBuffer::Free(DeviceMemoryBase dst) {
  return implementation_->Free(dst);
}

CommandBuffer::Mode CommandBuffer::mode() const {
  return implementation_->mode();
}

CommandBuffer::State CommandBuffer::state() const {
  return implementation_->state();
}

absl::Status CommandBuffer::Finalize() { return implementation_->Finalize(); }

absl::Status CommandBuffer::Update() { return implementation_->Update(); }

}  // namespace stream_executor
