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

#include "xla/stream_executor/gpu/gpu_command_buffer.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/path.h"

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
// Implementation details device kernels required by GpuCommandBuffer.
//===----------------------------------------------------------------------===//

using Mode = CommandBuffer::Mode;
using State = CommandBuffer::State;
using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;
using GraphConditionalHandle = GpuCommandBuffer::GraphConditionalHandle;
using GraphConditionalHandles = absl::Span<const GraphConditionalHandle>;

namespace {
absl::string_view to_string(State state) {
  switch (state) {
    case State::kCreate:
      return "create";
    case State::kUpdate:
      return "update";
    case State::kFinalized:
      return "finalized";
  }
}

absl::Status UnsupportedStateError(State state) {
  return absl::InternalError(
      absl::StrCat("Unsupported command buffer state: ", to_string(state)));
}
}  // namespace

//===----------------------------------------------------------------------===//
// GpuCommandBuffer resource usage tracking
//===----------------------------------------------------------------------===//

static std::atomic<int64_t> allocated_execs(0);
static std::atomic<int64_t> alive_execs(0);

int64_t GpuCommandBuffer::NotifyExecCreated() {
  alive_execs.fetch_add(1, std::memory_order_relaxed);
  return allocated_execs.fetch_add(1, std::memory_order_relaxed);
}

int64_t GpuCommandBuffer::NotifyExecDestroyed() {
  DCHECK_GE(alive_execs.load(std::memory_order_relaxed), 1);
  return alive_execs.fetch_sub(1, std::memory_order_relaxed) - 1;
}

int64_t GpuCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// GpuCommandBuffer implementation
//===----------------------------------------------------------------------===//

GpuCommandBuffer::GpuCommandBuffer(Mode mode, StreamExecutor* parent)
    : mode_(mode), parent_(parent) {}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetAutoDependencies() const {
  if (commands_.empty()) return Dependencies{};

  const Command* command = commands_.back().get();

  if (auto* gpu_command = dynamic_cast<const GpuCommand*>(command)) {
    return Dependencies{gpu_command->handle};
  }

  if (auto* gpu_command = dynamic_cast<const GpuIfCommand*>(command)) {
    return Dependencies{gpu_command->then_conditional_node.handle};
  }

  if (auto* gpu_command = dynamic_cast<const GpuIfElseCommand*>(command)) {
    return Dependencies{gpu_command->then_conditional_node.handle,
                        gpu_command->else_conditional_node.handle};
  }

  if (auto* gpu_command = dynamic_cast<const GpuCaseCommand*>(command)) {
    Dependencies dependencies;
    for (const auto& conditional_node : gpu_command->conditional_nodes) {
      dependencies.push_back(conditional_node.handle);
    }
    return dependencies;
  }

  if (auto* gpu_command = dynamic_cast<const GpuWhileCommand*>(command)) {
    return Dependencies{gpu_command->conditional_node.handle};
  }

  CHECK(false) << "Unsupported command type";  // Crash OK
  return Dependencies{};
}

absl::Status GpuCommandBuffer::CheckNotFinalized() {
  if (state_ == State::kFinalized)
    return absl::InternalError(
        "Command can't be added to a command buffer after it was finalized");
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*>
GpuCommandBuffer::LaunchWithPackedArgs(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& packed_args,
    absl::Span<const Command* const> dependencies) {
  CHECK_EQ(kernel.Arity() + (packed_args.number_of_shared_bytes() > 0),
           packed_args.number_of_arguments());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);
    TF_ASSIGN_OR_RETURN(
        GraphNodeHandle handle,
        CreateKernelNode(barrier, threads, blocks, kernel, packed_args));
    return AppendCommand(handle);
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(
        LaunchWithPackedArgs(&command, threads, blocks, kernel, packed_args));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::LaunchWithPackedArgs(
    const Command* command, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgsPackedArrayBase& packed_args) {
  auto* gpu_command = tsl::down_cast<const GpuCommand*>(command);
  return UpdateKernelNode(gpu_command->handle, threads, blocks, kernel,
                          packed_args);
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Launch(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgs& args, absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return LaunchWithPackedArgs(threads, blocks, kernel, *packed, dependencies);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return LaunchWithPackedArgs(threads, blocks, kernel, *packed, dependencies);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status GpuCommandBuffer::Launch(const Command* command,
                                      const ThreadDim& threads,
                                      const BlockDim& blocks,
                                      const Kernel& kernel,
                                      const KernelArgs& args) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return LaunchWithPackedArgs(command, threads, blocks, kernel, *packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return LaunchWithPackedArgs(command, threads, blocks, kernel, *packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::StatusOr<const CommandBuffer::Command*>
GpuCommandBuffer::AddNestedCommandBuffer(
    const CommandBuffer& nested,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);
    TF_ASSIGN_OR_RETURN(GraphNodeHandle handle,
                        CreateChildNode(barrier, nested));
    return AppendCommand(handle);
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(AddNestedCommandBuffer(&command, nested));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::AddNestedCommandBuffer(
    const Command* command, const CommandBuffer& nested) {
  auto* gpu_command = tsl::down_cast<const GpuCommand*>(command);
  return UpdateChildNode(gpu_command->handle, nested);
}

absl::StatusOr<const CommandBuffer::Command*>
GpuCommandBuffer::MemcpyDeviceToDevice(
    DeviceMemoryBase* dst, const DeviceMemoryBase& src, uint64_t size,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);
    TF_ASSIGN_OR_RETURN(GraphNodeHandle handle,
                        CreateMemcpyD2DNode(barrier, *dst, src, size));
    return AppendCommand(handle);
  }

  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(MemcpyDeviceToDevice(&command, dst, src, size));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::MemcpyDeviceToDevice(const Command* command,
                                                    DeviceMemoryBase* dst,
                                                    const DeviceMemoryBase& src,
                                                    uint64_t size) {
  auto* gpu_command = tsl::down_cast<const GpuCommand*>(command);
  return UpdateMemcpyD2DNode(gpu_command->handle, *dst, src, size);
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Memset(
    DeviceMemoryBase* dst, BitPattern bit_pattern, size_t num_elements,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);
    TF_ASSIGN_OR_RETURN(
        GraphNodeHandle handle,
        CreateMemsetNode(barrier, *dst, bit_pattern, num_elements));
    return AppendCommand(handle);
  }

  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(Memset(&command, dst, bit_pattern, num_elements));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Memset(const Command* command,
                                      DeviceMemoryBase* dst,
                                      const BitPattern& bit_pattern,
                                      size_t num_elements) {
  auto* gpu_command = tsl::down_cast<const GpuCommand*>(command);
  return UpdateMemsetNode(gpu_command->handle, *dst, bit_pattern, num_elements);
}

//--------------------------------------------------------------------------//
// Command buffer condtitional commands API
//--------------------------------------------------------------------------//

absl::StatusOr<std::vector<GraphConditionalHandle>>
GpuCommandBuffer::CreateConditionalHandles(size_t num_handles) {
  std::vector<GraphConditionalHandle> handles;
  handles.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    TF_ASSIGN_OR_RETURN(handles.emplace_back(), CreateConditionalHandle());
  }
  return handles;
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Case(
    DeviceMemory<uint8_t> index, bool index_is_bool,
    std::vector<Builder> branches,
    absl::Span<const Command* const> dependencies) {
  constexpr size_t kBranchBatchSize = 8;

  if (state_ == State::kCreate) {
    GpuCaseCommand command = {};

    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);

    int32_t batch_offset = 0;
    while (batch_offset < branches.size()) {
      // Conditionals will by default run branches[branchs.size()-1] if index is
      // `< 0` or `>= branches.size()`. See
      // https://openxla.org/xla/operation_semantics#conditional.
      // To break down a large case with back to back ConditionalCommands, only
      // the last batch should accept this default case.
      int32_t remaining_branches = branches.size() - batch_offset;
      int32_t batch_size;
      bool enable_conditional_default;
      if (remaining_branches <= kBranchBatchSize) {
        batch_size = remaining_branches;
        enable_conditional_default = true;
      } else {
        batch_size = kBranchBatchSize;
        enable_conditional_default = false;
      }

      TF_ASSIGN_OR_RETURN(auto conditionals,
                          CreateConditionalHandles(batch_size));

      TF_ASSIGN_OR_RETURN(auto set_condition_node,
                          CreateSetCaseConditionNode(
                              conditionals, index, index_is_bool, batch_offset,
                              enable_conditional_default, barrier));

      std::vector<GraphConditionalNodeHandle> conditional_nodes;
      for (int z = 0; z < batch_size; ++z) {
        int branch_offset = z + batch_offset;
        TF_ASSIGN_OR_RETURN(
            conditional_nodes.emplace_back(),
            CreateConditionalNode({set_condition_node}, conditionals[z],
                                  ConditionType::kIf));

        GpuCommandBuffer* case_command_buffer =
            conditional_nodes.back().command_buffer.get();
        TF_RETURN_IF_ERROR(branches[branch_offset](case_command_buffer));
        TF_RETURN_IF_ERROR(case_command_buffer->Finalize());
      }

      // Move the state into the recorded command.
      command.conditionals.insert(command.conditionals.end(),
                                  conditionals.begin(), conditionals.end());
      command.set_condition_nodes.push_back(set_condition_node);
      command.conditional_nodes.insert(
          command.conditional_nodes.end(),
          std::make_move_iterator(conditional_nodes.begin()),
          std::make_move_iterator(conditional_nodes.end()));

      batch_offset += batch_size;
    }

    return AppendCommand(std::move(command));
  }

  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(Case(&command, index, index_is_bool, branches));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Case(const Command* command,
                                    DeviceMemory<uint8_t> index,
                                    bool index_is_bool,
                                    std::vector<Builder> branches) {
  constexpr size_t kBranchBatchSize = 8;

  auto* gpu_command = tsl::down_cast<const GpuCaseCommand*>(command);

  // Update branch conditionals.
  size_t batch_index = 0;
  int32_t batch_offset = 0;
  while (batch_offset < branches.size()) {
    int32_t remaining_branches = branches.size() - batch_offset;
    int32_t batch_size;
    bool enable_conditional_default;
    if (remaining_branches <= kBranchBatchSize) {
      batch_size = remaining_branches;
      enable_conditional_default = true;
    } else {
      batch_size = kBranchBatchSize;
      enable_conditional_default = false;
    }

    TF_RETURN_IF_ERROR(UpdateSetCaseConditionNode(
        gpu_command->set_condition_nodes[batch_index],
        absl::MakeSpan(gpu_command->conditionals)
            .subspan(batch_offset, batch_size),
        index, index_is_bool, batch_offset, enable_conditional_default));

    batch_offset += batch_size;
    batch_index += 1;
  }

  // Update branch command buffers.
  for (size_t i = 0; i < gpu_command->conditional_nodes.size(); ++i) {
    GpuCommandBuffer* case_command_buffer =
        gpu_command->conditional_nodes[i].command_buffer.get();
    auto scoped_update_mode = ActivateUpdateMode(case_command_buffer);
    TF_RETURN_IF_ERROR(case_command_buffer->Update());
    TF_RETURN_IF_ERROR(branches[i](case_command_buffer));
    TF_RETURN_IF_ERROR(case_command_buffer->Finalize());
  }

  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Case(
    DeviceMemory<int32_t> index, std::vector<Builder> branches,
    absl::Span<const Command* const> dependencies) {
  return Case(
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/false, branches, dependencies);
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Case(
    DeviceMemory<bool> index, std::vector<Builder> branches,
    absl::Span<const Command* const> dependencies) {
  return Case(
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/true, branches, dependencies);
}

absl::Status GpuCommandBuffer::Case(const Command* command,
                                    DeviceMemory<int32_t> index,
                                    std::vector<Builder> branches) {
  return Case(
      command,
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/false, branches);
}

absl::Status GpuCommandBuffer::Case(const Command* command,
                                    DeviceMemory<bool> index,
                                    std::vector<Builder> branches) {
  return Case(
      command,
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/true, branches);
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::While(
    DeviceMemory<bool> pred, Builder cond_builder, Builder body_builder,
    absl::Span<const Command* const> dependencies) {
  if (state_ == State::kCreate) {
    GpuWhileCommand command = {};

    Dependencies barrier = dependencies.empty()
                               ? GetAutoDependencies()
                               : ToGraphNodeDependencies(dependencies);

    // TODO(ezhulenev): cond_builder should be able to take dependencies.
    (void)barrier;

    TF_RETURN_IF_ERROR(cond_builder(this));

    TF_ASSIGN_OR_RETURN(command.conditional, CreateConditionalHandle());
    TF_ASSIGN_OR_RETURN(command.set_init_condition_node,
                        CreateSetWhileConditionNode(command.conditional, pred,
                                                    GetAutoDependencies()));
    TF_ASSIGN_OR_RETURN(
        command.conditional_node,
        CreateConditionalNode({command.set_init_condition_node},
                              command.conditional, ConditionType::kWhile));

    GpuCommandBuffer* body = command.conditional_node.command_buffer.get();
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(cond_builder(body));
    TF_ASSIGN_OR_RETURN(
        command.set_body_condition_node,
        body->CreateSetWhileConditionNode(command.conditional, pred,
                                          body->GetAutoDependencies()));
    TF_RETURN_IF_ERROR(command.conditional_node.command_buffer->Finalize());

    return AppendCommand(std::move(command));
  }

  if (state_ == State::kUpdate) {
    Command& command = *commands_[update_state_.command_idx++];
    TF_RETURN_IF_ERROR(While(&command, pred, cond_builder, body_builder));
    return &command;
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::While(const Command* command,
                                     DeviceMemory<bool> pred,
                                     Builder cond_builder,
                                     Builder body_builder) {
  auto* gpu_command = tsl::down_cast<const GpuWhileCommand*>(command);

  TF_RETURN_IF_ERROR(cond_builder(this));

  TF_RETURN_IF_ERROR(UpdateSetWhileConditionNode(
      gpu_command->set_init_condition_node, gpu_command->conditional, pred));

  GpuCommandBuffer* body = gpu_command->conditional_node.command_buffer.get();
  auto body_update_mode = ActivateUpdateMode(body);

  // Update command buffer using user-provided builder callback.
  TF_RETURN_IF_ERROR(body->Update());
  TF_RETURN_IF_ERROR(body_builder(body));
  TF_RETURN_IF_ERROR(cond_builder(body));
  TF_RETURN_IF_ERROR(body->UpdateSetWhileConditionNode(
      gpu_command->set_body_condition_node, gpu_command->conditional, pred));
  TF_RETURN_IF_ERROR(body->Finalize());

  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
  TF_RETURN_IF_ERROR(PrepareFinalization());

  // Maybe dump created GPU graph to a dot file for debugging.
  if (state_ == State::kCreate && VLOG_IS_ON(10)) {
    std::string path = tsl::io::GetTempFilename(/*extension=*/"dot");
    TF_RETURN_IF_ERROR(WriteGraphToDotFile(path));
    if (VLOG_IS_ON(100)) {
      std::string dot_file_contents;
      TF_RETURN_IF_ERROR(
          tsl::ReadFileToString(tsl::Env::Default(), path, &dot_file_contents));
      VLOG(100) << "Contents of " << path << " is:\n" << dot_file_contents;
    }
  }

  size_t num_commands = commands_.size();

  if (mode_ == Mode::kPrimary && state_ == State::kCreate) {
    uint64_t start_nanos = tsl::Env::Default()->NowNanos();

    // If this is the first time we finalize command buffer after construction,
    // we need to instantiate it to an executable graph.
    auto instantiated = InstantiateGraph();

    if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Underlying backend ran out of memory trying to instantiate command "
          "buffer with %d (total of %d alive graphs in the process). You can "
          "try to (a) Give more memory to the driver by reducing "
          "XLA_CLIENT_MEM_FRACTION (b) Disable command buffers with "
          "'XLA_FLAGS=--xla_gpu_enable_command_buffer=' (empty set). Original "
          "error: %s",
          num_commands, AliveExecs(), instantiated.message()));
    }
    TF_RETURN_IF_ERROR(instantiated);

    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    auto exec_num = NotifyExecCreated();
    VLOG(5) << "Instantiated executable graph #" << exec_num << " in "
            << (end_nanos - start_nanos) / 1000 << " μs"
            << "; commands: " << num_commands
            << "; alive executable graphs: " << AliveExecs();

  } else if (mode_ == Mode::kPrimary && state_ == State::kUpdate) {
    // If this is a finalization after update, we don't have to do anything as
    // each individual command already updated executable graph.
    VLOG(5) << "Finalize executable graph of command buffer " << this
            << " update #" << num_updates_++ << " "
            << "(alive executable graphs: " << AliveExecs() << ")";

  } else if (mode_ == Mode::kNested) {
    // Nested command buffers do not have executable graphs.
    VLOG(5) << "Finalize nested command buffer without instantiating "
               "executable graph";
  }

  state_ = State::kFinalized;
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::Update() {
  TF_RETURN_IF_ERROR(CheckCanBeUpdated());

  if (state_ != State::kFinalized) {
    return absl::InternalError(
        "Command buffer has to be finalized first before it can be updated");
  }

  VLOG(5) << "Begin update of "
          << (mode_ == Mode::kPrimary ? "primary" : "nested")
          << " command buffer " << this;

  state_ = State::kUpdate;
  update_state_ = UpdateState();
  return absl::OkStatus();
}

absl::Span<const std::unique_ptr<CommandBuffer::Command>>
GpuCommandBuffer::commands() const {
  return commands_;
}

absl::Status GpuCommandBuffer::Submit(Stream* stream) {
  if (mode_ != CommandBuffer::Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  return LaunchGraph(stream);
}

}  // namespace stream_executor::gpu
