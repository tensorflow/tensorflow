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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
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
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"
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

/*static*/ int64_t GpuCommandBuffer::NotifyExecCreated() {
  alive_execs.fetch_add(1, std::memory_order_relaxed);
  return allocated_execs.fetch_add(1, std::memory_order_relaxed);
}

/*static*/ int64_t GpuCommandBuffer::NotifyExecDestroyed() {
  DCHECK_GE(alive_execs.load(std::memory_order_relaxed), 1);
  return alive_execs.fetch_sub(1, std::memory_order_relaxed) - 1;
}

/*static*/ int64_t GpuCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// GpuCommandBuffer implementation
//===----------------------------------------------------------------------===//

GpuCommandBuffer::GpuCommandBuffer(Mode mode, StreamExecutor* parent)
    : mode_(mode), parent_(parent) {}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrier() {
  return barriers_.empty() ? Dependencies{}
                           : Dependencies{barriers_.back().handle};
}

absl::Status GpuCommandBuffer::DisableBarriersExecution(
    GpuCommandBuffer& root_command_buffer) {
  for (GpuGraphBarrierInfo& barrier : barriers_) {
    if (barrier.is_barrier_node) {
      TF_RETURN_IF_ERROR(
          root_command_buffer.SetNodeExecutionEnabled(barrier.handle, false));
    }
  }
  for (ConditionalCommandBuffers& cmd_buffers : conditional_command_buffers_) {
    for (auto& cmd_buffer : cmd_buffers.command_buffers) {
      TF_RETURN_IF_ERROR(
          cmd_buffer->DisableBarriersExecution(root_command_buffer));
    }
  }
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::CheckNotFinalized() {
  if (state_ == State::kFinalized)
    return absl::InternalError(
        "Command can't be added to a command buffer after it was finalized");
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::CheckNumCommandBuffers(
    const ConditionalCommandBuffers& cmd_buffers, size_t num_cmd_buffers) {
  if (cmd_buffers.conditionals.size() != num_cmd_buffers) {
    return absl::InternalError(absl::StrCat(
        "Expected to have ", num_cmd_buffers,
        " conditional command buffers, got ", cmd_buffers.conditionals.size()));
  }
  return absl::OkStatus();
}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrierDependencies() {
  // Collect nodes that will become a new barrier dependencies starting from
  // the first command node added after the last barrier in the scope.
  Dependencies dependencies;
  for (size_t i = barriers_.empty() ? 0 : barriers_.back().nodes_offset;
       i < commands_.size(); ++i) {
    dependencies.push_back(commands_[i]->handle);
  }
  return dependencies;
}

absl::Status GpuCommandBuffer::Barrier() {
  if (state_ == State::kCreate) {
    // Nodes offset for a newly created barrier.
    size_t commands_offset = commands_.size();

    // Collect nodes that will become a new barrier dependencies starting from
    // the first command node added after the last barrier.
    Dependencies dependencies = GetBarrierDependencies();

    // If there are no new dependencies and we have an existing barrier simply
    // copy information from the last barrier to a new one.
    if (dependencies.empty() && !barriers_.empty()) {
      barriers_.push_back({barriers_.back()});
      return absl::OkStatus();
    }

    // If we have only one node added after the last barrier simply reuse the
    // last node corresponding to a command as a barrier.
    if (dependencies.size() == 1) {
      barriers_.push_back({commands_.back()->handle, false, commands_offset});
      return absl::OkStatus();
    }

    // If we have multiple dependencies or no existing barriers we have to
    // create a new empty node acting as an execution barrier.
    TF_ASSIGN_OR_RETURN(auto barrier_handle, CreateBarrierNode(dependencies));
    barriers_.push_back({barrier_handle, true, commands_offset});
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // Command buffer updates can't change the structure of the underlying gpu
    // graph (add or delete barriers). We simply do a sanity check that at
    // update time we didn't try to add more barriers than we had originally.
    if (update_state_.barrier_idx++ >= barriers_.size()) {
      return absl::InternalError("Barrier index out of range");
    }
    return absl::OkStatus();
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::LaunchWithPackedArgs(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& packed_args) {
  CHECK_EQ(kernel.Arity() + (packed_args.number_of_shared_bytes() > 0),
           packed_args.number_of_arguments());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    TF_ASSIGN_OR_RETURN(
        commands_.emplace_back(std::make_unique<GpuCommand>(nullptr))->handle,
        CreateKernelNode(barrier, threads, blocks, kernel, packed_args));
    return absl::OkStatus();
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    return UpdateKernelNode(commands_[update_state_.node_idx++]->handle,
                            threads, blocks, kernel, packed_args);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Launch(const ThreadDim& threads,
                                      const BlockDim& blocks,
                                      const Kernel& kernel,
                                      const KernelArgs& args) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return LaunchWithPackedArgs(threads, blocks, kernel, *packed);
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
    return LaunchWithPackedArgs(threads, blocks, kernel, *packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status GpuCommandBuffer::AddNestedCommandBuffer(
    const CommandBuffer& nested) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    TF_ASSIGN_OR_RETURN(
        commands_.emplace_back(std::make_unique<GpuCommand>(nullptr))->handle,
        CreateChildNode(barrier, nested));
    return absl::OkStatus();
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    GraphNodeHandle node = commands_[update_state_.node_idx++]->handle;
    return UpdateChildNode(node, nested);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                                    const DeviceMemoryBase& src,
                                                    uint64_t size) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    TF_ASSIGN_OR_RETURN(
        commands_.emplace_back(std::make_unique<GpuCommand>(nullptr))->handle,
        CreateMemcpyD2DNode(barrier, *dst, src, size));
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    GraphNodeHandle node = commands_[update_state_.node_idx++]->handle;
    return UpdateMemcpyD2DNode(node, *dst, src, size);
  }

  return UnsupportedStateError(state_);
}

absl::StatusOr<const CommandBuffer::Command*> GpuCommandBuffer::Memset(
    DeviceMemoryBase* dst, BitPattern bit_pattern, size_t num_elements,
    absl::Span<const Command* const> dependencies) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = dependencies.empty()
                               ? GetBarrier()
                               : ToGraphNodeDependencies(dependencies);
    TF_ASSIGN_OR_RETURN(
        GraphNodeHandle handle,
        CreateMemsetNode(barrier, *dst, bit_pattern, num_elements));
    return AppendCommand(handle);
  }

  if (state_ == State::kUpdate) {
    auto& command = commands_[update_state_.node_idx++];
    TF_RETURN_IF_ERROR(Memset(command.get(), dst, bit_pattern, num_elements));
    return command.get();
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

/*static*/ GpuCommandBuffer::ConditionBuilder
GpuCommandBuffer::ToConditionBuilder(Builder builder) {
  return [builder = std::move(builder)](CommandBuffer* cmd_buffer,
                                        GraphConditionalHandle condition) {
    return builder(cmd_buffer);
  };
}

absl::StatusOr<std::vector<GraphConditionalHandle>>
GpuCommandBuffer::CreateConditionalHandles(size_t num_handles) {
  std::vector<GraphConditionalHandle> handles;
  handles.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    TF_ASSIGN_OR_RETURN(handles.emplace_back(), CreateConditionalHandle());
  }
  return handles;
}

absl::StatusOr<std::vector<std::unique_ptr<GpuCommandBuffer>>>
GpuCommandBuffer::CreateConditionalCommandBuffers(
    ConditionType type, absl::Span<const GraphConditionalHandle> conditionals,
    absl::Span<const ConditionBuilder> builders) {
  std::vector<std::unique_ptr<GpuCommandBuffer>> cmd_buffers;
  cmd_buffers.reserve(conditionals.size());

  for (size_t i = 0; i < conditionals.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto command_buffer,
                        CreateConditionalCommandBuffer(type, conditionals[i]));
    TF_RETURN_IF_ERROR(builders[i](command_buffer.get(), conditionals[i]));
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
    cmd_buffers.push_back(std::move(command_buffer));
  }

  return cmd_buffers;
}

absl::Status GpuCommandBuffer::UpdateConditionalCommandBuffers(
    absl::Span<const GraphConditionalHandle> handles,
    absl::Span<const std::unique_ptr<GpuCommandBuffer>> command_buffers,
    absl::Span<const ConditionBuilder> builders) {
  for (size_t i = 0; i < command_buffers.size(); ++i) {
    // Use parent graph executable for conditional command buffer update.
    auto scoped_update_mode = ActivateUpdateMode(command_buffers[i].get());

    // Update command buffer using user-provided builder callback.
    TF_RETURN_IF_ERROR(command_buffers[i]->Update());
    TF_RETURN_IF_ERROR(builders[i](command_buffers[i].get(), handles[i]));
    TF_RETURN_IF_ERROR(command_buffers[i]->Finalize());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuCommandBuffer>>
GpuCommandBuffer::CreateConditionalCommandBuffer(
    ConditionType type, GraphConditionalHandle conditional) {
  TF_ASSIGN_OR_RETURN(auto result,
                      CreateConditionalNode(GetBarrier(), conditional, type));
  commands_.emplace_back(std::make_unique<GpuCommand>(nullptr))->handle =
      result.node_handle;
  return std::move(result.command_buffer);
}

absl::Status GpuCommandBuffer::AddConditionalCommandNode(
    ConditionType type, SetConditionFn set_condition,
    absl::Span<const ConditionBuilder> builders) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Every conditional command buffer is controlled by its own handle.
  size_t num_handles = builders.size();

  if (state_ == State::kCreate) {
    TF_ASSIGN_OR_RETURN(auto handles, CreateConditionalHandles(num_handles));

    // Add a kernel to update conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(handles));

    // Add a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier());

    // Create conditional command buffer for each builder.
    TF_ASSIGN_OR_RETURN(auto cmd_buffers, CreateConditionalCommandBuffers(
                                              type, handles, builders));

    // Keep track of created conditional handles and command buffers.
    conditional_command_buffers_.push_back(
        {std::move(handles), std::move(cmd_buffers)});

    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    ConditionalCommandBuffers& cond_cmd_buffers =
        conditional_command_buffers_[update_state_.conditional_idx++];

    // Sanity check that we got the correct conditional command buffers.
    TF_RETURN_IF_ERROR(CheckNumCommandBuffers(cond_cmd_buffers, num_handles));

    // Update a kernel that updates conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(cond_cmd_buffers.conditionals));

    // Update a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier());

    // Skip updating conditional nodes.
    update_state_.node_idx += num_handles;

    return UpdateConditionalCommandBuffers(
        cond_cmd_buffers.conditionals,
        absl::MakeSpan(cond_cmd_buffers.command_buffers), builders);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::If(DeviceMemory<bool> predicate,
                                  Builder then_builder) {
  auto set_cond_fn = [&](GraphConditionalHandles handles) {
    return LaunchSetIfConditionKernel(handles[0], predicate);
  };

  std::array<ConditionBuilder, 1> builders = {
      ToConditionBuilder(std::move(then_builder))};

  return AddConditionalCommandNode(ConditionType::kIf, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::IfElse(DeviceMemory<bool> predicate,
                                      Builder then_builder,
                                      Builder else_builder) {
  auto set_cond_fn = [&](GraphConditionalHandles handles) {
    return LaunchSetIfElseConditionKernel(handles[0], handles[1], predicate);
  };

  std::array<ConditionBuilder, 2> builders = {
      ToConditionBuilder(std::move(then_builder)),
      ToConditionBuilder(std::move(else_builder))};

  return AddConditionalCommandNode(ConditionType::kIf, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::Case(DeviceMemory<uint8_t> index,
                                    bool index_is_bool,
                                    std::vector<Builder> branches) {
  constexpr size_t kBranchBatchSize = 8;
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

    auto set_cond_fn = [&, batch_offset, enable_conditional_default](
                           GraphConditionalHandles conditionals) {
      return LaunchSetCaseConditionKernel(conditionals, index, index_is_bool,
                                          batch_offset,
                                          enable_conditional_default);
    };

    // Wrap all branches into conditional command buffer builders.
    absl::InlinedVector<ConditionBuilder, kBranchBatchSize> builders;
    builders.reserve(batch_size);
    for (int z = 0; z < batch_size; ++z) {
      int branch_offset = z + batch_offset;
      builders.push_back(
          ToConditionBuilder(std::move(branches[branch_offset])));
    }

    TF_RETURN_IF_ERROR(
        AddConditionalCommandNode(ConditionType::kIf, set_cond_fn, builders));
    batch_offset += batch_size;
  }
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::Case(DeviceMemory<bool> index,
                                    std::vector<Builder> branches) {
  return Case(
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/true, branches);
}

absl::Status GpuCommandBuffer::Case(DeviceMemory<int32_t> index,
                                    std::vector<Builder> branches) {
  return Case(
      DeviceMemory<uint8_t>::MakeFromByteSize(index.opaque(), index.size()),
      /*index_is_bool=*/false, branches);
}

absl::Status GpuCommandBuffer::For(int32_t num_iteration,
                                   DeviceMemory<int32_t> loop_counter,
                                   Builder body_builder) {
  // Reset loop counter to zero.
  TF_ASSIGN_OR_RETURN(auto memset, Memset(&loop_counter, uint32_t{0}, 1, {}));
  (void)memset;  // TODO(ezhulenev): Use memset as a dependency.
  TF_RETURN_IF_ERROR(Barrier());

  auto set_cond_fn = [&](GraphConditionalHandles handles) {
    return LaunchSetForConditionKernel(handles[0], loop_counter, num_iteration);
  };

  auto body = [&](GpuCommandBuffer* body, GraphConditionalHandle conditional) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier());

    // Decide if we want to continue loop iteration.
    return body->LaunchSetForConditionKernel(conditional, loop_counter,
                                             num_iteration);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return AddConditionalCommandNode(ConditionType::kWhile, set_cond_fn,
                                   builders);
}

absl::Status GpuCommandBuffer::While(DeviceMemory<bool> pred,
                                     Builder cond_builder,
                                     Builder body_builder) {
  // Record condition commands into the parent command buffer.
  TF_RETURN_IF_ERROR(cond_builder(this));
  TF_RETURN_IF_ERROR(Barrier());

  auto set_cond_fn = [&](GraphConditionalHandles handles) {
    return LaunchSetWhileConditionKernel(handles[0], pred);
  };

  auto body = [&](GpuCommandBuffer* body, GraphConditionalHandle conditional) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier());
    TF_RETURN_IF_ERROR(cond_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier());
    return body->LaunchSetWhileConditionKernel(conditional, pred);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return AddConditionalCommandNode(ConditionType::kWhile, set_cond_fn,
                                   builders);
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

  // Collect number of nodes and conditionals for logging below.
  size_t num_nodes = commands_.size();
  size_t num_cond_cmd_buffers = conditional_command_buffers_.size();

  if (mode_ == Mode::kPrimary && state_ == State::kCreate) {
    uint64_t start_nanos = tsl::Env::Default()->NowNanos();

    // If this is the first time we finalize command buffer after construction,
    // we need to instantiate it to an executable graph.
    auto instantiated = InstantiateGraph();

    if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Underlying backend ran out of memory trying to instantiate graph "
          "with %d nodes and %d conditionals (total of %d alive graphs "
          "in the process). You can try to (a) Give more memory to the "
          "driver by reducing XLA_CLIENT_MEM_FRACTION (b) Disable "
          "command buffers with 'XLA_FLAGS=--xla_gpu_enable_command_buffer=' "
          "(empty set). Original error: %s",
          num_nodes, num_cond_cmd_buffers, AliveExecs(),
          instantiated.message()));
    }
    TF_RETURN_IF_ERROR(instantiated);

    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    auto exec_num = NotifyExecCreated();
    VLOG(5) << "Instantiated executable graph #" << exec_num << " in "
            << (end_nanos - start_nanos) / 1000 << " μs"
            << "; nodes: " << num_nodes
            << "; conditionals: " << num_cond_cmd_buffers
            << "; alive executable graphs: " << AliveExecs();

    if (parent_->GetPlatform()->id() == cuda::kCudaPlatformId &&
        parent_->GetDeviceDescription().driver_version() <
            SemanticVersion(12, 4, 0)) {
      TF_RETURN_IF_ERROR(DisableBarriersExecution(*this));
    }

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

absl::Span<const std::unique_ptr<GpuCommandBuffer::GpuCommand>>
GpuCommandBuffer::commands() const {
  return commands_;
}

absl::Span<const GpuCommandBuffer::GpuGraphBarrierInfo>
GpuCommandBuffer::barriers() const {
  return barriers_;
}

absl::Status GpuCommandBuffer::Submit(Stream* stream) {
  if (mode_ != CommandBuffer::Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  return LaunchGraph(stream);
}

}  // namespace stream_executor::gpu
