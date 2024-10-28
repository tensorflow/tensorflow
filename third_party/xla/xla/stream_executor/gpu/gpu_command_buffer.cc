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
#include <string_view>
#include <utility>
#include <vector>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
// Implementation details device kernels required by GpuCommandBuffer.
//===----------------------------------------------------------------------===//

using Mode = CommandBuffer::Mode;
using State = CommandBuffer::State;
using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

std::string_view to_string(State state) {
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

//===----------------------------------------------------------------------===//
// GpuCommandBuffer resource usage tracking
//===----------------------------------------------------------------------===//

static std::atomic<int64_t> allocated_execs(0);
static std::atomic<int64_t> alive_execs(0);

static int64_t NotifyExecCreated() {
  alive_execs.fetch_add(1, std::memory_order_relaxed);
  return allocated_execs.fetch_add(1, std::memory_order_relaxed);
}

static int64_t NotifyExecDestroyed() {
  DCHECK_GE(alive_execs.load(std::memory_order_relaxed), 1);
  return alive_execs.fetch_sub(1, std::memory_order_relaxed) - 1;
}

/*static*/ int64_t GpuCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// GpuCommandBuffer implementation
//===----------------------------------------------------------------------===//

static std::string_view ModeToString(CommandBuffer::Mode mode) {
  switch (mode) {
    case CommandBuffer::Mode::kPrimary:
      return "primary";
    case CommandBuffer::Mode::kNested:
      return "nested";
  }
}

GpuCommandBuffer::GpuCommandBuffer(Mode mode, GpuExecutor* parent,
                                   GpuGraphHandle graph, bool is_owned_graph)
    : mode_(mode),
      parent_(parent),
      graph_(graph),
      is_owned_graph_(is_owned_graph) {
  VLOG(5) << "Created command buffer for graph " << graph_
          << "; mode=" << ModeToString(mode)
          << "; is_owned_graph=" << is_owned_graph_;
  execution_scopes_.try_emplace(kDefaulExecutionScope);
}

GpuCommandBuffer::~GpuCommandBuffer() {
  if (exec_ != nullptr && is_owned_graph_exec_) {
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << NotifyExecDestroyed()
            << ")";
    if (auto status = GpuDriver::DestroyGraphExec(exec_); !status.ok()) {
      LOG(ERROR) << "Failed to destroy GPU graph exec: " << status.message();
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status = GpuDriver::DestroyGraph(graph_); !status.ok()) {
      LOG(ERROR) << "Failed to destroy GPU graph: " << status.message();
    }
  }
}

GpuCommandBuffer::ScopedGpuGraphExec::ScopedGpuGraphExec(
    GpuCommandBuffer* cmd_buffer, GpuGraphExecHandle exec)
    : cmd_buffer(cmd_buffer),
      restore(cmd_buffer->exec_),
      restore_is_owned(cmd_buffer->is_owned_graph_exec_) {
  cmd_buffer->exec_ = exec;
  cmd_buffer->is_owned_graph_exec_ = false;
}

GpuCommandBuffer::ScopedGpuGraphExec::~ScopedGpuGraphExec() {
  cmd_buffer->exec_ = restore;
  cmd_buffer->is_owned_graph_exec_ = restore_is_owned;
}

// Converts a platform independent GraphNodeHandle into a platform specific
// GpuGraphNodeHandle. This function will be removed once all
// Node factory functions have been migrated into the subclasses.
static GpuGraphNodeHandle ToPlatformSpecificHandle(
    GpuCommandBuffer::GraphNodeHandle handle) {
  return absl::bit_cast<GpuGraphNodeHandle>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// platform specific GpuGraphNodeHandles. This function will be removed once
// all Node factory functions have been migrated into the subclasses.
static std::vector<GpuGraphNodeHandle> ToPlatformSpecificHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<GpuGraphNodeHandle> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToPlatformSpecificHandle(opaque_handle));
  }
  return handles;
}

// Converts a platform specific GpuGraphNodeHandle into a platform independent
// GraphNodeHandle. This function will be removed once all Node factory
// functions have been migrated into the subclasses.
static GpuCommandBuffer::GraphNodeHandle FromPlatformSpecificHandle(
    GpuGraphNodeHandle handle) {
  return absl::bit_cast<GpuCommandBuffer::GraphNodeHandle>(handle);
}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrier(
    ExecutionScopeId execution_scope_id) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
  return execution_scope.barriers.empty()
             ? Dependencies{}
             : Dependencies{execution_scope.barriers.back().handle};
}

absl::Status GpuCommandBuffer::DisableBarriersExecution(
    GpuGraphExecHandle exec) {
#if !defined(TENSORFLOW_USE_ROCM)
  ExecutionScope& execution_scope = execution_scopes_[kDefaulExecutionScope];

  for (GpuGraphBarrierInfo& barrier : execution_scope.barriers) {
    if (barrier.is_barrier_node) {
      TF_RETURN_IF_ERROR(GpuDriver::GraphNodeSetEnabled(
          exec, ToPlatformSpecificHandle(barrier.handle), false));
    }
  }
  for (ConditionalCommandBuffers& cmd_buffers :
       execution_scope.conditional_command_buffers) {
    for (auto& cmd_buffer : cmd_buffers.command_buffers) {
      TF_RETURN_IF_ERROR(cmd_buffer->DisableBarriersExecution(exec));
    }
  }
#endif  // TENSORFLOW_USE_ROCM
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
  if (cmd_buffers.handles.size() != num_cmd_buffers) {
    return absl::InternalError(absl::StrCat(
        "Expected to have ", num_cmd_buffers,
        " conditional command buffers, got ", cmd_buffers.handles.size()));
  }
  return absl::OkStatus();
}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrierDependencies(
    ExecutionScopeId execution_scope_id) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
  auto& barriers = execution_scope.barriers;

  // Collect nodes that will become a new barrier dependencies starting from
  // the first command node added after the last barrier in the scope.
  Dependencies dependencies;
  for (size_t i = barriers.empty() ? 0 : barriers.back().nodes_offset;
       i < execution_scope.nodes.size(); ++i) {
    dependencies.push_back(execution_scope.nodes[i].handle);
  }
  return dependencies;
}

absl::Status GpuCommandBuffer::Barrier(ExecutionScopeId execution_scope_id) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  if (state_ == State::kCreate) {
    // Nodes offset for a newly created barrier.
    size_t nodes_offset = execution_scope.nodes.size();

    // Collect nodes that will become a new barrier dependencies starting from
    // the first command node added after the last barrier.
    Dependencies dependencies = GetBarrierDependencies(execution_scope_id);

    // If there are no new dependencies and we have an existing barrier simply
    // copy information from the last barrier to a new one.
    if (dependencies.empty() && !execution_scope.barriers.empty()) {
      execution_scope.barriers.push_back({execution_scope.barriers.back()});
      return absl::OkStatus();
    }

    // If we have only one node added after the last barrier simply reuse the
    // last node corresponding to a command as a barrier.
    if (dependencies.size() == 1) {
      execution_scope.barriers.push_back(
          {execution_scope.nodes.back().handle, false, nodes_offset});
      return absl::OkStatus();
    }

    // If we have multiple dependencies or no existing barriers we have to
    // create a new empty node acting as an execution barrier.
    TF_ASSIGN_OR_RETURN(auto barrier_handle, CreateBarrierNode(dependencies));
    execution_scope.barriers.push_back({barrier_handle, true, nodes_offset});
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // Command buffer updates can't change the structure of the underlying gpu
    // graph (add or delete barriers). We simply do a sanity check that at
    // update time we didn't try to add more barriers than we had originally.
    if (execution_scope.update_state.barrier_idx++ >=
        execution_scope.barriers.size()) {
      return absl::InternalError(
          absl::StrFormat("Execution scope %d barrier index out of range",
                          execution_scope_id.value()));
    }
    return absl::OkStatus();
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Barrier(
    absl::Span<const ExecutionScopeId> execution_scope_ids) {
  // Nothing to synchronize here.
  if (execution_scope_ids.empty()) return absl::OkStatus();

  // Do not create two-level barriers for single execution scope.
  if (execution_scope_ids.size() == 1) {
    return Barrier(execution_scope_ids[0]);
  }

  // Add a new barrier to every synchronized execution scope.
  for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
    TF_RETURN_IF_ERROR(Barrier(execution_scope_id));
  }

  if (state_ == State::kCreate) {
    // Collect barriers from each scope as a dependencies.
    Dependencies dependencies;
    for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
      ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
      dependencies.push_back(execution_scope.barriers.back().handle);
    }

    // Create a new barrier that joins all per-scope barriers together.
    TF_ASSIGN_OR_RETURN(auto barrier_handle, CreateBarrierNode(dependencies));

    // Broadcast new barrier to all participating execution scopes.
    for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
      ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
      size_t nodes_offset = execution_scope.nodes.size();
      execution_scope.barriers.push_back({barrier_handle, true, nodes_offset});
    }

    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // Command buffer updates can't change the structure of the underlying gpu
    // graph (add or delete barriers). We simply do a sanity check that at
    // update time we didn't try to add more barriers than we had originally.
    for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
      ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
      if (execution_scope.update_state.barrier_idx++ >=
          execution_scope.barriers.size()) {
        return absl::InternalError(
            absl::StrFormat("Execution scope %d barrier index out of range",
                            execution_scope_id.value()));
      }
    }
    return absl::OkStatus();
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Barrier(ExecutionScopeId from_execution_scope_id,
                                       ExecutionScopeId to_execution_scope_id) {
  // If scopes are the same simply add a barrier to it.
  if (from_execution_scope_id == to_execution_scope_id) {
    return Barrier(from_execution_scope_id);
  }

  // Create new barriers in both execution scopes.
  TF_RETURN_IF_ERROR(Barrier(from_execution_scope_id));
  TF_RETURN_IF_ERROR(Barrier(to_execution_scope_id));

  if (state_ == State::kCreate) {
    // Collect barriers from each scope as dependencies.
    Dependencies dependencies = {
        execution_scopes_[from_execution_scope_id].barriers.back().handle,
        execution_scopes_[to_execution_scope_id].barriers.back().handle};

    // Create a new barrier that joins `from` and `to` scopes.
    TF_ASSIGN_OR_RETURN(auto barrier_handle, CreateBarrierNode(dependencies));

    // Add a new barrier only to the `to_execution_scope_id`.
    ExecutionScope& execution_scope = execution_scopes_[to_execution_scope_id];
    size_t nodes_offset = execution_scope.nodes.size();
    execution_scope.barriers.push_back({barrier_handle, true, nodes_offset});

    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // Command buffer updates can't change the structure of the underlying gpu
    // graph (add or delete barriers). We simply do a sanity check that at
    // update time we didn't try to add more barriers than we had originally.
    ExecutionScope& execution_scope = execution_scopes_[to_execution_scope_id];
    if (execution_scope.update_state.barrier_idx++ >=
        execution_scope.barriers.size()) {
      return absl::InternalError(
          absl::StrFormat("Execution scope %d barrier index out of range",
                          to_execution_scope_id.value()));
    }
    return absl::OkStatus();
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::LaunchWithPackedArgs(
    ExecutionScopeId execution_scope_id, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& packed_args) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  CHECK_EQ(kernel.Arity() + (packed_args.number_of_shared_bytes() > 0),
           packed_args.number_of_arguments());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    TF_ASSIGN_OR_RETURN(
        execution_scope.nodes.emplace_back().handle,
        CreateKernelNode(barrier, threads, blocks, kernel, packed_args));
    return absl::OkStatus();
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    return UpdateKernelNode(
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle,
        threads, blocks, kernel, packed_args);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Launch(ExecutionScopeId execution_scope_id,
                                      const ThreadDim& threads,
                                      const BlockDim& blocks,
                                      const Kernel& kernel,
                                      const KernelArgs& args) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return LaunchWithPackedArgs(execution_scope_id, threads, blocks, kernel,
                                *packed);
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
    return LaunchWithPackedArgs(execution_scope_id, threads, blocks, kernel,
                                *packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status GpuCommandBuffer::AddNestedCommandBuffer(
    ExecutionScopeId execution_scope_id, const CommandBuffer& nested) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    TF_ASSIGN_OR_RETURN(execution_scope.nodes.emplace_back().handle,
                        CreateChildNode(barrier, nested));
    return absl::OkStatus();
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    GraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return UpdateChildNode(node, nested);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::MemcpyDeviceToDevice(
    ExecutionScopeId execution_scope_id, DeviceMemoryBase* dst,
    const DeviceMemoryBase& src, uint64_t size) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    TF_ASSIGN_OR_RETURN(execution_scope.nodes.emplace_back().handle,
                        CreateMemcpyD2DNode(barrier, *dst, src, size));
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    GraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return UpdateMemcpyD2DNode(node, *dst, src, size);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Memset(ExecutionScopeId execution_scope_id,
                                      DeviceMemoryBase* dst,
                                      BitPattern bit_pattern,
                                      size_t num_elements) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    TF_ASSIGN_OR_RETURN(
        execution_scope.nodes.emplace_back().handle,
        CreateMemsetNode(barrier, *dst, bit_pattern, num_elements));
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    GraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return UpdateMemsetNode(node, *dst, bit_pattern, num_elements);
  }

  return UnsupportedStateError(state_);
}

//--------------------------------------------------------------------------//
// Command buffer condtitional commands API
//--------------------------------------------------------------------------//

using ConditionalHandles = absl::Span<const GpuGraphConditionalHandle>;

/*static*/ GpuCommandBuffer::ConditionBuilder
GpuCommandBuffer::ToConditionBuilder(Builder builder) {
  return [builder = std::move(builder)](CommandBuffer* cmd_buffer,
                                        GpuGraphConditionalHandle) {
    return builder(cmd_buffer);
  };
}

absl::StatusOr<std::vector<GpuGraphConditionalHandle>>
GpuCommandBuffer::CreateConditionalHandles(size_t num_handles) {
  std::vector<GpuGraphConditionalHandle> handles;
  for (size_t i = 0; i < num_handles; ++i) {
    TF_RETURN_IF_ERROR(GpuDriver::GraphConditionalHandleCreate(
        &handles.emplace_back(), graph_, parent_->gpu_context(), 0, 0));
  }
  return handles;
}

absl::StatusOr<std::vector<std::unique_ptr<GpuCommandBuffer>>>
GpuCommandBuffer::CreateConditionalCommandBuffers(
    absl::Span<const GpuGraphConditionalHandle> handles,
    absl::Span<const GpuGraphHandle> graphs,
    absl::Span<const ConditionBuilder> builders) {
  std::vector<std::unique_ptr<GpuCommandBuffer>> cmd_buffers;

  for (size_t i = 0; i < handles.size(); ++i) {
    auto command_buffer = CreateNestedCommandBuffer(graphs[i]);
    TF_RETURN_IF_ERROR(builders[i](command_buffer.get(), handles[i]));
    TF_RETURN_IF_ERROR(command_buffer->Finalize());

    cmd_buffers.push_back(std::move(command_buffer));
  }

  return cmd_buffers;
}

absl::Status GpuCommandBuffer::UpdateConditionalCommandBuffers(
    absl::Span<const GpuGraphConditionalHandle> handles,
    absl::Span<const std::unique_ptr<GpuCommandBuffer>> command_buffers,
    absl::Span<const ConditionBuilder> builders) {
  for (size_t i = 0; i < command_buffers.size(); ++i) {
    // Use parent graph executable for conditional command buffer update.
    ScopedGpuGraphExec scoped_exec(command_buffers[i].get(), exec_);

    // Update command buffer using user-provided builder callback.
    TF_RETURN_IF_ERROR(command_buffers[i]->Update());
    TF_RETURN_IF_ERROR(builders[i](command_buffers[i].get(), handles[i]));
    TF_RETURN_IF_ERROR(command_buffers[i]->Finalize());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<GpuGraphHandle>>
GpuCommandBuffer::CreateConditionalNodes(
    ExecutionScopeId execution_scope_id, ConditionType type,
    absl::Span<const GpuGraphConditionalHandle> handles) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  std::vector<GpuGraphHandle> conditional_graphs;

  using ConditionalParams = GpuDriver::GpuGraphConditionalNodeParams;
  using ConditionalResult = GpuDriver::GpuGraphConditionalNodeParams::Result;

  for (GpuGraphConditionalHandle handle : handles) {
    Dependencies barrier = GetBarrier(execution_scope_id);

    ConditionalParams params;
    params.type = type;
    params.handle = handle;
    params.context = parent_->gpu_context();

    GpuGraphNodeHandle node_handle = nullptr;

    TF_ASSIGN_OR_RETURN(
        GpuDriver::GpuGraphNodeResult result,
        GpuDriver::GraphAddNode(&node_handle, graph_,
                                ToPlatformSpecificHandles(barrier), params));

    conditional_graphs.push_back(std::get<ConditionalResult>(result).graph);
    execution_scope.nodes.emplace_back().handle =
        FromPlatformSpecificHandle(node_handle);
  }

  return conditional_graphs;
}

absl::Status GpuCommandBuffer::CreateConditionalCommand(
    ExecutionScopeId execution_scope_id, ConditionType type,
    SetConditionFn set_condition, absl::Span<const ConditionBuilder> builders) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Every conditional command buffer is controlled by its own handle.
  size_t num_handles = builders.size();

  if (state_ == State::kCreate) {
    TF_ASSIGN_OR_RETURN(auto handles, CreateConditionalHandles(num_handles));

    // Add a kernel to update conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(execution_scope_id, handles));

    // Add a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier(execution_scope_id));

    // Create conditional command buffer for each builder.
    TF_ASSIGN_OR_RETURN(
        auto graphs, CreateConditionalNodes(execution_scope_id, type, handles));
    TF_ASSIGN_OR_RETURN(auto cmd_buffers, CreateConditionalCommandBuffers(
                                              handles, graphs, builders));

    // Keep track of created conditional handles and command buffers.
    execution_scope.conditional_command_buffers.push_back(
        {std::move(handles), std::move(cmd_buffers)});

    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    ConditionalCommandBuffers& cond_cmd_buffers =
        execution_scope.conditional_command_buffers[execution_scope.update_state
                                                        .conditional_idx++];

    // Sanity check that we got the correct conditional command buffers.
    TF_RETURN_IF_ERROR(CheckNumCommandBuffers(cond_cmd_buffers, num_handles));

    // Update a kernel that updates conditional handles values.
    TF_RETURN_IF_ERROR(
        set_condition(execution_scope_id, cond_cmd_buffers.handles));

    // Update a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier(execution_scope_id));

    // Skip updating conditional nodes.
    execution_scope.update_state.node_idx += num_handles;

    return UpdateConditionalCommandBuffers(
        cond_cmd_buffers.handles,
        absl::MakeSpan(cond_cmd_buffers.command_buffers), builders);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::If(ExecutionScopeId execution_scope_id,
                                  DeviceMemory<bool> predicate,
                                  Builder then_builder) {
  TF_ASSIGN_OR_RETURN(SetIfConditionKernel * set_if_condition,
                      GetSetIfConditionKernel());

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_if_condition, id, ThreadDim(), BlockDim(),
                                 handles[0], predicate);
  };

  std::array<ConditionBuilder, 1> builders = {
      ToConditionBuilder(std::move(then_builder))};

  return CreateConditionalCommand(execution_scope_id, ConditionType::kIf,
                                  set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::IfElse(ExecutionScopeId execution_scope_id,
                                      DeviceMemory<bool> predicate,
                                      Builder then_builder,
                                      Builder else_builder) {
  TF_ASSIGN_OR_RETURN(SetIfElseConditionKernel * set_if_else_condition,
                      GetSetIfElseConditionKernel());

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_if_else_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], handles[1], predicate);
  };

  std::array<ConditionBuilder, 2> builders = {
      ToConditionBuilder(std::move(then_builder)),
      ToConditionBuilder(std::move(else_builder))};

  return CreateConditionalCommand(execution_scope_id, ConditionType::kIf,
                                  set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::Case(ExecutionScopeId execution_scope_id,
                                    DeviceMemory<int32_t> index,
                                    std::vector<Builder> branches) {
  TF_ASSIGN_OR_RETURN(SetCaseConditionKernel * set_case_condition,
                      GetSetCaseConditionKernel());

  constexpr size_t kBranchBatchSize = 8;
  int32_t batch_offset = 0;
  while (batch_offset < branches.size()) {
    // Conditionals will by default run branches[branchs.size()-1] if index is
    // <0 or >= branches.size(). See
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
                           ExecutionScopeId id, ConditionalHandles handles) {
      int32_t num_handles = handles.size();

      // Pad handles up to size 8 with a default initialized handle.
      std::vector<GpuGraphConditionalHandle> padded_handles(handles.begin(),
                                                            handles.end());
      padded_handles.resize(kBranchBatchSize);

      return CommandBuffer::Launch(
          *set_case_condition, id, ThreadDim(), BlockDim(), padded_handles[0],
          padded_handles[1], padded_handles[2], padded_handles[3],
          padded_handles[4], padded_handles[5], padded_handles[6],
          padded_handles[7], index, batch_offset, num_handles,
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

    TF_RETURN_IF_ERROR(CreateConditionalCommand(
        execution_scope_id, ConditionType::kIf, set_cond_fn, builders));
    batch_offset += batch_size;
  }
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::For(ExecutionScopeId execution_scope_id,
                                   int32_t num_iteration,
                                   DeviceMemory<int32_t> loop_counter,
                                   Builder body_builder) {
  TF_ASSIGN_OR_RETURN(SetForConditionKernel * set_for_condition,
                      GetSetForConditionKernel());

  // Reset loop counter to zero.
  TF_RETURN_IF_ERROR(Memset(execution_scope_id, &loop_counter, uint32_t{0}, 1));
  TF_RETURN_IF_ERROR(Barrier(execution_scope_id));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_for_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], loop_counter,
                                 num_iteration);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier());

    // Decide if we want to continue loop iteration.
    return body->Launch(*set_for_condition, ThreadDim(), BlockDim(), handle,
                        loop_counter, num_iteration);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(execution_scope_id, ConditionType::kWhile,
                                  set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::While(ExecutionScopeId execution_scope_id,
                                     DeviceMemory<bool> pred,
                                     ExecutionScopeBuilder cond_builder,
                                     Builder body_builder) {
  TF_ASSIGN_OR_RETURN(SetWhileConditionKernel * set_while_condition,
                      GetSetWhileConditionKernel());

  // Record condition commands into the parent command buffer.
  TF_RETURN_IF_ERROR(cond_builder(execution_scope_id, this));
  TF_RETURN_IF_ERROR(Barrier(execution_scope_id));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_while_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], pred);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier());
    TF_RETURN_IF_ERROR(cond_builder(kDefaulExecutionScope, body));
    TF_RETURN_IF_ERROR(body->Barrier());
    return body->Launch(*set_while_condition, ThreadDim(), BlockDim(), handle,
                        pred);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(execution_scope_id, ConditionType::kWhile,
                                  set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // TODO(b/362769658): Remove this workaround when cuda supports conditionals
  // with empty graphs.
#if !defined(TENSORFLOW_USE_ROCM)
  TF_ASSIGN_OR_RETURN(auto node_count, GpuDriver::GraphGetNodeCount(graph_));
  if (node_count == 0) {
    GpuGraphNodeHandle empty_node_handle = nullptr;
    TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel());

    TF_RETURN_IF_ERROR(GpuDriver::GraphAddKernelNode(
        &empty_node_handle, graph_, /*deps=*/{}, "noop",
        AsGpuKernel(&**noop)->gpu_function(), 1, 1, 1, 1, 1, 1, 0,
        /*kernel_params=*/nullptr, /*extra=*/nullptr));
  }
#endif

  // Maybe dump created CUDA graph to a dot file for debugging.
  if (state_ == State::kCreate && VLOG_IS_ON(10)) {
    std::string path = tsl::io::GetTempFilename(/*extension=*/"dot");
    auto printed = GpuDriver::GraphDebugDotPrint(
        graph_, path.c_str(), /*return_printed_graph=*/VLOG_IS_ON(100));
    if (VLOG_IS_ON(100) && printed.ok()) {
      VLOG(100) << "Printed Gpu graph " << graph_ << " to: " << path << "\n"
                << *printed;
    }
  }

  // Collect number of nodes and conditionals for logging below.
  size_t num_nodes = 0, num_cond_cmd_buffers = 0;
  for (auto& [_, execution_scope] : execution_scopes_) {
    num_nodes += execution_scope.nodes.size();
    num_cond_cmd_buffers += execution_scope.conditional_command_buffers.size();
  }

  if (mode_ == Mode::kPrimary && state_ == State::kCreate) {
    // If this is the first time we finalize command buffer after construction,
    // we need to instantiate it to an executable graph.
    GpuDriver::GraphInstantiateFlags flags;

    uint64_t start_nanos = tsl::Env::Default()->NowNanos();

    // If we get a "resource exhausted error" we retry instantiating Gpu graph
    // one more time after releasing unused device memory allocated for graphs.
    auto instantiated = GpuDriver::GraphInstantiate(&exec_, graph_, flags);
    if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
      LOG(WARNING) << "Retry CUDA graph instantiation after OOM error"
                   << "; execution_scopes: " << execution_scopes_.size()
                   << "; nodes: " << num_nodes
                   << "; conditionals: " << num_cond_cmd_buffers
                   << "; alive executable graphs: " << AliveExecs();

      TF_RETURN_IF_ERROR(parent_->TrimGraphMemory());

      auto retry = GpuDriver::GraphInstantiate(&exec_, graph_, flags);
      if (retry.code() == absl::StatusCode::kResourceExhausted) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "CUDA driver ran out of memory trying to instantiate CUDA graph "
            "with %d nodes and %d conditionals (total of %d alive CUDA graphs "
            "in the process). You can try to (a) Give more memory to CUDA "
            "driver by reducing XLA_CLIENT_MEM_FRACTION (b) Disable "
            "CUDA graph with 'XLA_FLAGS=--xla_gpu_enable_command_buffer=' "
            "(empty set). Original error: %s",
            num_nodes, num_cond_cmd_buffers, AliveExecs(), retry.message()));
      } else {
        TF_RETURN_IF_ERROR(retry);
      }
    } else {
      TF_RETURN_IF_ERROR(instantiated);
    }

    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    VLOG(5) << "Instantiated executable graph #" << NotifyExecCreated() << " "
            << exec_ << " in " << (end_nanos - start_nanos) / 1000 << " μs"
            << "; execution_scopes: " << execution_scopes_.size()
            << "; nodes: " << num_nodes
            << "; conditionals: " << num_cond_cmd_buffers
            << "; alive executable graphs: " << AliveExecs();

#if !defined(TENSORFLOW_USE_ROCM) && CUDA_VERSION < 12040
    TF_RETURN_IF_ERROR(DisableBarriersExecution(exec_));
#endif

  } else if (mode_ == Mode::kPrimary && state_ == State::kUpdate) {
    // If this is a finalization after update, we don't have to do anything as
    // each individual command already updated executable graph.
    VLOG(5) << "Finalize executable graph " << exec_ << " update #"
            << num_updates_++ << " "
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
  if (exec_ == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated");
  }

  if (state_ != State::kFinalized) {
    return absl::InternalError(
        "Command buffer has to be finalized first before it can be updated");
  }

  VLOG(5) << "Begin " << (mode_ == Mode::kPrimary ? "primary" : "nested")
          << " command buffer update for executable graph " << exec_;

  state_ = State::kUpdate;
  for (auto& [_, execution_scope] : execution_scopes_) {
    execution_scope.update_state = ExecutionScope::UpdateState();
  }
  return absl::OkStatus();
}

absl::Span<const GpuCommandBuffer::GpuGraphNodeInfo> GpuCommandBuffer::nodes(
    ExecutionScopeId id) const {
  if (auto it = execution_scopes_.find(id); it != execution_scopes_.end())
    return it->second.nodes;
  return {};
}

absl::Span<const GpuCommandBuffer::GpuGraphBarrierInfo>
GpuCommandBuffer::barriers(ExecutionScopeId id) const {
  if (auto it = execution_scopes_.find(id); it != execution_scopes_.end())
    return it->second.barriers;
  return {};
}

absl::Status GpuCommandBuffer::Submit(Stream* stream) {
  if (mode_ != CommandBuffer::Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return GpuDriver::GraphLaunch(exec_, AsGpuStreamValue(stream));
}

}  // namespace stream_executor::gpu
