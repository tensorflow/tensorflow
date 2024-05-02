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

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernels.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

using Mode = CommandBuffer::Mode;
using State = CommandBuffer::State;

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

/*static*/ int64_t GpuCommandBuffer::AllocatedExecs() {
  return allocated_execs.load(std::memory_order_relaxed);
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

static GpuDevicePtr AsDevicePtr(const DeviceMemoryBase& mem) {
  return reinterpret_cast<GpuDevicePtr>(const_cast<void*>(mem.opaque()));
}

absl::Status GpuCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
#if defined(TENSORFLOW_USE_ROCM)
  TF_ASSIGN_OR_RETURN(size_t count, GpuDriver::GraphGetNodeCount(graph_));
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");
#endif  // TENSORFLOW_USE_ROCM

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  auto gpu_stream = AsGpuStreamValue(stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
#if !defined(TENSORFLOW_USE_ROCM)
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCaptureToGraph(
      gpu_stream, graph_, GpuDriver::StreamCaptureMode::kThreadLocal));
#else
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCapture(
      gpu_stream, GpuDriver::StreamCaptureMode::kThreadLocal));
#endif  // TENSORFLOW_USE_ROCM
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  GpuGraphHandle captured_graph;
  TF_RETURN_IF_ERROR(GpuDriver::StreamEndCapture(gpu_stream, &captured_graph));
#if !defined(TENSORFLOW_USE_ROCM)
  DCHECK(captured_graph == graph_) << "Stream capture should update graph_";
#else
  TF_RETURN_IF_ERROR(
      GpuDriver::DestroyGraph(std::exchange(graph_, captured_graph)));
#endif  // TENSORFLOW_USE_ROCM
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrier(
    ExecutionScopeId execution_scope_id) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
  return execution_scope.barriers.empty()
             ? Dependencies{}
             : Dependencies{execution_scope.barriers.back().handle};
}

absl::StatusOr<GpuCommandBuffer::SetIfConditionKernel*>
GpuCommandBuffer::GetSetIfConditionKernel(StreamExecutor* executor) {
  if (!set_if_condition_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/2);
    spec.AddCudaPtxInMemory(gpu::GetSetIfConditionKernel(), "set_if_condition");
    TF_ASSIGN_OR_RETURN(set_if_condition_kernel_,
                        SetIfConditionKernel::Create(executor, spec));
  }
  return &set_if_condition_kernel_;
}

absl::StatusOr<GpuCommandBuffer::SetIfElseConditionKernel*>
GpuCommandBuffer::GetSetIfElseConditionKernel(StreamExecutor* executor) {
  if (!set_if_else_condition_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddCudaPtxInMemory(gpu::GetSetIfElseConditionKernel(),
                            "set_if_else_condition");
    TF_ASSIGN_OR_RETURN(set_if_else_condition_kernel_,
                        SetIfElseConditionKernel::Create(executor, spec));
  }
  return &set_if_else_condition_kernel_;
}

absl::StatusOr<GpuCommandBuffer::SetCaseConditionKernel*>
GpuCommandBuffer::GetSetCaseConditionKernel(StreamExecutor* executor) {
  if (!set_case_condition_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/10);
    spec.AddCudaPtxInMemory(gpu::GetSetCaseConditionKernel(),
                            "set_case_condition");
    TF_ASSIGN_OR_RETURN(set_case_condition_kernel_,
                        SetCaseConditionKernel::Create(executor, spec));
  }
  return &set_case_condition_kernel_;
}

absl::StatusOr<GpuCommandBuffer::SetForConditionKernel*>
GpuCommandBuffer::GetSetForConditionKernel(StreamExecutor* executor) {
  if (!set_for_condition_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddCudaPtxInMemory(gpu::GetSetForConditionKernel(),
                            "set_for_condition");
    TF_ASSIGN_OR_RETURN(set_for_condition_kernel_,
                        SetForConditionKernel::Create(executor, spec));
  }
  return &set_for_condition_kernel_;
}

absl::StatusOr<GpuCommandBuffer::SetWhileConditionKernel*>
GpuCommandBuffer::GetSetWhileConditionKernel(StreamExecutor* executor) {
  if (!set_while_condition_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/2);
    spec.AddCudaPtxInMemory(gpu::GetSetWhileConditionKernel(),
                            "set_while_condition");
    TF_ASSIGN_OR_RETURN(set_while_condition_kernel_,
                        SetWhileConditionKernel::Create(executor, spec));
  }
  return &set_while_condition_kernel_;
}

absl::StatusOr<GpuCommandBuffer::NoOpKernel*> GpuCommandBuffer::GetNoOpKernel(
    StreamExecutor* executor) {
#if !defined(TENSORFLOW_USE_ROCM)
  if (!noop_kernel_) {
    MultiKernelLoaderSpec spec(/*arity=*/0);
    spec.AddCudaPtxInMemory(gpu::kNoOpKernel, "noop");
    TF_ASSIGN_OR_RETURN(noop_kernel_, NoOpKernel::Create(executor, spec));
  }
  return &noop_kernel_;
#else
  return absl::UnimplementedError(
      "GpuCommandBuffer::GetNoOpKernel is not implemented.");
#endif  // TENSORFLOW_USE_ROCM
}

absl::Status GpuCommandBuffer::DisableBarriersExecution(
    GpuGraphExecHandle exec) {
#if !defined(TENSORFLOW_USE_ROCM)
  ExecutionScope& execution_scope = execution_scopes_[kDefaulExecutionScope];

  for (GpuGraphBarrierInfo& barrier : execution_scope.barriers) {
    if (barrier.is_barrier_node) {
      TF_RETURN_IF_ERROR(
          GpuDriver::GraphNodeSetEnabled(exec, barrier.handle, false));
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

absl::StatusOr<GpuGraphNodeHandle> GpuCommandBuffer::CreateBarrierNode(
    StreamExecutor* executor, const Dependencies& dependencies) {
  GpuGraphNodeHandle barrier_handle = nullptr;
#if !defined(TENSORFLOW_USE_ROCM)
  // TODO(b/316343054): Instead of empty nodes we create no-op kernel nodes as
  // barriers because CUDA 12.3 does not support empty nodes inside
  // conditional command buffers. This should be fixed in CUDA 12.4.
  TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel(executor));

  TF_RETURN_IF_ERROR(GpuDriver::GraphAddKernelNode(
      &barrier_handle, graph_, dependencies, "noop",
      AsGpuKernel(&**noop)->AsGpuFunctionHandle(), 1, 1, 1, 1, 1, 1, 0,
      /*kernel_params=*/nullptr, /*extra=*/nullptr));
#else
  TF_RETURN_IF_ERROR(
      GpuDriver::GraphAddEmptyNode(&barrier_handle, graph_, dependencies));
#endif  // TENSORFLOW_USE_ROCM

  return barrier_handle;
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

absl::Status GpuCommandBuffer::Barrier(StreamExecutor* executor,
                                       ExecutionScopeId execution_scope_id) {
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
    TF_ASSIGN_OR_RETURN(auto barrier_handle,
                        CreateBarrierNode(executor, dependencies));
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
    StreamExecutor* executor,
    absl::Span<const ExecutionScopeId> execution_scope_ids) {
  // Nothing to synchronize here.
  if (execution_scope_ids.empty()) return absl::OkStatus();

  // Do not create two-level barriers for single execution scope.
  if (execution_scope_ids.size() == 1) {
    return Barrier(executor, execution_scope_ids[0]);
  }

  // Add a new barrier to every synchronized execution scope.
  for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
    TF_RETURN_IF_ERROR(Barrier(executor, execution_scope_id));
  }

  if (state_ == State::kCreate) {
    // Collect barriers from each scope as a dependencies.
    Dependencies dependencies;
    for (ExecutionScopeId execution_scope_id : execution_scope_ids) {
      ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
      dependencies.push_back(execution_scope.barriers.back().handle);
    }

    // Create a new barrier that joins all per-scope barriers together.
    TF_ASSIGN_OR_RETURN(auto barrier_handle,
                        CreateBarrierNode(executor, dependencies));

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

absl::Status GpuCommandBuffer::Barrier(StreamExecutor* executor,
                                       ExecutionScopeId from_execution_scope_id,
                                       ExecutionScopeId to_execution_scope_id) {
  // If scopes are the same simply add a barrier to it.
  if (from_execution_scope_id == to_execution_scope_id) {
    return Barrier(executor, from_execution_scope_id);
  }

  // Create new barriers in both execution scopes.
  TF_RETURN_IF_ERROR(Barrier(executor, from_execution_scope_id));
  TF_RETURN_IF_ERROR(Barrier(executor, to_execution_scope_id));

  if (state_ == State::kCreate) {
    // Collect barriers from each scope as dependencies.
    Dependencies dependencies = {
        execution_scopes_[from_execution_scope_id].barriers.back().handle,
        execution_scopes_[to_execution_scope_id].barriers.back().handle};

    // Create a new barrier that joins `from` and `to` scopes.
    TF_ASSIGN_OR_RETURN(auto barrier_handle,
                        CreateBarrierNode(executor, dependencies));

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

  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle gpu_func = gpu_kernel->AsGpuFunctionHandle();

  void** kernel_params =
      const_cast<void**>(packed_args.argument_addresses().data());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    GpuGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddKernelNode(
        &node_info.handle, graph_, barrier, kernel.name(), gpu_func, blocks.x,
        blocks.y, blocks.z, threads.x, threads.y, threads.z,
        packed_args.number_of_shared_bytes(), kernel_params, /*extra=*/nullptr);
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecKernelNodeSetParams(
        exec_, node, kernel.name(), gpu_func, blocks.x, blocks.y, blocks.z,
        threads.x, threads.y, threads.z, packed_args.number_of_shared_bytes(),
        kernel_params, /*extra=*/nullptr);
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

  GpuGraphHandle child_graph = GpuCommandBuffer::Cast(&nested)->graph();

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    GpuGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddChildNode(&node_info.handle, graph_, barrier,
                                        child_graph);
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecChildNodeSetParams(exec_, node, child_graph);
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
    GpuGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddMemcpyD2DNode(
        parent_->gpu_context(), &node_info.handle, graph_, barrier,
        AsDevicePtr(*dst), AsDevicePtr(src), size);
  }

  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecMemcpyD2DNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst),
        AsDevicePtr(src), size);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Memset(ExecutionScopeId execution_scope_id,
                                      DeviceMemoryBase* dst,
                                      CommandBuffer::BitPattern bit_pattern,
                                      size_t num_elements) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    GpuGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddMemsetNode(
        parent_->gpu_context(), &node_info.handle, graph_, barrier,
        AsDevicePtr(*dst), bit_pattern, num_elements);
  }

  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecMemsetNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst), bit_pattern,
        num_elements);
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

  // Conditional command buffers always created in nested mode and with
  // underlying graphs owned by a conditional node.
  CommandBuffer::Mode nested = CommandBuffer::Mode::kNested;
  bool is_owned_graph = false;

  for (size_t i = 0; i < handles.size(); ++i) {
    auto command_buffer =
        parent_->CreateCommandBuffer(nested, graphs[i], is_owned_graph);
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
    GpuGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();

    ConditionalParams params;
    params.type = type;
    params.handle = handle;
    params.context = parent_->gpu_context();

    TF_ASSIGN_OR_RETURN(
        GpuDriver::GpuGraphNodeResult result,
        GpuDriver::GraphAddNode(&node_info.handle, graph_, barrier, params));

    conditional_graphs.push_back(std::get<ConditionalResult>(result).graph);
  }

  return conditional_graphs;
}

absl::Status GpuCommandBuffer::CreateConditionalCommand(
    ExecutionScopeId execution_scope_id, StreamExecutor* executor,
    ConditionType type, SetConditionFn set_condition,
    absl::Span<const ConditionBuilder> builders) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Every conditional command buffer is controlled by its own handle.
  size_t num_handles = builders.size();

  if (state_ == State::kCreate) {
    TF_ASSIGN_OR_RETURN(auto handles, CreateConditionalHandles(num_handles));

    // Add a kernel to update conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(execution_scope_id, handles));

    // Add a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier(executor, execution_scope_id));

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
    TF_RETURN_IF_ERROR(Barrier(executor, execution_scope_id));

    // Skip updating conditional nodes.
    execution_scope.update_state.node_idx += num_handles;

    return UpdateConditionalCommandBuffers(
        cond_cmd_buffers.handles,
        absl::MakeSpan(cond_cmd_buffers.command_buffers), builders);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::If(ExecutionScopeId execution_scope_id,
                                  StreamExecutor* executor,
                                  DeviceMemory<bool> predicate,
                                  Builder then_builder) {
  DCHECK(executor == parent_);

  TF_ASSIGN_OR_RETURN(SetIfConditionKernel * set_if_condition,
                      GetSetIfConditionKernel(executor));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_if_condition, id, ThreadDim(), BlockDim(),
                                 handles[0], predicate);
  };

  std::array<ConditionBuilder, 1> builders = {
      ToConditionBuilder(std::move(then_builder))};

  return CreateConditionalCommand(execution_scope_id, executor,
                                  ConditionType::kIf, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::IfElse(ExecutionScopeId execution_scope_id,
                                      StreamExecutor* executor,
                                      DeviceMemory<bool> predicate,
                                      Builder then_builder,
                                      Builder else_builder) {
  DCHECK(executor == parent_);

  TF_ASSIGN_OR_RETURN(SetIfElseConditionKernel * set_if_else_condition,
                      GetSetIfElseConditionKernel(executor));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_if_else_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], handles[1], predicate);
  };

  std::array<ConditionBuilder, 2> builders = {
      ToConditionBuilder(std::move(then_builder)),
      ToConditionBuilder(std::move(else_builder))};

  return CreateConditionalCommand(execution_scope_id, executor,
                                  ConditionType::kIf, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::Case(ExecutionScopeId execution_scope_id,
                                    StreamExecutor* executor,
                                    DeviceMemory<int32_t> index,
                                    std::vector<Builder> branches) {
  DCHECK(executor == parent_);

  // TODO(ezhulenev): Relax this constraint, we can launch multiple back to back
  // kernels to update conditional handles in batches of size 8.
  if (branches.size() > 8) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Case command supports only up to 8 branches, got: ", branches.size()));
  }

  TF_ASSIGN_OR_RETURN(SetCaseConditionKernel * set_case_condition,
                      GetSetCaseConditionKernel(executor));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    int32_t num_handles = handles.size();

    // Pad handles up to size 8 with a default initialized handle.
    std::vector<GpuGraphConditionalHandle> padded_handles(handles.begin(),
                                                          handles.end());
    padded_handles.resize(8);

    return CommandBuffer::Launch(
        *set_case_condition, id, ThreadDim(), BlockDim(), padded_handles[0],
        padded_handles[1], padded_handles[2], padded_handles[3],
        padded_handles[4], padded_handles[5], padded_handles[6],
        padded_handles[7], index, num_handles);
  };

  // Wrap all branches into conditional command buffer builders.
  absl::InlinedVector<ConditionBuilder, 8> builders;
  builders.reserve(branches.size());
  for (auto& branch : branches) {
    builders.push_back(ToConditionBuilder(std::move(branch)));
  }

  return CreateConditionalCommand(execution_scope_id, executor,
                                  ConditionType::kIf, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::For(ExecutionScopeId execution_scope_id,
                                   StreamExecutor* executor,
                                   int32_t num_iteration,
                                   DeviceMemory<int32_t> loop_counter,
                                   Builder body_builder) {
  DCHECK(executor == parent_);

  TF_ASSIGN_OR_RETURN(SetForConditionKernel * set_for_condition,
                      GetSetForConditionKernel(executor));

  // Reset loop counter to zero.
  TF_RETURN_IF_ERROR(Memset(execution_scope_id, &loop_counter, uint32_t{0}, 1));
  TF_RETURN_IF_ERROR(Barrier(executor, execution_scope_id));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_for_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], loop_counter,
                                 num_iteration);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));

    // Decide if we want to continue loop iteration.
    return body->Launch(*set_for_condition, ThreadDim(), BlockDim(), handle,
                        loop_counter, num_iteration);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(execution_scope_id, executor,
                                  ConditionType::kWhile, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::While(ExecutionScopeId execution_scope_id,
                                     StreamExecutor* executor,
                                     DeviceMemory<bool> pred,
                                     ExecutionScopeBuilder cond_builder,
                                     Builder body_builder) {
  DCHECK(executor == parent_);

  TF_ASSIGN_OR_RETURN(SetWhileConditionKernel * set_while_condition,
                      GetSetWhileConditionKernel(executor));

  // Record condition commands into the parent command buffer.
  TF_RETURN_IF_ERROR(cond_builder(execution_scope_id, this));
  TF_RETURN_IF_ERROR(Barrier(executor, execution_scope_id));

  auto set_cond_fn = [&](ExecutionScopeId id, ConditionalHandles handles) {
    return CommandBuffer::Launch(*set_while_condition, id, ThreadDim(),
                                 BlockDim(), handles[0], pred);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));
    TF_RETURN_IF_ERROR(cond_builder(kDefaulExecutionScope, body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));
    return body->Launch(*set_while_condition, ThreadDim(), BlockDim(), handle,
                        pred);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(execution_scope_id, executor,
                                  ConditionType::kWhile, set_cond_fn, builders);
}

absl::Status GpuCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

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

      TF_RETURN_IF_ERROR(GpuDriver::DeviceGraphMemTrim(parent_->device()));

      auto retry = GpuDriver::GraphInstantiate(&exec_, graph_, flags);
      if (retry.code() == absl::StatusCode::kResourceExhausted) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "CUDA driver ran out of memory trying to instantiate CUDA graph "
            "with %d nodes and %d conditionals (total of %d alive CUDA graphs "
            "in the process). You can try to (a) Give more memory to CUDA "
            "driver by reducing XLA_PYTHON_CLIENT_MEM_FRACTION (b) Disable "
            "CUDA graph with 'XLA_FLAGS=--xla_gpu_enable_command_buffer=' "
            "(empty set). Original error: %s",
            num_nodes, num_cond_cmd_buffers, AliveExecs(), retry.message()));
      } else {
        TF_RETURN_IF_ERROR(retry);
      }
    }

    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    VLOG(5) << "Instantiated executable graph #" << NotifyExecCreated() << " "
            << exec_ << " in " << (end_nanos - start_nanos) / 1000 << " μs"
            << "; execution_scopes: " << execution_scopes_.size()
            << "; nodes: " << num_nodes
            << "; conditionals: " << num_cond_cmd_buffers
            << "; alive executable graphs: " << AliveExecs();

    TF_RETURN_IF_ERROR(DisableBarriersExecution(exec_));

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

}  // namespace stream_executor::gpu
