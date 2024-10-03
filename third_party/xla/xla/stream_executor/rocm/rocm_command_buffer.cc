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

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
// Implementation details device kernels required by RocmCommandBuffer.
//===----------------------------------------------------------------------===//

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
// RocmCommandBuffer resource usage tracking
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

/*static*/ int64_t RocmCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// RocmCommandBuffer implementation
//===----------------------------------------------------------------------===//

static std::string_view ModeToString(CommandBuffer::Mode mode) {
  switch (mode) {
    case CommandBuffer::Mode::kPrimary:
      return "primary";
    case CommandBuffer::Mode::kNested:
      return "nested";
  }
}

RocmCommandBuffer::RocmCommandBuffer(Mode mode, GpuExecutor* parent,
                                     hipGraph_t graph, bool is_owned_graph)
    : mode_(mode),
      parent_(parent),
      graph_(graph),
      is_owned_graph_(is_owned_graph) {
  VLOG(5) << "Created command buffer for graph " << graph_
          << "; mode=" << ModeToString(mode)
          << "; is_owned_graph=" << is_owned_graph_;
  execution_scopes_.try_emplace(kDefaulExecutionScope);
}

RocmCommandBuffer::~RocmCommandBuffer() {
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

RocmCommandBuffer::ScopedGpuGraphExec::ScopedGpuGraphExec(
    RocmCommandBuffer* cmd_buffer, hipGraphExec_t exec)
    : cmd_buffer(cmd_buffer),
      restore(cmd_buffer->exec_),
      restore_is_owned(cmd_buffer->is_owned_graph_exec_) {
  cmd_buffer->exec_ = exec;
  cmd_buffer->is_owned_graph_exec_ = false;
}

RocmCommandBuffer::ScopedGpuGraphExec::~ScopedGpuGraphExec() {
  cmd_buffer->exec_ = restore;
  cmd_buffer->is_owned_graph_exec_ = restore_is_owned;
}

static hipDeviceptr_t AsDevicePtr(const DeviceMemoryBase& mem) {
  return reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(mem.opaque()));
}

absl::Status RocmCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
  TF_ASSIGN_OR_RETURN(size_t count, GpuDriver::GraphGetNodeCount(graph_));
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  auto gpu_stream = static_cast<hipStream_t>(
      AsGpuStream(stream)->platform_specific_handle().stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCapture(
      gpu_stream, GpuDriver::StreamCaptureMode::kThreadLocal));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  hipGraph_t captured_graph;
  TF_RETURN_IF_ERROR(GpuDriver::StreamEndCapture(gpu_stream, &captured_graph));
  TF_RETURN_IF_ERROR(
      GpuDriver::DestroyGraph(std::exchange(graph_, captured_graph)));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

RocmCommandBuffer::Dependencies RocmCommandBuffer::GetBarrier(
    ExecutionScopeId execution_scope_id) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];
  return execution_scope.barriers.empty()
             ? Dependencies{}
             : Dependencies{execution_scope.barriers.back().handle};
}

absl::Status RocmCommandBuffer::DisableBarriersExecution(hipGraphExec_t exec) {
  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::CheckNotFinalized() {
  if (state_ == State::kFinalized)
    return absl::InternalError(
        "Command can't be added to a command buffer after it was finalized");
  return absl::OkStatus();
}

absl::StatusOr<hipGraphNode_t> RocmCommandBuffer::CreateBarrierNode(
    const Dependencies& dependencies) {
  hipGraphNode_t barrier_handle = nullptr;
  TF_RETURN_IF_ERROR(
      GpuDriver::GraphAddEmptyNode(&barrier_handle, graph_, dependencies));

  return barrier_handle;
}

RocmCommandBuffer::Dependencies RocmCommandBuffer::GetBarrierDependencies(
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

absl::Status RocmCommandBuffer::Barrier(ExecutionScopeId execution_scope_id) {
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

absl::Status RocmCommandBuffer::Barrier(
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

absl::Status RocmCommandBuffer::Barrier(
    ExecutionScopeId from_execution_scope_id,
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

absl::Status RocmCommandBuffer::LaunchWithPackedArgs(
    ExecutionScopeId execution_scope_id, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& packed_args) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  CHECK_EQ(kernel.Arity() + (packed_args.number_of_shared_bytes() > 0),
           packed_args.number_of_arguments());

  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  hipFunction_t gpu_func = gpu_kernel->gpu_function();

  void** kernel_params =
      const_cast<void**>(packed_args.argument_addresses().data());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    RocmGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddKernelNode(
        &node_info.handle, graph_, barrier, kernel.name(), gpu_func, blocks.x,
        blocks.y, blocks.z, threads.x, threads.y, threads.z,
        packed_args.number_of_shared_bytes(), kernel_params, /*extra=*/nullptr);
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    hipGraphNode_t node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecKernelNodeSetParams(
        exec_, node, kernel.name(), gpu_func, blocks.x, blocks.y, blocks.z,
        threads.x, threads.y, threads.z, packed_args.number_of_shared_bytes(),
        kernel_params, /*extra=*/nullptr);
  }

  return UnsupportedStateError(state_);
}

absl::Status RocmCommandBuffer::Launch(ExecutionScopeId execution_scope_id,
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

absl::Status RocmCommandBuffer::AddNestedCommandBuffer(
    ExecutionScopeId execution_scope_id, const CommandBuffer& nested) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  hipGraph_t child_graph = RocmCommandBuffer::Cast(&nested)->graph();

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    RocmGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddChildNode(&node_info.handle, graph_, barrier,
                                        child_graph);
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    hipGraphNode_t node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecChildNodeSetParams(exec_, node, child_graph);
  }

  return UnsupportedStateError(state_);
}

absl::Status RocmCommandBuffer::MemcpyDeviceToDevice(
    ExecutionScopeId execution_scope_id, DeviceMemoryBase* dst,
    const DeviceMemoryBase& src, uint64_t size) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    RocmGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddMemcpyD2DNode(
        parent_->gpu_context(), &node_info.handle, graph_, barrier,
        AsDevicePtr(*dst), AsDevicePtr(src), size);
  }

  if (state_ == State::kUpdate) {
    hipGraphNode_t node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecMemcpyD2DNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst),
        AsDevicePtr(src), size);
  }

  return UnsupportedStateError(state_);
}

absl::Status RocmCommandBuffer::Memset(ExecutionScopeId execution_scope_id,
                                       DeviceMemoryBase* dst,
                                       CommandBuffer::BitPattern bit_pattern,
                                       size_t num_elements) {
  ExecutionScope& execution_scope = execution_scopes_[execution_scope_id];

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier(execution_scope_id);
    RocmGraphNodeInfo& node_info = execution_scope.nodes.emplace_back();
    return GpuDriver::GraphAddMemsetNode(
        parent_->gpu_context(), &node_info.handle, graph_, barrier,
        AsDevicePtr(*dst), bit_pattern, num_elements);
  }

  if (state_ == State::kUpdate) {
    hipGraphNode_t node =
        execution_scope.nodes[execution_scope.update_state.node_idx++].handle;
    return GpuDriver::GraphExecMemsetNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst), bit_pattern,
        num_elements);
  }

  return UnsupportedStateError(state_);
}

absl::Status RocmCommandBuffer::If(ExecutionScopeId execution_scope_id,
                                   DeviceMemory<bool> predicate,
                                   Builder then_builder) {
  return absl::UnimplementedError("Conditions are not supported.");
}

absl::Status RocmCommandBuffer::IfElse(ExecutionScopeId execution_scope_id,
                                       DeviceMemory<bool> predicate,
                                       Builder then_builder,
                                       Builder else_builder) {
  return absl::UnimplementedError("Conditions are not supported.");
}

absl::Status RocmCommandBuffer::Case(ExecutionScopeId execution_scope_id,
                                     DeviceMemory<int32_t> index,
                                     std::vector<Builder> branches) {
  return absl::UnimplementedError("Conditions are not supported.");
}

absl::Status RocmCommandBuffer::For(ExecutionScopeId execution_scope_id,
                                    int32_t num_iteration,
                                    DeviceMemory<int32_t> loop_counter,
                                    Builder body_builder) {
  return absl::UnimplementedError("Conditions are not supported.");
}

absl::Status RocmCommandBuffer::While(ExecutionScopeId execution_scope_id,
                                      DeviceMemory<bool> pred,
                                      ExecutionScopeBuilder cond_builder,
                                      Builder body_builder) {
  return absl::UnimplementedError("Conditions are not supported.");
}

absl::Status RocmCommandBuffer::Finalize() {
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

absl::Status RocmCommandBuffer::Update() {
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

absl::Span<const RocmCommandBuffer::RocmGraphNodeInfo> RocmCommandBuffer::nodes(
    ExecutionScopeId id) const {
  if (auto it = execution_scopes_.find(id); it != execution_scopes_.end())
    return it->second.nodes;
  return {};
}

absl::Span<const RocmCommandBuffer::RocmGraphBarrierInfo>
RocmCommandBuffer::barriers(ExecutionScopeId id) const {
  if (auto it = execution_scopes_.find(id); it != execution_scopes_.end())
    return it->second.barriers;
  return {};
}

absl::Status RocmCommandBuffer::Submit(Stream* stream) {
  if (mode_ != CommandBuffer::Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }

  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return GpuDriver::GraphLaunch(exec_, AsGpuStreamValue(stream));
}

}  // namespace stream_executor::gpu
