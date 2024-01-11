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
}

GpuCommandBuffer::~GpuCommandBuffer() {
  if (exec_ != nullptr && is_owned_graph_exec_) {
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << NotifyExecDestroyed()
            << ")";
    auto st = GpuDriver::DestroyGraphExec(exec_);
    CHECK(st.ok()) << "Failed to destroy GPU graph exec: " << st.message();
  }
  if (graph_ != nullptr && is_owned_graph_) {
    auto st = GpuDriver::DestroyGraph(graph_);
    CHECK(st.ok()) << "Failed to destroy GPU graph: " << st.message();
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
  // TODO(ezhulenev): Check that graph is empty, because we should not be mixing
  // graph tracing with explicit graph construction.
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream->DebugStreamPointers();

  auto gpu_stream = AsGpuStreamValue(stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCaptureToGraph(
      gpu_stream, graph_, GpuDriver::StreamCaptureMode::kThreadLocal));

  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  GpuGraphHandle captured_graph;
  TF_RETURN_IF_ERROR(GpuDriver::StreamEndCapture(gpu_stream, &captured_graph));
  DCHECK(captured_graph == graph_) << "Stream capture should update graph_";
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

GpuCommandBuffer::Dependencies GpuCommandBuffer::GetBarrier() {
  return barrier_ ? Dependencies{barrier_} : Dependencies{};
}

absl::StatusOr<GpuCommandBuffer::NoOpKernel*> GpuCommandBuffer::GetNoOpKernel(
    StreamExecutor* executor) {
  if (!noop_kernel_) {
    auto noop_kernel = std::make_unique<NoOpKernel>(executor);

    MultiKernelLoaderSpec spec(/*arity=*/0);
    spec.AddCudaPtxInMemory(gpu::kNoOpKernel, "noop");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, noop_kernel.get()));

    noop_kernel_ = std::move(noop_kernel);
  }

  return noop_kernel_.get();
}

absl::Status GpuCommandBuffer::DisableBarriersExecution(
    GpuGraphExecHandle exec) {
  for (GpuGraphNodeHandle barrier : barriers_) {
    if (barrier == nullptr) continue;
    TF_RETURN_IF_ERROR(GpuDriver::GraphNodeSetEnabled(exec, barrier, false));
  }
  for (ConditionalCommandBuffers& cmd_buffers : conditional_command_buffers_) {
    for (CommandBuffer& cmd_buffer : cmd_buffers.command_buffers) {
      TF_RETURN_IF_ERROR(Cast(&cmd_buffer)->DisableBarriersExecution(exec));
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
  if (cmd_buffers.handles.size() != num_cmd_buffers) {
    return absl::InternalError(absl::StrCat(
        "Expected to have ", num_cmd_buffers,
        " conditional command buffers, got ", cmd_buffers.handles.size()));
  }
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::Barrier(StreamExecutor* executor) {
  // We don't support adding barriers as root nodes and simply skip them.
  if ((state_ == State::kCreate && nodes_.empty()) ||
      (state_ == State::kUpdate && update_state_.node_idx == 0))
    return absl::OkStatus();

  if (state_ == State::kCreate) {
    // Collect nodes that will become a new barrier dependencies.
    Dependencies dependencies;
    for (int32_t i = nodes_.size() - 1; i >= 0; --i) {
      if (nodes_[i] == barrier_) break;
      dependencies.push_back(nodes_[i]);
    }

    // Add a noop kernel node acting as a barrier.
    if (dependencies.size() > 1) {
      // TODO(b/316343054): This should be an empty node, however CUDA 12.3 does
      // not support empty nodes inside conditional command buffers.
      GpuGraphNodeHandle* node = &nodes_.emplace_back();
      TF_ASSIGN_OR_RETURN(auto noop, GetNoOpKernel(executor));
      TF_RETURN_IF_ERROR(GpuDriver::GraphAddKernelNode(
          node, graph_, absl::MakeSpan(dependencies), noop->name(),
          AsGpuKernel(noop)->AsGpuFunctionHandle(), 1, 1, 1, 1, 1, 1, 0,
          /*kernel_params=*/nullptr, /*extra=*/nullptr));
      barriers_.push_back(*node);
    } else {
      barriers_.push_back(nullptr);
    }

    // Make the last node a barrier, if we didn't add a new no-op node acting
    // as a barrier we simply reuse the last node.
    barrier_ = nodes_.back();
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // Increment update node index only if we added a no-op node earlier and it
    // means that we just updated a "real" barrier node, otherwise barrier is
    // the last updated node.
    if (barriers_[update_state_.barrier_idx++]) {
      barrier_ = nodes_[update_state_.node_idx++];
    } else if (update_state_.node_idx) {
      barrier_ = nodes_[update_state_.node_idx - 1];
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

  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle gpu_func = gpu_kernel->AsGpuFunctionHandle();

  void** kernel_params =
      const_cast<void**>(packed_args.argument_addresses().data());

  // Adds a new kernel node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();
    return GpuDriver::GraphAddKernelNode(
        node, graph_, absl::MakeSpan(barrier), kernel.name(), gpu_func,
        blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
        packed_args.number_of_shared_bytes(), kernel_params, /*extra=*/nullptr);
  }

  // Updates kernel node in the executable graph.
  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node = nodes_[update_state_.node_idx++];
    return GpuDriver::GraphExecKernelNodeSetParams(
        exec_, node, kernel.name(), gpu_func, blocks.x, blocks.y, blocks.z,
        threads.x, threads.y, threads.z, packed_args.number_of_shared_bytes(),
        kernel_params, /*extra=*/nullptr);
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
    auto& pack = kernel.kernel_args_packing();
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

  GpuGraphHandle child_graph = GpuCommandBuffer::Cast(&nested)->graph();

  // Adds a child graph node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();
    return GpuDriver::GraphAddChildNode(node, graph_, absl::MakeSpan(barrier),
                                        child_graph);
  }

  // Updates child graph node in the executable graph.
  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node = nodes_[update_state_.node_idx++];
    return GpuDriver::GraphExecChildNodeSetParams(exec_, node, child_graph);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                                    const DeviceMemoryBase& src,
                                                    uint64_t size) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();
    return GpuDriver::GraphAddMemcpyD2DNode(
        parent_->gpu_context(), node, graph_, absl::MakeSpan(barrier),
        AsDevicePtr(*dst), AsDevicePtr(src), size);
  }

  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node = nodes_[update_state_.node_idx++];
    return GpuDriver::GraphExecMemcpyD2DNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst),
        AsDevicePtr(src), size);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Memset(DeviceMemoryBase* dst,
                                      CommandBuffer::BitPattern bit_pattern,
                                      size_t num_elements) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();
    return GpuDriver::GraphAddMemsetNode(
        parent_->gpu_context(), node, graph_, absl::MakeSpan(barrier),
        AsDevicePtr(*dst), bit_pattern, num_elements);
  }

  if (state_ == State::kUpdate) {
    GpuGraphNodeHandle node = nodes_[update_state_.node_idx++];
    return GpuDriver::GraphExecMemsetNodeSetParams(
        parent_->gpu_context(), exec_, node, AsDevicePtr(*dst), bit_pattern,
        num_elements);
  }

  return UnsupportedStateError(state_);
}

absl::StatusOr<DeviceMemoryBase> GpuCommandBuffer::Allocate(size_t bytes) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Adds a new memory allocation node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();

    GpuDevicePtr ptr;
    TF_RETURN_IF_ERROR(GpuDriver::GraphAddMemAllocNode(
        node, graph_, absl::MakeSpan(barrier),
        GpuDriver::MemAccessFlags::kReadWrite,
        GpuDriver::MemLocationType::kDevice, parent_->device_ordinal(),
        GpuDriver::MemAllocationType::kPinned, bytes, &ptr));
    // For CUDA impl, VA range is reserved when adding memory allocation node.
    CHECK(ptr) << "CUDA graph memory allocation node returned nullptr";

    VLOG(2) << "Setting device memory base with opaque pointer "
            << reinterpret_cast<void*>(ptr)
            << " device ordinal: " << parent_->device_ordinal();
    return DeviceMemoryBase(reinterpret_cast<void*>(ptr), bytes);
  }

  if (state_ == State::kUpdate) {
    // Memory allocation node implemented through CUDA graph does not allocate
    // new memory region on update, just return the memory region allocated
    // during the create step.
    TF_ASSIGN_OR_RETURN(AllocationResult params,
                        GpuDriver::GraphGetMemAllocNodeParams(
                            nodes_[update_state_.node_idx++]));
    return DeviceMemoryBase(reinterpret_cast<void*>(params.first),
                            params.second);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::Free(DeviceMemoryBase dst) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Adds a new memfree node to the graph under construction.
  if (state_ == State::kCreate) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();
    GpuDevicePtr gpu_dptr = AsDevicePtr(dst);
    TF_RETURN_IF_ERROR(GpuDriver::GraphAddMemFreeNode(
        node, graph_, absl::MakeSpan(barrier), gpu_dptr));
    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    // memfree node implemented through CUDA graph only free buffers that is
    // allocated through memory alloc node, so buffer address will not change,
    // no update is required.
    update_state_.node_idx++;
    return absl::OkStatus();
  }

  return UnsupportedStateError(state_);
}

//--------------------------------------------------------------------------//
// Command buffer condtitional commands API
//--------------------------------------------------------------------------//

/*static*/ GpuCommandBuffer::ConditionBuilder
GpuCommandBuffer::ToConditionBuilder(CommandBuffer::Builder builder) {
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

absl::StatusOr<std::vector<GpuGraphHandle>>
GpuCommandBuffer::CreateConditionalNodes(
    ConditionType type, absl::Span<const GpuGraphConditionalHandle> handles) {
  std::vector<GpuGraphHandle> conditional_graphs;

  using ConditionalParams = GpuDriver::GpuGraphConditionalNodeParams;
  using ConditionalResult = GpuDriver::GpuGraphConditionalNodeParams::Result;

  for (GpuGraphConditionalHandle handle : handles) {
    Dependencies barrier = GetBarrier();
    GpuGraphNodeHandle* node = &nodes_.emplace_back();

    ConditionalParams params;
    params.type = type;
    params.handle = handle;
    params.context = parent_->gpu_context();

    TF_ASSIGN_OR_RETURN(
        GpuDriver::GpuGraphNodeResult result,
        GpuDriver::GraphAddNode(node, graph_, absl::MakeSpan(barrier), params));

    conditional_graphs.push_back(std::get<ConditionalResult>(result).graph);
  }

  return conditional_graphs;
}

absl::StatusOr<std::vector<CommandBuffer>>
GpuCommandBuffer::CreateConditionalCommandBuffers(
    absl::Span<const GpuGraphConditionalHandle> handles,
    absl::Span<const GpuGraphHandle> graphs,
    absl::Span<const ConditionBuilder> builders) {
  std::vector<CommandBuffer> cmd_buffers;

  // Conditional command buffers always created in nested mode and with
  // underlying graphs owned by a conditional node.
  CommandBuffer::Mode nested = CommandBuffer::Mode::kNested;
  bool is_owned_graph = false;

  for (size_t i = 0; i < handles.size(); ++i) {
    auto command_buffer_impl = parent_->GetCommandBufferImplementation(
        nested, graphs[i], is_owned_graph);

    auto command_buffer = CommandBuffer::Create(std::move(command_buffer_impl));

    TF_RETURN_IF_ERROR(builders[i](&command_buffer, handles[i]));
    TF_RETURN_IF_ERROR(command_buffer.Finalize());

    cmd_buffers.push_back(std::move(command_buffer));
  }

  return cmd_buffers;
}

absl::Status GpuCommandBuffer::UpdateConditionalCommandBuffers(
    absl::Span<const GpuGraphConditionalHandle> handles,
    absl::Span<CommandBuffer> command_buffers,
    absl::Span<const ConditionBuilder> builders) {
  for (size_t i = 0; i < command_buffers.size(); ++i) {
    // Use parent graph executable for conditional command buffer update.
    ScopedGpuGraphExec scoped_exec(Cast(&command_buffers[i]), exec_);

    // Update command buffer using user-provided builder callback.
    TF_RETURN_IF_ERROR(command_buffers[i].Update());
    TF_RETURN_IF_ERROR(builders[i](&command_buffers[i], handles[i]));
    TF_RETURN_IF_ERROR(command_buffers[i].Finalize());
  }
  return absl::OkStatus();
}

absl::Status GpuCommandBuffer::CreateConditionalCommand(
    StreamExecutor* executor, ConditionType type, SetConditionFn set_condition,
    absl::Span<const ConditionBuilder> builders) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  // Every conditional command buffer is controlled by its own handle.
  size_t num_handles = builders.size();

  if (state_ == State::kCreate) {
    TF_ASSIGN_OR_RETURN(auto handles, CreateConditionalHandles(num_handles));

    // Add a kernel to update conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(handles));

    // Add a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier(executor));

    // Create conditional command buffer for each builder.
    TF_ASSIGN_OR_RETURN(auto graphs, CreateConditionalNodes(type, handles));
    TF_ASSIGN_OR_RETURN(auto cmd_buffers, CreateConditionalCommandBuffers(
                                              handles, graphs, builders));

    // Keep track of created conditional handles and command buffers.
    conditional_command_buffers_.emplace_back(std::move(handles),
                                              std::move(cmd_buffers));

    return absl::OkStatus();
  }

  if (state_ == State::kUpdate) {
    ConditionalCommandBuffers& cond_cmd_buffers =
        conditional_command_buffers_[update_state_.conditional_idx++];

    // Sanity check that we got the correct conditional command buffers.
    TF_RETURN_IF_ERROR(CheckNumCommandBuffers(cond_cmd_buffers, num_handles));

    // Update a kernel that updates conditional handles values.
    TF_RETURN_IF_ERROR(set_condition(cond_cmd_buffers.handles));

    // Update a barrier between conditional handles and conditional nodes.
    TF_RETURN_IF_ERROR(Barrier(executor));

    // Skip updating conditional nodes.
    update_state_.node_idx += num_handles;

    return UpdateConditionalCommandBuffers(
        cond_cmd_buffers.handles,
        absl::MakeSpan(cond_cmd_buffers.command_buffers), builders);
  }

  return UnsupportedStateError(state_);
}

absl::Status GpuCommandBuffer::If(StreamExecutor* executor,
                                  DeviceMemory<bool> predicate,
                                  CommandBuffer::Builder then_builder) {
  DCHECK(executor->implementation() == parent_);

  // TODO(ezhulenev): Keep kernel in `GpuCommandBuffer` to avoid loading it on
  // every call to `If`.
  SetIfConditionKernel set_if_condition(executor);

  {  // Load kernels that updates condition handle value.
    MultiKernelLoaderSpec spec(/*arity=*/2);
    spec.AddInProcessSymbol(gpu::GetSetIfConditionKernel(), "set_if_condition");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, &set_if_condition));
  }

  auto set_cond_fn = [&](absl::Span<const GpuGraphConditionalHandle> handles) {
    return Launch(set_if_condition, ThreadDim(), BlockDim(), handles[0],
                  predicate);
  };

  std::array<ConditionBuilder, 1> builders = {
      ToConditionBuilder(std::move(then_builder))};

  return CreateConditionalCommand(executor, ConditionType::kIf, set_cond_fn,
                                  builders);
}

absl::Status GpuCommandBuffer::IfElse(StreamExecutor* executor,
                                      DeviceMemory<bool> predicate,
                                      CommandBuffer::Builder then_builder,
                                      CommandBuffer::Builder else_builder) {
  DCHECK(executor->implementation() == parent_);

  // TODO(ezhulenev): Keep kernel in `GpuCommandBuffer` to avoid loading it on
  // every call to `IfElse`.
  SetIfElseConditionKernel set_if_else_condition(executor);

  {  // Load kernels that updates condition handle value.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(gpu::GetSetIfElseConditionKernel(),
                            "set_if_else_condition");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, &set_if_else_condition));
  }

  auto set_cond_fn = [&](absl::Span<const GpuGraphConditionalHandle> handles) {
    return Launch(set_if_else_condition, ThreadDim(), BlockDim(), handles[0],
                  handles[1], predicate);
  };

  std::array<ConditionBuilder, 2> builders = {
      ToConditionBuilder(std::move(then_builder)),
      ToConditionBuilder(std::move(else_builder))};

  return CreateConditionalCommand(executor, ConditionType::kIf, set_cond_fn,
                                  builders);
}

absl::Status GpuCommandBuffer::Case(
    StreamExecutor* executor, DeviceMemory<int32_t> index,
    std::vector<CommandBuffer::Builder> branches) {
  DCHECK(executor->implementation() == parent_);

  // TODO(ezhulenev): Relax this constraint, we can launch multiple back to back
  // kernels to update conditional handles in batches of size 8.
  if (branches.size() > 8) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Case command supports only up to 8 branches, got: ", branches.size()));
  }

  // TODO(ezhulenev): Keep kernel in `GpuCommandBuffer` to avoid loading it on
  // every call to `Case`.
  SetCaseConditionKernel set_case_condition(executor);

  {  // Load kernels that updates condition handle value.
    MultiKernelLoaderSpec spec(/*arity=*/10);
    spec.AddInProcessSymbol(gpu::GetSetCaseConditionKernel(),
                            "set_case_condition");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, &set_case_condition));
  }

  auto set_cond_fn = [&](absl::Span<const GpuGraphConditionalHandle> handles) {
    int32_t num_handles = handles.size();

    // Pad handles up to size 8 with a default initialized handle.
    std::vector<GpuGraphConditionalHandle> padded_handles(handles.begin(),
                                                          handles.end());
    padded_handles.resize(8);

    return Launch(set_case_condition, ThreadDim(), BlockDim(),
                  padded_handles[0], padded_handles[1], padded_handles[2],
                  padded_handles[3], padded_handles[4], padded_handles[5],
                  padded_handles[6], padded_handles[7], index, num_handles);
  };

  // Wrap all branches into conditional command buffer builders.
  absl::InlinedVector<ConditionBuilder, 8> builders;
  builders.reserve(branches.size());
  for (auto& branch : branches) {
    builders.push_back(ToConditionBuilder(std::move(branch)));
  }

  return CreateConditionalCommand(executor, ConditionType::kIf, set_cond_fn,
                                  builders);
}

absl::Status GpuCommandBuffer::For(StreamExecutor* executor,
                                   int32_t num_iteration,
                                   DeviceMemory<int32_t> loop_counter,
                                   CommandBuffer::Builder body_builder) {
  DCHECK(executor->implementation() == parent_);

  // TODO(ezhulenev): Keep kernel in `GpuCommandBuffer` to avoid loading it on
  // every call to `For`.
  SetForConditionKernel set_for_condition(executor);

  {  // Load kernels that updates condition handle value.
    MultiKernelLoaderSpec spec(/*arity=*/3);
    spec.AddInProcessSymbol(gpu::GetSetForConditionKernel(),
                            "set_for_condition");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, &set_for_condition));
  }

  // Reset loop counter to zero.
  TF_RETURN_IF_ERROR(Memset(&loop_counter, uint32_t{0}, 1));
  TF_RETURN_IF_ERROR(Barrier(executor));

  auto set_cond_fn = [&](absl::Span<const GpuGraphConditionalHandle> handles) {
    return Launch(set_for_condition, ThreadDim(), BlockDim(), handles[0],
                  loop_counter, num_iteration);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));

    // Decide if we want to continue loop iteration.
    return body->Launch(set_for_condition, ThreadDim(), BlockDim(), handle,
                        loop_counter, num_iteration);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(executor, ConditionType::kWhile, set_cond_fn,
                                  builders);
}

absl::Status GpuCommandBuffer::While(StreamExecutor* executor,
                                     DeviceMemory<bool> pred,
                                     CommandBuffer::Builder cond_builder,
                                     CommandBuffer::Builder body_builder) {
  DCHECK(executor->implementation() == parent_);

  // TODO(ezhulenev): Keep kernel in `GpuCommandBuffer` to avoid loading it on
  // every call to `While`.
  SetWhileConditionKernel set_while_condition(executor);

  {  // Load kernels that updates condition handle value.
    MultiKernelLoaderSpec spec(/*arity=*/2);
    spec.AddInProcessSymbol(gpu::GetSetWhileConditionKernel(),
                            "set_while_condition");
    TF_RETURN_IF_ERROR(executor->GetKernel(spec, &set_while_condition));
  }

  // Record condition commands into the parent command buffer.
  TF_RETURN_IF_ERROR(CommandBuffer::Build(this, cond_builder));
  TF_RETURN_IF_ERROR(Barrier(executor));

  auto set_cond_fn = [&](absl::Span<const GpuGraphConditionalHandle> handles) {
    return Launch(set_while_condition, ThreadDim(), BlockDim(), handles[0],
                  pred);
  };

  auto body = [&](CommandBuffer* body, GpuGraphConditionalHandle handle) {
    TF_RETURN_IF_ERROR(body_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));
    TF_RETURN_IF_ERROR(cond_builder(body));
    TF_RETURN_IF_ERROR(body->Barrier(executor));
    return body->Launch(set_while_condition, ThreadDim(), BlockDim(), handle,
                        pred);
  };

  std::array<ConditionBuilder, 1> builders = {std::move(body)};

  return CreateConditionalCommand(executor, ConditionType::kWhile, set_cond_fn,
                                  builders);
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
                   << "; nodes=" << nodes_.size()
                   << "; conditionals=" << conditional_command_buffers_.size()
                   << "; alive executable graphs: " << AliveExecs();

      TF_RETURN_IF_ERROR(GpuDriver::DeviceGraphMemTrim(parent_->device()));
      TF_RETURN_IF_ERROR(GpuDriver::GraphInstantiate(&exec_, graph_, flags));
    }

    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    VLOG(5) << "Instantiated executable graph #" << NotifyExecCreated() << " "
            << exec_ << " in " << (end_nanos - start_nanos) / 1000 << " μs"
            << "; nodes: " << nodes_.size()
            << "; conditionals: " << conditional_command_buffers_.size()
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
  barrier_ = nullptr;
  update_state_ = UpdateState();
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
