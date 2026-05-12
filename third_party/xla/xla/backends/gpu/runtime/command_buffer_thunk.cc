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

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

//===----------------------------------------------------------------------===//
// CommandBufferThunk
//===----------------------------------------------------------------------===//

CommandBufferThunk::ExecutorCommandBuffer::ExecutorCommandBuffer(
    std::unique_ptr<se::CommandBuffer> command_buffer)
    : command_buffer(std::move(command_buffer)) {}

CommandBufferThunk::CommandBufferThunk(
    CommandExecutor commands, ThunkInfo thunk_info,
    std::unique_ptr<SequentialThunk> thunks,
    bool enable_command_buffers_during_profiling,
    DebugOptions::CommandBufferUpdateMode command_buffer_update_mode)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      thunks_(std::move(thunks)),
      enable_command_buffers_during_profiling_(
          enable_command_buffers_during_profiling),
      command_buffer_update_mode_(command_buffer_update_mode),
      state_(std::make_shared<State>()) {
  if (VLOG_IS_ON(5)) {
    absl::StatusOr<std::string> graph = commands_.RenderExecutionGraph();
    if (graph.ok()) {
      VLOG(5) << "Rendered command buffer execution graph: " << *graph;
    } else {
      VLOG(5) << "Failed to render command buffer execution graph: "
              << graph.status();
    }
  }

  // When we create a new command buffer thunk (which happens when we
  // instantiate a new Gpu executable) we evict command buffers for all
  // previously instantiated executables. If previously instantiated executable
  // will be executed again, it will simply reconstruct command buffer from
  // a command buffer cmd sequence which is not terribly expensive (few
  // milliseconds for large command buffers). With this approach we keep command
  // buffers (CUDA graphs) resident in device memory only for executable that
  // are actually used.
  //
  // In a perfect world higher level framework (JAX, Tensorflow, PyTorch) would
  // be more aggressive with destroying unused executables, however today they
  // all have a pretty large LRU cache for keeping O(1000) XLA executables.
  EvictCommandBuffers();
  TrackCommandBuffers(state_);

  // Pre-compute the minimum allocation index of the first traced command so
  // GetOrCreateCommandBuffer() doesn't have to walk the command tree on every
  // call.
  if (command_buffer_update_mode_ == DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
    bool found = false;
    CHECK_OK(commands_.Walk([&](const Command* command) -> absl::Status {
      if (!found && command->IsTracedCommand()) {
        found = true;
        std::optional<BufferAllocation::Index> min_idx;
        for (const auto& use : command->buffer_uses()) {
          const BufferAllocation* alloc = use.slice().allocation();
          if (alloc->is_constant() || alloc->size() == 0) continue;
          if (!min_idx.has_value() || use.slice().index() < *min_idx) {
            min_idx = use.slice().index();
          }
        }
        if (min_idx.has_value()) {
          first_traced_cmd_alloc_idx_ = min_idx;
        }
      }
      return absl::OkStatus();
    }));
  }
}

std::vector<BufferAllocation::Index>
CommandBufferThunk::ExecutorCommandBuffer::UpdateBufferAllocations(
    const CommandExecutor& commands, const Thunk::ExecuteParams& params) {
  std::vector<BufferAllocation::Index> updated_allocs;
  const BufferAllocations* allocs = params.buffer_allocations;

  // We check only allocations referenced by commands in a cmd sequence, and
  // leave every other entry default initialized (nullptr device memory).
  for (BufferAllocation::Index index : commands.allocs_indices()) {
    se::DeviceAddressBase alloc = allocs->GetDeviceAddress(index);

    if (recorded_allocs.size() <= index) {
      recorded_allocs.resize(index + 1);
      updated_allocs.push_back(index);
    }

    if (!recorded_allocs[index].IsSameAs(alloc)) {
      recorded_allocs[index] = alloc;
      updated_allocs.push_back(index);
    }
  }

  return updated_allocs;
}

absl::Status CommandBufferThunk::Prepare(const PrepareParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Always prepare thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Prepare(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Prepare command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(commands_.Prepare(params));

  return absl::OkStatus();
}

absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Initialize commands.
  TF_RETURN_IF_ERROR(commands_.Initialize(params));

  // Always initialize thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Initialize(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Initialize command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
      GetOrCreateCommandBuffer(params.executor, *params.buffer_allocations));
  absl::MutexLock lock(cmd_buffer->mutex);

  // If there are no thunks, or command buffer does not require initialization,
  // we can mark warm up as done immediately.
  if (!thunks_ || !commands_.requires_initialization()) {
    cmd_buffer->warmup_done = true;
  }

  // Construct ExecuteParams with empty fields for everything that is not needed
  // for recording commands.
  Thunk::ExecuteParams execute_params(
      params.buffer_allocations, params.stream,
      params.command_buffer_trace_stream, params.collective_params,
      params.collective_cliques, params.collective_memory,
      /*device_to_host_stream=*/nullptr,
      /*host_to_device_stream=*/nullptr,
      /*send_device_memory_function=*/nullptr,
      /*recv_device_memory_function=*/nullptr, params.ffi_execution_context,
      /*additional_compute_streams=*/{}, params.execution_scoped_state,
      /*module_name=*/"",
      /*mock_collectives=*/false);

  if (!cmd_buffer->warmup_done) {
    return absl::OkStatus();
  }

  // If command buffer is in `kCreate` state it means that command buffer
  // sequence was never recorded into it. We initialize all command buffers
  // before execution, because command buffers when instantiated will allocate
  // memory on device and this might lead to deadlocks when we have concurrent
  // NCCL operations in flight.

  // If commands require initialization (and VA remapping is not enabled), we
  // also record them into the command buffer before execution. This is required
  // to guarantee that collective commands are recorded on all participating
  // ranks to avoid deadlocks.
  if (cmd_buffer->command_buffer->state() ==
          se::CommandBuffer::State::kCreate ||
      (command_buffer_update_mode_ != DebugOptions::NEVER_UPDATE &&
       commands_.requires_initialization())) {
    VLOG(3) << "Initialize command buffer on device #"
            << params.executor->device_ordinal()
            << " by recoding command buffer cmd sequence"
            << "; num_commands=" << commands_.size();

    TraceMe trace([&] {
      return TraceMeEncode("command_buffer::initialize",
                           {{"device", params.executor->device_ordinal()},
                            {"num_commands", commands_.size()}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    // Update recorded buffer allocations.
    auto updated_allocs =
        cmd_buffer->UpdateBufferAllocations(commands_, execute_params);

    Command::RecordParams record_params = {cmd_buffer->state,
                                           std::move(updated_allocs),
                                           /*is_initialization=*/true,
                                           /*command_buffer_update_mode=*/
                                           command_buffer_update_mode_};
    TF_RETURN_IF_ERROR(commands_.Record(execute_params, record_params,
                                        cmd_buffer->command_buffer.get()));

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    VLOG(3) << "Initialized command buffer on device #"
            << params.executor->device_ordinal() << " in "
            << (end_micros - start_micros)
            << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }
  return absl::OkStatus();
}

absl::Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // TODO(b/290773547): Profiler (CUPTI) + CUDA graphs lead to memory
  // corruption. As a work around disable command buffers (CUDA graphs) and run
  // everything in op-by-op mode.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Execute command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return thunks_->ExecuteOnStream(params);
  }

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
      GetOrCreateCommandBuffer(executor, *params.buffer_allocations));

  absl::MutexLock lock(cmd_buffer->mutex);

  // warm up iteration, run through thunks if they are present.
  if (!cmd_buffer->warmup_done && thunks_) {
    VLOG(2) << "Executing warm up iteration of command buffer thunk";
    TF_RETURN_IF_ERROR(thunks_->ExecuteOnStream(params));
    cmd_buffer->warmup_done = true;
    return absl::OkStatus();
  }

  auto updated_allocs = cmd_buffer->UpdateBufferAllocations(commands_, params);

  // Determine whether to (re-)record the command buffer and whether this is a
  // first-time initialization recording (VA remapping path).
  bool is_first_record =
      command_buffer_update_mode_ == DebugOptions::NEVER_UPDATE &&
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;
  bool needs_update =
      (command_buffer_update_mode_ == DebugOptions::ALWAYS_UPDATE ||
       command_buffer_update_mode_ == DebugOptions::CAPTURE_CMD_NEVER_UPDATE) &&
      !updated_allocs.empty();

  if (is_first_record || needs_update) {
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << "Create/Update command buffer"
        << " by recoding command buffer cmd sequence after "
        << cmd_buffer->num_executions << " executions since last update"
        << "; num_commands=" << commands_.size()
        << "; updated_allocs=" << updated_allocs.size()
        << "; is_first_record=" << is_first_record
        << "; needs_update=" << needs_update;

    TraceMe trace([&] {
      cmd_buffer->mutex.AssertHeld();
      return TraceMeEncode(needs_update
                               ? "command_buffer::update"
                               : "command_buffer::record_for_va_remapping",
                           {{"device", executor->device_ordinal()},
                            {"num_commands", commands_.size()}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    Command::RecordParams record_params = {
        cmd_buffer->state, std::move(updated_allocs),
        /*is_initialization=*/is_first_record,
        /*command_buffer_update_mode=*/
        command_buffer_update_mode_};
    TF_RETURN_IF_ERROR(commands_.Record(params, record_params,
                                        cmd_buffer->command_buffer.get()));

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << (needs_update ? "Updated" : "Recorded") << " command buffer in "
        << (end_micros - start_micros)
        << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }

  ++cmd_buffer->num_executions;

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Execute command buffer"
      << "; num_executions=" << cmd_buffer->num_executions;

  TraceMe trace([&] {
    cmd_buffer->mutex.AssertHeld();
    return TraceMeEncode("command_buffer::execute",
                         {{"device", executor->device_ordinal()},
                          {"num_commands", commands_.size()},
                          {"num_executions", cmd_buffer->num_executions}});
  });

  return cmd_buffer->command_buffer->Submit(params.stream);
}

absl::StatusOr<std::shared_ptr<CommandBufferThunk::ExecutorCommandBuffer>>
CommandBufferThunk::GetOrCreateCommandBuffer(
    se::StreamExecutor* executor, const BufferAllocations& buffer_allocations) {
  void* first_alloc_address = nullptr;
  if (command_buffer_update_mode_ == DebugOptions::NEVER_UPDATE &&
      !allocs_indices().empty()) {
    first_alloc_address =
        buffer_allocations.GetDeviceAddress(allocs_indices()[0]).opaque();
  } else if (command_buffer_update_mode_ ==
             DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
    // Use the cached minimum allocation index of the first traced command
    // (computed once at construction time) to look up the physical address of
    // its buffer allocation. This address serves as the key to identify which
    // VA reservation set is active for the current execution.
    //
    // This works because the VMM allocator assigns each VA reservation set a
    // distinct physical memory region: when execution alternates between two
    // VA ranges (indices 0 and 1), the physical address backing
    // first_traced_cmd_alloc_idx_ will differ between the two sets, uniquely
    // identifying the active VA range. Constants and zero-size allocations are
    // excluded (at construction time) to ensure the chosen index maps to a
    // real, varying physical address.
    if (first_traced_cmd_alloc_idx_.has_value()) {
      first_alloc_address =
          buffer_allocations.GetDeviceAddress(*first_traced_cmd_alloc_idx_)
              .opaque();
    }
  }
  auto key = std::make_pair(executor, first_alloc_address);
  absl::MutexLock lock(state_->mutex);
  // Check if command buffer already exists
  if (auto it = state_->command_buffers.find(key);
      it != state_->command_buffers.end()) {
    return it->second;
  }

  // Create a new empty command buffer.
  TF_ASSIGN_OR_RETURN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  auto emplaced = state_->command_buffers.emplace(
      key, std::make_shared<ExecutorCommandBuffer>(std::move(command_buffer)));
  // With kNumVaReservationSets=2, at most 2 command buffers should exist per
  // executor (one per VA reservation set). A CommandBufferThunk may be shared
  // across replicas (multiple executors), so count only entries for this
  // executor rather than the total map size.
  size_t count_for_executor = std::count_if(
      state_->command_buffers.begin(), state_->command_buffers.end(),
      [executor](const auto& entry) { return entry.first.first == executor; });
  DCHECK_LE(count_for_executor, static_cast<size_t>(2))
      << "command_buffers map has more entries than expected VA reservation "
      << "sets for executor " << executor;

  return emplaced.first->second;
}

//===----------------------------------------------------------------------===//
// Command buffer eviction
//===----------------------------------------------------------------------===//

struct CommandBufferThunk::GlobalState {
  absl::Mutex mutex;
  std::vector<std::weak_ptr<CommandBufferThunk::State>> state
      ABSL_GUARDED_BY(mutex);
};

CommandBufferThunk::GlobalState* CommandBufferThunk::GetGlobalState() {
  static auto* const global_state = new GlobalState();
  return global_state;
}

void CommandBufferThunk::TrackCommandBuffers(
    std::weak_ptr<CommandBufferThunk::State> state) {
  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(global_state->mutex);
  global_state->state.push_back(state);
}

void CommandBufferThunk::EvictCommandBuffers() {
  TraceMe trace("EvictCommandBuffers");

  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(global_state->mutex);
  VLOG(3) << "Evict command buffer thunk command buffers; tracked thunks = "
          << global_state->state.size();

  // Erase state for already destroyed thunks.
  global_state->state.erase(
      std::remove_if(global_state->state.begin(), global_state->state.end(),
                     [](auto& weak_ptr) { return weak_ptr.expired(); }),
      global_state->state.end());

  // Evict command buffers for all tracked thunks.
  int64_t num_evicted = 0;
  for (auto& weak_ptr : global_state->state) {
    auto ptr = weak_ptr.lock();
    if (!ptr) {
      continue;
    }

    // Evict all command buffers.
    absl::MutexLock state_lock(ptr->mutex);
    num_evicted += ptr->command_buffers.size();
    ptr->command_buffers.clear();
  }

  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}

absl::Status CommandBufferThunk::WalkNested(Walker callback) {
  if (thunks_ != nullptr) {
    TF_RETURN_IF_ERROR(thunks_->Walk(callback));
  }
  return absl::OkStatus();
}

std::string CommandBufferThunk::ToString(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks_->ToString(indent + 1));
  return result;
}
absl::StatusOr<ThunkProto> CommandBufferThunk::ToProto() const {
  return absl::InvalidArgumentError("CommandBufferThunk can't be serialized.");
}

}  // namespace xla::gpu
