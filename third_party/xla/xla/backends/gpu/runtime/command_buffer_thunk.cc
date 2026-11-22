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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

namespace {

// Returns the current module's top-level annotation, or an empty view when the
// thunk runs outside `GpuExecutable::ExecuteThunksImpl` (e.g. in tests), which
// installs the module annotations.
absl::string_view ModuleTopLevelAnnotation() {
  const ModuleAnnotations* mod = GetCurrentModuleAnnotations();
  return mod != nullptr ? static_cast<absl::string_view>(mod->top_level)
                        : absl::string_view();
}

}  // namespace

//===----------------------------------------------------------------------===//
// CommandBufferThunk
//===----------------------------------------------------------------------===//

CommandBufferThunk::ExecutorCommandBuffer::ExecutorCommandBuffer(
    std::unique_ptr<se::CommandBuffer> command_buffer)
    : command_buffer(std::move(command_buffer)) {}

bool CommandBufferThunk::ExecutorCommandBuffer::HasDynamicAllocations(
    const CommandExecutor& commands,
    absl::Span<const BufferAllocation::Index> persistent_alloc_indices) {
  DCHECK(absl::c_is_sorted(persistent_alloc_indices));
  return !absl::c_includes(persistent_alloc_indices, commands.allocs_indices());
}

bool CommandBufferThunk::ShouldRecordTelemetry(
    const se::CommandBuffer& cmd_buffer) const {
  if (!enable_command_buffer_telemetry_) {
    return false;
  }
  if (cmd_buffer.SupportsNodeAnnotations()) {
    return true;
  }
  LOG_FIRST_N(WARNING, 1)
      << "Ignoring --xla_gpu_enable_command_buffer_telemetry: the command "
         "buffer backend does not support node annotations (requires CUDA "
         "13.2+).";
  return false;
}

CommandBufferThunk::CommandBufferThunk(CommandExecutor commands,
                                       ThunkInfo thunk_info,
                                       int devices_in_process,
                                       std::unique_ptr<SequentialThunk> thunks,
                                       CommandBufferThunkOptions options)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      thunks_(std::move(thunks)),
      enable_command_buffers_during_profiling_(
          options.enable_command_buffers_during_profiling),
      enable_command_buffer_telemetry_(options.enable_command_buffer_telemetry),
      state_(std::make_shared<State>(devices_in_process)) {
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
}

std::vector<BufferAllocation::Index>
CommandBufferThunk::ExecutorCommandBuffer::UpdateBufferAllocations(
    const CommandExecutor& commands, const Thunk::ExecuteParams& params,
    absl::Span<const BufferAllocation::Index> persistent_alloc_indices) {
  std::vector<BufferAllocation::Index> updated_allocs;
  const BufferAllocations* allocs = params.buffer_allocations;
  std::vector<BufferAllocation::Index> dynamic_alloc_indices;

  absl::c_set_difference(commands.allocs_indices(), persistent_alloc_indices,
                         std::back_inserter(dynamic_alloc_indices));

  // We check only allocations referenced by commands in a cmd sequence, and
  // leave every other entry default initialized (nullptr device memory).
  for (BufferAllocation::Index index : dynamic_alloc_indices) {
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

  // `commands_` borrows pointers to the same Thunk objects owned by `thunks_`,
  // so preparing `thunks_` (when present) already prepares all commands.
  if (thunks_) {
    return thunks_->Prepare(params);
  }

  return commands_.Prepare(params);
}

absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Always initialize fallback thunks (or commands when there are no fallback
  // thunks). `commands_` borrows pointers to the same Thunk objects owned by
  // `thunks_`, so initializing `thunks_` (when present) already initializes all
  // commands.
  if (thunks_) {
    ABSL_RETURN_IF_ERROR(thunks_->Initialize(params));
  } else {
    ABSL_RETURN_IF_ERROR(commands_.Initialize(params));
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

  // Command buffer lowering is enabled only when persistent allocation
  // indices are available; until then the thunk executes its fallback thunk
  // sequence, so there is no command buffer to initialize.
  if (!params.persistent_alloc_indices.has_value()) {
    VLOG(1) << "Skip command buffer initialization because persistent "
               "allocation indices are not yet available (falling back to "
               "thunk sequence)";
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
      GetOrCreateCommandBuffer(params.executor, params.collective_cliques,
                               params.collective_memory));
  absl::MutexLock lock(cmd_buffer->mutex);

  // If there are no thunks, or command buffer does not require warmup,
  // we can mark warm up as done immediately.
  if (!cmd_buffer->warmup_done) {
    if (thunks_ == nullptr) {
      cmd_buffer->warmup_done = true;
    } else {
      TF_RET_CHECK(params.command_buffer_requests != nullptr);
      cmd_buffer->warmup_done =
          !commands_.requires_warmup(*params.command_buffer_requests);
    }
  }

  if (!cmd_buffer->warmup_done) {
    return absl::OkStatus();
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
      /*recv_device_memory_function=*/nullptr, params.custom_options,
      params.ffi_execution_context,
      /*additional_compute_streams=*/{}, params.execution_scoped_state,
      /*mock_collectives=*/false, /*execution_id=*/0,
      /*rng_seed=*/0, params.persistent_alloc_indices);

  // If command buffer is in `kCreate` state it means that command buffer
  // sequence was never recorded into it. We initialize all command buffers
  // before execution, because command buffers when instantiated will allocate
  // memory on device and this might lead to deadlocks when we have concurrent
  // NCCL operations in flight.

  // If commands require an update during initialization (and VA remapping is
  // not enabled), we also record them into the command buffer before execution.
  // This is required to guarantee that collective commands are recorded on all
  // participating ranks to avoid deadlocks.
  bool has_dynamic_allocations = cmd_buffer->HasDynamicAllocations(
      commands_, *params.persistent_alloc_indices);
  bool is_first_record =
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;
  if (is_first_record ||
      (has_dynamic_allocations && commands_.requires_update_on_initialize())) {
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
    std::optional<std::vector<BufferAllocation::Index>> updated_allocs =
        cmd_buffer->UpdateBufferAllocations(commands_, execute_params,
                                            *params.persistent_alloc_indices);

    Command::RecordParams record_params = {cmd_buffer->state,
                                           std::move(updated_allocs),
                                           /*is_initialization=*/true};
    // The root spans `Record`, including `Finalize`, so stream_executor sees
    // recording on for the whole recording when telemetry is enabled.
    auto telemetry_root = se::ScopedCommandBufferAnnotation::Root(
        ShouldRecordTelemetry(*cmd_buffer->command_buffer),
        ModuleTopLevelAnnotation());
    ABSL_RETURN_IF_ERROR(RecordCommands(params.executor, execute_params,
                                   std::move(record_params), cmd_buffer.get(),
                                   params.collective_memory));

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

  // Command buffer lowering is enabled only when persistent allocation
  // indices are available. Until then (e.g. during the VA remapping profiling
  // window) execute the fallback thunk sequence in op-by-op mode.
  if (!params.persistent_alloc_indices.has_value()) {
    if (thunks_ == nullptr) {
      return Internal(
          "CommandBufferThunk requires either valid persistent allocation "
          "indices or a fallback thunk sequence");
    }
    VLOG(2) << "Execute command buffer thunk as a regular thunk sequence "
               "because persistent allocation indices are not yet available";
    return thunks_->ExecuteOnStream(params);
  }

  se::StreamExecutor* executor = params.stream->parent();
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                   GetOrCreateCommandBuffer(executor, params.collective_cliques,
                                            params.collective_memory));

  absl::MutexLock lock(cmd_buffer->mutex);

  // warm up iteration, run through thunks if they are present.
  if (!cmd_buffer->warmup_done && thunks_) {
    VLOG(2) << "Executing warm up iteration of command buffer thunk";
    ABSL_RETURN_IF_ERROR(thunks_->ExecuteOnStream(params));
    cmd_buffer->warmup_done = true;
    return absl::OkStatus();
  }

  bool is_first_record =
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;

  bool has_dynamic_allocations = cmd_buffer->HasDynamicAllocations(
      commands_, *params.persistent_alloc_indices);
  auto updated_allocs = cmd_buffer->UpdateBufferAllocations(
      commands_, params, *params.persistent_alloc_indices);

  // TODO(ezhulenev): Commands captured into the command buffer can't depend on
  // `params.custom_options` as they can change between executions without
  // triggering an update. Commands that read custom options must currently
  // set `requires_update_on_execute()`.
  bool needs_update =
      commands_.requires_update_on_execute() || !updated_allocs.empty();

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
      return TraceMeEncode(
          is_first_record ? "command_buffer::record" : "command_buffer::update",
          {{"device", executor->device_ordinal()},
           {"num_commands", commands_.size()}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    Command::RecordParams record_params = {
        cmd_buffer->state, std::move(updated_allocs),
        /*is_initialization=*/is_first_record && !has_dynamic_allocations};
    // The root spans `Record`, including `Finalize`, so stream_executor sees
    // recording on for the whole recording when telemetry is enabled.
    auto telemetry_root = se::ScopedCommandBufferAnnotation::Root(
        ShouldRecordTelemetry(*cmd_buffer->command_buffer),
        ModuleTopLevelAnnotation());
    ABSL_RETURN_IF_ERROR(RecordCommands(executor, params, std::move(record_params),
                                   cmd_buffer.get(), params.collective_memory));

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

absl::Status CommandBufferThunk::RecordCommands(
    se::StreamExecutor* executor, const Thunk::ExecuteParams& execute_params,
    Command::RecordParams record_params, ExecutorCommandBuffer* cmd_buffer,
    const CollectiveMemory* collective_memory) {
  ABSL_RETURN_IF_ERROR(commands_.Record(execute_params, record_params,
                                   cmd_buffer->command_buffer.get()));
  SetRecordedSymmetricMemoryIds(executor, cmd_buffer, collective_memory);
  return absl::OkStatus();
}

// State tied to each acquired GPU clique. When any tied clique is destroyed,
// `~CliqueState()` resets `DeviceState::command_buffer` so the captured CUDA
// graph is destroyed before `ncclCommDestroy`.
struct CommandBufferThunk::CliqueState {
  CliqueState(std::weak_ptr<State> state, int device_ordinal,
              const ExecutorCommandBuffer* cmd_buffer)
      : state(std::move(state)),
        device_ordinal(device_ordinal),
        cmd_buffer(cmd_buffer) {}

  ~CliqueState() {
    std::shared_ptr<State> locked = state.lock();
    if (!locked) {
      return;
    }
    absl::StatusOr<DeviceState*> device_state =
        locked->device_states.GetOrCreate(device_ordinal);
    if (!device_state.ok()) {
      return;
    }
    std::shared_ptr<ExecutorCommandBuffer> to_destroy;
    {
      absl::MutexLock lock((*device_state)->mutex);
      if ((*device_state)->command_buffer.get() == cmd_buffer) {
        to_destroy = std::move((*device_state)->command_buffer);
      }
    }
    to_destroy.reset();
  }

  std::weak_ptr<State> state;
  int device_ordinal;
  const ExecutorCommandBuffer* cmd_buffer;
};

absl::StatusOr<std::shared_ptr<CommandBufferThunk::ExecutorCommandBuffer>>
CommandBufferThunk::GetOrCreateCommandBuffer(
    se::StreamExecutor* executor, CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  ABSL_ASSIGN_OR_RETURN(
      DeviceState * device_state,
      state_->device_states.GetOrCreate(executor->device_ordinal()));

  // A replaced command buffer is destroyed after `device_state->mutex` is
  // released (or later, by an execution that still holds a reference to it).
  std::shared_ptr<ExecutorCommandBuffer> stale_command_buffer;
  absl::MutexLock lock(device_state->mutex);

  // The command buffer is replaced rather than updated because commands cache
  // their recordings keyed by buffer addresses (CommandExecutor skips unchanged
  // or persistent commands, and TracedCommandBuffer reuses earlier captures
  // when buffer addresses match), so an in-place update would keep the stale
  // capture. Symmetric memories are created collectively, so all ranks
  // re-record in the same execution.
  if (device_state->command_buffer != nullptr && collective_memory != nullptr &&
      device_state->recorded_symmetric_memory_ids.has_value() &&
      !absl::c_equal(*device_state->recorded_symmetric_memory_ids,
                     collective_memory->symmetric_memory_ids())) {
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << "Symmetric memories changed since the command buffer was recorded; "
           "replace it with a new command buffer";
    stale_command_buffer = std::move(device_state->command_buffer);
  }

  if (device_state->command_buffer == nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<se::CommandBuffer> command_buffer,
        executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
    auto cmd_buffer =
        std::make_shared<ExecutorCommandBuffer>(std::move(command_buffer));
    if (collective_cliques != nullptr && !collective_cliques->empty()) {
      std::vector<std::unique_ptr<CliqueState>> clique_states;
      clique_states.reserve(collective_cliques->num_cliques());
      for (size_t i = 0; i < collective_cliques->num_cliques(); ++i) {
        clique_states.push_back(std::make_unique<CliqueState>(
            state_, executor->device_ordinal(), cmd_buffer.get()));
      }
      ABSL_ASSIGN_OR_RETURN(cmd_buffer->clique_refs,
                       collective_cliques->TieAll(std::move(clique_states)));
    }
    device_state->command_buffer = std::move(cmd_buffer);
    device_state->recorded_symmetric_memory_ids.reset();
  }
  return device_state->command_buffer;
}

void CommandBufferThunk::SetRecordedSymmetricMemoryIds(
    se::StreamExecutor* executor, const ExecutorCommandBuffer* cmd_buffer,
    const CollectiveMemory* collective_memory) {
  if (collective_memory == nullptr) {
    return;
  }
  absl::StatusOr<DeviceState*> device_state =
      state_->device_states.GetOrCreate(executor->device_ordinal());
  if (!device_state.ok()) {
    return;
  }
  absl::MutexLock lock((*device_state)->mutex);
  std::optional<std::vector<CollectiveMemory::SymmetricMemoryId>>& recorded =
      (*device_state)->recorded_symmetric_memory_ids;
  // Commands are re-recorded whenever buffer addresses change, but symmetric
  // memories change rarely: copy them only if they differ.
  if ((*device_state)->command_buffer.get() == cmd_buffer &&
      !(recorded.has_value() &&
        absl::c_equal(*recorded, collective_memory->symmetric_memory_ids()))) {
    recorded.emplace(collective_memory->symmetric_memory_ids().begin(),
                     collective_memory->symmetric_memory_ids().end());
  }
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
    ptr->device_states.ForEach([&](DeviceState& device_state) {
      std::shared_ptr<ExecutorCommandBuffer> to_destroy;
      {
        absl::MutexLock state_lock(device_state.mutex);
        to_destroy = std::move(device_state.command_buffer);
      }
      if (to_destroy != nullptr) {
        to_destroy.reset();
        ++num_evicted;
      }
    });
  }

  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}

absl::Status CommandBufferThunk::WalkNested(Walker pre_order,
                                            Walker post_order) {
  if (thunks_ != nullptr) {
    ABSL_RETURN_IF_ERROR(thunks_->Walk(pre_order, post_order));
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
