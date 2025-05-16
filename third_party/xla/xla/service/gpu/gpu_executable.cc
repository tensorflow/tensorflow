/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_executable.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/resource_requests.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/rendezvous.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::ScopedAnnotation;

// Returns the set of `ExecutionStreamIds` requested by all `Thunks` in the
// `GpuExecutable`. At run time `Thunks` may use additional streams to launch
// compute operations in parallel.
static absl::flat_hash_set<ExecutionStreamId> GetExecutionStreamIds(
    const SequentialThunk& thunks) {
  absl::flat_hash_set<ExecutionStreamId> stream_ids;
  thunks.ForAllThunks([&](const Thunk* thunk) {
    if (thunk->execution_stream_id() > 0) {
      stream_ids.insert(thunk->execution_stream_id());
    }
  });
  return stream_ids;
}

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(
    Params params) {
  return std::unique_ptr<GpuExecutable>(new GpuExecutable(std::move(params)));
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      dnn_compiled_graphs_(std::move(params.dnn_compiled_graphs)),
      gpu_version_(params.gpu_version),
      thunks_(std::move(params.executable)),
      execution_stream_ids_(GetExecutionStreamIds(*thunks_)),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.mlir_allocations)),
      buffer_assignment_(std::move(params.buffer_assignment)),
      debug_buffer_assignment_show_max_(
          params.debug_buffer_assignment_show_max),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)),
      enable_debug_info_manager_(params.enable_debug_info_manager) {
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(
          gpu_version_)) {
    // ROCm uses hsaco hashes to distinguish between modules.
    // Bad things happen if multiple modules with identical code are loaded.
    binary_.resize(binary_.size() + 16);
    *(uint64_t*)(&binary_[binary_.size() - 16]) = tsl::EnvTime::NowNanos();
    *(uint64_t*)(&binary_[binary_.size() - 8]) = tsl::random::New64();
  }
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               buffer_assignment_->ToProto());
  }
}

GpuExecutable::~GpuExecutable() {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

absl::Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::Platform::Id platform_id =
      main_stream->parent()->GetPlatform()->id();
  if (platform_id == stream_executor::rocm::kROCmPlatformId) {
    auto cc = main_stream->GetRocmComputeCapability();
    std::string stream_arch = cc.gcn_arch_name();
    std::string gpu_exec_arch =
        std::get<se::RocmComputeCapability>(gpu_version_).gcn_arch_name();
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_id == stream_executor::cuda::kCudaPlatformId) {
    se::GpuComputeCapability cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(std::get<se::CudaComputeCapability>(cc) ==
                 std::get<se::CudaComputeCapability>(gpu_version_))
        << "Compute capability mismatch; expected {"
        << std::get<se::CudaComputeCapability>(gpu_version_).ToString()
        << "}, but was {" << std::get<se::CudaComputeCapability>(cc).ToString()
        << "}";
  } else if (platform_id == stream_executor::sycl::kSyclPlatformId) {
    // TODO: Add check.
  } else {
    return Internal("Unknown platform");
  }

  return absl::OkStatus();
}

namespace {

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync);

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options);

absl::Status ExecuteThunksImpl(
    const DebugOptions* debug_options, const std::string& module_name,
    ModuleIdentifier module_id, SequentialThunk& thunk_sequence,
    Thunk::ExecutableSource executable_source,
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    const absl::flat_hash_set<ExecutionStreamId>& execution_stream_ids) {
  bool mock_collectives =
      run_options->run_options().gpu_executable_run_options()
          ? run_options->run_options()
                .gpu_executable_run_options()
                ->enable_mock_collectives()
          : false;

  int64_t collective_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_collective_max_nchannels()
                    : 0;
  int64_t p2p_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_p2p_max_nchannels() : 0;
  bool use_highest_priority_for_async_stream =
      debug_options
          ? debug_options->xla_gpu_enable_highest_priority_async_stream()
          : false;

  bool requires_exclusive_lock_on_gpu =
      run_options->run_options().gpu_executable_run_options()
          ? run_options->run_options()
                .gpu_executable_run_options()
                ->requires_exclusive_lock_on_gpu()
          : false;

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  stream_executor::StreamPriority stream_priority =
      stream_executor::StreamPriority::Default;
  if (use_highest_priority_for_async_stream) {
    stream_priority = stream_executor::StreamPriority::Highest;
  }

  // Borrow streams required for CollectiveThunk.
  absl::InlinedVector<se::Stream*, kAsyncStreamTotal> async_comms_streams(
      kAsyncStreamTotal, nullptr);
  se::Stream* command_buffer_trace_stream = nullptr;
  std::vector<StreamPool::Ptr> async_comms_streams_ownr;
  StreamPool::Ptr borrowed_command_buffer_trace_stream;
  if (run_options->HasStreamBorrower()) {
    TF_ASSIGN_OR_RETURN(
        async_comms_streams_ownr,
        run_options->BorrowStreams(executor->device_ordinal(),
                                   kAsyncStreamTotal, stream_priority));
    for (int64_t i = 0; i < kAsyncStreamTotal; ++i) {
      async_comms_streams[i] = async_comms_streams_ownr[i].get();
    }

    // Borrow stream for tracing command buffers.
    TF_ASSIGN_OR_RETURN(borrowed_command_buffer_trace_stream,
                        run_options->BorrowStream(executor->device_ordinal()));
    command_buffer_trace_stream = borrowed_command_buffer_trace_stream.get();
  }

  // Borrow stream for additional compute streams
  Thunk::ExecutionStreamIdMap additional_execution_streams;
  std::vector<StreamPool::Ptr> additional_streams;
  if (!execution_stream_ids.empty()) {
    TF_ASSIGN_OR_RETURN(additional_streams, run_options->BorrowStreams(
                                                executor->device_ordinal(),
                                                execution_stream_ids.size()));
    int64_t i = 0;
    for (ExecutionStreamId stream_id : execution_stream_ids) {
      additional_execution_streams[stream_id] = additional_streams.at(i).get();
      i++;
    }
    VLOG(2) << "Using " << additional_execution_streams.size()
            << " additional compute streams.";
  }

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::unique_ptr<se::EventBasedTimer> execution_timer;
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    TF_ASSIGN_OR_RETURN(execution_timer, main_stream->CreateEventBasedTimer(
                                             profile->warmup_run_executed()));
  }

  // Parameters for executing collective operations.
  TF_ASSIGN_OR_RETURN(Thunk::CollectiveExecuteParams collective_params,
                      Thunk::CollectiveExecuteParams::Create(
                          *run_options, async_comms_streams,
                          main_stream->parent()->device_ordinal(),
                          collective_max_nchannels, p2p_max_nchannels));

  ResourceRequests resource_requests;

  {  // Collect resource requirements from thunks.
    Thunk::PrepareParams prepare_params{&collective_params};

    tsl::profiler::TraceMe trace_prepare("Thunks::Prepare");
    TF_RETURN_IF_ERROR(
        thunk_sequence.Prepare(prepare_params, resource_requests));
  }

  // Acquire collective cliques requested by thunks.
  Thunk::CollectiveCliques collective_cliques;
  if (!mock_collectives) {
    TF_ASSIGN_OR_RETURN(
        collective_cliques,
        resource_requests.AcquireCollectiveCliques(
            collective_params,
            debug_options
                ? debug_options->xla_gpu_collectives_use_persistent_cliques()
                : false));
  }

  {  // Initialize thunks using prepared resources before execution.
    Thunk::InitializeParams initialize_params{
        executor,
        executable_source,
        &buffer_allocations,
        main_stream,
        command_buffer_trace_stream,
        &collective_params,
        &collective_cliques,
        run_options->run_options().ffi_execution_context(),
        run_options->local_device_count(),
        requires_exclusive_lock_on_gpu};

    tsl::profiler::TraceMe trace_initialize("Thunks::Initialize");
    TF_RETURN_IF_ERROR(thunk_sequence.Initialize(initialize_params));
  }

  // Maybe join a round of rendezvous after thunk initialization. We do this
  // only in presence of newly acquired collective cliques which means that we
  // have collective operations and clique initialization is famous for
  // introducing deadlocks if we try to execute it concurrently with other
  // potentially memory-allocating operations.
  if (collective_cliques.num_transient_cliques() > 0) {
    TF_RETURN_IF_ERROR(
        RendezvousAfterInitialization(run_options, debug_options));
  }

  // Prepare parameters for thunks execution.
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      *run_options, buffer_allocations, main_stream,
      command_buffer_trace_stream, &collective_params, &collective_cliques,
      std::move(additional_execution_streams));

  VLOG(1) << "[" << run_options->device_ordinal() << "] "
          << "Start GpuExecutable::ExecuteOnStream module: " << module_name;
  TF_RETURN_IF_ERROR(thunk_sequence.ExecuteOnStream(execute_params));
  VLOG(1) << "[" << run_options->device_ordinal() << "] "
          << "End GpuExecutable::ExecuteOnStream module: " << module_name;

  return MaybeSyncAndProfile(run_options, execution_timer.get(),
                             block_host_until_done ? main_stream : nullptr);
}

namespace {
// Wrap RunId into a unique struct to guarantee we do not accidentally try to
// run multiple unrelated rendezvous for a same key.
struct InitializationKey {
  RunId run_id;

  template <typename H>
  friend H AbslHashValue(H h, const InitializationKey& key) {
    return H::combine(std::move(h), key.run_id);
  }
};

bool operator==(const InitializationKey& a, const InitializationKey& b) {
  return a.run_id == b.run_id;
}
}  // namespace

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options) {
  // Thunk initialization can allocate new control data structures on device
  // that can lead to deadlocks if other replicas are executing concurrently
  // (i.e. this happens if we try to instantiate CUDA graph when other replica
  // is executing NCCL kernels). If we detect that we are running in multi-gpu
  // setup we synchronize after first initialization to make sure that all
  // replicas completed initialization process before we start execution.
  auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
  auto* device_assn = run_options->run_options().device_assignment();

  // If we don't have Gpu executable options or device assignment it means we
  // are running in a single Gpu config and don't need a rendezvous.
  if (!gpu_opts || !device_assn) return absl::OkStatus();

  // Assume that all participants execute locally first, if we have a local
  // device id to global device id map we will use it to get the real number of
  // participating local devices.
  int64_t num_local_participants =
      device_assn->replica_count() * device_assn->computation_count();

  // Find what local devices are part of the device assignment.
  if (gpu_opts->gpu_global_device_ids()) {
    auto d2l_map = device_assn->GetDeviceToLogicalIdMap();

    num_local_participants = 0;
    for (auto& [local_id, global_id] : *gpu_opts->gpu_global_device_ids()) {
      num_local_participants += d2l_map.contains(global_id);
    }

    if (num_local_participants == 0) {
      return absl::InternalError(
          "Cound't find the number of local participants");
    }
  }

  VLOG(1) << "Join thunks initialization rendezvous with "
          << num_local_participants << " local participants"
          << "; device_ordinal=" << run_options->device_ordinal();

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "RendezvousAfterInitialization",
        {{"run_id", run_options->run_options().run_id().ToInt()},
         {"num_local_participants", num_local_participants}});
  });

  auto rendezvous_key = InitializationKey{run_options->run_options().run_id()};
  auto rendezvous_name = absl::StrFormat(
      "thunk initialization completion for device ordinal %d; run_id=%d",
      run_options->device_ordinal(),
      run_options->run_options().run_id().ToInt());

  return Rendezvous(
      rendezvous_name, rendezvous_key, num_local_participants,
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_warn_stuck_timeout_seconds()
              : 10),
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_terminate_timeout_seconds()
              : 30));
}

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync = nullptr) {
  // If we're measuring the execution time then it's important to queue the
  // stop event before triggering any synchronization.
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed,
                        execution_timer->GetElapsedDuration());
    profile->set_compute_time_ns(
        std::max(absl::ToDoubleNanoseconds(elapsed), 1.0));
  }

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    absl::Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return Internal(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.message());
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(&module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() ==
            stream_executor::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    TF_ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<stream_executor::DeviceMemoryBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceMemoryBase global;
    if (static_cast<bool>(module_handle) && global_status.ok()) {
      // The constant was defined in the PTX and has been allocated by the CUDA
      // driver.
      global = *global_status;
      VLOG(3) << "Resolved global " << info.symbol_name << " to "
              << global.opaque();

      if (!info.content.span().empty()) {
        // This means the constant did not have an initializer in the PTX and
        // therefore must be initialized by XLA here.
        TF_RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                          info.content.span().size()));
        submitted_mem_copies = true;
      }
    } else {
      // The constant was not defined in the PTX and therefore must be both
      // allocated and initialized by XLA here.
      CHECK(!info.content.span().empty());

      TF_ASSIGN_OR_RETURN(auto shared, executor->CreateOrShareConstant(
                                           stream, info.content.span()));
      global = *shared;
      VLOG(3) << "Allocated (or shared) global " << info.symbol_name << " at "
              << global.opaque();
      // XLA will continue to own this global at least until this executable is
      // destroyed (longer if another, longer-lived executable shares the same
      // constant).
      shared_constants_.push_back(std::move(shared));
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    TF_CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

absl::StatusOr<se::DeviceMemoryBase> GpuExecutable::BufferForAllocation(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx) {
  if (allocation.is_thread_local()) {
    return se::DeviceMemoryBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64_t param_no = allocation.parameter_number();
    se::DeviceMemoryBase registered_buffer = [&] {
      if (auto unowned_shapedbuffers =
              std::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return std::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceMemoryBase();
      }
    }();
    if (registered_buffer.is_null() && registered_buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(), param_no);
    }
    return registered_buffer;
  } else if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceMemoryBase();
    }
    return it->second;
  } else {
    // Allocate each allocation that might escape, or is the temp buffer.
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    const int64_t buffer_size = allocation.size();
    se::DeviceMemoryBase buffer_address;
    if (buffer_size > 0) {
      TF_ASSIGN_OR_RETURN(
          se::OwningDeviceMemory buffer,
          memory_allocator->Allocate(device_ordinal, buffer_size,
                                     /*retry_on_failure=*/true,
                                     /*memory_space=*/allocation.color()));
      buffer_address = buffer.Release();
    }
    return buffer_address;
  }
}

static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceMemoryBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    } else {
      return kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

absl::StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::Span<const BufferAllocation> allocations = GetAllocations();
  const int64_t num_buffers = allocations.size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations[i];
    TF_ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(arguments, globals, allocations[i],
                            memory_allocator, device_ordinal, i));
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

absl::StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, &GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(&GetGpuMutex(executor));
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/output_shape_, memory_allocator,
                         device_ordinal, executor->device_ordinal());

  TF_ASSIGN_OR_RETURN(
      BufferAllocations buffer_allocations,
      GenerateBufferAllocations(arguments, globals, memory_allocator,
                                device_ordinal));
  VLOG(3) << buffer_allocations.ToString();
  absl::Span<const BufferAllocation> allocations = GetAllocations();

  std::set<se::DeviceMemoryBase> buffers_in_result;

  const bool is_entire_tuple_contents_aliased = [&] {
    for (auto& p : result.MutableResult()->buffers().leaves()) {
      if (!output_info_.contains(p.first)) {
        continue;
      }
      const OutputInfo& output_info = output_info_.at(p.first);
      if (!output_info.alias_config.has_value()) {
        return false;
      }
    }
    return true;
  }();

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        &allocations[output_info.allocation_index];
    se::DeviceMemoryBase& result_buffer = p.second;

    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceMemory* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceMemory* {
        // ScopedBuffer is never an owned buffer.
        if (std::holds_alternative<absl::Span<const ShapedBuffer* const>>(
                arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              std::get<absl::Span<ExecutionInput>>(arguments);
          ExecutionInput& input =
              unowned_execution_input[allocation->parameter_number()];
          return input.MutableBuffer(allocation->param_shape_index());
        }
      }();
      if (output_info.alias_config->must_alias() && maybe_owning_memory &&
          !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (maybe_owning_memory && maybe_owning_memory->HasOwnership()) {
        std::optional<tensorflow::se::OwningDeviceMemory> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error from the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vector, which will be fed to the ExecutionOutput, which will use
        // the indices to drop the addresses from its own ScopedShapedBuffer
        // result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(output_shape_, index).IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(output_shape_, index));
        absl::StatusOr<se::OwningDeviceMemory> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return VerboseAllocationError(allocated_buffer.status());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  TF_RETURN_IF_ERROR(ExecuteThunks(buffer_allocations, run_options));

  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, GetAllocations()));

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

absl::Status GpuExecutable::VerboseAllocationError(absl::Status s) {
  return ResourceExhausted(
      "%s\n%s\n", s.message(),
      buffer_assignment_->ToVerboseString(debug_buffer_assignment_show_max_));
}

absl::Status GpuExecutable::ExecuteThunks(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("GpuExecutable::ExecuteThunks",
                                        {{"module_name", module_name_}});
  });

  if (VLOG_IS_ON(5)) {
    se::StreamExecutor* executor = run_options->stream()->parent();
    // Debug code to compare current allocation's address with previous run's
    // address, and report the allocation info if memory addressed changed.
    // Useful for identify in user's model if it is command buffer perf friendly
    // (no command buffer update cost).
    absl::MutexLock lock(&module_handle_mutex_);
    if (module_allocations_.find(executor) == module_allocations_.end()) {
      std::vector<se::DeviceMemoryBase> allocs_addr;
      allocs_addr.reserve(buffer_allocations.size());
      for (int i = 0; i < buffer_allocations.size(); i++) {
        allocs_addr.push_back(buffer_allocations.GetDeviceAddress(i));
      }
      module_allocations_[executor] = std::move(allocs_addr);
    } else {
      for (int i = 0; i < buffer_allocations.size(); i++) {
        if (module_allocations_[executor][i].IsSameAs(
                buffer_allocations.GetDeviceAddress(i))) {
          continue;
        }
        module_allocations_[executor][i] =
            buffer_allocations.GetDeviceAddress(i);
        VLOG(5) << "Gpu address changed for module " << module_name_;
      }
    }
  }

  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  ScopedAnnotation annotation([&] { return module_annotations_.top_level; });
  ScopedModuleAnnotations module_annotations(&module_annotations_);

  ModuleIdentifier unique_id = has_module() ? module().unique_id() : -1;
  Thunk::ExecutableSource executable_source = {text_, binary_,
                                               dnn_compiled_graphs_};

  TF_RETURN_IF_ERROR(ExecuteThunksImpl(
      has_module() ? &module_config().debug_options() : nullptr, module_name_,
      unique_id, *thunks_, executable_source, run_options, buffer_allocations,
      block_host_until_done, execution_stream_ids_));
  return absl::OkStatus();
}

int64_t GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (const auto& allocation : GetAllocations()) {
    if (allocation.is_constant()) {
      size += allocation.size();
    }
  }
  return size;
}

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment) {
  const HloInstruction* root =
      hlo_module.entry_computation()->root_instruction();

  InstructionValueSet root_value_set =
      assignment.dataflow_analysis().GetInstructionValueSet(root);

  if (root_value_set.IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  using OutputInfoMap =
      absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
  OutputInfoMap output;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> absl::Status {
        const auto& sources = root_value_set.element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        GpuExecutable::OutputInfo& info = output[index];
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter;
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();

        output[index].alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);

        return absl::OkStatus();
      }));
  return output;
}

OutputInfoProto GpuExecutable::OutputInfo::ToProto() const {
  OutputInfoProto proto;
  proto.set_allocation_index(allocation_index);
  proto.set_passthrough(passthrough);

  if (alias_config.has_value()) {
    proto.mutable_alias_config()->set_parameter_number(
        alias_config->parameter_number);
    proto.mutable_alias_config()->mutable_parameter_shape_index()->Assign(
        alias_config->parameter_index.begin(),
        alias_config->parameter_index.end());

    switch (alias_config->kind) {
      case xla::HloInputOutputAliasConfig::AliasKind::kMayAlias:
        proto.mutable_alias_config()->set_kind(Kind::MAY_ALIAS);
        break;
      case xla::HloInputOutputAliasConfig::AliasKind::kMustAlias:
        proto.mutable_alias_config()->set_kind(Kind::MUST_ALIAS);
        break;
    }
  }

  return proto;
}

absl::StatusOr<GpuExecutable::OutputInfo> GpuExecutable::OutputInfo::FromProto(
    const OutputInfoProto& proto) {
  OutputInfo output_info;
  output_info.allocation_index = proto.allocation_index();
  output_info.passthrough = proto.passthrough();
  if (proto.has_alias_config()) {
    xla::HloInputOutputAliasConfig::AliasKind alias_kind;
    switch (proto.alias_config().kind()) {
      case Kind::MAY_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMayAlias;
        break;
      case Kind::MUST_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMustAlias;
        break;
      default:
        return absl::InvalidArgumentError("Given alias kind is invalid");
    }
    const auto& parameter_shape_index =
        proto.alias_config().parameter_shape_index();
    output_info.alias_config.emplace(
        proto.alias_config().parameter_number(),
        ShapeIndex{parameter_shape_index.begin(), parameter_shape_index.end()},
        alias_kind);
  }
  return output_info;
}
}  // namespace gpu
}  // namespace xla
