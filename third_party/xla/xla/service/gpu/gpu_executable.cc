/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/map_util.h"
#include "xla/mlir/runtime/ir/rt_ops.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "xla/mlir/runtime/transforms/type_converter.h"
#include "xla/runtime/executable.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "xla/service/gpu/runtime/executable.h"
#include "xla/service/gpu/runtime/tracing.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/rendezvous.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

#if TENSORFLOW_USE_ROCM
#include "tsl/platform/random.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#else
namespace stream_executor::gpu {
class GpuTimer {};
}  // namespace stream_executor::gpu
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

bool IsXlaRuntimeExecutableEnabled(const HloModuleConfig& config) {
  return config.debug_options().xla_gpu_enable_xla_runtime_executable();
}

namespace {

using ::tsl::profiler::ScopedAnnotation;
using ::tsl::profiler::ScopedAnnotationAlways;

bool NeedsAsyncCommsStream(Thunk& thunk) {
  switch (thunk.kind()) {
    case Thunk::Kind::kNcclAllReduceStart:
    case Thunk::Kind::kNcclAllReduceDone:
      return true;
    default:
      return false;
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(
    Params params) {
  auto executable = std::move(params.executable);
  std::unique_ptr<GpuExecutable> result(new GpuExecutable(std::move(params)));

  if (std::holds_alternative<OwnedThunkSequence>(executable)) {
    result->thunks_ = std::move(std::get<OwnedThunkSequence>(executable));
    return result;
  }

  if (std::holds_alternative<OwnedGpuRuntimeProgram>(executable)) {
    auto& program = std::get<OwnedGpuRuntimeProgram>(executable);
    TF_ASSIGN_OR_RETURN(
        result->gpu_runtime_executable_,
        GpuRuntimeExecutable::Create(result->module_name_, std::move(program)));
    return result;
  }

  return InternalError("No XLA gpu executable was provided");
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      gpu_version_(params.gpu_version),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.mlir_allocations)),
      buffer_assignment_(std::move(params.buffer_assignment)),
      debug_buffer_assignment_show_max_(
          params.debug_buffer_assignment_show_max),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)),
      enable_debug_info_manager_(params.enable_debug_info_manager) {
#if TENSORFLOW_USE_ROCM
  // ROCm uses hsaco hashes to distinguish between modules.
  // Bad things happen if multiple modules with identical code are loaded.
  binary_.resize(binary_.size() + 16);
  *(uint64_t*)(&binary_[binary_.size() - 16]) = tsl::EnvTime::NowNanos();
  *(uint64_t*)(&binary_[binary_.size() - 8]) = tsl::random::New64();
#endif
  if (has_module()) {
    annotation_info_.emplace(module());
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
      main_stream->parent()->platform()->id();
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
  } else {
    return InternalError("Unknown platform");
  }

  return absl::OkStatus();
}

namespace {

absl::Status MaybeSyncAndProfile(
    const ServiceExecutableRunOptions* run_options,
    std::optional<se::gpu::GpuTimer> execution_timer,
    se::Stream* stream_to_sync);

absl::Status MaybeRendezvousAfterInitialization(
    const ServiceExecutableRunOptions* run_options,
    std::atomic<int64_t>* thunks_initialized);

absl::Status ExecuteThunks(const std::string& module_name,
                           ModuleIdentifier module_id,
                           const ThunkSequence& thunk_sequence,
                           Thunk::ExecutableSource executable_source,
                           const ServiceExecutableRunOptions* run_options,
                           const BufferAllocations& buffer_allocations,
                           bool block_host_until_done,
                           bool use_highest_priority_for_async_stream,
                           std::atomic<int64_t>* thunks_initialized) {
  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  stream_executor::StreamPriority stream_priority =
      stream_executor::StreamPriority::Default;
  if (use_highest_priority_for_async_stream) {
    stream_priority = stream_executor::StreamPriority::Highest;
  }

  // Borrow streams required for NcclCollectiveThunk.
  absl::InlinedVector<se::Stream*, kAsyncStreamTotal> async_comms_streams(
      kAsyncStreamTotal, nullptr);
  absl::StatusOr<std::vector<StreamPool::Ptr>> streams =
      run_options->BorrowStreams(executor->device_ordinal(), kAsyncStreamTotal,
                                 stream_priority);
  if (streams.ok()) {
    for (int64_t i = 0; i < kAsyncStreamTotal; ++i) {
      async_comms_streams[i] = streams->at(i).get();
    }
  }

  // Borrow stream for tracing command buffers.
  se::Stream* command_buffer_trace_stream = nullptr;
  absl::StatusOr<StreamPool::Ptr> borrowed_command_buffer_trace_stream =
      run_options->BorrowStream(executor->device_ordinal());
  if (borrowed_command_buffer_trace_stream.ok()) {
    command_buffer_trace_stream = borrowed_command_buffer_trace_stream->get();
  }

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::optional<se::gpu::GpuTimer> execution_timer;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    TF_ASSIGN_OR_RETURN(
        execution_timer,
        se::gpu::GpuTimer::Create(se::gpu::AsGpuStream(main_stream)));
  }
#endif

  Thunk::ExecuteParams execute_params{*run_options, buffer_allocations,
                                      main_stream, command_buffer_trace_stream,
                                      async_comms_streams};

  // Initialize thunks to prepare them for execution.
  Thunk::InitializeParams initialize_params{
      executor,    executable_source,           &buffer_allocations,
      main_stream, command_buffer_trace_stream, &execute_params.nccl_params};

  for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
    TF_RETURN_IF_ERROR(thunk->Initialize(initialize_params));
  }

  // Maybe join a round of rendezvous after thunk initialization.
  TF_RETURN_IF_ERROR(
      MaybeRendezvousAfterInitialization(run_options, thunks_initialized));

  for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
    VLOG(2) << "Executing the thunk for " << thunk->profile_annotation();
    if (NeedsAsyncCommsStream(*thunk)) {
      for (se::Stream* async_stream : async_comms_streams) {
        TF_RET_CHECK(async_stream != nullptr)
            << "`run_options` must have a stream borrower for async thunks.";
      }
    }

    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(execute_params));
  }
  return MaybeSyncAndProfile(run_options, std::move(execution_timer),
                             block_host_until_done ? main_stream : nullptr);
}

absl::Status MaybeRendezvousAfterInitialization(
    const ServiceExecutableRunOptions* run_options,
    std::atomic<int64_t>* thunks_initialized) {
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

  // If `thunks_initialized` value is `-1` it means that all thunks are
  // initialized and we can go ahead and execute all of them. All other values
  // signal how many threads are executing rendezvous (they can be from
  // different run ids).
  if (thunks_initialized->load() < 0) return absl::OkStatus();

  // We rely on CAS operations to make sure that all participants of
  // potentially multiple concurrent XLA executions join the rendezvous or
  // none of them join, because otherwise we will get a dead lock.
  int64_t participant_id = thunks_initialized->load();
  while (participant_id >= 0 && !thunks_initialized->compare_exchange_weak(
                                    participant_id, participant_id + 1)) {
  }

  // If we exited a CAS loop with participant id less than 0 it means that some
  // other thread completed initialization rendezvous.
  if (participant_id < 0) return absl::OkStatus();

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

  RendezvousSingle(run_options->run_options().run_id(), num_local_participants,
                   absl::Seconds(10), absl::Seconds(30));

  // Reload participant_id and use CAS to decide if we are the one who
  // should mark initialization completed.
  participant_id = thunks_initialized->load();

  // Check that no one completed initialization process without us, and the
  // number of participants inside the critical section is greater than 0 (we
  // are here, so it can't be 0).
  CHECK_GT(participant_id, 0);  // NOLINT

  // If we are the last one, we try to mark executable initialization as
  // completed by writing `-1` into the flag.
  while (!thunks_initialized->compare_exchange_weak(
      participant_id, participant_id == 1 ? -1 : participant_id - 1)) {
    // Check precondition for participant id after CAS failure reloaded it.
    CHECK_GT(participant_id, 0);  // NOLINT
  }

  return absl::OkStatus();
}

absl::Status MaybeSyncAndProfile(
    const ServiceExecutableRunOptions* run_options,
    std::optional<se::gpu::GpuTimer> execution_timer,
    se::Stream* stream_to_sync = nullptr) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // If we're measuring the execution time then it's important to queue the
  // stop event before triggering any synchronization.
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    CHECK(execution_timer.has_value());
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed,
                        execution_timer->GetElapsedDuration());
    profile->set_compute_time_ns(
        std::max(absl::ToDoubleNanoseconds(elapsed), 1.0));
  }
#endif

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    absl::Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
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
  if (!(executor->platform()->id() == stream_executor::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<stream_executor::DeviceMemoryBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status =
          executor->GetUntypedSymbol(info.symbol_name, module_handle);
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
        stream->ThenMemcpy(&global, info.content.span().data(),
                           info.content.span().size());
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
      absl::StatusOr<se::OwningDeviceMemory> buffer =
          memory_allocator->Allocate(device_ordinal, buffer_size,
                                     /*retry_on_failure=*/true,
                                     /*memory_space=*/allocation.color());
      if (!buffer.ok()) {
        return ResourceExhausted("%s\n%s\n", buffer.status().message(),
                                 buffer_assignment_->ToVerboseString(
                                     debug_buffer_assignment_show_max_));
      }
      buffer_address = buffer->Release();
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
    return InternalError(
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
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

absl::StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

static absl::Status ExecuteXlaRuntime(
    const std::string& module_name, ModuleIdentifier module_id,
    GpuRuntimeExecutable& gpu_runtime_executable,
    const ServiceExecutableRunOptions* run_options, const std::string& asm_text,
    const std::vector<uint8_t>& binary,
    const BufferAllocations& buffer_allocations,
    const BufferAllocation* temp_buffer, bool block_host_until_done,
    NonAtomicallyUpgradeableRWLock& gpu_lock) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::optional<se::gpu::GpuTimer> execution_timer;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    TF_ASSIGN_OR_RETURN(
        execution_timer,
        se::gpu::GpuTimer::Create(se::gpu::AsGpuStream(run_options->stream())));
  }
#endif
  auto executed = gpu_runtime_executable.Execute(
      run_options, asm_text, binary, buffer_allocations, gpu_lock, temp_buffer);
  if (!executed.ok()) return executed;

  return MaybeSyncAndProfile(
      run_options, std::move(execution_timer),
      block_host_until_done ? run_options->stream() : nullptr);
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  se::gpu::GpuExecutor* gpu_executor = se::gpu::ExtractGpuExecutor(executor);
  se::gpu::ScopedActivateExecutorContext activation(gpu_executor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  NonAtomicallyUpgradeableRWLock gpu_lock(&GetGpuMutex(executor));
  std::optional<NonAtomicallyUpgradeableRWLock::WriterLock> exclusive_gpu_lock;
  const gpu::GpuExecutableRunOptions* gpu_opts =
      run_options->run_options().gpu_executable_run_options();
  if (gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    exclusive_gpu_lock.emplace(&gpu_lock);
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  auto device_ordinal = executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/output_shape_, memory_allocator,
                         device_ordinal);

  TF_ASSIGN_OR_RETURN(
      BufferAllocations buffer_allocations,
      GenerateBufferAllocations(arguments, globals, memory_allocator,
                                device_ordinal));
  VLOG(2) << buffer_allocations.ToString();
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

  absl::Span<const BufferAllocation> allocations = GetAllocations();
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
          return ResourceExhausted("%s\n%s\n",
                                   allocated_buffer.status().message(),
                                   buffer_assignment_->ToVerboseString(
                                       debug_buffer_assignment_show_max_));
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        run_options->stream()->ThenMemcpyD2D(&result_buffer, aliased_buffer,
                                             aliased_buffer.size());
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

  TF_RETURN_IF_ERROR(ExecuteThunksOrXlaRuntime(
      run_options, buffer_allocations, block_host_until_done, gpu_lock));

  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, GetAllocations()));

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

namespace {
struct ModuleAnnotationManager {
  ModuleAnnotationManager(const std::optional<ModuleAnnotations>& annotations) {
    if (annotations.has_value()) {
      m_old_annotations = SetCurrentModuleAnnotations(&(*annotations));
    }
  }
  ~ModuleAnnotationManager() {
    if (m_old_annotations.has_value()) {
      SetCurrentModuleAnnotations(*m_old_annotations);
    }
  }

 private:
  std::optional<const ModuleAnnotations*> m_old_annotations;
};
}  // namespace

absl::Status GpuExecutable::ExecuteThunksOrXlaRuntime(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    NonAtomicallyUpgradeableRWLock& gpu_lock) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  // There isn't always an HLO module.
  ModuleIdentifier unique_id = -1;
  if (has_module()) {
    unique_id = module().unique_id();
  }

  ScopedAnnotationAlways annotation([&]() -> ModuleAnnotation {
    if (annotation_info_) {
      return annotation_info_->top_level;
    } else {
      return {module_name_, unique_id};
    }
  });
  ModuleAnnotationManager set_current_kernel_annotations{annotation_info_};

  if (thunks_) {
    Thunk::ExecutableSource executable_source = {text_, binary_};

    return ExecuteThunks(
        module_name_, unique_id, *thunks_, executable_source, run_options,
        buffer_allocations, block_host_until_done,
        /*use_highest_priority_for_async_stream*/
        has_module() ? module_config()
                           .debug_options()
                           .xla_gpu_enable_highest_priority_async_stream()
                     : false,
        &thunks_initialized_);
  }

  // Match IrEmitter's temp buffer allocation for kernel launches. See
  // IrEmitterUnnested::BuildKernelThunkImpl().
  const BufferAllocation* temp_buffer = nullptr;
  for (const BufferAllocation& alloc : GetAllocations()) {
    if (alloc.IsPreallocatedTempBuffer()) {
      // Retrieve the first seen temp buffer.
      if (temp_buffer == nullptr) temp_buffer = &alloc;
    }
  }

  if (gpu_runtime_executable_) {
    return ExecuteXlaRuntime(module_name_, unique_id, *gpu_runtime_executable_,
                             run_options, text_, binary_, buffer_allocations,
                             temp_buffer, block_host_until_done, gpu_lock);
  }

  return FailedPrecondition("Expected XLA gpu executable is not supplied.");
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

absl::Status GpuExecutable::SetUpMlirAllocation(
    mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
    std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape) {
  for (int i = 0; i < buffer_sizes.size(); i++) {
    // This code path is taken when using the non-thunk based runtime. Memory
    // space is being set to 0 for all allocations. We need to copy over the
    // value from BufferAssignment instead.
    allocations->emplace_back(i, buffer_sizes[i], /*memory_space=*/0);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      xla::ShapeIndex shape_index;
      if (auto shape_index_attr =
              func.getArgAttrOfType<mlir::DenseIntElementsAttr>(
                  i, "lmhlo.param_shape_index")) {
        for (const llvm::APInt& element : shape_index_attr) {
          shape_index.push_back(element.getSExtValue());
        }
      }
      allocations->at(i).set_entry_computation_parameter(
          param_attr.cast<mlir::IntegerAttr>().getInt(), shape_index,
          static_cast<bool>(func.getArgAttr(i, "lmhlo.output_index")));
    }
    // TODO(timshen): this information is redundant. This is here only for
    // smooth migration to LMHLO. Remove it.
    if (func.getArgAttr(i, "lmhlo.constant_name")) {
      allocations->at(i).set_constant(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      allocations->at(i).set_maybe_live_out(true);

      // Reconstruct a shape index from output_index.
      ShapeIndex shape_index;
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        shape_index.push_back(element.getSExtValue());
      }
      auto& o = (*output_info)[shape_index];
      o.allocation_index = i;
      if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
        HloInputOutputAliasConfig::AliasKind kind =
            HloInputOutputAliasConfig::kMayAlias;
        if (func.getArgAttr(i, "lmhlo.must_alias")) {
          kind = HloInputOutputAliasConfig::kMustAlias;
        }
        o.alias_config.emplace(param_attr.cast<mlir::IntegerAttr>().getInt(),
                               ShapeIndex{}, kind);
      }
      if (func.getArgument(i).use_empty()) {
        o.passthrough = true;
      }
    }
  }
  // Expects result_xla_shape as a XLA shape in string form.
  //
  // The attribute is necessary, because GpuExecutable/ExecutionOutput supports
  // tuples / tree-like shapes, while the LMHLO argument list loses the tree
  // form.
  //
  // The string format is necessary since MLIR doesn't support XLA shape with
  // dynamic_dimension.
  //
  // TODO(timshen): now this field is mandatory. Make it optional for
  // non-GpuExecutable outputs.
  TF_ASSIGN_OR_RETURN(
      *output_shape,
      ParseShape(func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
                     .getValue()
                     .str()));

  return absl::OkStatus();
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

GpuExecutable::GpuExecutable(
    std::shared_ptr<HloModule> hlo_module, std::string asm_text,
    std::vector<uint8_t> binary, std::vector<ConstantInfo> constants,
    se::GpuComputeCapability gpu_version, absl::string_view module_name,
    Shape xla_output_shape, std::vector<BufferAllocation> allocations,
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
    std::unique_ptr<GpuRuntimeExecutable> gpu_runtime_executable)
    : Executable(std::move(hlo_module)),
      text_(std::move(asm_text)),
      binary_(std::move(binary)),
      gpu_version_(gpu_version),
      gpu_runtime_executable_(std::move(gpu_runtime_executable)),
      module_name_(module_name),
      output_shape_(xla_output_shape),
      allocations_(std::move(allocations)),
      constants_(std::move(constants)),
      output_info_(std::move(output_info)),
      enable_debug_info_manager_(true) {
  if (has_module()) {
    annotation_info_.emplace(module());
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               BufferAssignmentProto());
  }
}

// Returns a list of functions exported from the `module` that should be loaded
// from the object file. Entrypoint functions always loaded with ordinal 0.
static absl::StatusOr<std::vector<runtime::Executable::LoadFunction>>
GetFunctionsToLoad(mlir::ModuleOp module, std::string_view entry) {
  std::vector<runtime::Executable::LoadFunction> functions;

  // Use canonical type converter because we currently do not support any
  // user-defined types in XLA:GPU executables.
  runtime::TypeConverter type_converter;

  // Converts function type and adds load function metadata. In XLA:GPU exported
  // function runtime signature is the same as regular signature with an extra
  // execution context argument at index 0.
  auto convert = [&](mlir::func::FuncOp func) -> absl::Status {
    auto signature = type_converter.Convert(func.getFunctionType());
    if (!signature.ok())
      return InternalError("Failed to convert entry function type: %s",
                           signature.status().message());

    // TODO(ezhulenev): Copy `signature` once FunctionType is copyable.
    auto rt_signature = type_converter.Convert(func.getFunctionType());
    rt_signature->insert_operand(
        0, std::make_unique<runtime::ExecutionContextOperandType>());

    functions.push_back({func.getName().str(), std::move(*signature),
                         std::move(*rt_signature)});

    return absl::OkStatus();
  };

  mlir::SymbolTable sym_table(module);

  // Load entrypoint function first at ordinal 0.
  TF_CHECK_OK(convert(module.lookupSymbol<mlir::func::FuncOp>(entry)));

  // Load all functions explicitly exported from the module (in XLA:GPU it's
  // always CUDA graph capture functions). We explicitly sort them by ordinal,
  // to make sure they are loaded in correct order.
  auto export_ops = llvm::to_vector(module.getOps<runtime::ExportOp>());
  llvm::sort(export_ops, [](runtime::ExportOp a, runtime::ExportOp b) {
    return a.getOrdinal()->getSExtValue() < b.getOrdinal()->getSExtValue();
  });
  for (runtime::ExportOp exported : export_ops) {
    TF_CHECK_OK(convert(
        sym_table.lookup<mlir::func::FuncOp>(exported.getFunctionRef())));
  }

  return functions;
}

// Get arguments buffer sizes from the entry function signature.
static absl::StatusOr<std::vector<int64_t>> GetBufferSizes(
    runtime::FunctionType& f) {
  std::vector<int64_t> buffer_sizes;
  for (unsigned i = 0; i < f.num_operands(); ++i) {
    auto* memref = llvm::dyn_cast<runtime::MemrefType>(f.operand(i));

    // Entry function argument must be a statically shaped 1d I8 memref.
    if (memref == nullptr || memref->element_type() != PrimitiveType::S8 ||
        memref->rank() != 1 || runtime::MemrefType::IsDynamic(memref->size(0)))
      return InternalError("Illegal buffer argument type: %s",
                           f.operand(0)->ToString());

    buffer_sizes.push_back(memref->size(0));
  }
  return buffer_sizes;
}

// TODO(ezhulenev): This is a copy of `GetAllocationIndices` from
// `mlir/backends/gpu/transforms/passes.h`. We can't depend on that file because
// of a dependency cycle, and this is a short term work around the cuda graph
// capture bug. This code should not survive beyond Q1 2024.
static std::vector<std::vector<int64_t>> GetAllocationIndices(
    mlir::ModuleOp module) {
  std::vector<std::vector<int64_t>> res;

  mlir::SymbolTable sym_table(module);
  for (auto op : module.getOps<runtime::ExportOp>()) {
    unsigned ordinal = *op.ordinal();
    if (ordinal >= res.size()) res.resize(ordinal + 1);

    auto func = sym_table.lookup<mlir::func::FuncOp>(op.getFunctionRef());
    res[ordinal].resize(func.getNumArguments(), -1);

    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      auto idx =
          func.getArgAttrOfType<mlir::IntegerAttr>(i, "rt.allocation_index");
      if (idx) res[ordinal][i] = idx.getInt();
    }
  }

  return res;
}

absl::StatusOr<std::unique_ptr<Executable>> GpuExecutable::LoadFromObjFile(
    std::shared_ptr<HloModule> hlo_module, absl::string_view obj_file,
    absl::string_view mlir_module, DebugOptions debug_options,
    absl::string_view asm_text, absl::string_view binary,
    std::vector<ConstantInfo> constants, se::GpuComputeCapability gpu_version) {
  VLOG(1) << "Load serialized Gpu executable from object file: module="
          << hlo_module->name();

  std::string_view entry = hlo_module->entry_computation()->name();

  // Load MLIR module behind the compiled object file to recover XLA allocations
  // and output info details. Also recover buffer sizes from the entrypoint
  // function signature.
  mlir::MLIRContext context;
  runtime::AppendXlaGpuDialectRegistry(context);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_module, &context);
  if (!module) return InternalError("Failed to parse AOT compiled module");

  // Get the list of functions to be loaded from the object file.
  TF_ASSIGN_OR_RETURN(std::vector<runtime::Executable::LoadFunction> functions,
                      GetFunctionsToLoad(*module, entry));
  VLOG(2) << "Found " << functions.size() << " functions to load";

  // Get the buffer sizes from the entry function signature.
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> buffer_sizes,
                      GetBufferSizes(functions[0].signature));

  // Get allocation indices from graph capture functions.
  auto allocation_indices = GetAllocationIndices(*module);

  // Get the XLA module entrypoint function.
  auto func = mlir::cast<mlir::func::FuncOp>(module->lookupSymbol(entry));

  // Infer XLA allocations and output info from the MLIR module.
  std::vector<BufferAllocation> allocations;
  absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
  Shape result_xla_shape;
  TF_RETURN_IF_ERROR(SetUpMlirAllocation(func, buffer_sizes, &allocations,
                                         &output_info, &result_xla_shape));

  // Create a named buffer from compiled object file.
  llvm::StringRef data(obj_file.data(), obj_file.size());
  auto buffer = llvm::MemoryBuffer::getMemBuffer(data, hlo_module->name());

  auto symbol_map = runtime::ToSymbolsBinding(RegisterXlaGpuRuntimeCustomCalls,
                                              RegisterXlaGpuTypeIdNames);

  // Load XLA Runtime executable from an object file, and link it with Gpu
  // runtime intrinsics implementing Gpu custom calls.
  auto executable = runtime::Executable::LoadFromObjFile(
      hlo_module->name(), std::move(buffer), std::move(functions), symbol_map);

  if (!executable.ok())
    return InternalError("Failed to load XLA Runtime executable: %s",
                         executable.status().message());

  // Move runtime::Executable ownership to the GpuRuntimeExecutable.
  TF_ASSIGN_OR_RETURN(auto gpu_runtime_executable,
                      GpuRuntimeExecutable::Create(
                          hlo_module->name(), std::move(buffer_sizes),
                          std::move(allocation_indices), std::move(*executable),
                          std::move(debug_options)));

  // Construct GpuExecutable for the loaded XLA Runtime executable.
  std::string name = hlo_module->name();
  std::string asm_text_string = std::string(asm_text);
  std::vector<uint8_t> binary_vector(binary.begin(), binary.end());
  return std::unique_ptr<Executable>(new GpuExecutable(
      std::move(hlo_module), std::move(asm_text_string),
      std::move(binary_vector), std::move(constants), gpu_version, name,
      result_xla_shape, std::move(allocations), std::move(output_info),
      std::move(gpu_runtime_executable)));
}

absl::StatusOr<std::string_view> GpuExecutable::GetObjFile() const {
  if (!gpu_runtime_executable_)
    return Internal("gpu_runtime_executable is null");
  return gpu_runtime_executable_->GetObjFile();
}

absl::StatusOr<std::string_view> GpuExecutable::GetMlirModule() const {
  if (!gpu_runtime_executable_)
    return Internal("gpu_runtime_executable is null");
  return gpu_runtime_executable_->GetMlirModule();
}

}  // namespace gpu
}  // namespace xla
