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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/platform.h"

#if BEF_EXECUTABLE
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tfrt/gpu/gpu_executor.h"  // from @tf_runtime
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#endif  // BEF_EXECUTABLE

namespace xla {
namespace gpu {
namespace {

using ::tensorflow::profiler::ScopedAnnotation;

bool NeedsAsyncCommsStream(Thunk& thunk) {
  switch (thunk.kind()) {
    case Thunk::Kind::kNcclAllReduceStart:
    case Thunk::Kind::kNcclAllReduceDone:
      return true;
    default:
      return false;
  }
}

static std::string ModuleUniqueName(absl::string_view module_name,
                                    const HloModule* module) {
  std::string unique_id;
  if (module != nullptr) {
    unique_id = absl::StrCat("module.", module->unique_id(), ".");
  }
  return absl::StrCat(unique_id, module_name);
}

}  // namespace

void GpuExecutable::BefBufferDeleter::operator()(uint8_t* ptr) const {
#if BEF_EXECUTABLE
  tfrt::AlignedFree(ptr);
#else
  assert(false && "OwnedBefBuffer only supported with BEF_EXECUTABLE");
#endif
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module),
                 std::move(params.hlo_profile_printer_data),
                 std::move(params.hlo_profile_index_map)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      gpu_version_(params.gpu_version),
      thunks_or_bef_(std::move(params.thunks_or_bef)),
      entry_func_attrs_(params.entry_func_attrs),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.allocations)),
      debug_buffer_assignment_(std::move(params.debug_buffer_assignment)),
      verbose_buffer_assignment_string_dumper_(
          params.verbose_buffer_assignment_string_dumper),
      entry_computation_profile_index_(params.entry_computation_profile_index),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)) {
  XlaDebugInfoManager::Get()->RegisterModule(
      ModuleUniqueName(module_name_, shared_module().get()), shared_module(),
      debug_buffer_assignment_);
}

GpuExecutable::~GpuExecutable() {
  XlaDebugInfoManager::Get()->UnregisterModule(
      ModuleUniqueName(module_name_, shared_module().get()), shared_module(),
      debug_buffer_assignment_);

  {
    // We could have issued host->device mem copies in ResolveConstantGlobals.
    // Wait for those to finish so that we can safely deallocate the backing HLO
    // module.
    //
    // We need for the host->device memcpies to finish they are concurrently
    // reading memory (xla::Literal's) owned by the HLO module.
    tensorflow::mutex_lock lock(module_handle_mutex_);
    for (const auto& pair : module_globals_) {
      CHECK(pair.first->SynchronizeAllActivity());
    }
  }
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::PlatformKind platform_kind =
      main_stream->parent()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kROCm) {
    std::string stream_arch = main_stream->parent()
                                  ->GetDeviceDescription()
                                  .rocm_amdgpu_gcn_arch_name();
    std::string gpu_exec_arch = absl::get<std::string>(gpu_version_);
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_kind == stream_executor::PlatformKind::kCuda) {
    GpuVersion cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(absl::get<se::CudaComputeCapability>(cc) ==
                 absl::get<se::CudaComputeCapability>(gpu_version_))
        << "Compute capability mismatch; expected {"
        << absl::get<se::CudaComputeCapability>(gpu_version_).ToString()
        << "}, but was {" << absl::get<se::CudaComputeCapability>(cc).ToString()
        << "}";
  } else {
    return InternalError("Unknown platform: %d", platform_kind);
  }

  return Status::OK();
}

namespace {

Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                           uint64_t start_micros, se::Stream* stream_to_sync);

Status ExecuteThunks(const std::string& module_name,
                     const ThunkSchedule& thunk_schedule,
                     const ServiceExecutableRunOptions* run_options,
                     const BufferAllocations& buffer_allocations,
                     bool block_host_until_done) {
  XlaDebugInfoManager::Get()->OnModuleStart(module_name);
  auto cleanup = absl::MakeCleanup(
      [&]() { XlaDebugInfoManager::Get()->OnModuleStop(module_name); });

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();

  StatusOr<StreamPool::Ptr> async_comms_stream =
      run_options->BorrowStream(executor->device_ordinal());

  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  std::vector<StreamPool::Ptr> sub_streams;
  sub_streams.reserve(thunk_schedule.StreamCount() - 1);
  while (sub_streams.size() + 1 < thunk_schedule.StreamCount()) {
    sub_streams.emplace_back();
    TF_ASSIGN_OR_RETURN(sub_streams.back(),
                        run_options->BorrowStream(executor->device_ordinal()));
    // Require substreams to wait for the main stream, otherwise substreams may
    // execute before the program is scheduled to start on the main stream.
    sub_streams.back()->ThenWaitFor(main_stream);
  }

  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  absl::flat_hash_map<const Thunk*, std::unique_ptr<se::Event>>
      thunk_to_finish_event;
  for (const std::unique_ptr<Thunk>& thunk : thunk_schedule.TotalOrder()) {
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });

    int32_t stream_no = thunk_schedule.StreamNumberForThunk(thunk.get());
    se::Stream* stream =
        (stream_no == 0 ? main_stream : sub_streams[stream_no - 1].get());

    for (const Thunk* dependency : thunk_schedule.DependsOn(thunk.get())) {
      stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
    }

    VLOG(2) << "Executing the thunk for " << thunk->profile_annotation()
            << " on stream " << stream_no;

    TF_RET_CHECK(async_comms_stream.ok() || !NeedsAsyncCommsStream(*thunk))
        << "`run_options` must have a stream borrower for async thunks.";

    const GpuExecutableRunOptions* gpu_options =
        run_options->run_options().gpu_executable_run_options();
    Thunk::ExecuteParams thunk_params{
        &buffer_allocations,
        stream,
        async_comms_stream.ok() ? async_comms_stream->get() : nullptr,
        run_options->run_options().run_id(),
        run_options->run_options().device_assignment(),
        gpu_options && gpu_options->gpu_global_device_ids()
            ? &*gpu_options->gpu_global_device_ids()
            : nullptr,
        gpu_options && gpu_options->nccl_unique_id_callback()
            ? &gpu_options->nccl_unique_id_callback()
            : nullptr};
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    if (thunk_schedule.Depended(thunk.get())) {
      auto finish_event = absl::make_unique<se::Event>(main_stream->parent());
      finish_event->Init();
      stream->ThenRecordEvent(finish_event.get());
      thunk_to_finish_event[thunk.get()] = std::move(finish_event);
    }
  }

  main_stream->ThenWaitFor(&sub_streams);
  return MaybeSyncAndProfile(run_options, start_micros,
                             block_host_until_done ? main_stream : nullptr);
}

Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                           uint64_t start_micros,
                           se::Stream* stream_to_sync = nullptr) {
  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.error_message());
    }
  }

  // FinishExecution() blocks until main_stream has completed if profiling is
  // enabled; we therefore do not need to defer profile collection onto a
  // stream.
  uint64_t end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->run_options().execution_profile()) {
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return Status::OK();
}

}  // namespace

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  tensorflow::mutex_lock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return &it->second;
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  absl::flat_hash_map<int64_t, se::DeviceMemoryBase> globals;
  if (executor->platform_kind() == se::PlatformKind::kCuda &&
      module_spec.cuda_ptx_in_memory() == nullptr) {
    // No custom PTX => no globals.
    return &module_globals_.emplace(executor, std::move(globals)).first->second;
  }

  se::ModuleHandle module_handle;
  TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));

  for (const auto& info : constants_) {
    TF_ASSIGN_OR_RETURN(auto global, executor->GetUntypedSymbol(
                                         info.symbol_name, module_handle));
    VLOG(3) << "Resolved global " << info.symbol_name << " to "
            << global.opaque();

    if (!info.content.empty()) {
      stream->ThenMemcpy(&global, info.content.data(), info.content.size());
    }

    if (info.allocation_index != -1) {
      InsertOrDie(&globals, info.allocation_index, global);
    }
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return &module_globals_.emplace(executor, std::move(globals)).first->second;
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::BufferForAllocation(
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
              absl::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return absl::get<absl::Span<ExecutionInput>>(arguments)[param_no]
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
      StatusOr<se::OwningDeviceMemory> buffer =
          memory_allocator->Allocate(device_ordinal, buffer_size);
      if (!buffer.ok()) {
        return ResourceExhausted("%s\n%s\n", buffer.status().error_message(),
                                 verbose_buffer_assignment_string_dumper_());
      }
      buffer_address = buffer->Release();
    }
    return buffer_address;
  }
}

static Status CheckAlignment(const BufferAllocation& allocation,
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
  return Status::OK();
}

StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator,
    se::StreamExecutor* executor) {
  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  const int64_t num_buffers = allocations_.size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations_[i];
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buffer,
        BufferForAllocation(arguments, globals, allocation, memory_allocator,
                            executor->device_ordinal(), i));
    buffers.push_back(buffer);
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
  }
  return {{buffers, executor->device_ordinal(), memory_allocator}};
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

#if BEF_EXECUTABLE
static const char kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

struct CoreRuntimeAndWorkQueue {
  mlir::MLIRContext* mlir_ctx;
  tfrt::CoreRuntime* core_runtime;
  tensorflow::tfrt_stub::WorkQueueInterface* work_queue;
};

// TODO(hanbinyoon): Deduplicate with that in bef_thunk.cc when
// tensorflow/core/tfrt/runtime is generally available in OSS.
StatusOr<CoreRuntimeAndWorkQueue> GetCoreRuntimeAndWorkQueue() {
  static auto runtime_and_queue_or =
      [&]() -> StatusOr<CoreRuntimeAndWorkQueue> {
    // TODO(hanbinyoon): Make these configurable.
    int num_threads = tensorflow::port::MaxParallelism();
    int num_blocking_threads = 16;

    // Create work queue.
    auto work_queue = tensorflow::tfrt_stub::WrapDefaultWorkQueue(
        tfrt::CreateMultiThreadedWorkQueue(num_threads, num_blocking_threads));
    if (work_queue == nullptr) {
      return tensorflow::errors::Internal("Failed to create TFRT work queue.");
    }
    auto* work_queue_ptr = work_queue.get();
    auto* mlir_ctx = new mlir::MLIRContext;

    // Create core runtime.
    auto expected_core_runtime = tfrt::CoreRuntime::Create(
        tfrt::gpu::GetDiagHandler(mlir_ctx), tfrt::CreateMallocAllocator(),
        std::move(work_queue), kDefaultHostDeviceName);
    if (!expected_core_runtime) {
      auto error = expected_core_runtime.takeError();
      return tensorflow::errors::Internal(llvm::toString(std::move(error)));
    }

    return CoreRuntimeAndWorkQueue{mlir_ctx, expected_core_runtime->release(),
                                   work_queue_ptr};
  }();
  return runtime_and_queue_or;
}

// TODO(hanbinyoon): Deduplicate with that in bef_thunk.cc when
// tf_runtime/backends/gpu:gpu_types can be built in OSS.
StatusOr<std::unique_ptr<tfrt::gpu::BorrowedGpuStream>> CreateGpuStream(
    stream_executor::Stream* stream) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto se_gpu_executor = static_cast<stream_executor::gpu::GpuExecutor*>(
      stream->parent()->implementation());
  auto se_gpu_stream =
      static_cast<stream_executor::gpu::GpuStream*>(stream->implementation());
  stream_executor::gpu::GpuContextHandle context_handle =
      stream_executor::gpu::GpuDriver::GetContextHandle(
          se_gpu_executor->gpu_context());
  return absl::make_unique<tfrt::gpu::BorrowedGpuStream>(
      tfrt::gpu::wrapper::Context(context_handle),
      tfrt::gpu::wrapper::Stream(se_gpu_stream->gpu_stream()));
#else
  return tensorflow::errors::Unimplemented("GPU is not configured.");
#endif
}

// TODO(hanbinyoon): Deduplicate with that in bef_thunk.cc.
tfrt::RCReference<tfrt::AsyncValue> CreateGpuBuffer(
    stream_executor::DeviceMemoryBase* data) {
  tfrt::gpu::wrapper::Pointer<void> pointer(data->opaque(),
                                            tfrt::gpu::wrapper::Platform::CUDA);
  auto allocator =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuOneShotAllocator<void>>(
          pointer);
  auto buffer =
      tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), data->size());
  if (!buffer)
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(buffer.takeError()));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
      std::move(*buffer));
}

static Status ExecuteBef(const std::string& module_name,
                         const tfrt::RCReference<tfrt::BEFFile>& bef_file,
                         const ServiceExecutableRunOptions* run_options,
                         const BufferAllocations& buffer_allocations,
                         size_t num_allocations, bool block_host_until_done) {
  XlaDebugInfoManager::Get()->OnModuleStart(module_name);
  auto cleanup = absl::MakeCleanup(
      [&]() { XlaDebugInfoManager::Get()->OnModuleStop(module_name); });

  se::Stream* main_stream = run_options->stream();

  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  // TODO(hanbinyoon): Expand on the annotation.
  ScopedAnnotation annotation("BefExecution");

  // Create execution context.
  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  auto resource_ctx = std::make_unique<tfrt::ResourceContext>();
  tfrt::RequestContextBuilder request_context_builder(
      runtime_and_queue.core_runtime->GetHostContext(), resource_ctx.get());
  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    auto error = expected_req_ctx.takeError();
    return tensorflow::errors::Internal(llvm::toString(std::move(error)));
  }
  tfrt::ExecutionContext exec_ctx(std::move(*expected_req_ctx));

  auto entry_point = tfrt::gpu::GetEntryPoint(*bef_file, exec_ctx);
  if (!entry_point) {
    auto error = entry_point.takeError();
    return tensorflow::errors::Internal(llvm::toString(std::move(error)));
  }
  absl::string_view entry_function_name(entry_point->function_name);

  // Signature: (chain, stream, inputs..., outputs...) -> (chain).
  const tfrt::Function* function = bef_file->GetFunction(entry_function_name);
  if (!function) {
    return InternalError("Failed to get '%s' function.", entry_function_name);
  }

  // Create owning handles for arguments and add pointer to them to 'args'.
  tfrt::SmallVector<tfrt::AsyncValue*, 8> args;
  args.reserve(function->num_arguments());
  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain();
  args.push_back(chain.GetAsyncValue());
  TF_ASSIGN_OR_RETURN(auto borrowed_stream, CreateGpuStream(main_stream));
  args.push_back(
      static_cast<tfrt::AsyncValueRef<tfrt::gpu::GpuStream>>(*borrowed_stream)
          .GetAsyncValue());
  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 8> buffers;
  for (size_t i = 0; i < num_allocations; i++) {
    auto input = buffer_allocations.GetDeviceAddress(i);
    buffers.push_back(CreateGpuBuffer(&input));
    args.push_back(buffers.back().get());
  }
  if (args.size() != function->num_arguments())
    return InternalError("Unexpected argument count.");

  // Create return chain.
  tfrt::RCReference<tfrt::AsyncValue> result;
  if (function->num_results() != 1)
    return InternalError("Unexpected result count.");

  // Capture errors and augment with source.
  std::string diag_str;
  llvm::raw_string_ostream diag_os(diag_str);
  llvm::SourceMgr src_mgr;
  mlir::SourceMgrDiagnosticHandler handler(src_mgr, runtime_and_queue.mlir_ctx,
                                           diag_os);

  // Execute the function.
  function->Execute(exec_ctx, args, {result});

  // Wait for async execution to complete.
  tfrt::Await(exec_ctx, llvm::makeArrayRef(result));

  // Report error if any, from handler and result.
  if (diag_os.tell()) return tensorflow::errors::Internal(diag_os.str());
  if (auto* error = result->GetErrorIfPresent())
    return tensorflow::errors::Internal(tfrt::StrCat(*error));

  return MaybeSyncAndProfile(run_options, start_micros,
                             block_host_until_done ? main_stream : nullptr);
}
#endif  // BEF_EXECUTABLE

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  se::StreamExecutor* executor = run_options->stream()->parent();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously.
  auto gpu_lock = LockGpuShared(executor);

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tensorflow::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  auto device_ordinal = executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/output_shape_, memory_allocator,
                         device_ordinal);

  TF_ASSIGN_OR_RETURN(BufferAllocations buffer_allocations,
                      GenerateBufferAllocations(arguments, globals,
                                                memory_allocator, executor));
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

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        &allocations_[output_info.allocation_index];
    se::DeviceMemoryBase& result_buffer = p.second;

    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceMemory* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceMemory* {
        // ScopedBuffer is never an owned buffer.
        if (auto* unowned_shapedbuffers =
                absl::get_if<absl::Span<const ShapedBuffer* const>>(
                    &arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              absl::get<absl::Span<ExecutionInput>>(arguments);
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
        absl::optional<tensorflow::se::OwningDeviceMemory> owning =
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
        StatusOr<se::OwningDeviceMemory> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size);
        if (!allocated_buffer.ok()) {
          return ResourceExhausted("%s\n%s\n",
                                   allocated_buffer.status().error_message(),
                                   verbose_buffer_assignment_string_dumper_());
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

  TF_RETURN_IF_ERROR(ExecuteThunksOrBef(run_options, buffer_allocations,
                                        block_host_until_done));

  // Free all temporary allocations.
  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, allocations_));

  // Free allocations for arguments.
  if (auto args = absl::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

Status GpuExecutable::ExecuteThunksOrBef(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done) {
#if BEF_EXECUTABLE
  if (!absl::holds_alternative<OwnedBefBuffer>(thunks_or_bef_)) {
    return FailedPrecondition("Expected BefBuffer is not supplied.");
  }
  const auto& bef_buffer = absl::get<OwnedBefBuffer>(thunks_or_bef_);

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  tfrt::HostContext* host = runtime_and_queue.core_runtime->GetHostContext();
  tfrt::RCReference<tfrt::BEFFile> bef_file = tfrt::BEFFile::Open(
      {bef_buffer.get(), bef_buffer.get_deleter().size},
      host->GetKernelRegistry(), host->diag_handler(), host->allocator());
  if (!bef_file) {
    return InternalError("Failed to load BEF file.");
  }
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));
  return ExecuteBef(module_name_, bef_file, run_options, buffer_allocations,
                    allocations_.size(), block_host_until_done);
#else   // BEF_EXECUTABLE
  if (!absl::holds_alternative<OwnedThunkSchedule>(thunks_or_bef_)) {
    return FailedPrecondition("Expected ThunkSchedule is not supplied.");
  }
  const ThunkSchedule& thunk_schedule =
      *absl::get<OwnedThunkSchedule>(thunks_or_bef_);

  se::StreamExecutor* executor = run_options->stream()->parent();
  for (const std::unique_ptr<Thunk>& thunk : thunk_schedule.TotalOrder()) {
    TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
  }
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));
  return ExecuteThunks(module_name_, thunk_schedule, run_options,
                       buffer_allocations, block_host_until_done);
#endif  // BEF_EXECUTABLE
}

int64_t GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (BufferAllocation::Index i = 0; i < allocations_.size(); ++i) {
    const BufferAllocation& allocation = allocations_[i];
    if (allocation.is_constant()) {
      size += allocation.size();
    }
  }
  return size;
}

Status GpuExecutable::SetUpMlirAllocation(
    mlir::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
    std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape) {
  for (int i = 0; i < buffer_sizes.size(); i++) {
    allocations->emplace_back(i, buffer_sizes[i], 0);
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

  return Status::OK();
}

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
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
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> Status {
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

        return Status::OK();
      }));
  return output;
}

}  // namespace gpu
}  // namespace xla
