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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
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
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/platform.h"

#if XLA_ENABLE_XLIR
#include "tensorflow/compiler/xla/service/gpu/jitrt_custom_calls.h"
#include "tfrt/jitrt/diagnostics.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime
#endif  // XLA_ENABLE_XLIR

namespace xla {
namespace gpu {

bool IsJitRtExecutableEnabled(const HloModuleConfig& config) {
#if !XLA_ENABLE_XLIR
  CHECK(!config.debug_options().xla_gpu_jitrt_executable())
      << "Failed to enable JitRt backend, because it was not compiled.";
#endif  // !XLA_ENABLE_XLIR
  return config.debug_options().xla_gpu_jitrt_executable();
}

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

}  // namespace

#if XLA_ENABLE_XLIR
namespace jitrt = ::tfrt::jitrt;

class GpuExecutable::JitRtExecutable {
 public:
  static StatusOr<JitRtExecutable*> Create(OwnedJitRtProgram program) {
    // Options for the default JitRt compilation pipeline.
    jitrt::CompilationPipelineOptions copts;
    // We do not expect any parallel loops on the GPU program, so we disable
    // all concurrency (async parallel for loops).
    copts.num_worker_threads = 1;

    // For passing LMHLO attributes as XLA(SE) enums/structs to custom calls.
    copts.populate_attr_encodings = PopulateLmhloToXlaAttrEncoding;

    // Options for constructing JitRt JitExecutable.
    jitrt::CompilationOptions opts;
    opts.specialization = jitrt::CompilationOptions::Specialization::kDisabled;
    opts.register_dialects = [](mlir::DialectRegistry& registry) {
      jitrt::RegisterDefaultJitRtDialects(registry);
      // For the encoding of attributes to custom calls.
      registry.insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
    };

    // Register JitRt Gpu runtime custom calls with the linker.
    opts.runtime_symbol_map = GetSymbolsBinding(JitRtGpuCustomCalls());

    // We just use the default compilation pipeline provided by the JitRt.
    // Alternatively instead of having a separate JitRtProgram (LMHLO lowered to
    // JitRt dialects), we can assemble a pipeline that will compile starting
    // from the LMHLO dialect. However this intermediate step helps with
    // debugging, by materializing IR with XLA runtime custom calls.
    opts.create_compilation_pipeline = [copts](mlir::PassManager& pm) {
      jitrt::CreateDefaultJitRtCompilationPipeline(pm, copts);
    };

    // TODO(b/241296710): LLVM optimizations interact badly with the memory
    // loads and stores pattern generated in very large XLA programs, and can
    // take minutes to run. Currently we do not expect any expensive code
    // running on the host, so we can safely disable optimization passes.
    opts.jit_code_opt_level = llvm::CodeGenOpt::None;

    // Instantiate new JitExecutable from the MLIR source.
    auto jit_executable = jitrt::JitExecutable::Instantiate(
        program->module, program->entry_point, opts);
    if (auto err = jit_executable.takeError())
      return InternalError("Failed to compile JitRt program: %s",
                           tfrt::StrCat(err));

    // Pass ownership to the GpuExecutable.
    return new JitRtExecutable(std::move(program->buffer_sizes),
                               std::move(*jit_executable),
                               std::move(program->debug_options));
  }

  jitrt::JitExecutable& jit_executable() { return jit_executable_; }
  jitrt::Executable& default_executable() { return *default_executable_; }
  JitRtKernelsCache& kernels_cache() { return kernels_cache_; }
  JitRtGemmConfigCache& gemm_configs_cache() { return gemm_configs_cache_; }
  JitRtCollectiveSupport& collectives() { return collectives_; }

  // We pass a pointer to the buffer size to the compiled function, so we return
  // a reference to a stable memory location.
  const int64_t& buffer_size(size_t offset) const {
    return buffer_sizes_[offset];
  }

  const DebugOptions& debug_options() const { return debug_options_; }

 private:
  explicit JitRtExecutable(std::vector<int64_t> buffer_sizes,
                           jitrt::JitExecutable jit_executable,
                           DebugOptions debug_options)
      : buffer_sizes_(std::move(buffer_sizes)),
        jit_executable_(std::move(jit_executable)),
        default_executable_(&jit_executable_.DefaultExecutable().get()),
        debug_options_(std::move(debug_options)) {}

  std::vector<int64_t> buffer_sizes_;
  jitrt::JitExecutable jit_executable_;
  jitrt::Executable* default_executable_;  // owned by `jit_executable`
  DebugOptions debug_options_;

  // Keep a cache of kernels instantiated by this executable.
  JitRtKernelsCache kernels_cache_;

  // Keep a cache of gemm configs for all gemm operation in the program.
  JitRtGemmConfigCache gemm_configs_cache_;

  // Support for running collective operations.
  JitRtCollectiveSupport collectives_;
};
#endif  // XLA_ENABLE_XLIR

StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(Params params) {
  auto executable = std::move(params.executable);
  std::unique_ptr<GpuExecutable> result(new GpuExecutable(std::move(params)));

  if (std::holds_alternative<OwnedThunkSchedule>(executable)) {
    result->thunks_ = std::move(std::get<OwnedThunkSchedule>(executable));
    return result;
  }

#if XLA_ENABLE_XLIR
  if (std::holds_alternative<OwnedJitRtProgram>(executable)) {
    auto& program = std::get<OwnedJitRtProgram>(executable);
    TF_ASSIGN_OR_RETURN(result->jitrt_executable_,
                        JitRtExecutable::Create(std::move(program)));
    return result;
  }
#endif  // XLA_ENABLE_XLIR

  return InternalError("No XLA gpu executable was provided");
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      gpu_version_(params.gpu_version),
      entry_func_attrs_(params.entry_func_attrs),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.allocations)),
      debug_buffer_assignment_(std::move(params.debug_buffer_assignment)),
      verbose_buffer_assignment_string_dumper_(
          params.verbose_buffer_assignment_string_dumper),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)) {
  if (has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(
        module().unique_id(), shared_module(), debug_buffer_assignment_);
  }
}

GpuExecutable::~GpuExecutable() {
  if (has_module()) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }

  {
    // We could have issued host->device mem copies in ResolveConstantGlobals.
    // Wait for those to finish so that we can safely deallocate the backing HLO
    // module.
    //
    // We need for the host->device memcpies to finish they are concurrently
    // reading memory (xla::Literal's) owned by the HLO module.
    absl::MutexLock lock(&module_handle_mutex_);
    for (const auto& pair : module_globals_) {
      CHECK(pair.first->SynchronizeAllActivity());
    }
  }

#if XLA_ENABLE_XLIR
  delete jitrt_executable_;
#endif
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::PlatformKind platform_kind =
      main_stream->parent()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kROCm) {
    auto cc = main_stream->GetRocmComputeCapability();
    std::string stream_arch = cc.gcn_arch_name();
    std::string gpu_exec_arch =
        std::get<se::RocmComputeCapability>(gpu_version_).gcn_arch_name();
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_kind == stream_executor::PlatformKind::kCuda) {
    GpuVersion cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(std::get<se::CudaComputeCapability>(cc) ==
                 std::get<se::CudaComputeCapability>(gpu_version_))
        << "Compute capability mismatch; expected {"
        << std::get<se::CudaComputeCapability>(gpu_version_).ToString()
        << "}, but was {" << std::get<se::CudaComputeCapability>(cc).ToString()
        << "}";
  } else {
    return InternalError("Unknown platform: %d", platform_kind);
  }

  return OkStatus();
}

namespace {

Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                           uint64_t start_micros, se::Stream* stream_to_sync);

Status ExecuteThunks(const std::string& module_name,
                     const ThunkSchedule& thunk_schedule,
                     const ServiceExecutableRunOptions* run_options,
                     const BufferAllocations& buffer_allocations,
                     bool block_host_until_done) {
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

    Thunk::ExecuteParams thunk_params{
        *run_options, buffer_allocations, stream,
        async_comms_stream.ok() ? async_comms_stream->get() : nullptr};
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    if (thunk_schedule.Depended(thunk.get())) {
      auto finish_event = std::make_unique<se::Event>(main_stream->parent());
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

  return OkStatus();
}

}  // namespace

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(&module_handle_mutex_);
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
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load empty PTX. It's okay if we skip loading
  // in this case; if the module isn't loaded, all symbol lookups will fail,
  // just as they should for an empty module.
  if (!(executor->platform_kind() == se::PlatformKind::kCuda &&
        module_spec.cuda_ptx_in_memory() == nullptr)) {
    TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));
  }

  for (const ConstantInfo& info : constants_) {
    StatusOr<stream_executor::DeviceMemoryBase> global_status;
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

      if (!info.content.empty()) {
        // This means the constant did not have an initializer in the PTX and
        // therefore must be initialized by XLA here.
        stream->ThenMemcpy(&global, info.content.data(), info.content.size());
      }
    } else {
      // The constant was not defined in the PTX and therefore must be both
      // allocated and initialized by XLA here.
      CHECK(!info.content.empty());

      TF_ASSIGN_OR_RETURN(
          auto shared, executor->CreateOrShareConstant(stream, info.content));
      global = *shared;
      VLOG(3) << "Allocated (or shared) global " << info.symbol_name << " at "
              << global.opaque();
      // XLA will continue to own this global at least until this executable is
      // destroyed (longer if another, longer-lived executable shares the same
      // constant).
      shared_constants_.push_back(std::move(shared));
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
  return OkStatus();
}

StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal) {
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
                            device_ordinal, i));
    buffers.push_back(buffer);
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
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

#if XLA_ENABLE_XLIR
static Status ExecuteJitRt(const std::string& module_name,
                           GpuExecutable::JitRtExecutable* jitrt_executable,
                           const ServiceExecutableRunOptions* run_options,
                           const BufferAllocations& buffer_allocations,
                           size_t num_allocations, bool block_host_until_done) {
  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  ScopedAnnotation annotation(
      []() -> std::string { return "JitRtExecutable"; });

  // TODO(ezhulenev): Here we rely on implementation details of passing memrefs
  // to the compiled kernel. We should have a nicer API to do this, without
  // creating a vector of temporary MemrefDesc for passing operands.

  // Pack buffer allocations as executable arguments. It is guaranteed that
  // compiled function will make a copy of all arguments and will write all
  // results after the call to `Execute` completes, so it is safe to keep in on
  // the stack.
  jitrt::Executable::CallFrame call_frame;

  // Each buffer allocation pased as 1d memref to the compiled kernel:
  //   {basePtr, dataPtr, offset, [sizes, ...], [strides, ...]}
  size_t num_args_ptrs = 1 + num_allocations * 5;
  call_frame.args.resize_for_overwrite(num_args_ptrs);

  // Pass pointers to these constants as a memref offset and stride.
  int64_t zero = 0;
  int64_t one = 1;
  void* offset = &zero;
  void* stride = &one;

  // Add a placeholder for the kernel context as the first argument.
  call_frame.args[0] = nullptr;

  // Storage for data pointers.
  llvm::SmallVector<void*, 16> ptrs;
  ptrs.resize_for_overwrite(num_allocations);

  // Initialize arguments for the buffer operands.
  for (unsigned i = 0; i < num_allocations; ++i) {
    void* data = &(ptrs[i] = buffer_allocations.GetDeviceAddress(i).opaque());
    void* size = const_cast<int64_t*>(&jitrt_executable->buffer_size(i));
    unsigned idx = 1 + i * 5;
    call_frame.args[idx + 0] = data;
    call_frame.args[idx + 1] = data;
    call_frame.args[idx + 2] = offset;
    call_frame.args[idx + 3] = size;
    call_frame.args[idx + 4] = stride;
  }

  // JitRt executables do not return any values.
  jitrt::NoResultConverter converter;

  // Prepare options for executing JitRt program.
  jitrt::Executable::ExecuteOpts opts;

  // We don't expect to see any async tasks in the JitRt executable.
  opts.async_task_runner =
      reinterpret_cast<jitrt::AsyncTaskRunner*>(0XDEADBEEF);

  // Get the async communications stream for async collectives.
  int device_ordinal = run_options->stream()->parent()->device_ordinal();
  StatusOr<StreamPool::Ptr> async_comms_stream =
      run_options->BorrowStream(device_ordinal);

  // Async collective support instantiated for each Gpu executable run, so that
  // concurrent executions can run independenty using a separate set of events
  // for communication.
  JitRtAsyncCollectiveSupport async_collectives(
      async_comms_stream.ok() ? async_comms_stream->get() : nullptr);

  // Pass auxiliary data to the custom call handlers.
  jitrt::CustomCall::UserData user_data;
  user_data.insert_all(
      run_options, &jitrt_executable->debug_options(),
      &jitrt_executable->kernels_cache(),
      &jitrt_executable->gemm_configs_cache(), &jitrt_executable->collectives(),
      async_collectives.async_comm_stream() ? &async_collectives : nullptr);
  opts.custom_call_data = &user_data;

  // Collect all emitted diagnostic messages.
  jitrt::DiagnosticEngine diagnostic_engine;
  std::string diagnostic;
  diagnostic_engine.AddHandler([&](jitrt::Diagnostic& d) {
    llvm::raw_string_ostream(diagnostic) << d.str();
    return mlir::success();
  });

  opts.diagnostic_engine = &diagnostic_engine;

  // Get the default executable. We do not support specialization because
  // all shapes are static. Default executable is guaranteed to be available.
  jitrt::Executable& executable = jitrt_executable->default_executable();

  // Execute with the prepared call frame.
  executable.Execute(call_frame, opts);
  if (auto err = executable.ReturnResults(converter, &call_frame)) {
    return InternalError(
        "Failed to execute JitRt executable: %s.",
        tfrt::StrCat(err,
                     diagnostic.empty() ? "" : tfrt::StrCat(": ", diagnostic)));
  }

  return MaybeSyncAndProfile(
      run_options, start_micros,
      block_host_until_done ? run_options->stream() : nullptr);
}
#endif  // XLA_ENABLE_XLIR

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
  absl::ReaderMutexLock gpu_lock(&GetGpuMutex(executor));

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
                std::get_if<absl::Span<const ShapedBuffer* const>>(
                    &arguments)) {
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

  TF_RETURN_IF_ERROR(ExecuteThunksOrJitRt(run_options, buffer_allocations,
                                          block_host_until_done));

  // Free all temporary allocations.
  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, allocations_));

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

Status GpuExecutable::ExecuteThunksOrJitRt(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  if (thunks_) {
    se::StreamExecutor* executor = run_options->stream()->parent();
    for (const std::unique_ptr<Thunk>& thunk : thunks_->TotalOrder()) {
      TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
    }
    return ExecuteThunks(module_name_, *thunks_, run_options,
                         buffer_allocations, block_host_until_done);
  }

#if XLA_ENABLE_XLIR
  if (jitrt_executable_) {
    return ExecuteJitRt(module_name_, jitrt_executable_, run_options,
                        buffer_allocations, allocations_.size(),
                        block_host_until_done);
  }
#endif  // XLA_ENABLE_XLIR

  return FailedPrecondition("Expected XLA gpu executable is not supplied.");
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
    mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
    std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape, int buffer_param_offset) {
  for (int i = 0; i < buffer_sizes.size(); i++) {
    allocations->emplace_back(i, buffer_sizes[i], 0);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    if (i < buffer_param_offset) {
      continue;
    }
    const int buffer_index = i - buffer_param_offset;

    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      xla::ShapeIndex shape_index;
      if (auto shape_index_attr =
              func.getArgAttrOfType<mlir::DenseIntElementsAttr>(
                  i, "lmhlo.param_shape_index")) {
        for (const llvm::APInt& element : shape_index_attr) {
          shape_index.push_back(element.getSExtValue());
        }
      }
      allocations->at(buffer_index)
          .set_entry_computation_parameter(
              param_attr.cast<mlir::IntegerAttr>().getInt(), shape_index,
              static_cast<bool>(func.getArgAttr(i, "lmhlo.output_index")));
    }
    // TODO(timshen): this information is redundant. This is here only for
    // smooth migration to LMHLO. Remove it.
    if (func.getArgAttr(i, "lmhlo.constant_name")) {
      allocations->at(buffer_index).set_constant(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      allocations->at(buffer_index).set_maybe_live_out(true);

      // Reconstruct a shape index from output_index.
      ShapeIndex shape_index;
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        shape_index.push_back(element.getSExtValue());
      }
      auto& o = (*output_info)[shape_index];
      o.allocation_index = buffer_index;
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

  return OkStatus();
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

        return OkStatus();
      }));
  return output;
}

}  // namespace gpu
}  // namespace xla
