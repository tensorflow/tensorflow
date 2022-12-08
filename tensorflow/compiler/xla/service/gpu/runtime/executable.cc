/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/executable.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/ffi.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cholesky.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/custom_call.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/io_feed.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memcpy.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memset.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/tracing.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/tsl/protobuf/dnn.pb.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif  // #if GOOGLE_CUDA

namespace xla {

#if GOOGLE_CUDA
namespace runtime {
namespace ffi {

// Override weak symbol defined in the `xla/runtime/ffi.cc` with a strong one
// that provides implementation for the XLA:GPU backend.
XLA_FFI_Stream* GetXlaFfiStream(const CustomCall::UserData* user_data,
                                const DiagnosticEngine* diagnostic) {
  auto run_opts = user_data->getIfExists<const ServiceExecutableRunOptions>();
  auto stream = se::gpu::AsGpuStreamValue(run_opts->stream());
  return reinterpret_cast<XLA_FFI_Stream*>(stream);
}

}  // namespace ffi
}  // namespace runtime
#endif  // GOOGLE_CUDA

namespace gpu {

using ::xla::runtime::CustomCallAttrEncodingSet;
using ::xla::runtime::DirectCustomCallRegistry;
using ::xla::runtime::Executable;
using ::xla::runtime::JitExecutable;
using ::xla::runtime::success;
using ::xla::runtime::Tagged;
using ::xla::runtime::TypeIDNameRegistry;

using ::xla::runtime::ExportModules;
using ::xla::runtime::ffi::ExportFfiModules;
using ::xla::runtime::ffi::FfiStateVector;

void RegisterXlaGpuRuntimeCustomCalls(DirectCustomCallRegistry& registry) {
  RegisterKernelLaunchCustomCalls(registry);
  RegisterTracingCustomCalls(registry);
  RegisterFftCustomCalls(registry);
  RegisterCholeskyCustomCalls(registry);
  RegisterCollectiveCustomCalls(registry);
  RegisterGemmCustomCalls(registry);
  RegisterConvCustomCalls(registry);
  RegisterMemcpyCustomCalls(registry);
  RegisterIoFeedCustomCalls(registry);
  RegisterMemsetCustomCalls(registry);

#if GOOGLE_CUDA
  // Graph launch kernels depend on Cuda Graph API.
  RegisterGraphLaunchCustomCalls(registry);
  RegisterMatmulCustomCalls(registry);
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  RegisterCustomCall(registry);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void RegisterXlaGpuTypeIdNames(TypeIDNameRegistry& registry) {
  registry.Register<Tagged<se::dnn::ActivationMode>>(
      "__type_id_se_dnn_activation");
  registry.Register<Tagged<DotDimensionNumbers>>(
      "__type_id_dot_dimension_numbers");
  registry.Register<Tagged<ConvDimensionNumbers>>(
      "__type_id_conv_dimension_numbers");
  registry.Register<Tagged<se::fft::Type>>("__type_id_se_fft_type");
  registry.Register<Tagged<ConvBackendConfig>>("__type_id_conv_backend_config");

  RegisterTracingTypeIdNames(registry);

#if GOOGLE_CUDA
  registry.Register<Tagged<se::cuda::BlasLt::Epilogue>>(
      "__type_id_se_cublas_lt_epilogue");
#endif  // GOOGLE_CUDA
}

void RegisterXlaGpuAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  PopulateConvAttrEncoding(encoding);
  PopulateFftAttrEncoding(encoding);
  PopulateDotDimsAttrEncoding(encoding);

#if GOOGLE_CUDA
  PopulateCublasLtMatmulAttrEncoding(encoding);
#endif  // GOOGLE_CUDA
}

//===----------------------------------------------------------------------===//

GpuRuntimeExecutable::GpuRuntimeExecutable(
    std::vector<int64_t> buffer_sizes,
    std::unique_ptr<JitExecutable> jit_executable, DebugOptions debug_options,
    ModulesState modules_state, FfiModulesState ffi_modules_state)
    : buffer_sizes_(std::move(buffer_sizes)),
      executable_(std::move(jit_executable)),
      debug_options_(std::move(debug_options)),
      modules_state_(std::move(modules_state)),
      ffi_modules_state_(std::move(ffi_modules_state)) {
  ExportModules(dynamic_custom_calls_);     // export runtime modules
  ExportFfiModules(dynamic_custom_calls_);  // export FFI modules
}

GpuRuntimeExecutable::GpuRuntimeExecutable(
    std::vector<int64_t> buffer_sizes,
    std::unique_ptr<Executable> aot_executable, DebugOptions debug_options,
    ModulesState modules_state, FfiModulesState ffi_modules_state)
    : buffer_sizes_(std::move(buffer_sizes)),
      executable_(std::move(aot_executable)),
      debug_options_(std::move(debug_options)),
      modules_state_(std::move(modules_state)),
      ffi_modules_state_(std::move(ffi_modules_state)) {
  ExportModules(dynamic_custom_calls_);     // export runtime modules
  ExportFfiModules(dynamic_custom_calls_);  // export FFI modules
}

//===----------------------------------------------------------------------===//
// Compile Xla program lowered to runtime dialects to Gpu runtime executable.
//===----------------------------------------------------------------------===//

/*static*/ StatusOr<std::unique_ptr<GpuRuntimeExecutable>>
GpuRuntimeExecutable::Create(std::unique_ptr<GpuRuntimeProgram> program) {
  // Options for the default XLA Runtim compilation pipeline.
  runtime::CompilationPipelineOptions copts;

  // Populate mapping from XLA (SE) enums/structs type id to symbol names.
  copts.populate_type_id_names = RegisterXlaGpuTypeIdNames;

  // For passing LMHLO attributes as XLA (SE) enums/structs to custom calls.
  copts.populate_attr_encodings = RegisterXlaGpuAttrEncoding;

  // Options for constructing XLA runtime JitExecutable.
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects =
      runtime::RegisterDefaultXlaGpuRuntimeDialects;

  // Register XLA Gpu runtime custom calls with the linker.
  opts.compiler.symbols_binding = runtime::ToSymbolsBinding(
      RegisterXlaGpuRuntimeCustomCalls, RegisterXlaGpuTypeIdNames);

  // We just use the default compilation pipeline provided by the XLA runtime.
  // Alternatively instead of having a separate Xla Runtime program (LMHLO
  // lowered to canonical dialects), we can assemble a pipeline that will
  // compile starting from the LMHLO dialect. However this intermediate step
  // helps with debugging, by materializing IR with XLA runtime custom calls.
  opts.compiler.create_compilation_pipeline =
      [copts](xla::runtime::PassManager& passes) {
        runtime::CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
      };

  // TODO(b/241296710): LLVM optimizations interact badly with the memory
  // loads and stores pattern generated in very large XLA programs, and can
  // take minutes to run. Currently we do not expect any expensive code
  // running on the host, so we can safely disable optimization passes.
  opts.compiler.jit_code_opt_level = llvm::CodeGenOpt::None;

  // Instantiate new JitExecutable from the MLIR source.
  auto jit_executable =
      JitExecutable::Instantiate(program->module, program->entry_point, opts);
  if (!jit_executable.ok())
    return InternalError("Failed to compile XLA Runtime program: %s",
                         jit_executable.status().message());

  // Instantiate state for all registered runtime modules.
  auto modules_state = ModulesState::Instantiate();
  if (!modules_state.ok())
    return InternalError("Failed to instantiate modules state: %s",
                         modules_state.status().message());

  // Instantiate state for all registered FFI modules.
  auto ffi_modules_state = FfiModulesState::Instantiate();
  if (!ffi_modules_state.ok())
    return InternalError("Failed to instantiate FFI modules state: %s",
                         ffi_modules_state.status().message());

  return std::unique_ptr<GpuRuntimeExecutable>(new GpuRuntimeExecutable(
      std::move(program->buffer_sizes),
      std::make_unique<JitExecutable>(std::move(*jit_executable)),
      std::move(program->debug_options), std::move(*modules_state),
      std::move(*ffi_modules_state)));
}

//===----------------------------------------------------------------------===//
// Constructs Gpu runtime executable from AOT compiled runtime artifact.
//===----------------------------------------------------------------------===//

/*static*/ StatusOr<std::unique_ptr<GpuRuntimeExecutable>>
GpuRuntimeExecutable::Create(absl::Span<const int64_t> buffer_sizes,
                             Executable executable,
                             DebugOptions debug_options) {
  // Instantiate state for all registered runtime modules.
  auto modules_state = ModulesState::Instantiate();
  if (!modules_state.ok())
    return InternalError("Failed to instantiate modules state: %s",
                         modules_state.status().message());

  // Instantiate state for all registered FFI modules.
  auto ffi_modules_state = FfiModulesState::Instantiate();
  if (!ffi_modules_state.ok())
    return InternalError("Failed to instantiate FFI modules state: %s",
                         ffi_modules_state.status().message());

  return std::unique_ptr<GpuRuntimeExecutable>(new GpuRuntimeExecutable(
      std::vector<int64_t>(buffer_sizes.begin(), buffer_sizes.end()),
      std::make_unique<Executable>(std::move(executable)),
      std::move(debug_options), std::move(*modules_state),
      std::move(*ffi_modules_state)));
}

//===----------------------------------------------------------------------===//
// Executes with the given buffer arguments.
//===----------------------------------------------------------------------===//

static runtime::AsyncTaskRunner* NoAsyncTaskRunner() {
  return reinterpret_cast<runtime::AsyncTaskRunner*>(0XDEADBEEF);
}

// TODO(ezhulenev): We rely on implementation details of passing memrefs to the
// compiled kernel. We should have a nicer API to do this, without creating a
// vector of temporary MemrefDesc for passing operands.
static void InitializeCallFrame(runtime::Executable::CallFrame& call_frame,
                                const BufferAllocations& buffer_allocations,
                                absl::Span<const int64_t> buffer_sizes,
                                llvm::SmallVectorImpl<void*>& ptrs) {
  size_t num_allocations = buffer_allocations.size();
  assert(ptrs.empty() && "pointers storage must be empty");
  ptrs.resize_for_overwrite(num_allocations);

  // Each buffer allocation pased as 1d memref to the compiled function:
  //   {basePtr, dataPtr, offset, [sizes, ...], [strides, ...]}
  size_t num_args_ptrs = 1 + num_allocations * 5;
  call_frame.args.resize_for_overwrite(num_args_ptrs);

  // Pass pointers to these constants as a memref offset and stride.
  static int64_t zero = 0;
  static int64_t one = 1;
  void* offset = &zero;
  void* stride = &one;

  // Add a placeholder for the kernel context as the first argument.
  call_frame.args[0] = nullptr;

  // Initialize arguments for the buffer operands.
  for (unsigned i = 0; i < num_allocations; ++i) {
    void* data = &(ptrs[i] = buffer_allocations.GetDeviceAddress(i).opaque());
    void* size = const_cast<int64_t*>(&buffer_sizes[i]);
    unsigned idx = 1 + i * 5;
    call_frame.args[idx + 0] = data;
    call_frame.args[idx + 1] = data;
    call_frame.args[idx + 2] = offset;
    call_frame.args[idx + 3] = size;
    call_frame.args[idx + 4] = stride;
  }
}

Status GpuRuntimeExecutable::Execute(
    const ServiceExecutableRunOptions* run_options, const std::string& asm_text,
    const std::vector<uint8_t>& binary,
    const BufferAllocations& buffer_allocations,
    const BufferAllocation* temp_alloc) {
  // Pack buffer allocations as executable arguments. It is guaranteed that
  // the compiled function will make a copy of all arguments and will write all
  // results after the call to `Execute` completes, so it is safe to keep them
  // on the stack.
  runtime::Executable::CallFrame call_frame;

  llvm::SmallVector<void*, 16> ptrs;  // storage for device address pointers
  InitializeCallFrame(call_frame, buffer_allocations, buffer_sizes_, ptrs);

  // XLA Runtime executables do not return any values.
  runtime::NoResultConverter converter;

  // Get the async communications stream for async collectives.
  se::StreamExecutor* executor = run_options->stream()->parent();
  int device_ordinal = executor->device_ordinal();
  StatusOr<StreamPool::Ptr> async_comms_stream =
      run_options->BorrowStream(device_ordinal);

  // Async collective support instantiated for each Gpu executable run, so that
  // concurrent executions can run independenty using a separate set of events
  // for communication.
  JitRtAsyncCollectiveSupport async_collectives(
      async_comms_stream.ok() ? async_comms_stream->get() : nullptr);

  // Always pass in the temp buffer, even if it is null, to accommodate the
  // 0-sized buffer corner case.
  se::DeviceMemoryBase temp_buffer;
  if (temp_alloc)
    temp_buffer = buffer_allocations.GetDeviceAddress(temp_alloc->index());

  // We pass a pointer to the executable through UserData, so that we can
  // get access to other exported functions from custom call handlers.
  runtime::Executable& executable = this->executable();

  // Take snapshots of every state required by custom calls.
  StreamExecutorKernels::Snapshot kernels = gpu_kernels_(executor)->snapshot();
  GemmConfigs::Snapshot gemm_configs = gemm_configs_.snapshot();

  // Initialize state required for running functions exported from FFI modules.
  FfiStateVector ffi_state = ffi_modules_state_.state_vector();

  // Pass auxiliary data to the custom call handlers.
  runtime::CustomCall::UserData user_data(
      run_options, &executable, &debug_options_, &temp_buffer, &asm_text,
      &ffi_state, &binary, &kernels, &gemm_configs, &conv_runners_cache_,
      &collectives_,
      // Null pointer will be interpreted as an absence of async collectives
      // support and custom calls will safely return an error.
      async_collectives.async_comm_stream() ? &async_collectives : nullptr);

  // Initialize state required for running functions from registered modules.
  auto state_ref = modules_state_.InitializeUserData(user_data);
  if (!state_ref.ok())
    return InternalError("Failed to initialize runtime modules state: %s",
                         state_ref.status().message());

#if GOOGLE_CUDA
  // Add auxiliary data that is available only if compiled with CUDA support.
  MatmulPlans::Snapshot matmul_plans = cublas_lt_matmul_plans_.snapshot();
  GraphInstances::Snapshot graph_instances = graph_instances_.snapshot();
  user_data.insert_all(&matmul_plans, &graph_instances);
#endif  // GOOGLE_CUDA

  // Collect all emitted diagnostic messages.
  std::string diagnostic;
  runtime::DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([&](runtime::Diagnostic& d) {
    absl::StrAppend(&diagnostic, d.status().message());
    return success();
  });

  // Prepare options for executing XLA Runtime program.
  runtime::Executable::ExecuteOpts opts;
  opts.async_task_runner = NoAsyncTaskRunner();
  opts.custom_call_data = &user_data;
  opts.diagnostic_engine = &diagnostic_engine;
  opts.custom_call_registry = &dynamic_custom_calls_;

  // Execute with the prepared call frame.
  executable.Execute(call_frame, opts);

  if (auto st = executable.ReturnResults(converter, &call_frame); !st.ok()) {
    return InternalError("Failed to execute XLA Runtime executable: %s%s%s.",
                         st.message(), diagnostic.empty() ? "" : ": ",
                         diagnostic);
  }

  return OkStatus();
}

//===----------------------------------------------------------------------===//

Executable& GpuRuntimeExecutable::executable() {
  if (auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_)) {
    return *(*jit)->DefaultExecutable();
  }
  return *std::get<std::unique_ptr<Executable>>(executable_);
}

StatusOr<std::string_view> GpuRuntimeExecutable::GetObjFile() const {
  const auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_);
  if (!jit) return InternalError("ObjFile is not available");

  if (auto obj_file = (*jit)->DefaultExecutable()->obj_file())
    return std::string_view(obj_file->getBuffer());

  return InternalError("gpu runtime executable didn't save the obj file");
}

StatusOr<std::string_view> GpuRuntimeExecutable::GetMlirModule() const {
  const auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_);
  if (!jit) return InternalError("MLIR module is not available");

  return (*jit)->mlir_module();
}

}  // namespace gpu
}  // namespace xla
