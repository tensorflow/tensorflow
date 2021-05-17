/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/bef_thunk.h"

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "tensorflow/core/platform/errors.h"

#if BEF_THUNKS
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_converter/mlir_to_bef_translate.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gpu_passes.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#endif  // BEF_THUNKS

namespace xla {
namespace gpu {

namespace {

#if BEF_THUNKS
StatusOr<std::unique_ptr<tfrt::gpu::Program>> ConvertToGpuProgram(
    mlir::Operation* op, tfrt::HostContext* host) {
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(op->getContext()));
  const std::string func_name = "main";
  // TODO(hanbinyoon): Merge with the async lhlo->tfrt_gpu lowering pipeline.
  if (tensorflow::LhloGpuOpToTfrtCudaModule(op, module.get(), func_name)
          .failed()) {
    return tensorflow::errors::Internal(
        "Failed to lower lmhlo_gpu op to tfrt_gpu dialect.");
  }

  std::string bef;
  llvm::raw_string_ostream bef_ostream(bef);
  if (tfrt::MLIRToBEFTranslate(*module, bef_ostream).failed()) {
    return tensorflow::errors::Internal("Failed to translate MLIR to BEF.");
  }

  auto buffer = tfrt::BefBuffer(bef.data(), bef.data() + bef.size());
  return absl::make_unique<tfrt::gpu::Program>(std::move(buffer), func_name,
                                               host);
}
#endif  // BEF_THUNKS

}  // namespace

bool BefThunk::SupportsOp(mlir::Operation* op) {
#if !BEF_THUNKS
  return false;
#else
  return GetThunkKind(op).ok();
#endif  // BEF_THUNKS
}

StatusOr<std::unique_ptr<BefThunk>> BefThunk::Create(
    ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> inputs,
    std::vector<BufferAllocation::Slice> outputs) {
#if !BEF_THUNKS
  LOG(FATAL) << "BefThunk is disabled.";
#else
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto thunk = absl::WrapUnique(
      new BefThunk(thunk_info, kind, std::move(inputs), std::move(outputs)));
  TF_ASSIGN_OR_RETURN(
      thunk->gpu_program_,
      ConvertToGpuProgram(op, runtime().core_runtime()->GetHostContext()));
  return thunk;
#endif  // BEF_THUNKS
}

BefThunk::BefThunk(ThunkInfo thunk_info, Thunk::Kind kind,
                   std::vector<BufferAllocation::Slice> inputs,
                   std::vector<BufferAllocation::Slice> outputs)
    : Thunk(kind, thunk_info)
#if BEF_THUNKS
      ,
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs))
#endif  // BEF_THUNKS
{
}

Status BefThunk::ExecuteOnStream(const ExecuteParams& params) {
#if !BEF_THUNKS
  LOG(FATAL) << "BefThunk is disabled.";
#else
  VLOG(2) << "Executing BEF thunk.";

  auto* host = runtime().core_runtime()->GetHostContext();
  auto gpu_system = tfrt::gpu::System::Instantiate(host);
  host->Await({gpu_system.CopyRCRef()});

  // Use the GPU resources specified in ExecuteParams (initialized by the
  // StreamExecutor).
  auto se_gpu_executor = down_cast<stream_executor::gpu::GpuExecutor*>(
      params.stream->parent()->implementation());
  auto se_gpu_stream = down_cast<stream_executor::gpu::GpuStream*>(
      params.stream->implementation());
  auto context =
      tfrt::gpu::wrapper::Context(se_gpu_executor->gpu_context()->context());
  auto stream = tfrt::gpu::wrapper::Stream(se_gpu_stream->gpu_stream());
  tfrt::gpu::BorrowedGpuStream gpu_stream(context, stream);

  // Prepare arguments for BEF execution.
  auto get_async_value_ref = [&](const BufferAllocation::Slice& slice)
      -> tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer> {
    se::DeviceMemoryBase data =
        params.buffer_allocations->GetDeviceAddress(slice);
    tfrt::gpu::wrapper::Pointer<void> pointer(
        data.opaque(), tfrt::gpu::wrapper::Platform::CUDA);
    auto allocator =
        tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuOneShotAllocator<void>>(
            pointer);
    auto buffer =
        tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), data.size());
    if (!buffer)
      return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(buffer.takeError()));
    return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
        std::move(*buffer));
  };

  llvm::SmallVector<tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer>, 4> inputs;
  llvm::SmallVector<tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer>, 4> outputs;
  for (auto input : inputs_) {
    inputs.push_back(get_async_value_ref(input));
  }
  for (auto output : outputs_) {
    outputs.push_back(get_async_value_ref(output));
  }

  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain(host);
  chain = gpu_system->Execute(exec_ctx(), *gpu_program_, gpu_stream, inputs,
                              outputs, std::move(chain));

  // TODO(hanbinyoon): BorrowedGpuStream releases context/stream information
  // upon destruction. Change it to not use wrapper::OwningContext and
  // wrapper::OwningStream.
  // TODO(b/184696034): Remove this. Right now we need this to ensure kernels
  // in bef function have been dispatched to stream.
  host->Quiesce();
  return Status::OK();
#endif  // BEF_THUNKS
}

StatusOr<Thunk::Kind> BefThunk::GetThunkKind(mlir::Operation* op) {
  if (mlir::isa<mlir::lmhlo_gpu::GEMMOp, mlir::lmhlo_gpu::GEMM_BiasOp>(op)) {
    return Thunk::Kind::kGemm;
  }
  return tensorflow::errors::Unimplemented(
      "Operation is not supported by BefThunk.");
}

#if BEF_THUNKS
tensorflow::tfrt_stub::Runtime& BefThunk::runtime() {
  static auto runtime = tensorflow::tfrt_stub::Runtime::Create().release();
  return *runtime;
}

tfrt::ExecutionContext& BefThunk::exec_ctx() {
  static tfrt::ExecutionContext* exec_ctx = [] {
    // Create request context and prepare deadline tracker.
    tfrt::RequestContextBuilder request_context_builder(
        runtime().core_runtime()->GetHostContext(),
        /*resource_context=*/nullptr);
    tensorflow::thread::ThreadPoolInterface* intra_op_threadpool = nullptr;
    DCHECK(
        runtime()
            .work_queue()
            ->InitializeRequest(&request_context_builder, &intra_op_threadpool)
            .ok());
    auto req_ctx = std::move(request_context_builder).build();
    DCHECK(req_ctx);
    return new tfrt::ExecutionContext(std::move(*req_ctx));
  }();
  return *exec_ctx;
}
#endif  // BEF_THUNKS

}  // namespace gpu
}  // namespace xla
