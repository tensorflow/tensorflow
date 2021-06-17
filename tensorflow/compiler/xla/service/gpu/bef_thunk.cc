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

#include "tensorflow/core/platform/errors.h"

#if BEF_THUNKS
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gpu_passes.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "third_party/tf_runtime/backends/gpu/include/tfrt/gpu/gpu_types.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {
namespace gpu {

bool IsBefThunkEnabled() { return true; }

namespace {
class BefThunk : public Thunk {
 public:
  BefThunk(Thunk::Kind kind, ThunkInfo thunk_info,
           std::vector<BufferAllocation::Slice> inputs,
           std::vector<BufferAllocation::Slice> outputs,
           tfrt::BefBuffer bef_buffer,
           tfrt::RCReference<tfrt::BEFFile> bef_file)
      : Thunk(kind, thunk_info),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)) {}

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::vector<BufferAllocation::Slice> inputs_;
  std::vector<BufferAllocation::Slice> outputs_;
  tfrt::BefBuffer bef_buffer_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;
};
}  // namespace

static const char kFuncName[] = "main";

// Clones 'op' into a function within a new module.
static mlir::OwningOpRef<mlir::ModuleOp> CreateModule(mlir::Operation* op) {
  mlir::OpBuilder builder(op->getContext());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      builder.create<mlir::ModuleOp>(op->getLoc());

  builder.setInsertionPointToEnd(module->getBody());
  auto func_type = builder.getType<mlir::FunctionType>(op->getOperandTypes(),
                                                       op->getResultTypes());
  auto func = builder.create<mlir::FuncOp>(op->getLoc(), kFuncName, func_type);
  func.setPublic();

  builder.setInsertionPointToEnd(func.addEntryBlock());
  mlir::BlockAndValueMapping mapping;
  for (const auto& pair :
       llvm::zip_first(op->getOperands(), func.getArguments())) {
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  }
  builder.clone(*op, mapping);

  builder.create<mlir::lmhlo::TerminatorOp>(op->getLoc());

  return module;
}

// Lowers 'module' to BEF.
static StatusOr<tfrt::BefBuffer> ConvertToBef(mlir::ModuleOp module) {
  mlir::PassManager pass_manager(module->getContext(),
                                 mlir::PassManager::Nesting::Implicit);
  pass_manager.addPass(tensorflow::createLmhloGpuAsyncConversionPass());
  pass_manager.addPass(mlir::createGpuAsyncRegionPass());
  pass_manager.addPass(tensorflow::createAsyncGpuTfrtConversionPass());
  if (failed(pass_manager.run(module)))
    return tensorflow::errors::Internal("Failed to run pass pipeline.");

  std::string bef;
  llvm::raw_string_ostream bef_ostream(bef);
  if (failed(tfrt::MLIRToBEFTranslate(module, bef_ostream)))
    return tensorflow::errors::Internal("Failed to translate MLIR to BEF.");

  return tfrt::BefBuffer(bef.data(), bef.data() + bef.size());
}

static StatusOr<Thunk::Kind> GetThunkKind(mlir::Operation* op) {
  if (mlir::isa<mlir::lmhlo_gpu::GEMMOp, mlir::lmhlo_gpu::GEMM_BiasOp>(op)) {
    return Thunk::Kind::kGemm;
  }
  return tensorflow::errors::Unimplemented(
      "Operation is not supported by BefThunk.");
}

// TODO(hanbinyoon): Pass in ExecutionContext at construction time when TF/XLA
// can depend on TFRT in OSS.
static const tfrt::ExecutionContext* bef_thunk_exec_ctx = nullptr;
void SetExecutionContext(const tfrt::ExecutionContext* exec_ctx) {
  bef_thunk_exec_ctx = exec_ctx;
}
static StatusOr<const tfrt::ExecutionContext*> GetExecutionContext() {
  if (bef_thunk_exec_ctx != nullptr) return bef_thunk_exec_ctx;
  return FailedPrecondition("BefThunk ExecutionContext has not been set");
}

StatusOr<std::unique_ptr<Thunk>> CreateBefThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> inputs,
    std::vector<BufferAllocation::Slice> outputs) {
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto module = CreateModule(op);
  TF_ASSIGN_OR_RETURN(tfrt::BefBuffer bef_buffer, ConvertToBef(*module));

  TF_ASSIGN_OR_RETURN(const auto* exec_ctx, GetExecutionContext());
  tfrt::HostContext* host = exec_ctx->host();
  auto bef_file = tfrt::BEFFile::Open(bef_buffer, host->GetKernelRegistry(),
                                      host->diag_handler(), host->allocator());
  if (!bef_file)
    return tensorflow::errors::Internal("Failed to load BEF file.");

  return std::unique_ptr<Thunk>(
      new BefThunk(kind, thunk_info, std::move(inputs), std::move(outputs),
                   std::move(bef_buffer), std::move(bef_file)));
}

// Wrap the GPU stream specified in 'params' (initialized by the StreamExecutor)
// to be passed to BEF functions as AsyncValueRef<GpuStream>.
static auto CreateGpuStream(const Thunk::ExecuteParams& params) {
  auto se_gpu_executor = static_cast<stream_executor::gpu::GpuExecutor*>(
      params.stream->parent()->implementation());
  auto se_gpu_stream = static_cast<stream_executor::gpu::GpuStream*>(
      params.stream->implementation());
  return tfrt::gpu::BorrowedGpuStream(
      tfrt::gpu::wrapper::Context(se_gpu_executor->gpu_context()->context()),
      tfrt::gpu::wrapper::Stream(se_gpu_stream->gpu_stream()));
}

// Wrap the GPU buffer specified in 'slice' to be passed to BEF functions as
// AsyncValueRef<GpuBuffer>.
static tfrt::RCReference<tfrt::AsyncValue> CreateGpuBuffer(
    const Thunk::ExecuteParams& params, const BufferAllocation::Slice& slice) {
  se::DeviceMemoryBase data =
      params.buffer_allocations->GetDeviceAddress(slice);
  tfrt::gpu::wrapper::Pointer<void> pointer(data.opaque(),
                                            tfrt::gpu::wrapper::Platform::CUDA);
  auto allocator =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuOneShotAllocator<void>>(
          pointer);
  auto buffer =
      tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), data.size());
  if (!buffer)
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(buffer.takeError()));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
      std::move(*buffer));
}

Status BefThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(2) << "Executing BEF thunk.";

  // Signature: (chain, stream, inputs..., outputs...) -> (chain).
  const tfrt::Function* function = bef_file_->GetFunction(kFuncName);
  if (!function) {
    return tensorflow::errors::Internal("Failed to get '", kFuncName,
                                        "' function.");
  }

  TF_ASSIGN_OR_RETURN(const auto* exec_ctx, GetExecutionContext());

  // Create owning handles for arguments and add pointer to them to 'args'.
  tfrt::SmallVector<tfrt::AsyncValue*, 8> args;
  args.reserve(function->num_arguments());
  tfrt::AsyncValueRef<tfrt::Chain> chain =
      tfrt::GetReadyChain(exec_ctx->host());
  args.push_back(chain.GetAsyncValue());
  tfrt::gpu::BorrowedGpuStream stream = CreateGpuStream(params);
  args.push_back(static_cast<tfrt::AsyncValueRef<tfrt::gpu::GpuStream>>(stream)
                     .GetAsyncValue());
  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 8> buffers;
  for (auto input : inputs_) {
    buffers.push_back(CreateGpuBuffer(params, input));
  }
  for (auto output : outputs_) {
    buffers.push_back(CreateGpuBuffer(params, output));
  }
  for (auto& buffer : buffers) {
    args.push_back(buffer.get());
  }
  if (args.size() != function->num_arguments())
    return tensorflow::errors::Internal("Unexpected argument count.");

  // Create return chain.
  tfrt::RCReference<tfrt::AsyncValue> result;
  if (function->num_results() != 1)
    return tensorflow::errors::Internal("Unexpected result count.");

  // Execute the function.
  function->Execute(*exec_ctx, args, {result});

  // Wait for async execution to complete.
  tfrt::Await(*exec_ctx, llvm::makeArrayRef(result));

  // Report error if any.
  if (auto* error = result->GetErrorIfPresent())
    return tensorflow::errors::Internal(error->message);

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
#else   // BEF_THUNKS
namespace xla {

bool gpu::IsBefThunkEnabled() { return false; }

void gpu::SetExecutionContext(const tfrt::ExecutionContext*) {}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>,
    std::vector<BufferAllocation::Slice>) {
  return tensorflow::errors::FailedPrecondition("BefThunks are disabled.");
}

}  // namespace xla
#endif  // BEF_THUNKS
