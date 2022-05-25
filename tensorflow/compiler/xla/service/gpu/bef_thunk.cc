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

#include <string>

#include "tensorflow/core/platform/errors.h"

#if XLA_ENABLE_XLIR
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_tfrt_gpu.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tfrt/gpu/gpu_executor.h"  // from @tf_runtime
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace xla {
namespace gpu {
namespace {

struct MlirAndTfrtHostCtx {
  mlir::MLIRContext* mlir_ctx;
  tfrt::HostContext* host_ctx;
};

class BefThunk : public Thunk {
 public:
  BefThunk(Thunk::Kind kind, ThunkInfo thunk_info,
           std::vector<BufferAllocation::Slice> buffers,
           tfrt::BefBuffer bef_buffer,
           tfrt::RCReference<tfrt::BEFFile> bef_file)
      : Thunk(kind, thunk_info),
        buffers_(std::move(buffers)),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)) {}

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<BufferAllocation::Slice> buffers_;
  tfrt::BefBuffer bef_buffer_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;

  // The module data will be set in the execution context for kernel thunk to
  // use during execution. The resource contexts cache the loaded modules.
  absl::Mutex mutex_;
  absl::optional<GpuModuleData> gpu_module_data_ ABSL_GUARDED_BY(mutex_);
  tfrt::gpu::GpuContextCache gpu_context_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace

static const char kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

static const char kFuncName[] = "main";

// Clones 'op' into a function within a new module.
static mlir::OwningOpRef<mlir::ModuleOp> CreateModule(mlir::Operation* op) {
  mlir::OpBuilder builder(op->getContext());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      builder.create<mlir::ModuleOp>(op->getLoc());

  // Copy module attributes over to the newly created module.
  (*module)->setAttrs(op->getParentOfType<mlir::ModuleOp>()->getAttrs());

  builder.setInsertionPointToEnd(module->getBody());
  auto func_type = builder.getType<mlir::FunctionType>(op->getOperandTypes(),
                                                       op->getResultTypes());
  auto func =
      builder.create<mlir::func::FuncOp>(op->getLoc(), kFuncName, func_type);
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

static Status RunLmhloGpuToTfrtConversionPipeline(mlir::ModuleOp module) {
  mlir::PassManager pass_manager(module->getContext(),
                                 mlir::PassManager::Nesting::Implicit);
  tensorflow::populateLmhloToTfrtGpuPasses(pass_manager);
  if (failed(pass_manager.run(module)))
    return tensorflow::errors::Internal("Failed to run pass pipeline.");
  return Status::OK();
}

// Converts `module` to BEF.
static StatusOr<std::pair<tfrt::BefBuffer, tfrt::RCReference<tfrt::BEFFile>>>
ConvertToBef(mlir::ModuleOp module, tfrt::HostContext* host) {
  std::string bef;
  llvm::raw_string_ostream bef_ostream(bef);
  if (failed(tfrt::MLIRToBEFTranslate(module, bef_ostream)))
    return tensorflow::errors::Internal("Failed to translate MLIR to BEF.");

  tfrt::BefBuffer bef_buffer(bef.data(), bef.data() + bef.size());
  auto bef_file = tfrt::BEFFile::Open(bef_buffer, host->GetKernelRegistry(),
                                      host->diag_handler(), host->allocator());
  if (!bef_file)
    return tensorflow::errors::Internal("Failed to load BEF file.");

  return std::pair<tfrt::BefBuffer, tfrt::RCReference<tfrt::BEFFile>>(
      std::move(bef_buffer), std::move(bef_file));
}

static StatusOr<Thunk::Kind> GetThunkKind(mlir::Operation* op) {
  if (mlir::isa<mlir::lmhlo_gpu::GEMMOp, mlir::lmhlo_gpu::GEMM_BiasOp>(op)) {
    return Thunk::Kind::kGemm;
  }
  if (mlir::isa<mlir::lmhlo::AllGatherOp>(op)) {
    return Thunk::Kind::kNcclAllGather;
  }
  if (mlir::isa<mlir::lmhlo::AllReduceOp>(op)) {
    return Thunk::Kind::kNcclAllReduce;
  }
  if (mlir::isa<mlir::lmhlo::ReduceScatterOp>(op)) {
    return Thunk::Kind::kNcclReduceScatter;
  }
  if (mlir::isa<mlir::lmhlo::AllToAllOp>(op)) {
    return Thunk::Kind::kNcclAllToAll;
  }
  if (mlir::isa<mlir::lmhlo::CollectivePermuteOp>(op)) {
    return Thunk::Kind::kCollectivePermute;
  }
  if (mlir::isa<mlir::lmhlo::CustomCallOp>(op)) {
    return Thunk::Kind::kCustomCall;
  }
  if (mlir::isa<mlir::lmhlo_gpu::CholeskyOp>(op)) {
    return Thunk::Kind::kCholesky;
  }
  if (mlir::isa<mlir::lmhlo_gpu::ConvForwardOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvBackwardInputOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvBackwardFilterOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvForwardFusedOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvForwardFusedSideInputOp>(op)) {
    return Thunk::Kind::kConvolution;
  }
  if (mlir::isa<mlir::lmhlo::ReplicaIdOp>(op)) {
    return Thunk::Kind::kReplicaId;
  }
  if (mlir::isa<mlir::lmhlo::PartitionIdOp>(op)) {
    return Thunk::Kind::kPartitionId;
  }
  if (mlir::isa<mlir::lmhlo::InfeedOp>(op)) {
    return Thunk::Kind::kInfeed;
  }
  if (mlir::isa<mlir::lmhlo::OutfeedOp>(op)) {
    return Thunk::Kind::kOutfeed;
  }
  if (mlir::isa<mlir::lmhlo::FftOp>(op)) {
    return Thunk::Kind::kFft;
  }
  return tensorflow::errors::Unimplemented(
      "Operation is not supported by BefThunk.");
}

static MlirAndTfrtHostCtx GetMlirAndTfrtHostCtx() {
  static auto* mlir_ctx = new mlir::MLIRContext;
  static auto* host_ctx =
      tfrt::gpu::CreateHostContext(tfrt::gpu::GetDiagHandler(mlir_ctx))
          .release();
  return {mlir_ctx, host_ctx};
}

// Creates a TFRT module that loads the GPU module and launches the target
// kernel function.
static mlir::OwningOpRef<mlir::ModuleOp> CreateTfrtKernelLaunchModule(
    mlir::MLIRContext* mlir_context, const std::string& kernel_name,
    int num_buffers, const LaunchDimensions& launch_dimensions) {
  mlir::OpBuilder builder(mlir_context);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::OwningOpRef<ModuleOp> tfrt_module = builder.create<mlir::ModuleOp>(loc);

  mlir::Type chain_type = builder.getType<tfrt::compiler::ChainType>();
  mlir::Type stream_type = builder.getType<tfrt::gpu::StreamType>();
  mlir::Type buffer_type = builder.getType<tfrt::gpu::BufferType>();
  mlir::Type module_type = builder.getType<tfrt::gpu::ModuleType>();

  // (chain, stream, buffers...) -> chain
  llvm::SmallVector<mlir::Type, 4> input_types = {chain_type, stream_type};
  input_types.resize(input_types.size() + num_buffers, buffer_type);

  // Add a function that loads the module and main function.
  builder.setInsertionPointToEnd(tfrt_module->getBody());
  mlir::func::FuncOp module_func = builder.create<mlir::func::FuncOp>(
      loc, "module_load",
      builder.getFunctionType(builder.getType<tfrt::gpu::ContextType>(),
                              module_type));
  mlir::func::FuncOp main_func = builder.create<mlir::func::FuncOp>(
      loc, kFuncName, builder.getFunctionType(input_types, chain_type));
  main_func.setPublic();

  builder.setInsertionPointToEnd(module_func.addEntryBlock());
  // The module data will be provided by the execution context.
  auto module_load_op =
      builder.create<ModuleLoadOp>(loc, module_func.getArgument(0));
  builder.create<tfrt::compiler::ReturnOp>(loc, module_load_op.getResult());

  builder.setInsertionPointToEnd(main_func.addEntryBlock());
  mlir::Value in_chain = main_func.getArgument(0);
  mlir::Value stream_arg = main_func.getArgument(1);

  auto get_context_op =
      builder.create<tfrt::gpu::StreamGetContextOp>(loc, stream_arg);
  auto once_op = builder.create<tfrt::compiler::OnceOp>(
      loc, module_type, get_context_op.getResult(), module_func.getName());

  auto module_function_op = builder.create<tfrt::gpu::ModuleGetFunctionOp>(
      loc, once_op.getResult(0), builder.getStringAttr(kernel_name));

  auto grid_dim_x = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.block_counts().x);
  auto grid_dim_y = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.block_counts().y);
  auto grid_dim_z = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.block_counts().z);
  auto block_dim_x = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.thread_counts_per_block().x);
  auto block_dim_y = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.thread_counts_per_block().y);
  auto block_dim_z = builder.create<tfrt::compiler::ConstantUI32Op>(
      loc, launch_dimensions.thread_counts_per_block().z);
  // XLA does not use dynamic shared memory, so it's always zero.
  auto shared_mem_size = builder.create<tfrt::compiler::ConstantUI32Op>(loc, 0);

  mlir::Value launch_op = builder.create<tfrt::gpu::FunctionLaunchOp>(
      loc, chain_type, stream_arg, module_function_op, grid_dim_x, grid_dim_y,
      grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_size,
      in_chain, main_func.getArguments().drop_front(2));

  builder.create<tfrt::compiler::ReturnOp>(loc, launch_op);

  return tfrt_module;
}

StatusOr<std::unique_ptr<Thunk>> CreateBefThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> buffers) {
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto module = CreateModule(op);
  TF_RETURN_IF_ERROR(RunLmhloGpuToTfrtConversionPipeline(*module));

  auto mlir_and_host_ctx = GetMlirAndTfrtHostCtx();
  TF_ASSIGN_OR_RETURN(auto bef_result,
                      ConvertToBef(*module, mlir_and_host_ctx.host_ctx));

  return std::unique_ptr<Thunk>(
      new BefThunk(kind, thunk_info, std::move(buffers),
                   std::move(bef_result.first), std::move(bef_result.second)));
}

StatusOr<std::unique_ptr<Thunk>> CreateBefCollectiveThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> buffers, int64_t replica_count,
    int64_t partition_count) {
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto module = CreateModule(op);
  // Forward collective permute attributes for use by the lowering pipeline.
  mlir::OpBuilder builder(module->getContext());
  mlir::IntegerAttr replica_count_attr =
      builder.getI64IntegerAttr(replica_count);
  mlir::IntegerAttr num_partitions_attr =
      builder.getI64IntegerAttr(partition_count);
  mlir::func::FuncOp func = module->lookupSymbol<mlir::func::FuncOp>(kFuncName);
  func->setAttr("replica_count", replica_count_attr);
  func->setAttr("num_partitions", num_partitions_attr);

  TF_RETURN_IF_ERROR(RunLmhloGpuToTfrtConversionPipeline(*module));

  auto mlir_and_host_ctx = GetMlirAndTfrtHostCtx();
  TF_ASSIGN_OR_RETURN(auto bef_result,
                      ConvertToBef(*module, mlir_and_host_ctx.host_ctx));

  return std::unique_ptr<Thunk>(
      new BefThunk(kind, thunk_info, std::move(buffers),
                   std::move(bef_result.first), std::move(bef_result.second)));
}

StatusOr<std::unique_ptr<Thunk>> CreateBefKernelThunk(
    Thunk::ThunkInfo thunk_info, absl::Span<const BufferAllocation* const> args,
    const std::string& kernel_name, const LaunchDimensions& launch_dimensions) {
  // Construct the TFRT module and convert it to BEF.
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::func::FuncDialect, tfrt::compiler::TFRTDialect,
                           tfrt::gpu::GpuDialect, xla::gpu::XlirDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> tfrt_module = CreateTfrtKernelLaunchModule(
      &mlir_context, kernel_name, args.size(), launch_dimensions);

  auto mlir_and_host_ctx = GetMlirAndTfrtHostCtx();
  TF_ASSIGN_OR_RETURN(auto bef_result,
                      ConvertToBef(*tfrt_module, mlir_and_host_ctx.host_ctx));

  std::vector<BufferAllocation::Slice> arg_buffers;
  for (auto arg : args) {
    arg_buffers.emplace_back(arg, /*offset=*/0, arg->size());
  }

  return std::unique_ptr<Thunk>(
      new BefThunk(Thunk::Kind::kKernel, thunk_info, std::move(arg_buffers),
                   std::move(bef_result.first), std::move(bef_result.second)));
}

// Wrap the GPU buffer specified in 'slice' to be passed to BEF functions as
// AsyncValueRef<GpuBuffer>.
static tfrt::RCReference<tfrt::AsyncValue> CreateGpuBuffer(
    const Thunk::ExecuteParams& params, const BufferAllocation::Slice& slice) {
  se::DeviceMemoryBase data =
      params.buffer_allocations->GetDeviceAddress(slice);

  // TODO(hanbinyoon): This should be moved to a function in a central place.
#if TENSORFLOW_USE_ROCM
  auto platform = tfrt::gpu::wrapper::Platform::ROCm;
#else
  auto platform = tfrt::gpu::wrapper::Platform::CUDA;
#endif

  tfrt::gpu::wrapper::Pointer<void> pointer(data.opaque(), platform);
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

static StatusOr<std::unique_ptr<tfrt::ExecutionContext>> CreateExecutionContext(
    const Thunk::ExecuteParams& params,
    tfrt::RequestContextBuilder request_context_builder) {
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  request_context_builder.context_data().emplace<XlaGpuParams>(XlaGpuParams{
      params.run_id, params.device_assn, params.gpu_global_device_ids,
      params.nccl_unique_id_callback, global_device_id,
      GetOrCreateInfeedManager(params.stream->parent()),
      GetOrCreateOutfeedManager(params.stream->parent())});

  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    auto error = expected_req_ctx.takeError();
    return tensorflow::errors::Internal(llvm::toString(std::move(error)));
  }
  return std::make_unique<tfrt::ExecutionContext>(std::move(*expected_req_ctx));
}

static Status InsertKernelRequestContext(
    absl::optional<GpuModuleData> gpu_module_data,
    tfrt::RequestContextBuilder* request_context_builder) {
  if (!gpu_module_data.has_value()) {
    return tensorflow::errors::Internal(
        "GPU module data is not set for the kernel thunk.");
  }
  request_context_builder->context_data().emplace<GpuModuleData>(
      *gpu_module_data);
  return Status::OK();
}

Status BefThunk::Initialize(const GpuExecutable& executable,
                            se::StreamExecutor* executor) {
  // Save the module data for kernel thunk to use during execution.
  if (kind() == Thunk::kKernel) {
    absl::MutexLock lock(&mutex_);
    if (!gpu_module_data_.has_value()) {
      GpuModuleData module_data;
      // The module data should be null-terminated, so the length of the
      // inserted data is incremented by 1 to include '\0'.
      module_data.blob = llvm::ArrayRef<uint8_t>(
          executable.binary().data(), executable.binary().size() + 1);
      for (const auto& constant : executable.constants()) {
        module_data.constants.push_back(GpuModuleData::ConstantInfo{
            constant.symbol_name, constant.content});
      }
      gpu_module_data_ = module_data;
    }
  }
  return Status::OK();
}

Status BefThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(2) << "Executing BEF thunk.";

  // Signature: (chain, stream, inputs..., outputs...) -> (chain).
  const tfrt::Function* function = bef_file_->GetFunction(kFuncName);
  if (!function) {
    return tensorflow::errors::Internal("Failed to get '", kFuncName,
                                        "' function.");
  }

  // Look up or create a cached GpuContext and ResourceContext from CUcontext.
  // The ResourceContext holds the results of `tfrt.once @...(%context)`.
  se::gpu::GpuStream* stream = se::gpu::AsGpuStream(params.stream);
  auto gpu_context = [&] {
    absl::MutexLock lock(&mutex_);
    return gpu_context_cache_.GetOrCreate(
        se::gpu::GpuDriver::GetContextHandle(stream->parent()->gpu_context()));
  }();
  auto gpu_stream =
      tfrt::gpu::MakeBorrowedStream(gpu_context.first, stream->gpu_stream());

  // Create execution context.
  auto mlir_and_host_ctx = GetMlirAndTfrtHostCtx();
  tfrt::RequestContextBuilder request_context_builder(
      mlir_and_host_ctx.host_ctx, gpu_context.second);
  if (kind() == Thunk::kKernel) {
    absl::MutexLock lock(&mutex_);
    TF_RETURN_IF_ERROR(
        InsertKernelRequestContext(gpu_module_data_, &request_context_builder));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<tfrt::ExecutionContext> exec_ctx,
      CreateExecutionContext(params, std::move(request_context_builder)));

  // Create owning handles for arguments and add pointer to them to 'args'.
  llvm::SmallVector<tfrt::AsyncValue*, 8> args;
  args.reserve(function->num_arguments());
  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain();
  args.push_back(chain.GetAsyncValue());
  args.push_back(gpu_stream.get().value());
  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 8> buffers;
  for (auto& buffer : buffers_) {
    buffers.push_back(CreateGpuBuffer(params, buffer));
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

  // Capture errors and augment with source.
  std::string diag_str;
  llvm::raw_string_ostream diag_os(diag_str);
  llvm::SourceMgr src_mgr;
  mlir::SourceMgrDiagnosticHandler handler(src_mgr, mlir_and_host_ctx.mlir_ctx,
                                           diag_os);

  // Execute the function.
  function->Execute(*exec_ctx, args, {result});

  // Wait for async execution to complete.
  tfrt::Await(*exec_ctx, llvm::makeArrayRef(result));

  // Report error if any, from handler and result.
  if (diag_os.tell()) return tensorflow::errors::Internal(diag_os.str());
  if (auto* error = result->GetErrorIfPresent())
    return tensorflow::errors::Internal(tfrt::StrCat(*error));

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
#else  // XLA_ENABLE_XLIR

namespace xla {

static Status GetXlirDisabledError() {
  return tensorflow::errors::FailedPrecondition(
      "Built without XLA_ENABLE_XLIR");
}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>) {
  return GetXlirDisabledError();
}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefCollectiveThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>,
    int64_t, int64_t) {
  return GetXlirDisabledError();
}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefKernelThunk(
    Thunk::ThunkInfo, absl::Span<const BufferAllocation* const>,
    const std::string&, const LaunchDimensions&) {
  return GetXlirDisabledError();
}

}  // namespace xla

#endif  // XLA_ENABLE_XLIR
