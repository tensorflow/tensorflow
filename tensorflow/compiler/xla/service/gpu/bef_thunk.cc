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

#if BEF_THUNKS
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_tfrt_gpu.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
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
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace xla {
namespace gpu {

bool IsBefThunkEnabled() { return true; }

namespace {

struct CoreRuntimeAndWorkQueue {
  mlir::MLIRContext* mlir_ctx;
  tfrt::CoreRuntime* core_runtime;
  tensorflow::tfrt_stub::WorkQueueInterface* work_queue;
};

class BefThunk : public Thunk {
 public:
  BefThunk(Thunk::Kind kind, ThunkInfo thunk_info,
           std::vector<BufferAllocation::Slice> buffers,
           tfrt::BefBuffer bef_buffer,
           tfrt::RCReference<tfrt::BEFFile> bef_file,
           mlir::Operation* op = nullptr)
      : Thunk(kind, thunk_info),
        buffers_(std::move(buffers)),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)) {
    if (auto all_gather_op =
            mlir::dyn_cast_or_null<mlir::lmhlo::AllGatherOp>(op)) {
      xccl_config_ = GetNcclCollectiveConfigForMlir(
          all_gather_op, all_gather_op.use_global_device_ids());
    }
    if (auto all_reduce_op =
            mlir::dyn_cast_or_null<mlir::lmhlo::AllReduceOp>(op)) {
      xccl_config_ = GetNcclCollectiveConfigForMlir(
          all_reduce_op, all_reduce_op.use_global_device_ids());
    }
    if (auto reduce_scatter_op =
            mlir::dyn_cast_or_null<mlir::lmhlo::ReduceScatterOp>(op)) {
      xccl_config_ = GetNcclCollectiveConfigForMlir(
          reduce_scatter_op, reduce_scatter_op.use_global_device_ids());
    }
    if (auto all_to_all_op =
            mlir::dyn_cast_or_null<mlir::lmhlo::AllToAllOp>(op)) {
      xccl_config_ = GetNcclCollectiveConfigForMlir(
          all_to_all_op, all_to_all_op.use_global_device_ids());
    }
  }

  // Constructor for performing Collective Permute.
  BefThunk(Thunk::Kind kind, ThunkInfo thunk_info,
           std::vector<BufferAllocation::Slice> buffers,
           tfrt::BefBuffer bef_buffer,
           tfrt::RCReference<tfrt::BEFFile> bef_file, int64_t replica_count,
           int64_t partition_count, mlir::Operation* op = nullptr)
      : Thunk(kind, thunk_info),
        buffers_(std::move(buffers)),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)) {
    if (auto collective_permute_op =
            mlir::dyn_cast_or_null<mlir::lmhlo::CollectivePermuteOp>(op)) {
      auto config = NcclCollectivePermuteThunk::GetNcclCollectivePermuteConfig(
          collective_permute_op, replica_count, partition_count);
      id_to_collective_permute_source_target_ =
          std::move(config.id_to_source_target);
      xccl_config_ = std::move(config);
    }
  }

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<BufferAllocation::Slice> buffers_;
  tfrt::BefBuffer bef_buffer_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;

  // Used only when performing collective ops.
  absl::optional<NcclCollectiveConfig> xccl_config_;
  absl::flat_hash_map<int64_t,
                      NcclCollectivePermuteConfig::SourceTargetMapEntry>
      id_to_collective_permute_source_target_;

  // The module data will be set in the execution context for kernel thunk to
  // use during execution. The resource contexts cache the loaded modules.
  tensorflow::mutex mutex_;
  absl::optional<GpuModuleData> gpu_module_data_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<CUcontext, std::unique_ptr<tfrt::ResourceContext>>
      resource_contexts_ TF_GUARDED_BY(mutex_);
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
  if (mlir::isa<mlir::gpu::MemcpyOp>(op)) {
    return Thunk::Kind::kCopy;
  }
  if (mlir::isa<mlir::gpu::MemsetOp>(op)) {
    return Thunk::Kind::kMemset32BitValue;
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
  if (mlir::isa<mlir::lmhlo::TriangularSolveOp>(op)) {
    return Thunk::Kind::kTriangularSolve;
  }
  if (mlir::isa<mlir::lmhlo_gpu::ConvForwardOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvBackwardInputOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvBackwardFilterOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvForwardFusedOp>(op) ||
      mlir::isa<mlir::lmhlo_gpu::ConvForwardFusedSideInputOp>(op)) {
    return Thunk::Kind::kConvolution;
  }
  return tensorflow::errors::Unimplemented(
      "Operation is not supported by BefThunk.");
}

static StatusOr<CoreRuntimeAndWorkQueue> GetCoreRuntimeAndWorkQueue() {
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
  mlir::FuncOp module_func = builder.create<mlir::FuncOp>(
      loc, "module_load",
      builder.getFunctionType(builder.getType<tfrt::gpu::ContextType>(),
                              module_type));
  mlir::FuncOp main_func = builder.create<mlir::FuncOp>(
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

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  TF_ASSIGN_OR_RETURN(
      auto bef_result,
      ConvertToBef(*module, runtime_and_queue.core_runtime->GetHostContext()));

  return std::unique_ptr<Thunk>(new BefThunk(
      kind, thunk_info, std::move(buffers), std::move(bef_result.first),
      std::move(bef_result.second), op));
}

StatusOr<std::unique_ptr<Thunk>> CreateBefCollectivePermuteThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> buffers, int64_t replica_count,
    int64_t partition_count) {
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto module = CreateModule(op);
  TF_RETURN_IF_ERROR(RunLmhloGpuToTfrtConversionPipeline(*module));

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  TF_ASSIGN_OR_RETURN(
      auto bef_result,
      ConvertToBef(*module, runtime_and_queue.core_runtime->GetHostContext()));

  return std::unique_ptr<Thunk>(new BefThunk(
      kind, thunk_info, std::move(buffers), std::move(bef_result.first),
      std::move(bef_result.second), replica_count, partition_count, op));
}

StatusOr<std::unique_ptr<Thunk>> CreateBefKernelThunk(
    Thunk::ThunkInfo thunk_info, absl::Span<const BufferAllocation* const> args,
    const std::string& kernel_name, const LaunchDimensions& launch_dimensions) {
  // Construct the TFRT module and convert it to BEF.
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<tfrt::compiler::TFRTDialect, tfrt::gpu::GpuDialect,
                           xla::gpu::XlirDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> tfrt_module = CreateTfrtKernelLaunchModule(
      &mlir_context, kernel_name, args.size(), launch_dimensions);

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  TF_ASSIGN_OR_RETURN(
      auto bef_result,
      ConvertToBef(*tfrt_module,
                   runtime_and_queue.core_runtime->GetHostContext()));

  std::vector<BufferAllocation::Slice> arg_buffers;
  for (auto arg : args) {
    arg_buffers.emplace_back(arg, /*offset=*/0, arg->size());
  }

  return std::unique_ptr<Thunk>(
      new BefThunk(Thunk::Kind::kKernel, thunk_info, std::move(arg_buffers),
                   std::move(bef_result.first), std::move(bef_result.second)));
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

static StatusOr<std::unique_ptr<tfrt::ExecutionContext>> CreateExecutionContext(
    std::function<Status(tfrt::RequestContextBuilder&)> build_request_context,
    tfrt::ResourceContext* resource_context = nullptr) {
  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  tfrt::RequestContextBuilder request_context_builder(
      runtime_and_queue.core_runtime->GetHostContext(), resource_context);
  TF_RETURN_IF_ERROR(build_request_context(request_context_builder));

  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    auto error = expected_req_ctx.takeError();
    return tensorflow::errors::Internal(llvm::toString(std::move(error)));
  }
  return std::make_unique<tfrt::ExecutionContext>(std::move(*expected_req_ctx));
}

static StatusOr<std::unique_ptr<tfrt::ExecutionContext>>
CreateDefaultExecutionContext(
    tfrt::ResourceContext* resource_context = nullptr) {
  return CreateExecutionContext(
      [](tfrt::RequestContextBuilder& request_context_builder) {
        return Status::OK();
      },
      resource_context);
}

#if XLA_ENABLE_XCCL
static StatusOr<std::unique_ptr<tfrt::ExecutionContext>>
CreateXcclExecutionContext(const Thunk::ExecuteParams& params,
                           const NcclCollectiveConfig& xccl_config) {
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(std::vector<GlobalDeviceId> participants,
                      GetParticipatingDevices(
                          global_device_id, *params.device_assn,
                          xccl_config.replica_groups, xccl_config.group_mode));

  if (IsGlobalNcclConfig() &&
      (participants.size() != params.device_assn->replica_count())) {
    return InvalidArgument(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  auto it = absl::c_find(participants, global_device_id);
  TF_RET_CHECK(it != participants.end());
  int rank = it - participants.begin();

  OpId op_id(xccl_config.op_id);
  size_t num_local_participants = GetNumLocalParticipants(
      participants, /*local_devices=*/params.gpu_global_device_ids);

  bool is_local = participants.size() == num_local_participants;
  TF_ASSIGN_OR_RETURN(
      const NcclUniqueIdCallback* unique_id_callback,
      GetNcclUniqueIdCallback(params.nccl_unique_id_callback, is_local));

  TF_ASSIGN_OR_RETURN(
      NcclComm::Lock comm,
      AcquireNcclComm(params.run_id, op_id, std::move(participants),
                      num_local_participants, *unique_id_callback, rank));

  return CreateExecutionContext(
      [&](tfrt::RequestContextBuilder& request_context_builder) {
        request_context_builder.context_data().emplace<XcclContext>(
            std::move(comm));
        return Status::OK();
      });
}

static StatusOr<XcclContext::CollectivePermuteSourceTarget>
GetCollectivePermuteSourceTarget(
    const Thunk::ExecuteParams& params, const NcclCollectiveConfig& xccl_config,
    const absl::flat_hash_map<
        int64_t, NcclCollectivePermuteConfig::SourceTargetMapEntry>&
        id_to_collective_permute_source_target) {
  // NCCL 2.8.x has an issue with point-to-point communication primitives if
  // different ranks process different amounts of data. This can happen in the
  // case of a collective permute as certain nodes may not do any send or
  // receives, or do only send or only receive. Sending and receiving to self
  // as well (identity pair) causes this imbalance. NCCL 2.8.x requires the
  // use of NCCL_LAUNCH_MODE=PARALLEL to avoid these issues. See
  // https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-8-4.html#rel_2-8-4
  if (!IsNcclLaunchModeParallel()) {
    LOG(WARNING) << "NCCL based collective permute may not work correctly if "
                    "NCCL_LAUNCH_MODE is not set to PARALLEL";
  }

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(DeviceAssignment::LogicalID current_logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  const int64_t current_id =
      xccl_config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;

  auto it = id_to_collective_permute_source_target.find(current_id);
  if (it != id_to_collective_permute_source_target.end())
    return XcclContext::CollectivePermuteSourceTarget{it->second.source,
                                                      it->second.target};
  return XcclContext::CollectivePermuteSourceTarget{};
}
#endif  // XLA_ENABLE_XCCL

static StatusOr<std::unique_ptr<tfrt::ExecutionContext>>
CreateKernelExecutionContext(absl::optional<GpuModuleData> gpu_module_data,
                             tfrt::ResourceContext* resource_context) {
  if (!gpu_module_data.has_value()) {
    return tensorflow::errors::Internal(
        "GPU module data is not set for the kernel thunk.");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<tfrt::ExecutionContext> exec_ctx,
      CreateExecutionContext(
          [&](tfrt::RequestContextBuilder& request_context_builder) {
            request_context_builder.context_data().emplace<GpuModuleData>(
                *gpu_module_data);
            return Status::OK();
          },
          resource_context));
  return std::move(exec_ctx);
}

Status BefThunk::Initialize(const GpuExecutable& executable,
                            se::StreamExecutor* executor) {
  // Save the module data for kernel thunk to use during execution.
  if (kind() == Thunk::kKernel) {
    tensorflow::mutex_lock lock(mutex_);
    if (!gpu_module_data_.has_value()) {
      GpuModuleData module_data;
      // The module data should be null-terminated, so the length of the
      // inserted data is incremented by 1 to include '\0'.
      module_data.blob = llvm::StringRef(executable.text().c_str(),
                                         executable.text().size() + 1);
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

  tfrt::gpu::BorrowedGpuStream stream = CreateGpuStream(params);

  // Create execution context.
  std::unique_ptr<tfrt::ExecutionContext> exec_ctx;
#if XLA_ENABLE_XCCL
  if (xccl_config_.has_value()) {
    TF_ASSIGN_OR_RETURN(exec_ctx,
                        CreateXcclExecutionContext(params, *xccl_config_));
    if (!id_to_collective_permute_source_target_.empty()) {
      auto& xccl_ctx = exec_ctx->request_ctx()->GetData<XcclContext>();
      TF_ASSIGN_OR_RETURN(
          xccl_ctx.collective_permute_source_target,
          GetCollectivePermuteSourceTarget(
              params, *xccl_config_, id_to_collective_permute_source_target_));
    }
  }
#endif  // XLA_ENABLE_XCCL
  if (!exec_ctx) {
    if (kind() == Thunk::kKernel || kind() == Thunk::kConvolution) {
      tensorflow::mutex_lock lock(mutex_);
      using AsyncStreamRef = tfrt::AsyncValueRef<tfrt::gpu::GpuStream>;
      CUcontext context = static_cast<AsyncStreamRef>(stream)->context()->get();
      auto it = resource_contexts_.find(context);
      if (it == resource_contexts_.end()) {
        it = resource_contexts_.emplace_hint(it, context,
                                             new tfrt::ResourceContext());
      }
      if (kind() == Thunk::kKernel) {
        TF_ASSIGN_OR_RETURN(exec_ctx, CreateKernelExecutionContext(
                                          gpu_module_data_, it->second.get()));
      } else {  // kind() == Thunk::kConvolution
        TF_ASSIGN_OR_RETURN(exec_ctx,
                            CreateDefaultExecutionContext(it->second.get()));
      }
    } else {
      TF_ASSIGN_OR_RETURN(exec_ctx, CreateDefaultExecutionContext());
    }
  }

  // Create owning handles for arguments and add pointer to them to 'args'.
  tfrt::SmallVector<tfrt::AsyncValue*, 8> args;
  args.reserve(function->num_arguments());
  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain();
  args.push_back(chain.GetAsyncValue());
  args.push_back(static_cast<tfrt::AsyncValueRef<tfrt::gpu::GpuStream>>(stream)
                     .GetAsyncValue());
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

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  // Capture errors and augment with source.
  std::string diag_str;
  llvm::raw_string_ostream diag_os(diag_str);
  llvm::SourceMgr src_mgr;
  mlir::SourceMgrDiagnosticHandler handler(src_mgr, runtime_and_queue.mlir_ctx,
                                           diag_os);

  // Execute the function.
  function->Execute(*exec_ctx, args, {result});

  // Wait for async execution to complete.
  tfrt::Await(*exec_ctx, llvm::makeArrayRef(result));

#if XLA_ENABLE_XCCL
  if (xccl_config_.has_value()) {
    auto& xccl_ctx = exec_ctx->request_ctx()->GetData<XcclContext>();
    // Release the ownership of comms lent to tfrt::gpu::GpuCclHandle.
    xccl_ctx.ccl_handle->release();
    xccl_ctx.ccl_handle.reset();
  }
#endif  // XLA_ENABLE_XCCL

  // Report error if any, from handler and result.
  if (diag_os.tell()) return tensorflow::errors::Internal(diag_os.str());
  if (auto* error = result->GetErrorIfPresent())
    return tensorflow::errors::Internal(tfrt::StrCat(*error));

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
#else   // BEF_THUNKS
namespace xla {

bool gpu::IsBefThunkEnabled() { return false; }

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>) {
  return tensorflow::errors::FailedPrecondition("BefThunks are disabled.");
}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefCollectivePermuteThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>,
    int64_t, int64_t) {
  return tensorflow::errors::FailedPrecondition("BefThunks are disabled.");
}

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefKernelThunk(
    Thunk::ThunkInfo, absl::Span<const BufferAllocation* const>,
    const std::string&, const LaunchDimensions&) {
  return tensorflow::errors::FailedPrecondition(
      "BefKernelThunks are disabled.");
}

}  // namespace xla
#endif  // BEF_THUNKS
