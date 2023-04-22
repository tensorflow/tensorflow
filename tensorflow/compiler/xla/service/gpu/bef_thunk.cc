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
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/tfrt/gpu/gpu_shared_context.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "third_party/tf_runtime/backends/gpu/include/tfrt/gpu/gpu_types.h"
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
#include "tfrt/support/error_util.h"  // from @tf_runtime

// Common place for all collective thunks to source nccl/rccl headers.
// Also, all the RunNcclCollective() functions for various thunks should
// use XLA_ENABLE_XCCL to guard use NCCL/RCCL usage (and not use GOOGLE_XCCL).
#if GOOGLE_XCCL
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define XLA_ENABLE_XCCL 1
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // GOOGLE_XCCL

#if XLA_ENABLE_XCCL
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#error "Neither CUDA nor ROCm enabled but NCCL/RCCL enabled"
#endif

// Also include this file required by all collective thunks.
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#endif  // XLA_ENABLE_XCCL

namespace xla {
namespace gpu {

bool IsBefThunkEnabled() { return true; }

namespace {

struct CoreRuntimeAndWorkQueue {
  tfrt::CoreRuntime* core_runtime;
  tensorflow::tfrt_stub::WorkQueueInterface* work_queue;
};

class BefThunk : public Thunk {
 public:
  BefThunk(Thunk::Kind kind, ThunkInfo thunk_info,
           std::vector<BufferAllocation::Slice> inputs,
           std::vector<BufferAllocation::Slice> outputs,
           tfrt::BefBuffer bef_buffer,
           tfrt::RCReference<tfrt::BEFFile> bef_file, mlir::Operation* op)
      : Thunk(kind, thunk_info),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)) {
    // TODO(hanbinyoon): Also handle other collective ops.
    if (auto all_reduce_op = mlir::dyn_cast<mlir::lmhlo::AllReduceOp>(*op)) {
      xccl_config_ = GetNcclCollectiveConfigForMlir(
          all_reduce_op, all_reduce_op.use_global_device_ids());
    }
  }

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::vector<BufferAllocation::Slice> inputs_;
  std::vector<BufferAllocation::Slice> outputs_;
  tfrt::BefBuffer bef_buffer_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;
  absl::optional<NcclCollectiveConfig> xccl_config_;
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

static StatusOr<CoreRuntimeAndWorkQueue> GetCoreRuntimeAndWorkQueue() {
  // TODO(hanbinyoon): Make these configurable.
  int tfrt_num_threads = tensorflow::port::MaxParallelism();
  int tfrt_num_blocking_threads = 16;

  static StatusOr<CoreRuntimeAndWorkQueue>* runtime_and_queue_or =
      [&](int num_threads, int num_blocking_threads) {
        // Create work queue.
        auto work_queue = tensorflow::tfrt_stub::WrapDefaultWorkQueue(
            tfrt::CreateMultiThreadedWorkQueue(num_threads,
                                               num_blocking_threads));
        if (work_queue == nullptr) {
          auto status =
              tensorflow::errors::Internal("Failed to create TFRT work queue.");
          return new StatusOr<CoreRuntimeAndWorkQueue>(status);
        }
        auto* work_queue_ptr = work_queue.get();

        // Create core runtime.
        auto expected_core_runtime = tfrt::CoreRuntime::Create(
            [](const tfrt::DecodedDiagnostic& diag) {
              LOG(ERROR) << diag.message;
            },
            tfrt::CreateMallocAllocator(), std::move(work_queue),
            kDefaultHostDeviceName);
        if (!expected_core_runtime) {
          auto error = expected_core_runtime.takeError();
          auto status =
              tensorflow::errors::Internal(llvm::toString(std::move(error)));
          return new StatusOr<CoreRuntimeAndWorkQueue>(status);
        }

        auto runtime_and_queue = CoreRuntimeAndWorkQueue{
            expected_core_runtime->release(), work_queue_ptr};
        return new StatusOr<CoreRuntimeAndWorkQueue>(runtime_and_queue);
      }(tfrt_num_threads, tfrt_num_blocking_threads);

  TF_RETURN_IF_ERROR(runtime_and_queue_or->status());
  return runtime_and_queue_or->ValueOrDie();
}

StatusOr<std::unique_ptr<Thunk>> CreateBefThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> inputs,
    std::vector<BufferAllocation::Slice> outputs) {
  TF_ASSIGN_OR_RETURN(auto kind, GetThunkKind(op));
  auto module = CreateModule(op);
  TF_ASSIGN_OR_RETURN(tfrt::BefBuffer bef_buffer, ConvertToBef(*module));

  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  tfrt::HostContext* host = runtime_and_queue.core_runtime->GetHostContext();
  auto bef_file = tfrt::BEFFile::Open(bef_buffer, host->GetKernelRegistry(),
                                      host->diag_handler(), host->allocator());
  if (!bef_file)
    return tensorflow::errors::Internal("Failed to load BEF file.");

  return std::unique_ptr<Thunk>(
      new BefThunk(kind, thunk_info, std::move(inputs), std::move(outputs),
                   std::move(bef_buffer), std::move(bef_file), op));
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

static Status CreateXcclContext(
    const Thunk::ExecuteParams& params, const NcclCollectiveConfig& xccl_config,
    tfrt::RequestContextBuilder* request_context_builder) {
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

  TF_ASSIGN_OR_RETURN(
      std::vector<LocalParticipant> local_participants,
      GetLocalParticipants(participants, params.gpu_global_device_ids));
  absl::flat_hash_map<tfrt::gpu::GpuSharedContext::LocalDeviceIdentifier, int>
      local_ids_to_rank;
  for (const auto& participant : local_participants) {
    local_ids_to_rank[participant.device_ordinal] = participant.rank;
  }

  std::vector<int64> gpu_global_device_ids;
  if (params.gpu_global_device_ids != nullptr) {
    for (const auto& global_device_id : *params.gpu_global_device_ids) {
      gpu_global_device_ids.push_back(global_device_id.value());
    }
  }

  tfrt::gpu::XcclUniqueIdCallback xccl_unique_id_callback;
  if (params.nccl_unique_id_callback != nullptr) {
    xccl_unique_id_callback = [&](const tfrt::gpu::XcclCliqueKey& kernel_key)
        -> llvm::Expected<std::string> {
      std::vector<GlobalDeviceId> devices;
      for (const int64_t device : kernel_key) {
        devices.push_back(GlobalDeviceId(device));
      }
      auto nccl_unique_id_or =
          (*params.nccl_unique_id_callback)(NcclCliqueKey(devices));
      if (!nccl_unique_id_or.ok()) {
        return tfrt::MakeStringError(
            nccl_unique_id_or.status().error_message());
      }
      return nccl_unique_id_or.ValueOrDie();
    };
  }

  request_context_builder->context_data().emplace<tfrt::gpu::GpuSharedContext>(
      params.run_id.ToInt(), std::move(local_ids_to_rank),
      std::move(gpu_global_device_ids), std::move(xccl_unique_id_callback),
      /*compiled_code=*/nullptr);
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

  // Create execution context.
  TF_ASSIGN_OR_RETURN(auto runtime_and_queue, GetCoreRuntimeAndWorkQueue());
  tfrt::RequestContextBuilder request_context_builder(
      runtime_and_queue.core_runtime->GetHostContext(),
      /*resource_context=*/nullptr);
  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool = nullptr;
  TF_RETURN_IF_ERROR(runtime_and_queue.work_queue->InitializeRequest(
      &request_context_builder, &intra_op_threadpool));
  if (xccl_config_.has_value()) {
    TF_RETURN_IF_ERROR(
        CreateXcclContext(params, *xccl_config_, &request_context_builder));
  }
  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    auto error = expected_req_ctx.takeError();
    return tensorflow::errors::Internal(llvm::toString(std::move(error)));
  }
  tfrt::ExecutionContext exec_ctx(std::move(*expected_req_ctx));

  // Create owning handles for arguments and add pointer to them to 'args'.
  tfrt::SmallVector<tfrt::AsyncValue*, 8> args;
  args.reserve(function->num_arguments());
  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain(exec_ctx.host());
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
  function->Execute(exec_ctx, args, {result});

  // Wait for async execution to complete.
  tfrt::Await(exec_ctx, llvm::makeArrayRef(result));

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

StatusOr<std::unique_ptr<gpu::Thunk>> gpu::CreateBefThunk(
    Thunk::ThunkInfo, mlir::Operation*, std::vector<BufferAllocation::Slice>,
    std::vector<BufferAllocation::Slice>) {
  return tensorflow::errors::FailedPrecondition("BefThunks are disabled.");
}

}  // namespace xla
#endif  // BEF_THUNKS
