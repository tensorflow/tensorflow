// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pattern to lower lmhlo.fusion ops to gpu dialect.

#include <iterator>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"

namespace tensorflow {
namespace {

// Replaces all lmhlo.fusion ops within a module with tfrt_gpu.launch ops.
struct FusionRewritePattern : mlir::OpRewritePattern<mlir::ModuleOp> {
  using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const override;
};

struct RewriteData {
  mlir::lmhlo::FusionOp fusion_op;
  mlir::SetVector<mlir::Value> captures;
  xla::gpu::LaunchDimensions launch_dims;
  std::string gpu_module_data;
};

}  // namespace

static llvm::Error MakeError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
static llvm::Error MakeError(xla::Status status) {
  return MakeError(status.error_message());
}

// Clones 'fusion_op' into a function taking 'arguments' within a module.
static std::tuple<mlir::OwningModuleRef, mlir::FuncOp> CloneToModule(
    mlir::lmhlo::FusionOp fusion_op, mlir::ValueRange arguments) {
  auto loc = fusion_op->getLoc();
  mlir::OpBuilder builder(fusion_op->getContext());

  mlir::OwningModuleRef module_op = builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToEnd(module_op->getBody());

  auto func_type = builder.getType<mlir::FunctionType>(
      mlir::TypeRange(arguments), mlir::TypeRange());
  auto func_op = builder.create<mlir::FuncOp>(loc, "func", func_type);
  func_op.setPublic();

  builder.setInsertionPointToEnd(func_op.addEntryBlock());
  mlir::BlockAndValueMapping mapping;
  for (const auto& pair : llvm::zip_first(arguments, func_op.getArguments())) {
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  }
  builder.clone(*fusion_op, mapping);
  builder.create<mlir::lmhlo::TerminatorOp>(loc);

  return std::make_tuple(std::move(module_op), func_op);
}

// Converts the argument's shaped types into buffer allocations.
static llvm::Expected<mlir::SmallVector<xla::BufferAllocation, 4>>
GetAllocations(mlir::ValueRange arguments) {
  mlir::SmallVector<xla::BufferAllocation, 4> allocations;
  allocations.reserve(arguments.size());
  for (mlir::Value argument : arguments) {
    mlir::ShapedType type = argument.getType().dyn_cast<mlir::ShapedType>();
    if (!type || !type.hasStaticShape())
      return MakeError("Expected static shapes");
    auto element_size_bytes = xla::GetElementTypeBytes(type.getElementType());
    if (!element_size_bytes.ok()) return MakeError(element_size_bytes.status());
    size_t size = *element_size_bytes * type.getNumElements();
    allocations.emplace_back(allocations.size(), size, 0);
  }
  return allocations;
}

// Emits thunks and an llvm device code module for the given func_op.
static llvm::Expected<std::unique_ptr<xla::gpu::IrEmitterUnnested>> Emit(
    mlir::FuncOp func_op, absl::Span<const xla::BufferAllocation> allocations,
    const stream_executor::CudaComputeCapability& cuda_compute_capability,
    const xla::HloModuleConfig& hlo_module_config, llvm::Module* llvm_module) {
  // Hardcoded values for now...
  const char target_triple[] = "nvptx64-nvidia-cuda";
  const char data_layout[] = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  const char platform_name[] = "CUDA";
  xla::gpu::GpuDeviceInfo gpu_device_info = {1024, 32,         49152, 2048,
                                             30,   2147483647, 65535, 65535};
  const xla::HloProfileIndexMap* profile_index_map = nullptr;

  llvm_module->setTargetTriple(target_triple);
  llvm_module->setDataLayout(data_layout);

  xla::gpu::IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, profile_index_map,
      func_op->getContext(), llvm_module);

  ir_emitter_context.set_allocations(allocations);

  auto ir_emitter = xla::gpu::IrEmitterUnnested::Create(hlo_module_config,
                                                        &ir_emitter_context);
  if (!ir_emitter.ok()) return MakeError(ir_emitter.status());

  auto emit_status = (*ir_emitter)->EmitLmhloRegion(&func_op.body());
  if (!emit_status.ok()) return MakeError(emit_status);

  if (!ir_emitter_context.constants().empty())
    return MakeError("constants not yet supported");

  return std::move(*ir_emitter);
}

// Returns the data to rewrite 'fusion_op' without changing the IR.
static llvm::Expected<RewriteData> Match(mlir::lmhlo::FusionOp fusion_op) {
  mlir::SetVector<mlir::Value> captures;
  getUsedValuesDefinedAbove(fusion_op->getRegions(), captures);
  auto arguments = captures.getArrayRef();

  auto allocations = GetAllocations(arguments);
  if (!allocations) return allocations.takeError();
  auto module_op = CloneToModule(fusion_op, arguments);

  xla::HloModuleConfig hlo_module_config;
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  hlo_module_config.set_debug_options(options);
  stream_executor::CudaComputeCapability cuda_compute_capability = {6, 1};
  llvm::LLVMContext llvm_context;
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);
  auto ir_emitter =
      Emit(std::get<mlir::FuncOp>(module_op), *allocations,
           cuda_compute_capability, hlo_module_config, llvm_module.get());

  auto libdevice_dir = xla::gpu::GetLibdeviceDir(hlo_module_config);
  auto ptx =
      xla::gpu::nvptx::CompileToPtx(llvm_module.get(), cuda_compute_capability,
                                    hlo_module_config, libdevice_dir);
  if (!ptx.ok()) return MakeError(ptx.status());

  auto thunks = (*ir_emitter)->ConsumeThunkSequence();
  if (thunks->size() != 1 ||
      thunks->front()->kind() != xla::gpu::Thunk::kKernel)
    return MakeError("Expected single kernel thunk");
  const auto* kernel_thunk =
      static_cast<const xla::gpu::KernelThunk*>(thunks->front().get());

  return RewriteData{fusion_op, std::move(captures),
                     kernel_thunk->GetLaunchDimensions(), std::move(*ptx)};
}

// Replaces 'fusion_op' with 'gpu.launch_func'.
static void Rewrite(mlir::lmhlo::FusionOp fusion_op,
                    mlir::PatternRewriter& rewriter,
                    mlir::ArrayRef<mlir::Value> arguments,
                    const xla::gpu::LaunchDimensions& launch_dims,
                    mlir::StringRef gpu_module_data) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto loc = fusion_op->getLoc();

  rewriter.setInsertionPoint(fusion_op->getParentOfType<mlir::FuncOp>());
  auto gpu_module = rewriter.create<mlir::gpu::GPUModuleOp>(loc, "gpu_module");
  gpu_module->setAttr("nvvm.cubin", rewriter.getStringAttr(gpu_module_data));
  rewriter.setInsertionPointToStart(gpu_module.getBody());
  auto func_type = rewriter.getType<mlir::FunctionType>(
      mlir::TypeRange(arguments), mlir::TypeRange());

  mlir::gpu::GPUFuncOp kernel_func =
      rewriter.create<mlir::gpu::GPUFuncOp>(loc, "kernel", func_type);
  kernel_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                       rewriter.getUnitAttr());
  rewriter.setInsertionPointToEnd(&kernel_func.getBody().back());
  rewriter.create<mlir::gpu::ReturnOp>(loc);

  rewriter.setInsertionPoint(fusion_op);
  auto make_const_idx = [&](int64_t value) {
    auto attr = rewriter.getIndexAttr(value);
    return rewriter.create<mlir::ConstantOp>(loc, attr).getResult();
  };
  auto make_kernel_dim3 = [&](const xla::gpu::LaunchDimensions::Dim3D& dim3) {
    return mlir::gpu::KernelDim3{make_const_idx(dim3.x), make_const_idx(dim3.y),
                                 make_const_idx(dim3.z)};
  };
  auto grid_size = make_kernel_dim3(launch_dims.block_counts());
  auto block_size = make_kernel_dim3(launch_dims.thread_counts_per_block());

  rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
      fusion_op, kernel_func, grid_size, block_size,
      /*shared_memory_size_bytes=*/nullptr, arguments);
}

mlir::LogicalResult FusionRewritePattern::matchAndRewrite(
    mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const {
  mlir::SmallVector<RewriteData, 4> rewrites;

  // Gather data to rewrite each lmhlo.fusion op without changing the IR.
  auto callback = [&](mlir::lmhlo::FusionOp fusion_op) -> mlir::WalkResult {
    auto data = Match(fusion_op);
    if (!data)
      return rewriter.notifyMatchFailure(fusion_op, toString(data.takeError()));
    rewrites.emplace_back(std::move(*data));
    return mlir::success();
  };
  if (module_op.walk(callback).wasInterrupted()) return mlir::failure();

  if (rewrites.empty())
    return rewriter.notifyMatchFailure(module_op, "No lmhlo.fusion ops");

  // Mark module as 'gpu.container_module'.
  rewriter.updateRootInPlace(module_op, [&] {
    module_op->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                       rewriter.getUnitAttr());
  });

  // Replace the lmhlo.fusion ops with gpu.launch_func.
  for (const auto& data : rewrites) {
    Rewrite(data.fusion_op, rewriter, data.captures.getArrayRef(),
            data.launch_dims, data.gpu_module_data);
  }

  return mlir::success();
}

void populateFusionConversionPattern(mlir::RewritePatternSet& patterns) {
  patterns.add<FusionRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
