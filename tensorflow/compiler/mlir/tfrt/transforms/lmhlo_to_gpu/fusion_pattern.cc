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
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_helper.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {

using mlir::ArrayRef;
using mlir::SmallVector;
using mlir::Value;
using mlir::memref::GetGlobalOp;
using xla::gpu::DeviceToDeviceCopyThunk;
using xla::gpu::IrEmitterContext;
using xla::gpu::IrEmitterUnnested;
using xla::gpu::KernelThunk;
using xla::gpu::Thunk;
using xla::gpu::ThunkSequence;
using ConstantInfo = xla::gpu::GpuExecutable::ConstantInfo;

namespace {

// Replaces all lmhlo.fusion ops within a module with tfrt_gpu.launch ops.
struct FusionRewritePattern : mlir::OpRewritePattern<mlir::ModuleOp> {
  using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const override;
};

struct RewriteData {
  mlir::lmhlo::FusionOp fusion_op;
  mlir::SetVector<Value> captures;
  std::vector<xla::BufferAllocation> allocations;
  std::unique_ptr<ThunkSequence> thunks;
  std::vector<ConstantInfo> constants;
  std::string gpu_module_data;
};

}  // namespace

static llvm::Error MakeError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
static llvm::Error MakeError(xla::Status status) {
  return MakeError(status.error_message());
}

// Clones `fusion_op` into a function within a module with `captures` arguments.
// The `get_global_ops` are the def ops of `captures`, or null otherwise.
static std::tuple<mlir::OwningModuleRef, mlir::FuncOp> CloneToModule(
    mlir::lmhlo::FusionOp fusion_op, mlir::ValueRange captures,
    mlir::MutableArrayRef<GetGlobalOp> get_global_ops) {
  auto loc = fusion_op->getLoc();
  auto* context = fusion_op->getContext();
  mlir::OpBuilder builder(context);

  mlir::OwningModuleRef module_op = builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToEnd(module_op->getBody());
  // Clone and annotate the memref.global ops that the memref.get_global ops
  // refer to. The lmhlo.alloc index refers to one of the function arguments.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    auto symbol = mlir::SymbolTable::lookupNearestSymbolFrom(
        pair.value(), pair.value().nameAttr());
    auto attr = builder.getIndexAttr(pair.index());
    builder.clone(*symbol)->setAttr("lmhlo.alloc", attr);
  }

  auto func_type = builder.getType<mlir::FunctionType>(
      mlir::TypeRange(captures), mlir::TypeRange());
  auto func_name = fusion_op->getParentOfType<mlir::FuncOp>().getName();
  auto func_op = builder.create<mlir::FuncOp>(loc, func_name, func_type);
  // Annotate the function arguments if they refer to a memref.global op.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    auto attr = builder.getStringAttr(pair.value().name());
    func_op.setArgAttr(pair.index(), "lmhlo.constant_name", attr);
  }
  func_op.setPublic();

  builder.setInsertionPointToEnd(func_op.addEntryBlock());
  mlir::BlockAndValueMapping mapping;
  for (const auto& pair : llvm::zip_first(captures, func_op.getArguments()))
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  // Clone the memref.get_global ops.
  for (auto get_global_op : get_global_ops) {
    if (!get_global_op) continue;
    mapping.map(get_global_op, builder.clone(*get_global_op)->getResult(0));
  }
  auto* clone = builder.clone(*fusion_op, mapping);
  auto name_loc = mlir::NameLoc::get(builder.getIdentifier(func_name));
  clone->setLoc(mlir::FusedLoc::get(context, {loc, name_loc}));
  builder.create<mlir::lmhlo::TerminatorOp>(loc);

  return std::make_tuple(std::move(module_op), func_op);
}

// Converts the argument's shaped types into buffer allocations.
static llvm::Expected<std::vector<xla::BufferAllocation>> GetAllocations(
    const mlir::SetVector<Value>& captures,
    ArrayRef<GetGlobalOp> get_global_ops) {
  std::vector<xla::BufferAllocation> allocations;
  allocations.reserve(captures.size());
  for (Value argument : captures) {
    mlir::ShapedType type = argument.getType().dyn_cast<mlir::ShapedType>();
    if (!type || !type.hasStaticShape())
      return MakeError("Expected static shapes");
    auto element_size_bytes = xla::GetElementTypeBytes(type.getElementType());
    if (!element_size_bytes.ok()) return MakeError(element_size_bytes.status());
    size_t size = *element_size_bytes * type.getNumElements();
    allocations.emplace_back(allocations.size(), size, 0);
  }
  for (auto pair : llvm::zip_first(allocations, get_global_ops))
    std::get<0>(pair).set_constant(std::get<1>(pair));
  return allocations;
}

// Emits thunks and an llvm device code module for the given func_op.
static llvm::Expected<
    std::tuple<std::unique_ptr<ThunkSequence>, std::vector<ConstantInfo>>>
Emit(mlir::FuncOp func_op, absl::Span<const xla::BufferAllocation> allocations,
     const stream_executor::CudaComputeCapability& cuda_compute_capability,
     const xla::HloModuleConfig& hlo_module_config, llvm::Module* llvm_module) {
  // Hardcoded values for now...
  const char target_triple[] = "nvptx64-nvidia-cuda";
  const char data_layout[] = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  const char platform_name[] = "CUDA";
  xla::gpu::GpuDeviceInfo gpu_device_info = {};
  gpu_device_info.threads_per_block_limit = 1024;
  gpu_device_info.threads_per_warp = 32;
  gpu_device_info.shared_memory_per_block = 49152;  // static shmem limit.
  // Should be 1024 for sm7.5, 1536 for sm8.6. This results in more blocks than
  // SMs on those architectures, but doesn't hit any resource limit.
  gpu_device_info.threads_per_core_limit = 2048;
  // This is higher than any SKU, resulting in more blocks than SMs.
  gpu_device_info.core_count = 128;
  gpu_device_info.block_dim_limit_x = 2147483647;
  gpu_device_info.block_dim_limit_y = 65535;
  gpu_device_info.block_dim_limit_z = 65535;
  const xla::HloProfileIndexMap* profile_index_map = nullptr;

  llvm_module->setTargetTriple(target_triple);
  llvm_module->setDataLayout(data_layout);

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, profile_index_map,
      func_op->getContext(), llvm_module);

  ir_emitter_context.set_allocations(allocations);

  auto ir_emitter =
      IrEmitterUnnested::Create(hlo_module_config, &ir_emitter_context);
  if (!ir_emitter.ok()) return MakeError(ir_emitter.status());

  auto emit_status = (*ir_emitter)->EmitLmhloRegion(&func_op.body());
  if (!emit_status.ok()) return MakeError(emit_status);

  return std::make_tuple((*ir_emitter)->ConsumeThunkSequence(),
                         std::move(ir_emitter_context.constants()));
}

// Returns the data to rewrite fusion_op without changing the IR.
static llvm::Expected<RewriteData> Match(mlir::lmhlo::FusionOp fusion_op) {
  mlir::SetVector<Value> captures;
  getUsedValuesDefinedAbove(fusion_op->getRegions(), captures);

  // Collect captures that are defined by a memref.get_global op. The created
  // module's annotations make the ir emitter recognize them as constants.
  SmallVector<GetGlobalOp, 4> get_global_ops;
  get_global_ops.reserve(captures.size());
  llvm::transform(
      captures, std::back_inserter(get_global_ops),
      [](Value argument) { return argument.getDefiningOp<GetGlobalOp>(); });

  auto allocations = GetAllocations(captures, get_global_ops);
  if (!allocations) return allocations.takeError();
  auto module_op =
      CloneToModule(fusion_op, captures.getArrayRef(), get_global_ops);

  xla::HloModuleConfig hlo_module_config;
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  hlo_module_config.set_debug_options(options);
  stream_executor::CudaComputeCapability cuda_compute_capability = {5, 2};
  llvm::LLVMContext llvm_context;
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);

  auto emit_result =
      Emit(std::get<mlir::FuncOp>(module_op), *allocations,
           cuda_compute_capability, hlo_module_config, llvm_module.get());
  if (!emit_result) return emit_result.takeError();
  auto thunks = std::move(std::get<0>(*emit_result));
  auto constants = std::move(std::get<1>(*emit_result));
  // Inline sequential thunks into the `thunks` vector.
  for (auto it = thunks->begin(); it != thunks->end();) {
    if (it->get()->kind() == Thunk::kSequential) {
      auto sequence = std::move(
          static_cast<xla::gpu::SequentialThunk*>(it->get())->thunks());
      it = thunks->erase(it);
      it = thunks->insert(it, std::make_move_iterator(sequence.begin()),
                          std::make_move_iterator(sequence.end()));
    } else {
      ++it;
    }
  }
  if (!llvm::all_of(*thunks, [](const auto& thunk) {
        Thunk::Kind kinds[] = {Thunk::kKernel, Thunk::kCopy};
        auto equal = [&](Thunk::Kind kind) { return thunk->kind() == kind; };
        return llvm::any_of(kinds, equal);
      })) {
    return MakeError("Expected only kernel and copy thunks");
  }

  auto libdevice_dir = xla::gpu::GetLibdeviceDir(hlo_module_config);
  auto ptx =
      xla::gpu::nvptx::CompileToPtx(llvm_module.get(), cuda_compute_capability,
                                    hlo_module_config, libdevice_dir);
  if (!ptx.ok()) return MakeError(ptx.status());

  return RewriteData{
      fusion_op,         std::move(captures),  std::move(*allocations),
      std::move(thunks), std::move(constants), std::move(*ptx)};
}

// Replaces fusion_op with gpu.launch_func.
static void Rewrite(mlir::lmhlo::FusionOp fusion_op,
                    mlir::PatternRewriter& rewriter,
                    mlir::SymbolTable& symbol_table, ArrayRef<Value> captures,
                    ThunkSequence* thunks, ArrayRef<ConstantInfo> constants,
                    mlir::StringRef gpu_module_data) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto loc = fusion_op->getLoc();

  rewriter.setInsertionPoint(fusion_op->getParentOfType<mlir::FuncOp>());
  auto gpu_module = rewriter.create<mlir::gpu::GPUModuleOp>(loc, "gpu_module");
  symbol_table.insert(gpu_module);
  gpu_module->setAttr(tfrt::gpu::getGpuBinaryAttrName(),
                      rewriter.getStringAttr(gpu_module_data));
  SmallVector<mlir::NamedAttribute, 4> const_attrs;
  for (const auto& constant : constants) {
    if (constant.content.empty()) continue;
    auto type = mlir::RankedTensorType::get(constant.content.size(),
                                            rewriter.getIntegerType(8));
    auto attr = mlir::DenseIntElementsAttr::get(type, constant.content);
    const_attrs.emplace_back(rewriter.getNamedAttr(constant.symbol_name, attr));
  }
  if (!const_attrs.empty())
    gpu_module->setAttr(tfrt::gpu::getGpuConstantsAttrName(),
                        rewriter.getDictionaryAttr(const_attrs));

  for (const auto& thunk : *thunks) {
    if (thunk->kind() == Thunk::kCopy) {
      const auto* copy_thunk =
          static_cast<const DeviceToDeviceCopyThunk*>(thunk.get());
      auto get_argument = [&](const xla::BufferAllocation::Slice& slice) {
        assert(slice.offset() == 0 && slice.size() == copy_thunk->size_bytes());
        Value result = captures[slice.index()];
        // Annotate defining memref.get_global with the gpu_module symbol.
        // Unlike kernel thunks below, which use the global in the kernel only.
        if (auto op = result.getDefiningOp<GetGlobalOp>()) {
          op->setAttr(tfrt::gpu::getGpuModuleAttrName(),
                      mlir::SymbolRefAttr::get(gpu_module));
        }
        return result;
      };
      rewriter.setInsertionPoint(fusion_op);
      rewriter.create<mlir::gpu::MemcpyOp>(
          loc, mlir::TypeRange(), mlir::ValueRange(),
          get_argument(copy_thunk->destination()),
          get_argument(copy_thunk->source()));
      continue;
    }

    const auto* kernel_thunk = static_cast<const KernelThunk*>(thunk.get());
    rewriter.setInsertionPointToStart(gpu_module.getBody());
    SmallVector<Value, 4> arguments;
    for (auto argument : kernel_thunk->arguments())
      arguments.push_back(captures[argument->index()]);
    auto func_type = rewriter.getType<mlir::FunctionType>(
        mlir::TypeRange(mlir::ValueRange(arguments)), mlir::TypeRange());
    mlir::gpu::GPUFuncOp kernel_func = rewriter.create<mlir::gpu::GPUFuncOp>(
        loc, kernel_thunk->kernel_name(), func_type);
    kernel_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                         rewriter.getUnitAttr());
    rewriter.setInsertionPointToEnd(&kernel_func.getBody().back());
    rewriter.create<mlir::gpu::ReturnOp>(loc);

    rewriter.setInsertionPoint(fusion_op);
    auto make_const_idx = [&](int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, attr).getResult();
    };
    auto make_kernel_dim3 = [&](const auto& dim3) {
      return mlir::gpu::KernelDim3{make_const_idx(dim3.x),
                                   make_const_idx(dim3.y),
                                   make_const_idx(dim3.z)};
    };
    const auto& launch_dims = kernel_thunk->launch_dimensions();
    auto grid_size = make_kernel_dim3(launch_dims.block_counts());
    auto block_size = make_kernel_dim3(launch_dims.thread_counts_per_block());

    rewriter.create<mlir::gpu::LaunchFuncOp>(
        loc, kernel_func, grid_size, block_size,
        /*shared_memory_size_bytes=*/nullptr, arguments);
  }

  rewriter.eraseOp(fusion_op);
}

mlir::LogicalResult FusionRewritePattern::matchAndRewrite(
    mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const {
  SmallVector<RewriteData, 4> rewrites;

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

  // Mark module as gpu.container_module.
  rewriter.updateRootInPlace(module_op, [&] {
    module_op->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                       rewriter.getUnitAttr());
  });

  // Replace the lmhlo.fusion ops with gpu.launch_func.
  mlir::SymbolTable symbol_table(module_op);
  for (const auto& data : rewrites) {
    Rewrite(data.fusion_op, rewriter, symbol_table, data.captures.getArrayRef(),
            data.thunks.get(), data.constants, data.gpu_module_data);
  }

  return mlir::success();
}

void populateFusionConversionPattern(mlir::RewritePatternSet& patterns) {
  patterns.add<FusionRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
