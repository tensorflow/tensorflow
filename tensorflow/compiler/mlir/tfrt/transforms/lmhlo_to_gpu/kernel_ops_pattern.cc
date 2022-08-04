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

// Pattern to lower lmhlo ops with help of the ir emitter to gpu device code
// and gpu dialect ops (gpu.launch_func and gpu.memcpy).

#include <iterator>
#include <numeric>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

#if TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#else
#include "tensorflow/compiler/xla/service/gpu/nvptx_helper.h"
#endif

namespace tensorflow {

using mlir::ArrayRef;
using mlir::FloatType;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Value;
using mlir::arith::ConstantFloatOp;
using mlir::arith::ConstantIntOp;
using mlir::arith::ConstantOp;
using mlir::memref::GetGlobalOp;
using xla::gpu::DeviceToDeviceCopyThunk;
using xla::gpu::IrEmitterContext;
using xla::gpu::IrEmitterUnnested;
using xla::gpu::KernelThunk;
using xla::gpu::Thunk;
using xla::gpu::ThunkSequence;
using ConstantInfo = xla::gpu::GpuExecutable::ConstantInfo;

namespace {

mlir::Value MakeBitPatternConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type type, uint32_t bit_pattern) {
  // In XLA a 1-byte bit pattern copied to fill a 32-byte word when
  // `Memset32BitValueThunk` is constructed, so to get back an `i1` constant we
  // only need to check if any bit is set to `1`.
  if (type.isInteger(1)) {
    return builder.create<ConstantOp>(loc, builder.getBoolAttr(bit_pattern));
  }

  if (type.isInteger(32)) {
    llvm::APInt i32(32, bit_pattern);
    return builder.create<ConstantIntOp>(loc, i32.getSExtValue(), type);
  }

  if (type.isF32()) {
    llvm::APFloat f32(llvm::APInt(32, bit_pattern).bitsToFloat());
    return builder.create<ConstantFloatOp>(loc, f32, type.cast<FloatType>());
  }

  llvm_unreachable("unsupported type");
}

// Replaces lmhlo ops within a module with gpu.launch_func and gpu.memcpy ops.
struct KernelOpsPattern : mlir::OpRewritePattern<mlir::ModuleOp> {
  using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const override;
};

struct RewriteData {
  Operation* op;
  mlir::SmallVector<Value, 4> arguments;
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

// Clones `op` into a function within a module with `arguments`.
// The `get_global_ops` are the def ops of `arguments`, or null otherwise.
static std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, mlir::func::FuncOp>
CloneToModule(Operation* op, mlir::ValueRange arguments,
              mlir::MutableArrayRef<GetGlobalOp> get_global_ops) {
  auto loc = op->getLoc();
  auto* context = op->getContext();
  mlir::OpBuilder builder(context);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToEnd(module_op->getBody());
  // Clone and annotate the memref.global ops that the memref.get_global ops
  // refer to. The lmhlo.alloc index refers to one of the function arguments.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    Operation* global_op = mlir::SymbolTable::lookupNearestSymbolFrom(
        pair.value(), pair.value().getNameAttr());
    auto attr = builder.getIndexAttr(pair.index());
    builder.clone(*global_op)->setAttr("lmhlo.alloc", attr);
  }

  // If 'op' is a gpu.launch_func, clone referenced gpu.module.
  if (auto launch_func_op = llvm::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
    builder.clone(*mlir::SymbolTable::lookupNearestSymbolFrom(
        op, launch_func_op.getKernelModuleName()));
  }

  auto func_type = builder.getType<mlir::FunctionType>(
      mlir::TypeRange(arguments), mlir::TypeRange());
  auto func_name = op->getParentOfType<mlir::func::FuncOp>().getName();
  auto func_op = builder.create<mlir::func::FuncOp>(loc, func_name, func_type);
  // Annotate the function arguments if they refer to a memref.global op.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    auto attr = builder.getStringAttr(pair.value().getName());
    func_op.setArgAttr(pair.index(), "lmhlo.constant_name", attr);
  }
  func_op.setPublic();

  builder.setInsertionPointToEnd(func_op.addEntryBlock());
  mlir::BlockAndValueMapping mapping;
  for (const auto& pair : llvm::zip_first(arguments, func_op.getArguments()))
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  // Clone the memref.get_global ops.
  for (auto get_global_op : get_global_ops) {
    if (!get_global_op) continue;
    mapping.map(get_global_op, builder.clone(*get_global_op)->getResult(0));
  }
  auto* clone = builder.clone(*op, mapping);
  auto name_loc = mlir::NameLoc::get(builder.getStringAttr(func_name));
  clone->setLoc(mlir::FusedLoc::get(context, {loc, name_loc}));
  builder.create<mlir::lmhlo::TerminatorOp>(loc);

  return std::make_tuple(std::move(module_op), func_op);
}

// Converts the argument's shaped types into buffer allocations.
static llvm::Expected<std::vector<xla::BufferAllocation>> GetAllocations(
    ArrayRef<Value> arguments, ArrayRef<GetGlobalOp> get_global_ops) {
  std::vector<xla::BufferAllocation> allocations;
  allocations.reserve(arguments.size());
  for (Value argument : arguments) {
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
Emit(mlir::func::FuncOp func_op,
     absl::Span<const xla::BufferAllocation> allocations,
     const stream_executor::CudaComputeCapability& cuda_compute_capability,
     const stream_executor::RocmComputeCapability& rocm_compute_capability,
     const xla::HloModuleConfig& hlo_module_config, llvm::Module* llvm_module) {
#if TENSORFLOW_USE_ROCM
  const char target_triple[] = "amdgcn-amd-amdhsa";
  const char data_layout[] =
      "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-"
      "v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-"
      "v2048:2048-n32:64-A5";
  const char platform_name[] = "ROCm";
#else
  const char target_triple[] = "nvptx64-nvidia-cuda";
  const char data_layout[] = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  const char platform_name[] = "CUDA";
#endif
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

  llvm_module->setTargetTriple(target_triple);
  llvm_module->setDataLayout(data_layout);

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, rocm_compute_capability,
      func_op->getContext(), llvm_module);

  ir_emitter_context.set_allocations(allocations);

  auto ir_emitter =
      IrEmitterUnnested::Create(hlo_module_config, &ir_emitter_context);
  if (!ir_emitter.ok()) return MakeError(ir_emitter.status());

  auto emit_status = (*ir_emitter)->EmitLmhloRegion(&func_op.getBody());
  if (!emit_status.ok()) return MakeError(emit_status);

  return std::make_tuple((*ir_emitter)->ConsumeThunkSequence(),
                         std::move(ir_emitter_context.constants()));
}

// Returns the data to rewrite op without changing the IR.
static llvm::Expected<RewriteData> Match(Operation* op) {
  mlir::SmallVector<Value> arguments;
  llvm::copy_if(
      op->getOperands(), std::back_inserter(arguments),
      // Filter block/thread size arguments of gpu.launch_func.
      [](Value value) { return value.getType().isa<mlir::ShapedType>(); });
  mlir::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  llvm::copy(captures, std::back_inserter(arguments));

  // Collect arguments that are defined by a memref.get_global op. The
  // created module's annotations make the ir emitter recognize them as
  // constants.
  SmallVector<GetGlobalOp, 4> get_global_ops;
  get_global_ops.reserve(arguments.size());
  llvm::transform(
      arguments, std::back_inserter(get_global_ops),
      [](Value argument) { return argument.getDefiningOp<GetGlobalOp>(); });

  auto allocations = GetAllocations(arguments, get_global_ops);
  if (!allocations) return allocations.takeError();
  auto module_op = CloneToModule(op, arguments, get_global_ops);

  xla::HloModuleConfig hlo_module_config;
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  hlo_module_config.set_debug_options(options);
  // TODO(b/228163857): pass down capability from CompileModuleToLlvmIrImpl().
  stream_executor::CudaComputeCapability cuda_compute_capability = {5, 2};
  stream_executor::RocmComputeCapability rocm_compute_capability("gfx900");
#if TENSORFLOW_USE_ROCM
  auto platform = xla::PlatformUtil::GetPlatform("gpu");
  if (!platform.ok()) return MakeError(platform.status());
  auto stream_executors = xla::PlatformUtil::GetStreamExecutors(*platform);
  if (!stream_executors.ok()) return MakeError(stream_executors.status());
  if (stream_executors->empty()) return MakeError("No gpu stream executors");
  rocm_compute_capability = stream_executors->front()
                                ->GetDeviceDescription()
                                .rocm_compute_capability();
#endif
  llvm::LLVMContext llvm_context;
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);

  auto emit_result = Emit(std::get<mlir::func::FuncOp>(module_op), *allocations,
                          cuda_compute_capability, rocm_compute_capability,
                          hlo_module_config, llvm_module.get());
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
        Thunk::Kind kinds[] = {Thunk::kKernel, Thunk::kCopy,
                               Thunk::kMemset32BitValue, Thunk::kMemzero};
        auto equal = [&](Thunk::Kind kind) { return thunk->kind() == kind; };
        return llvm::any_of(kinds, equal);
      })) {
    return MakeError("Expected only kernel, copy, memset, and memzero thunks");
  }

#if TENSORFLOW_USE_ROCM
  auto libdevice_dir = tensorflow::RocdlRoot();
  xla::gpu::GpuVersion gpu_version{rocm_compute_capability};
  auto hsaco = xla::gpu::amdgpu::CompileToHsaco(
      llvm_module.get(), gpu_version, hlo_module_config, libdevice_dir);
  if (!hsaco.ok()) return MakeError(hsaco.status());
  StatusOr<std::string> ptx(std::string(hsaco->begin(), hsaco->end()));
#else
  auto libdevice_dir = xla::gpu::GetLibdeviceDir(hlo_module_config);
  auto ptx =
      xla::gpu::nvptx::CompileToPtx(llvm_module.get(), cuda_compute_capability,
                                    hlo_module_config, libdevice_dir);
  if (!ptx.ok()) return MakeError(ptx.status());
#endif

  return RewriteData{op,
                     std::move(arguments),
                     std::move(*allocations),
                     std::move(thunks),
                     std::move(constants),
                     std::move(*ptx)};
}

// Replaces op with gpu.launch_func and gpu.memcpy ops.
static void Rewrite(Operation* op, mlir::PatternRewriter& rewriter,
                    mlir::SymbolTable& symbol_table, ArrayRef<Value> arguments,
                    ThunkSequence* thunks, ArrayRef<ConstantInfo> constants,
                    mlir::StringRef gpu_module_data) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto loc = op->getLoc();

  rewriter.setInsertionPoint(op->getParentOfType<mlir::func::FuncOp>());
  auto gpu_module = rewriter.create<mlir::gpu::GPUModuleOp>(loc, "gpu_module");
  symbol_table.insert(gpu_module);
  gpu_module->setAttr(tfrt::gpu::GetGpuBinaryAttrName(),
                      rewriter.getStringAttr(gpu_module_data));

  // Annotate memref.global ops with the gpu.module symbol, and annotate the
  // gpu.module op with memref.global symbols which require initialization.
  SmallVector<mlir::Attribute, 4> const_attrs;
  for (const auto& constant : constants) {
    auto global_op = mlir::SymbolTable::lookupNearestSymbolFrom(
        op, rewriter.getStringAttr(constant.symbol_name));
    if (!global_op) {
      LOG(WARNING) << "memref.global op not found for constant. Possibly "
                   << "unused (spurious) constant.";
      continue;
    }
    global_op->setAttr(tfrt::gpu::GetGpuModuleAttrName(),
                       mlir::SymbolRefAttr::get(gpu_module));
    if (!constant.content.empty())
      const_attrs.emplace_back(mlir::SymbolRefAttr::get(global_op));
  }
  if (!const_attrs.empty()) {
    gpu_module->setAttr(tfrt::gpu::GetGpuConstantsAttrName(),
                        rewriter.getArrayAttr(const_attrs));
  }

  for (const auto& thunk : *thunks) {
    if (thunk->kind() == Thunk::kCopy) {
      const auto* copy_thunk =
          static_cast<const DeviceToDeviceCopyThunk*>(thunk.get());
      auto get_argument = [&](const xla::BufferAllocation::Slice& slice) {
        assert(slice.offset() == 0 && slice.size() == copy_thunk->size_bytes());
        return arguments[slice.index()];
      };
      rewriter.setInsertionPoint(op);
      rewriter.create<mlir::gpu::MemcpyOp>(
          loc, mlir::TypeRange(), mlir::ValueRange(),
          get_argument(copy_thunk->destination()),
          get_argument(copy_thunk->source()));
      continue;
    }

    auto rewrite_memset = [&](const xla::BufferAllocation::Slice& slice,
                              uint32_t memset_value) {
      assert(slice.offset() == 0);
      Value buffer_arg = arguments[slice.index()];
      auto element_type =
          buffer_arg.getType().cast<mlir::MemRefType>().getElementType();
      rewriter.setInsertionPoint(op);
      Value value =
          MakeBitPatternConstant(rewriter, loc, element_type, memset_value);
      rewriter.create<mlir::gpu::MemsetOp>(
          loc, mlir::TypeRange(), mlir::ValueRange(), buffer_arg, value);
    };

    if (thunk->kind() == Thunk::kMemset32BitValue) {
      const auto* memset_thunk =
          static_cast<const xla::gpu::Memset32BitValueThunk*>(thunk.get());
      rewrite_memset(memset_thunk->destination(), memset_thunk->value());
      continue;
    }
    if (thunk->kind() == Thunk::kMemzero) {
      const auto* memzero_thunk =
          static_cast<const xla::gpu::MemzeroThunk*>(thunk.get());
      rewrite_memset(memzero_thunk->destination(), 0);
      continue;
    }

    const auto* kernel_thunk = static_cast<const KernelThunk*>(thunk.get());
    rewriter.setInsertionPointToStart(gpu_module.getBody());
    SmallVector<Value, 4> kernel_args;
    for (auto kernel_arg : kernel_thunk->arguments())
      kernel_args.push_back(arguments[kernel_arg->index()]);
    auto func_type = rewriter.getType<mlir::FunctionType>(
        mlir::TypeRange(mlir::ValueRange(kernel_args)), mlir::TypeRange());
    mlir::gpu::GPUFuncOp kernel_func = rewriter.create<mlir::gpu::GPUFuncOp>(
        loc, kernel_thunk->kernel_name(), func_type);
    kernel_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                         rewriter.getUnitAttr());
    rewriter.setInsertionPointToEnd(&kernel_func.getBody().back());
    rewriter.create<mlir::gpu::ReturnOp>(loc);

    rewriter.setInsertionPoint(op);
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
        /*shared_memory_size_bytes=*/nullptr, kernel_args);
  }

  rewriter.eraseOp(op);
}

// An overload set for defining predicates for operations that should
// conditionally go through the XLA GPU code emitters.
template <typename OpTy>
static bool HasGpuEmitter(OpTy) {
  return true;
}

// Select custom calls that have corresponding GPU emitters.
static bool HasGpuEmitter(mlir::lmhlo::CustomCallOp custom_call) {
  llvm::StringRef target = custom_call.getCallTargetName();
  return target == "SliceToDynamic" || target == "PadToStatic";
}

mlir::LogicalResult KernelOpsPattern::matchAndRewrite(
    mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const {
  SmallVector<RewriteData, 4> rewrites;

  // Get data to rewrite kernel ops without changing the IR.
  auto walk = [&](auto op_type_tag) {
    using OpTy = decltype(op_type_tag);

    return module_op.walk([&](OpTy op) -> mlir::WalkResult {
      if (!HasGpuEmitter(op)) return mlir::success();

      auto data = Match(op);
      if (auto err = data.takeError())
        return rewriter.notifyMatchFailure(op, toString(std::move(err)));

      rewrites.emplace_back(std::move(*data));
      return mlir::success();
    });
  };

  // Compile all operations that have GPU code emitters to the GPU binary,
  if (walk(mlir::lmhlo::FusionOp()).wasInterrupted() ||
      walk(mlir::lmhlo::RngGetAndUpdateStateOp()).wasInterrupted() ||
      walk(mlir::lmhlo::ScatterOp()).wasInterrupted() ||
      walk(mlir::lmhlo::SelectAndScatterOp()).wasInterrupted() ||
      walk(mlir::lmhlo::SortOp()).wasInterrupted() ||
      walk(mlir::lmhlo::CustomCallOp()).wasInterrupted() ||
      walk(mlir::gpu::LaunchFuncOp()).wasInterrupted())
    return mlir::failure();

  if (rewrites.empty()) {
    return rewriter.notifyMatchFailure(module_op, "No kernel ops");
  }

  // Mark module as gpu.container_module.
  rewriter.updateRootInPlace(module_op, [&] {
    module_op->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                       rewriter.getUnitAttr());
  });

  // Replace the kernel ops with gpu.launch_func.
  mlir::SymbolTable symbol_table(module_op);
  for (const auto& data : rewrites) {
    Rewrite(data.op, rewriter, symbol_table, data.arguments, data.thunks.get(),
            data.constants, data.gpu_module_data);
  }

  return mlir::success();
}

void populateKernelOpsPattern(mlir::RewritePatternSet& patterns) {
  patterns.add<KernelOpsPattern>(patterns.getContext());
}

}  // namespace tensorflow
