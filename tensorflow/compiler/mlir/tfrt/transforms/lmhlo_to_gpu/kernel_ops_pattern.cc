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
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
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
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu_binary.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"

namespace tensorflow {

using mlir::FloatType;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Value;
using mlir::arith::ConstantFloatOp;
using mlir::arith::ConstantIntOp;
using mlir::arith::ConstantOp;
using xla::gpu::ConditionalThunk;
using xla::gpu::DeviceToDeviceCopyThunk;
using xla::gpu::KernelThunk;
using xla::gpu::Memset32BitValueThunk;
using xla::gpu::MemzeroThunk;
using xla::gpu::SequentialThunk;
using xla::gpu::Thunk;
using xla::gpu::ThunkSequence;
using xla::gpu::WhileThunk;

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
  KernelOpsPattern(mlir::MLIRContext* context, ThunkSequence* thunk_sequence)
      : mlir::OpRewritePattern<mlir::ModuleOp>(context),
        thunk_sequence(thunk_sequence) {}

  using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const override;

  ThunkSequence* thunk_sequence;
};

}  // namespace

static llvm::Error MakeError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
static llvm::Error MakeError(xla::Status status) {
  return MakeError(status.error_message());
}

static void ExtractThunksForOp(mlir::Operation* op,
                               ThunkSequence& thunk_sequence,
                               ThunkSequence* thunks_for_op) {
  for (std::unique_ptr<Thunk>& thunk : thunk_sequence) {
    if (thunk == nullptr) {
      // This thunk has already been std::move()'ed out of the ThunkSequence
      // (see below). Do nothing.
    } else if (thunk->kind() == Thunk::kWhile) {
      // Search for thunks for the op in while loop.
      auto* while_thunk = static_cast<WhileThunk*>(thunk.get());
      ExtractThunksForOp(op, while_thunk->condition_thunk_sequence()->thunks(),
                         thunks_for_op);
      ExtractThunksForOp(op, while_thunk->body_thunk_sequence()->thunks(),
                         thunks_for_op);
    } else if (thunk->kind() == Thunk::kConditional) {
      // Search for thunks for the op in conditional branches.
      auto* cond_thunk = static_cast<ConditionalThunk*>(thunk.get());
      for (const std::unique_ptr<SequentialThunk>& branch_thunks :
           cond_thunk->branch_thunks()) {
        ExtractThunksForOp(op, branch_thunks->thunks(), thunks_for_op);
      }
    } else if (thunk->op() == op) {
      // Found a thunk for the op.
      thunks_for_op->push_back(std::move(thunk));
    } else {
      // Thunk is not relevant to the op. Do nothing.
    }
  }
}

// Returns the data to rewrite op without changing the IR.
static llvm::Expected<std::unique_ptr<ThunkSequence>> Match(
    Operation* op, ThunkSequence& thunk_sequence) {
  auto thunks_for_op = std::make_unique<ThunkSequence>();
  ExtractThunksForOp(op, thunk_sequence, thunks_for_op.get());

  if (!llvm::all_of(*thunks_for_op, [](const auto& thunk) {
        Thunk::Kind kinds[] = {Thunk::kKernel, Thunk::kCopy,
                               Thunk::kMemset32BitValue, Thunk::kMemzero,
                               Thunk::kSequential};
        auto equal = [&](Thunk::Kind kind) { return thunk->kind() == kind; };
        return llvm::any_of(kinds, equal);
      })) {
    return MakeError(
        "Expected only kernel, copy, memset, memzero, and sequential thunks");
  }

  return std::move(thunks_for_op);
}

static void LowerThunkToGpuOp(Operation* op, mlir::PatternRewriter& rewriter,
                              mlir::gpu::GPUModuleOp gpu_module, Thunk* thunk);

// Replaces op with gpu.launch_func, gpu.memcpy, gpu.memset ops.
static void Rewrite(Operation* op, mlir::PatternRewriter& rewriter,
                    mlir::SymbolTable& symbol_table, ThunkSequence* thunks) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto loc = op->getLoc();

  rewriter.setInsertionPoint(op->getParentOfType<mlir::func::FuncOp>());
  auto gpu_module = rewriter.create<mlir::gpu::GPUModuleOp>(loc, "gpu_module");
  symbol_table.insert(gpu_module);

  for (const std::unique_ptr<Thunk>& thunk : *thunks) {
    LowerThunkToGpuOp(op, rewriter, gpu_module, thunk.get());
  }

  rewriter.eraseOp(op);
}

static void LowerThunkToGpuOp(Operation* op, mlir::PatternRewriter& rewriter,
                              mlir::gpu::GPUModuleOp gpu_module, Thunk* thunk) {
  auto loc = op->getLoc();

  if (thunk->kind() == Thunk::kSequential) {
    const auto* seq_thunk = static_cast<const SequentialThunk*>(thunk);
    for (const std::unique_ptr<Thunk>& thunk : seq_thunk->thunks()) {
      LowerThunkToGpuOp(op, rewriter, gpu_module, thunk.get());
    }
    return;
  }

  if (thunk->kind() == Thunk::kCopy) {
    const auto* copy_thunk = static_cast<const DeviceToDeviceCopyThunk*>(thunk);
    rewriter.setInsertionPoint(op);
    rewriter.create<mlir::gpu::MemcpyOp>(
        loc, mlir::TypeRange(), mlir::ValueRange(),
        copy_thunk->destination_value(), copy_thunk->source_value());
    return;
  }

  auto rewrite_memset = [&](const xla::BufferAllocation::Slice& slice,
                            uint32_t memset_value, Value buffer_arg) {
    assert(slice.offset() == 0);
    auto element_type =
        buffer_arg.getType().cast<mlir::MemRefType>().getElementType();
    rewriter.setInsertionPoint(op);
    Value value =
        MakeBitPatternConstant(rewriter, loc, element_type, memset_value);
    rewriter.create<mlir::gpu::MemsetOp>(loc, mlir::TypeRange(),
                                         mlir::ValueRange(), buffer_arg, value);
  };

  if (thunk->kind() == Thunk::kMemset32BitValue) {
    const auto* memset_thunk = static_cast<const Memset32BitValueThunk*>(thunk);
    rewrite_memset(memset_thunk->destination(), memset_thunk->value(),
                   memset_thunk->dest_value());
    return;
  }
  if (thunk->kind() == Thunk::kMemzero) {
    const auto* memzero_thunk = static_cast<const MemzeroThunk*>(thunk);
    rewrite_memset(memzero_thunk->destination(), 0,
                   memzero_thunk->dest_value());
    return;
  }

  const auto* kernel_thunk = static_cast<const KernelThunk*>(thunk);
  rewriter.setInsertionPointToStart(gpu_module.getBody());
  SmallVector<Value, 4> kernel_args;
  for (auto kernel_arg : kernel_thunk->values())
    kernel_args.push_back(kernel_arg);
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
    return mlir::gpu::KernelDim3{make_const_idx(dim3.x), make_const_idx(dim3.y),
                                 make_const_idx(dim3.z)};
  };
  const auto& launch_dims = kernel_thunk->launch_dimensions();
  auto grid_size = make_kernel_dim3(launch_dims.block_counts());
  auto block_size = make_kernel_dim3(launch_dims.thread_counts_per_block());

  rewriter.create<mlir::gpu::LaunchFuncOp>(
      loc, kernel_func, grid_size, block_size,
      /*shared_memory_size_bytes=*/nullptr, kernel_args);
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
  if (thunk_sequence == nullptr) {
    // No thunks to lower from. Skip pass.
    return mlir::failure();
  }

  absl::flat_hash_map<Operation*, std::unique_ptr<ThunkSequence>> rewrites;

  // Get data to rewrite kernel ops without changing the IR.
  auto walk = [&](auto op_type_tag) {
    using OpTy = decltype(op_type_tag);

    return module_op.walk([&](OpTy op) -> mlir::WalkResult {
      if (!HasGpuEmitter(op)) return mlir::success();

      auto data = Match(op, *thunk_sequence);
      if (auto err = data.takeError())
        return rewriter.notifyMatchFailure(op, toString(std::move(err)));

      rewrites[op] = std::move(*data);
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
  for (const auto& rewrite : rewrites) {
    Rewrite(rewrite.first, rewriter, symbol_table, rewrite.second.get());
  }
  return mlir::success();
}

void populateKernelOpsPattern(mlir::RewritePatternSet& patterns,
                              ThunkSequence* thunk_sequence) {
  patterns.add<KernelOpsPattern>(patterns.getContext(), thunk_sequence);
}

}  // namespace tensorflow
