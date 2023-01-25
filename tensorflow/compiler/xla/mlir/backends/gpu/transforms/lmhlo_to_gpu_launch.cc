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

#include <array>
#include <iterator>
#include <memory>
#include <numeric>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTLMHLOTOGPULAUNCHPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::GPUDialect;
using mlir::gpu::GPUFuncOp;
using mlir::gpu::GPUModuleOp;
using mlir::gpu::KernelDim3;
using mlir::gpu::LaunchFuncOp;
using mlir::gpu::MemcpyOp;
using mlir::gpu::MemsetOp;
using mlir::gpu::ReturnOp;

class ConvertLmhloToGpuLaunchPass
    : public impl::ConvertLmhloToGpuLaunchPassBase<
          ConvertLmhloToGpuLaunchPass> {
 public:
  explicit ConvertLmhloToGpuLaunchPass(ThunkSequence* thunk_sequence)
      : thunk_sequence_(thunk_sequence) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::gpu::GPUDialect, xla::runtime::RuntimeDialect>();
  }

 private:
  ThunkSequence* thunk_sequence_;
};

// XLA some times (ab)uses custom calls to represent operations for which we do
// not want to define a separate `HloOpcode`. These operations emitted as device
// kernels (similar to fusions), and we detect such custom calls by name, and
// handle them similar to how we handle fusions.
static std::array<std::string_view, 3> kCustomCallIntrinsics = {
    "SliceToDynamic", "PadToStatic", "__triton"};

//===-----------------------------------------------------------------------===/

static Value MakeBitPatternConstant(OpBuilder& b, Location loc, Type type,
                                    uint32_t bit_pattern) {
  mlir::MLIRContext* ctx = type.getContext();

  // For zero bit pattern always memset with a zero value of the same type.
  if (bit_pattern == 0) {
    // Because `arith` dialect doesn't support unsigned constants, we have to
    // create signless constant first, and then use `rt.unsigned_cast` operation
    // to make it unsigned. When lowering to LLVM and function calls, this
    // casting operation will be erased.
    if (type.isUnsignedInteger()) {
      auto signless = IntegerType::get(ctx, type.getIntOrFloatBitWidth());
      auto zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(signless));
      return b.create<runtime::UnsignedCastOp>(loc, type, zero.getResult());
    }

    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(type));
  }

  // In XLA a 1-byte bit pattern copied to fill a 32-byte word when
  // `Memset32BitValueThunk` is constructed, so to get back an `i1` constant we
  // only need to check if any bit is set to `1`.
  if (type.isInteger(1)) {
    return b.create<arith::ConstantOp>(loc, b.getBoolAttr(bit_pattern));
  }

  // Xla IR emitter copies integers of smaller width to fill 32 bits, so we can
  // safely truncate the bit pattern. For integers larger than 32 bits we can
  // construct a wider integer, as Xla guarantees that all 32-bit words are
  // equal.
  if (auto integer = type.dyn_cast<mlir::IntegerType>()) {
    llvm::APInt i32(32, bit_pattern);

    assert(integer.getWidth() <= 64 && "integer value must be <= 64 bits");
    llvm::APInt value = integer.getWidth() <= 32 ? i32.trunc(integer.getWidth())
                                                 : i32.concat(i32);

    // See unsigned-to-signed cast documentation above.
    if (integer.isUnsigned()) {
      auto signless = IntegerType::get(ctx, integer.getWidth());
      auto cst =
          b.create<arith::ConstantOp>(loc, b.getIntegerAttr(signless, value));
      return b.create<runtime::UnsignedCastOp>(loc, type, cst.getResult());
    }

    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(integer, value));
  }

  // Similar to integer type we can safely truncate or concat bit pattern.
  if (auto fp = type.dyn_cast<mlir::FloatType>()) {
    llvm::APInt i32(32, bit_pattern);

    assert(fp.getWidth() <= 64 && "floating point value must be <= 64 bits");
    llvm::APInt ivalue =
        fp.getWidth() <= 32 ? i32.trunc(fp.getWidth()) : i32.concat(i32);

    llvm::APFloat fvalue = [&]() -> llvm::APFloat {
      if (fp.isBF16()) return {llvm::APFloat::BFloat(), ivalue};
      if (fp.isF16()) return {llvm::APFloat::IEEEhalf(), ivalue};
      if (fp.isF32()) return {llvm::APFloat::IEEEsingle(), ivalue};
      if (fp.isF64()) return {llvm::APFloat::IEEEdouble(), ivalue};

      assert(false && "unsupported floating point type");
      return llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
    }();

    return b.create<arith::ConstantFloatOp>(loc, fvalue, fp);
  }

  // Return a constant index value, that will safely fail verification (there is
  // no memset operation for `index` type), so that we do not accidentally crash
  // the binary in optimized builds.
  assert(false && "unsupported memset type");
  return b.create<arith::ConstantIndexOp>(loc, 0);
}

static void ExtractThunksForOp(Operation* op, ThunkSequence& thunk_sequence,
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
static absl::StatusOr<std::unique_ptr<ThunkSequence>> Match(
    Operation* op, ThunkSequence& thunk_sequence) {
  auto thunks_for_op = std::make_unique<ThunkSequence>();
  ExtractThunksForOp(op, thunk_sequence, thunks_for_op.get());

  // Check if we know how to lower a Thunk to Gpu operation(s).
  auto is_supported = [](const std::unique_ptr<Thunk>& thunk) -> bool {
    Thunk::Kind kinds[] = {Thunk::kKernel, Thunk::kCopy,
                           Thunk::kMemset32BitValue, Thunk::kMemzero,
                           Thunk::kSequential};
    return llvm::any_of(
        kinds, [&](Thunk::Kind kind) { return thunk->kind() == kind; });
  };

  if (!llvm::all_of(*thunks_for_op, is_supported)) {
    return absl::InternalError("Unsupported Thunk kind");
  }

  return std::move(thunks_for_op);
}

static void LowerThunkToGpuOp(Operation* op, OpBuilder& b,
                              GPUModuleOp gpu_module, Thunk* thunk);

// Replaces op with gpu.launch_func, gpu.memcpy, gpu.memset ops.
static void Rewrite(Operation* op, OpBuilder& b, SymbolTable& symbol_table,
                    ThunkSequence* thunks) {
  OpBuilder::InsertionGuard guard(b);
  auto loc = op->getLoc();

  b.setInsertionPoint(op->getParentOfType<func::FuncOp>());
  auto gpu_module = b.create<GPUModuleOp>(loc, "gpu_module");
  symbol_table.insert(gpu_module);

  for (const std::unique_ptr<Thunk>& thunk : *thunks) {
    LowerThunkToGpuOp(op, b, gpu_module, thunk.get());
  }

  op->erase();
}

static void LowerThunkToGpuOp(Operation* op, OpBuilder& b,
                              GPUModuleOp gpu_module, Thunk* thunk) {
  auto loc = op->getLoc();

  if (thunk->kind() == Thunk::kSequential) {
    const auto* seq_thunk = static_cast<const SequentialThunk*>(thunk);
    for (const std::unique_ptr<Thunk>& thunk : seq_thunk->thunks()) {
      LowerThunkToGpuOp(op, b, gpu_module, thunk.get());
    }
    return;
  }

  if (thunk->kind() == Thunk::kCopy) {
    const auto* copy_thunk = static_cast<const DeviceToDeviceCopyThunk*>(thunk);
    b.setInsertionPoint(op);
    b.create<MemcpyOp>(loc, TypeRange(), ValueRange(),
                       copy_thunk->destination_value(),
                       copy_thunk->source_value());
    return;
  }

  auto rewrite_memset = [&](const xla::BufferAllocation::Slice& slice,
                            uint32_t memset_value, Value buffer_arg) {
    auto element_type =
        buffer_arg.getType().cast<MemRefType>().getElementType();
    b.setInsertionPoint(op);
    Value value = MakeBitPatternConstant(b, loc, element_type, memset_value);
    b.create<MemsetOp>(loc, TypeRange(), ValueRange(), buffer_arg, value);
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
  b.setInsertionPointToStart(gpu_module.getBody());

  SmallVector<Value> kernel_args;
  for (auto kernel_arg : kernel_thunk->values())
    kernel_args.push_back(kernel_arg);

  auto func_type =
      b.getType<FunctionType>(TypeRange(ValueRange(kernel_args)), TypeRange());

  gpu::GPUFuncOp kernel_func =
      b.create<gpu::GPUFuncOp>(loc, kernel_thunk->kernel_name(), func_type);
  kernel_func->setAttr(GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
  b.setInsertionPointToEnd(&kernel_func.getBody().back());
  b.create<ReturnOp>(loc);

  auto make_const_idx = [&](int64_t value) {
    auto attr = b.getIndexAttr(value);
    return b.create<arith::ConstantOp>(loc, attr).getResult();
  };

  auto make_kernel_dim3 = [&](const auto& dim3) {
    return KernelDim3{make_const_idx(dim3.x), make_const_idx(dim3.y),
                      make_const_idx(dim3.z)};
  };

  const auto& launch_dims = kernel_thunk->launch_dimensions();

  b.setInsertionPoint(op);
  auto grid_size = make_kernel_dim3(launch_dims.block_counts());
  auto block_size = make_kernel_dim3(launch_dims.thread_counts_per_block());
  auto shmem_size = b.create<arith::ConstantOp>(
      loc,
      b.getI32IntegerAttr(kernel_thunk->launch_dimensions().SharedMemBytes()));

  b.create<LaunchFuncOp>(loc, kernel_func, grid_size, block_size, shmem_size,
                         kernel_args);
}

// An overload set for defining predicates for operations that should
// conditionally go through the XLA GPU code emitters.
template <typename OpTy>
static bool HasGpuEmitter(OpTy) {
  return true;
}

// Select custom calls that have corresponding GPU emitters.
static bool HasGpuEmitter(lmhlo::CustomCallOp custom_call) {
  return llvm::any_of(kCustomCallIntrinsics, [&](std::string_view name) {
    return custom_call.getCallTargetName().equals(name);
  });
}

//===-----------------------------------------------------------------------===/

void ConvertLmhloToGpuLaunchPass::runOnOperation() {
  ModuleOp module = getOperation();

  // No thunks to lower from. Skip pass.
  if (thunk_sequence_ == nullptr) return signalPassFailure();

  // Collect thunks for rewriting each compatible operation in the module into
  // the sequence of device kernel launches. Some operation might have an empty
  // thunk sequence (e.g. redundant copy operation that does not require running
  // anything on device).
  absl::flat_hash_map<Operation*, std::unique_ptr<ThunkSequence>> rewrites;

  // Get data to rewrite kernel ops without changing the IR.
  auto walk = [&](auto op_type_tag) {
    return module.walk([&](decltype(op_type_tag) op) -> WalkResult {
      if (!HasGpuEmitter(op)) return success();

      auto data = Match(op, *thunk_sequence_);
      if (!data.ok()) return op.emitOpError(data.status().message());

      rewrites[op] = std::move(*data);
      return success();
    });
  };

  // Collect all operations that have GPU code emitters.
  if (walk(lmhlo::FusionOp()).wasInterrupted() ||
      walk(lmhlo::RngGetAndUpdateStateOp()).wasInterrupted() ||
      walk(lmhlo::ScatterOp()).wasInterrupted() ||
      walk(lmhlo::SelectAndScatterOp()).wasInterrupted() ||
      walk(lmhlo::SortOp()).wasInterrupted() ||
      walk(lmhlo::CustomCallOp()).wasInterrupted() ||
      walk(LaunchFuncOp()).wasInterrupted())
    return signalPassFailure();

  // No operations that should be lowered to sequence of device launches.
  if (rewrites.empty()) return;

  OpBuilder b(module);
  SymbolTable symbol_table(module);

  // Replace matched operations with gpu.launch_func's.
  for (const auto& [op, thunks] : rewrites) {
    Rewrite(op, b, symbol_table, thunks.get());
  }

  // Mark module as gpu.container_module.
  module->setAttr(GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToGpuLaunchPass(ThunkSequence* thunk_sequence) {
  return std::make_unique<ConvertLmhloToGpuLaunchPass>(thunk_sequence);
}

}  // namespace gpu
}  // namespace xla
