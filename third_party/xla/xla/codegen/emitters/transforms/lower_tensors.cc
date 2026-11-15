/* Copyright 2024 The OpenXLA Authors.

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
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/util.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

using llvm::dyn_cast_or_null;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpResult;
using mlir::OpRewritePattern;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace ml = ::mlir::LLVM;
namespace vector = ::mlir::vector;

constexpr int kGPUSharedMemoryAddrSpace = 3;

Value GetPtr(Value value) {
  if (!mlir::isa<mlir::RankedTensorType>(value.getType())) {
    return nullptr;
  }
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() == 1 && cast.getNumResults() == 1 &&
        mlir::isa<ml::LLVMPointerType>(cast.getOperand(0).getType())) {
      return cast.getOperand(0);
    }
  }
  return nullptr;
}

// Unwraps a single step of an SSA operation that updates or threads an
// existing tensor buffer in place, returning the underlying tensor operand or a
// null `Value` if `def` is not a pass-through buffer operation.
//
// Shared by `TraceDestinationBuffer` (destination-passing pointer resolution
// for stores/atomics) and `GetBaseBufferAlignment` (base-pointer alignment
// provenance for reads and writes). Control-flow merges (`scf.if`) and terminal
// allocations (`xla_gpu.allocate_shared`, `llvm.mlir.addressof`) are handled
// separately by each caller because writes require identical destination
// buffers across branches whereas reads may select between distinct buffers
// with different alignments.
Value UnwrapPassThroughBufferOp(Operation* def, unsigned result_number) {
  if (auto insert = mlir::dyn_cast<mlir::tensor::InsertOp>(def)) {
    return insert.getDest();
  }
  if (auto copy_slice = mlir::dyn_cast<CopySliceOp>(def)) {
    return copy_slice.getDest();
  }
  if (auto set_dim = mlir::dyn_cast<SetDynamicDimSizeOp>(def)) {
    return set_dim.getTensor();
  }
  if (auto atomic_store = mlir::dyn_cast<AtomicStoreOp>(def)) {
    return atomic_store.getDest();
  }
  if (auto atomic_rmw = mlir::dyn_cast<AtomicRMWOp>(def)) {
    return atomic_rmw.getInput();
  }
  if (auto transfer_write = mlir::dyn_cast<vector::TransferWriteOp>(def)) {
    return transfer_write.getBase();
  }
  if (auto sync_threads = mlir::dyn_cast<gpu::SyncThreadsOp>(def)) {
    return sync_threads.getOperand(result_number);
  }
  if (auto scf_for = mlir::dyn_cast<scf::ForOp>(def)) {
    return scf_for.getInitArgs()[result_number];
  }
  return nullptr;
}

}  // namespace

Value TraceDestinationBuffer(Value dest, bool emit_error) {
  llvm::SmallPtrSet<Value, 8> visited;
  while (dest) {
    if (!visited.insert(dest).second) {
      return nullptr;
    }
    if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(dest)) {
      if (auto scf_for = mlir::dyn_cast_or_null<scf::ForOp>(
              block_arg.getOwner()->getParentOp())) {
        if (block_arg.getArgNumber() >= scf_for.getNumInductionVars()) {
          unsigned idx =
              block_arg.getArgNumber() - scf_for.getNumInductionVars();
          dest = scf_for.getInitArgs()[idx];
          continue;
        }
      }
      break;
    }
    Operation* def = dest.getDefiningOp();
    if (!def) {
      break;
    }
    unsigned result_number = mlir::cast<OpResult>(dest).getResultNumber();
    if (Value unwrapped = UnwrapPassThroughBufferOp(def, result_number)) {
      dest = unwrapped;
    } else if (auto scf_if = mlir::dyn_cast<scf::IfOp>(def)) {
      // Pick one of the branches, they're required to yield the same buffers.
      dest = scf_if.getThenRegion().front().getTerminator()->getOperand(
          result_number);
    } else if (auto scf_switch = mlir::dyn_cast<scf::IndexSwitchOp>(def)) {
      dest = scf_switch.getDefaultBlock().getTerminator()->getOperand(
          result_number);
    } else if (auto cast = mlir::dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getNumOperands() == 1 &&
          mlir::isa<mlir::RankedTensorType>(cast.getOperand(0).getType())) {
        dest = cast.getOperand(0);
      } else {
        break;
      }
    } else if (mlir::isa<gpu::AllocateSharedOp>(def)) {
      break;
    } else {
      if (emit_error) {
        def->emitOpError("unsupported dest type");
      }
      return nullptr;
    }
  }
  return dest;
}

namespace {

Value GetDestinationBuffer(Value dest) {
  return TraceDestinationBuffer(dest, /*emit_error=*/true);
}

bool IsSharedMemoryBuffer(Value dest) {
  Value root = TraceDestinationBuffer(dest);
  if (!root) {
    return false;
  }
  if (root.getDefiningOp<gpu::AllocateSharedOp>() != nullptr) {
    return true;
  }
  Value ptr = GetPtr(root);
  llvm::SmallPtrSet<Value, 8> visited_ptrs;
  while (ptr) {
    if (!visited_ptrs.insert(ptr).second) {
      break;
    }
    auto ptr_ty = mlir::dyn_cast<ml::LLVMPointerType>(ptr.getType());
    if (ptr_ty && ptr_ty.getAddressSpace() == kGPUSharedMemoryAddrSpace) {
      return true;
    }
    if (auto cast = ptr.getDefiningOp<ml::AddrSpaceCastOp>()) {
      ptr = cast.getArg();
      continue;
    }
    if (auto gep = ptr.getDefiningOp<ml::GEPOp>()) {
      ptr = gep.getBase();
      continue;
    }
    break;
  }
  return false;
}

template <typename Op>
bool IsSupportedTransfer(Op op) {
  return !absl::c_linear_search(op.getInBoundsValues(), false) &&
         op.getVectorType().getRank() == 1 && !op.getMask() &&
         op.getPermutationMap().isMinorIdentity();
}

class RewriteFunctionSignatures : public OpRewritePattern<mlir::func::FuncOp> {
 public:
  RewriteFunctionSignatures(mlir::MLIRContext* context,
                            const DeviceSpec& device_spec)
      : OpRewritePattern<mlir::func::FuncOp>(context),
        device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto is_tensor = [](Type ty) {
      return mlir::isa<mlir::RankedTensorType>(ty);
    };
    if (!llvm::any_of(op.getFunctionType().getInputs(), is_tensor)) {
      return rewriter.notifyMatchFailure(op,
                                         "the function has no input tensors");
    }

    bool some_tensor_result =
        llvm::any_of(op.getFunctionType().getResults(), is_tensor);

    if (!device_spec_.IsCpu()) {
      bool all_tensor_results =
          llvm::all_of(op.getFunctionType().getResults(), is_tensor);
      if (some_tensor_result && !all_tensor_results) {
        op->emitOpError("function has a mix of tensor and non-tensor results");
        return failure();
      }
    }

    TypeRange new_results = op.getFunctionType().getResults();
    if (some_tensor_result) {
      new_results = {};
      auto terminator = op.getFunctionBody().front().getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);
    }

    SmallVector<Type> new_operands(op.getFunctionType().getInputs());
    for (auto&& [index, operand] : llvm::enumerate(new_operands)) {
      if (is_tensor(operand)) {
        rewriter.setInsertionPointToStart(&op.getBody().front());
        auto cast = UnrealizedConversionCastOp::create(
            rewriter, op.getLoc(), operand, op.getArgument(index));
        op.getArgument(index).replaceAllUsesExcept(cast.getResult(0), cast);
        operand = ml::LLVMPointerType::get(op.getContext());
        // The kernel does not write the buffer, but other kernels may write it
        // between launches, so its loads are not !invariant.load. Backends
        // derive what they need (e.g. ld.global.nc) from readonly + noalias.
        if (op.getArgAttr(index, kXlaInvariantAttr)) {
          op.setArgAttr(index, ml::LLVMDialect::getReadonlyAttrName(),
                        rewriter.getUnitAttr());
        }
      }
    }

    op.setFunctionType(rewriter.getFunctionType(new_operands, new_results));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }

    return success();
  }

 private:
  const DeviceSpec& device_spec_;
};

struct RewriteFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::ForOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallBitVector inits_to_remove(op.getNumRegionIterArgs(), false);
    SmallVector<Value> new_inits;
    new_inits.reserve(op.getNumResults());
    SmallVector<Value> ptrs;
    ptrs.reserve(op.getNumRegionIterArgs());
    for (auto [index, init] : llvm::enumerate(op.getInitArgs())) {
      Value ptr = GetPtr(init);
      if (ptr) {
        ptrs.push_back(ptr);
        inits_to_remove.set(index);
        continue;
      }
      new_inits.push_back(init);
    }
    if (inits_to_remove.none()) {
      return rewriter.notifyMatchFailure(op, "no args to remove");
    }
    // Create new ForOp with updated init args. The empty body builder is needed
    // to avoid implicit construction of scf.yield in the body block.
    Location loc = op.getLoc();
    auto new_for_op = scf::ForOp::create(
        rewriter, loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        new_inits, [](OpBuilder&, Location, Value, ValueRange) {});
    new_for_op->setAttrs(op->getAttrs());

    // Collect a mapping for block arguments and results. If the init is
    // removed, we can use the init of the original scf.for for replacement,
    // since it was provided by the `builtin.unrealized_conversion_cast` cast to
    // the correct type.
    mlir::Block* new_body = new_for_op.getBody();
    mlir::Block* old_body = op.getBody();
    rewriter.setInsertionPoint(new_body, new_body->begin());

    SmallVector<Value, 4> bb_args_mapping;
    bb_args_mapping.reserve(old_body->getNumArguments());
    bb_args_mapping.push_back(new_for_op.getInductionVar());
    SmallVector<Value, 4> results_replacement;
    results_replacement.reserve(old_body->getNumArguments());
    int num_removed_args = 0;
    for (auto [index, arg] : llvm::enumerate(op.getRegionIterArgs())) {
      if (!inits_to_remove.test(index)) {
        bb_args_mapping.push_back(
            new_for_op.getRegionIterArg(index - num_removed_args));
        results_replacement.push_back(
            new_for_op.getResult(index - num_removed_args));
        continue;
      }
      bb_args_mapping.push_back(op.getInitArgs()[index]);
      results_replacement.push_back(op.getInitArgs()[index]);
      ++num_removed_args;
    }

    // Move the body of the old ForOp to the new one.
    rewriter.mergeBlocks(old_body, new_body, bb_args_mapping);

    // Update the terminator.
    auto new_terminator = mlir::cast<scf::YieldOp>(new_body->getTerminator());
    SmallVector<Value> new_yielded_values;
    new_yielded_values.reserve(new_terminator->getNumOperands());
    rewriter.setInsertionPoint(new_terminator);
    for (auto [index, yielded_value] :
         llvm::enumerate(new_terminator.getResults())) {
      if (inits_to_remove.test(index)) {
        continue;
      }
      new_yielded_values.push_back(yielded_value);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(new_terminator,
                                              new_yielded_values);

    // Replace the op.
    rewriter.replaceOp(op, results_replacement);
    return mlir::success();
  }
};

Value GetLinearIndex(ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  CHECK_LE(indices.size(), 1) << "Only 0D and 1D tensors are supported";
  auto index = indices.empty() ? mlir::arith::ConstantIndexOp::create(b, 0)
                               : indices.front();
  auto index_ty = b.getIntegerType(
      mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
          .getTypeSizeInBits(index.getType()));
  return mlir::arith::IndexCastUIOp::create(b, index_ty, index);
}

// If the provided type is a sub-byte type get its bit-width, else nullopt.
std::optional<int> GetSubByteBitWidth(Type element_type) {
  if (element_type.isIntOrFloat()) {
    int bit_width = element_type.getIntOrFloatBitWidth();
    if (bit_width == 4 || bit_width == 2) {
      return bit_width;
    }
  }
  return std::nullopt;
}

constexpr int kBitsPerByte = 8;
constexpr int kMaxPackedTransferBits = 128;
constexpr int kMaxProvenanceDepth = 4096;
constexpr int kMaxBranchDepth = 8;
constexpr int kMaxIndexDivisorDepth = 32;
constexpr unsigned kMaxSafePowerOfTwoShift = 62;
constexpr uint64_t kMaxLLVMAlignment = uint64_t{1} << 30;
constexpr uint64_t kSharedMemSubWordAlignmentBytes = 4;

// Extracts the `llvm.align` byte alignment when `block_arg` is an entry-block
// parameter of a `func.func`. Non-entry CFG block arguments are rejected
// because their values arrive along arbitrary control-flow edges rather than
// directly from the function signature.
std::optional<uint64_t> GetFuncEntryArgAlignment(
    mlir::BlockArgument block_arg) {
  auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      block_arg.getOwner()->getParentOp());
  if (!func || !block_arg.getOwner()->isEntryBlock()) {
    return std::nullopt;
  }
  auto align_attr = func.getArgAttrOfType<mlir::IntegerAttr>(
      block_arg.getArgNumber(), ml::LLVMDialect::getAlignAttrName());
  if (!align_attr) {
    return std::nullopt;
  }
  int64_t align = align_attr.getValue().getSExtValue();
  if (align <= 0) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(align);
}

// Returns the base byte alignment guaranteed for an `xla_gpu.allocate_shared`
// buffer before it is lowered to `llvm.mlir.global`. Elements narrower than 32
// bits are word-aligned (4 bytes) by `RewriteAllocateShared` so sub-word atomic
// CAS `ptrmask` operations remain within the allocation; wider types use their
// target DataLayout ABI alignment.
std::optional<uint64_t> GetSharedMemoryAlignment(Type element_type,
                                                 Operation* op) {
  if (element_type.isIntOrFloat() &&
      element_type.getIntOrFloatBitWidth() < 32) {
    return kSharedMemSubWordAlignmentBytes;
  }
  mlir::LLVMTypeConverter converter(op->getContext());
  Type llvm_elem_type = converter.convertType(element_type);
  if (!llvm_elem_type) {
    return std::nullopt;
  }
  return mlir::DataLayout::closest(op).getTypeABIAlignment(llvm_elem_type);
}

// Resolves the byte alignment of an `llvm.mlir.addressof` symbol reference.
// Because `RewriteAllocateShared` and `RewriteNonScalarConstants` run in the
// same greedy pattern rewrite as vector transfer lowering, shared-memory and
// constant tensors may already be lowered to `llvm.mlir.global` symbols when
// `GetBaseBufferAlignment` inspects them. Looking up the `GlobalOp` directly
// preserves explicit `alignment` attributes (e.g., 4, 8, or 16 bytes) and falls
// back to the global type's ABI alignment when no explicit attribute is set.
std::optional<uint64_t> GetGlobalAlignment(ml::AddressOfOp addr_of,
                                           Operation* op) {
  auto module = addr_of->getParentOfType<mlir::ModuleOp>();
  auto global = module
                    ? module.lookupSymbol<ml::GlobalOp>(addr_of.getGlobalName())
                    : ml::GlobalOp{};
  if (!global) {
    return std::nullopt;
  }
  if (std::optional<uint64_t> align = global.getAlignment();
      align && *align > 0) {
    return *align;
  }
  return mlir::DataLayout::closest(op).getTypeABIAlignment(
      global.getGlobalType());
}

// Traces a tensor SSA value back through in-place updates, casts, loops, and
// branches to its underlying allocation or kernel argument to determine its
// guaranteed base-pointer byte alignment. Returns `std::nullopt` if provenance
// reaches an unknown source or exceeds traversal bounds.
std::optional<uint64_t> GetBaseBufferAlignment(Value base, Operation* op,
                                               int branch_depth = 0) {
  for (int depth = 0; base && depth < kMaxProvenanceDepth; ++depth) {
    if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(base)) {
      if (std::optional<uint64_t> align = GetFuncEntryArgAlignment(block_arg)) {
        return align;
      }
      // Tensor `iter_args` of `scf.for` carry in-place updates to the same
      // underlying buffer initialized before the loop.
      if (auto scf_for = mlir::dyn_cast_or_null<scf::ForOp>(
              block_arg.getOwner()->getParentOp())) {
        unsigned arg_num = block_arg.getArgNumber();
        unsigned num_ivs = scf_for.getNumInductionVars();
        if (arg_num >= num_ivs) {
          base = scf_for.getInitArgs()[arg_num - num_ivs];
          continue;
        }
      }
      return std::nullopt;
    }

    Operation* def = base.getDefiningOp();
    if (!def) {
      return std::nullopt;
    }
    unsigned result_number = mlir::cast<OpResult>(base).getResultNumber();
    if (Value unwrapped = UnwrapPassThroughBufferOp(def, result_number)) {
      base = unwrapped;
    } else if (auto scf_if = mlir::dyn_cast<scf::IfOp>(def)) {
      // Reads may select between distinct buffers across `scf.if` branches; the
      // merged pointer is aligned to `gcd(then_align, else_align)`.
      if (branch_depth >= kMaxBranchDepth || scf_if.getThenRegion().empty() ||
          scf_if.getElseRegion().empty()) {
        return std::nullopt;
      }
      Value then_val =
          scf_if.getThenRegion().front().getTerminator()->getOperand(
              result_number);
      Value else_val =
          scf_if.getElseRegion().front().getTerminator()->getOperand(
              result_number);
      if (then_val == else_val) {
        base = then_val;
        continue;
      }
      std::optional<uint64_t> then_align =
          GetBaseBufferAlignment(then_val, op, branch_depth + 1);
      std::optional<uint64_t> else_align =
          GetBaseBufferAlignment(else_val, op, branch_depth + 1);
      return (then_align && else_align)
                 ? std::optional<uint64_t>(std::gcd(*then_align, *else_align))
                 : std::nullopt;
    } else if (auto alloc = mlir::dyn_cast<gpu::AllocateSharedOp>(def)) {
      return GetSharedMemoryAlignment(
          alloc.getResult().getType().getElementType(), op);
    } else if (auto addr_of = mlir::dyn_cast<ml::AddressOfOp>(def)) {
      return GetGlobalAlignment(addr_of, op);
    } else if (auto addr_cast = mlir::dyn_cast<ml::AddrSpaceCastOp>(def)) {
      base = addr_cast.getArg();
    } else if (auto cast = mlir::dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getNumOperands() != 1) {
        return std::nullopt;
      }
      base = cast.getOperand(0);
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// Returns `|value|` as `uint64_t` without signed overflow on `INT64_MIN`.
uint64_t UnsignedAbs(int64_t value) {
  return value < 0 ? (uint64_t{0} - static_cast<uint64_t>(value))
                   : static_cast<uint64_t>(value);
}

// Returns the largest power-of-two factor dividing a non-zero `value`.
uint64_t LargestPowerOfTwoDivisor(uint64_t value) {
  DCHECK_GT(value, 0);
  return uint64_t{1} << absl::countr_zero(value);
}

// Multiplies two known divisors `lhs` and `rhs`, where `0` represents a value
// known to be identically zero (divisible by every integer). If `lhs * rhs`
// overflows `uint64_t`, falls back to the largest power-of-two divisor of the
// exact product (`2^min(62, ctz(lhs) + ctz(rhs))`), which remains a sound
// divisor while preserving power-of-two byte alignment.
uint64_t MultiplyDivisors(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    unsigned trailing_zeros =
        std::min<unsigned>(kMaxSafePowerOfTwoShift,
                           absl::countr_zero(lhs) + absl::countr_zero(rhs));
    return uint64_t{1} << trailing_zeros;
  }
  return lhs * rhs;
}

// Propagates known integer divisors through a `SymbolicExpr` AST given known
// divisors of its input variables (`var_divisors`).
//
// Convention: `0` means the expression is identically zero (`std::gcd(a, 0) ==
// a`), and `1` means no non-trivial divisor is guaranteed.
uint64_t GetExprKnownDivisor(SymbolicExpr expr,
                             absl::Span<const uint64_t> var_divisors) {
  if (!expr) {
    return 1;
  }
  switch (expr.GetType()) {
    case SymbolicExprType::kConstant:
      return UnsignedAbs(expr.GetValue());
    case SymbolicExprType::kVariable: {
      int64_t variable_id = expr.GetValue();
      if (variable_id < 0 ||
          static_cast<size_t>(variable_id) >= var_divisors.size()) {
        return 1;
      }
      return var_divisors[variable_id];
    }
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMin:
    case SymbolicExprType::kMax:
      return std::gcd(GetExprKnownDivisor(expr.GetLHS(), var_divisors),
                      GetExprKnownDivisor(expr.GetRHS(), var_divisors));
    case SymbolicExprType::kMod: {
      // `a mod b = a - q * b` for some integer `q`, so any common divisor of
      // `a` and `b` also divides the remainder.
      uint64_t lhs = GetExprKnownDivisor(expr.GetLHS(), var_divisors);
      if (lhs == 0) {
        return 0;
      }
      uint64_t rhs = GetExprKnownDivisor(expr.GetRHS(), var_divisors);
      return rhs == 0 ? 1 : std::gcd(lhs, rhs);
    }
    case SymbolicExprType::kMul:
      return MultiplyDivisors(GetExprKnownDivisor(expr.GetLHS(), var_divisors),
                              GetExprKnownDivisor(expr.GetRHS(), var_divisors));
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv: {
      // When `lhs` is a multiple of `lhs_div` and a positive constant `c`
      // divides `lhs_div` evenly, `lhs = k * (lhs_div / c) * c` has zero
      // remainder upon division by `c`, so `lhs_div / c` divides both
      // `floordiv(lhs, c)` and `ceildiv(lhs, c)`. Otherwise (e.g., `4 floordiv
      // 8 == 0` vs `12 floordiv 8 == 1`), no divisor `> 1` is guaranteed.
      if (expr.GetRHS().GetType() == SymbolicExprType::kConstant &&
          expr.GetRHS().GetValue() > 0) {
        uint64_t divisor = static_cast<uint64_t>(expr.GetRHS().GetValue());
        uint64_t lhs = GetExprKnownDivisor(expr.GetLHS(), var_divisors);
        if (lhs == 0) {
          return 0;
        }
        if (lhs % divisor == 0) {
          return lhs / divisor;
        }
      }
      return 1;
    }
  }
  return 1;
}

// Computes a guaranteed non-negative divisor of a 1D index SSA value (`0` if
// identically zero, `1` if unknown) across `arith` ops, `scf.for` induction
// variables, and `xla.apply_indexing` affine maps.
uint64_t GetIndexKnownDivisor(Value index_value, int depth = 0) {
  if (!index_value || depth > kMaxIndexDivisorDepth) {
    return 1;
  }
  llvm::APInt const_val;
  if (mlir::matchPattern(index_value, mlir::m_ConstantInt(&const_val))) {
    if (const_val.isZero()) {
      return 0;
    }
    llvm::APInt abs_val = const_val.isNegative() ? -const_val : const_val;
    if (abs_val.getActiveBits() <= 64) {
      return abs_val.getZExtValue();
    }
    return uint64_t{1} << std::min<unsigned>(kMaxSafePowerOfTwoShift,
                                             const_val.countr_zero());
  }
  if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(index_value)) {
    // An `scf.for` induction variable takes values `lb + k * step` (`k >= 0`),
    // so `gcd(divisor(lb), divisor(step))` divides every iteration's index.
    if (auto scf_for = mlir::dyn_cast_or_null<scf::ForOp>(
            block_arg.getOwner()->getParentOp());
        scf_for && block_arg == scf_for.getInductionVar()) {
      return std::gcd(GetIndexKnownDivisor(scf_for.getLowerBound(), depth + 1),
                      GetIndexKnownDivisor(scf_for.getStep(), depth + 1));
    }
    return 1;
  }
  Operation* def = index_value.getDefiningOp();
  if (!def) {
    return 1;
  }
  if (auto apply = mlir::dyn_cast<ApplyIndexingOp>(def)) {
    unsigned result_index = mlir::cast<OpResult>(index_value).getResultNumber();
    SmallVector<uint64_t, 8> var_divisors;
    var_divisors.reserve(apply.getNumOperands());
    for (Value operand : apply.getOperands()) {
      var_divisors.push_back(GetIndexKnownDivisor(operand, depth + 1));
    }
    return GetExprKnownDivisor(apply.getSymbolicMap().GetResult(result_index),
                               var_divisors);
  }
  if (auto add = mlir::dyn_cast<arith::AddIOp>(def)) {
    return std::gcd(GetIndexKnownDivisor(add.getLhs(), depth + 1),
                    GetIndexKnownDivisor(add.getRhs(), depth + 1));
  }
  if (auto mul = mlir::dyn_cast<arith::MulIOp>(def)) {
    return MultiplyDivisors(GetIndexKnownDivisor(mul.getLhs(), depth + 1),
                            GetIndexKnownDivisor(mul.getRhs(), depth + 1));
  }
  if (auto shl = mlir::dyn_cast<arith::ShLIOp>(def)) {
    llvm::APInt shift;
    if (mlir::matchPattern(shl.getRhs(), mlir::m_ConstantInt(&shift)) &&
        shift.ult(63)) {
      return MultiplyDivisors(GetIndexKnownDivisor(shl.getLhs(), depth + 1),
                              uint64_t{1} << shift.getZExtValue());
    }
  }
  return 1;
}

// Returns the in-memory storage size in bits of a single tensor element: the
// exact packed bit-width (`2` or `4`) for sub-byte element types, or the target
// DataLayout allocation size in bits for byte-addressable types.
std::optional<uint64_t> GetElementStorageBitWidth(Type element_type,
                                                  Operation* op) {
  if (std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type)) {
    return static_cast<uint64_t>(*sub_byte_width);
  }
  mlir::LLVMTypeConverter converter(op->getContext());
  Type llvm_elem_type = converter.convertType(element_type);
  if (!llvm_elem_type) {
    return std::nullopt;
  }
  llvm::TypeSize alloc_size =
      mlir::DataLayout::closest(op).getTypeAllocSize(llvm_elem_type);
  if (alloc_size.isScalable()) {
    return std::nullopt;
  }
  return alloc_size.getFixedValue() * kBitsPerByte;
}

// Infers the guaranteed power-of-two byte alignment for a 0D/1D vector transfer
// from its base buffer alignment and starting element index offset, or returns
// `std::nullopt` when the base buffer alignment is unknown.
//
// Alignment is computed in bits (`gcd(base_byte_align * 8, elem_div *
// elem_bits)`) before converting back to bytes so packed sub-byte types (e.g.,
// `i4` at an index divisible by `8` -> `32`-bit / `4`-byte offset) and
// multi-byte types (e.g., `i64` at index `2` -> `16`-byte offset) are handled
// uniformly.
std::optional<uint64_t> GetTransferAlignment(Value base, ValueRange indices,
                                             Type element_type, Operation* op) {
  CHECK_LE(indices.size(), 1) << "Only 0D and 1D tensors are supported";
  std::optional<uint64_t> base_align = GetBaseBufferAlignment(base, op);
  if (!base_align || *base_align == 0) {
    return std::nullopt;
  }
  uint64_t base_byte_align = std::min<uint64_t>(
      LargestPowerOfTwoDivisor(*base_align), kMaxLLVMAlignment);
  uint64_t elem_div = indices.empty() ? 0 : GetIndexKnownDivisor(indices[0]);
  if (elem_div == 0) {
    return base_byte_align;
  }

  std::optional<uint64_t> elem_bits =
      GetElementStorageBitWidth(element_type, op);
  if (!elem_bits || *elem_bits == 0) {
    return std::nullopt;
  }

  uint64_t bit_offset_div = MultiplyDivisors(elem_div, *elem_bits);
  uint64_t bit_align = std::gcd(base_byte_align * kBitsPerByte, bit_offset_div);
  if (bit_align < kBitsPerByte) {
    return uint64_t{1};
  }
  return LargestPowerOfTwoDivisor(bit_align / kBitsPerByte);
}

// Get the number of bits used to index into a sub-byte type.
int SubByteIndexingBits(int bit_width) {
  CHECK_LT(bit_width, 8) << "Passed width is not a sub-byte";
  return absl::bit_width<unsigned int>(8 / bit_width) - 1;
}

std::tuple<Value, Value> GetSubByteIndex(Value linear_index, int bit_width,
                                         mlir::ImplicitLocOpBuilder& b) {
  CHECK_LT(bit_width, 8) << "Passed width is not a sub-byte";
  int sub_byte_indexing_bits = SubByteIndexingBits(bit_width);
  // Get the sub-byte index by just masking out the high bits.
  Value sub_byte_index = mlir::arith::TruncIOp::create(
      b, b.getI8Type(),
      mlir::arith::AndIOp::create(
          b, linear_index,
          mlir::arith::ConstantIntOp::create(
              b, linear_index.getType(), (1 << sub_byte_indexing_bits) - 1)));
  //  Calculate the bit shift (assume little-endianness).
  Value sub_byte_shift = mlir::arith::MulIOp::create(
      b, mlir::arith::ConstantIntOp::create(b, b.getI8Type(), bit_width),
      sub_byte_index);
  Value byte_shift = mlir::arith::ConstantIntOp::create(
      b, linear_index.getType(), sub_byte_indexing_bits);
  Value i8_index = mlir::arith::ShRUIOp::create(b, linear_index, byte_shift);
  return {i8_index, sub_byte_shift};
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    Value linear_index, mlir::ImplicitLocOpBuilder& b) {
  mlir::RankedTensorType tensor_type = tensor.getType();
  Type element_type = tensor_type.getElementType();
  int64_t num_elements = tensor_type.getNumElements();
  std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type);
  if (sub_byte_width) {
    element_type = b.getI8Type();
    // Elements are packed.
    num_elements = CeilOfRatio<int64_t>(num_elements, 8 / *sub_byte_width);
  }
  auto ptr = ml::LLVMPointerType::get(b.getContext());
  Value tensor_ptr =
      UnrealizedConversionCastOp::create(b, ptr, tensor).getResult(0);
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  auto array_type =
      b.getType<ml::LLVMArrayType>(llvm_element_type, num_elements);
  auto gep =
      ml::GEPOp::create(b, ptr, array_type, tensor_ptr,
                        llvm::SmallVector<mlir::LLVM::GEPArg>{0, linear_index});
  gep.setNoWrapFlags(mlir::LLVM::GEPNoWrapFlags::inbounds);
  return gep;
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  return CreateGep(tensor, GetLinearIndex(indices, b), b);
}

// Returns a tensor of the type of `tensor` backed by the buffer that `tensor`
// forwards, which the caller must have traced successfully. An address computed
// from `tensor` itself would keep forwarding ops such as scf.index_switch
// results alive.
TypedValue<mlir::RankedTensorType> ResolveAddressTensor(
    TypedValue<mlir::RankedTensorType> tensor, mlir::ImplicitLocOpBuilder& b) {
  Value buffer = TraceDestinationBuffer(tensor);
  CHECK(buffer) << "Untraceable destinations must be rejected before lowering";
  if (buffer.getType() == tensor.getType()) {
    return mlir::cast<TypedValue<mlir::RankedTensorType>>(buffer);
  }
  // The type of `tensor` defines the byte stride of the address.
  return mlir::cast<TypedValue<mlir::RankedTensorType>>(
      UnrealizedConversionCastOp::create(b, tensor.getType(), buffer)
          .getResult(0));
}

ml::GEPOp CreateAtomicGep(TypedValue<mlir::RankedTensorType> tensor,
                          ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  return CreateGep(ResolveAddressTensor(tensor, b), indices, b);
}

struct RewriteTensorExtract : OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto linear_index = GetLinearIndex(op.getIndices(), b);
    Type element_type = op.getTensor().getType().getElementType();
    Value sub_byte_shift = nullptr;
    std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type);
    if (sub_byte_width) {
      std::tie(linear_index, sub_byte_shift) =
          GetSubByteIndex(linear_index, *sub_byte_width, b);
    }

    auto gep = CreateGep(op.getTensor(), linear_index, b);
    Type llvm_type =
        mlir::dyn_cast<ml::LLVMArrayType>(gep.getElemType()).getElementType();
    auto load_op = ml::LoadOp::create(rewriter, gep.getLoc(), llvm_type, gep);
    if (auto no_alias_attr = op->getAttrOfType<mlir::ArrayAttr>(
            ml::LLVMDialect::getNoAliasAttrName())) {
      load_op.setNoaliasScopesAttr(no_alias_attr);
    }
    auto load = load_op.getResult();

    if (sub_byte_shift) {
      load = mlir::arith::TruncIOp::create(
          b, b.getIntegerType(*sub_byte_width),
          mlir::arith::ShRUIOp::create(b, load, sub_byte_shift));
    }

    if (op.getType().isIntOrFloat()) {
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, op.getType(), load);
    } else {
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                              load);
    }
    return success();
  }
};

// Builds an `inbounds` GEP to the starting address of a 0D/1D vector transfer.
// For packed sub-byte element types (e.g., `i4` packed 2 per byte), converts
// the element index into a byte index by shifting right by
// `SubByteIndexingBits(bit_width)`; vectorized sub-byte transfers always start
// at a byte-aligned element offset.
ml::GEPOp CreateTransferGep(TypedValue<mlir::RankedTensorType> tensor,
                            ValueRange indices, Type element_type,
                            mlir::ImplicitLocOpBuilder& b) {
  Value linear_index = GetLinearIndex(indices, b);
  if (std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type)) {
    linear_index = arith::ShRUIOp::create(
        b, linear_index,
        arith::ConstantIntOp::create(b, linear_index.getType(),
                                     SubByteIndexingBits(*sub_byte_width)));
  }
  return CreateGep(tensor, linear_index, b);
}

// Returns a scalar integer type (`i8`, `i16`, `i32`, `i64`, or `i128`) matching
// the total bit-width of `vector_type` when `vector_type` is a fixed-length
// vector of packed sub-byte elements whose total width is a byte-sized power of
// two up to 128 bits; otherwise returns a null `IntegerType`.
//
// Sub-byte tensor elements (`i2`, `i4`, `f4E2M1FN`) are tightly packed in
// little-endian bit order within each byte, matching LLVM's bitcast layout
// between `<N x iM>` and `i(N*M)`. Emitting `llvm.load` / `llvm.store` directly
// on sub-byte LLVM vector types such as `<8 x i4>` causes LLVM's SelectionDAG
// legalizer to split or scalarize the access into narrow byte loads/stores and
// bit shifts even when wide alignment is proven. Loading/storing through a
// single `i(N*M)` integer and bitcasting in registers allows backends (e.g.,
// NVPTX/AMDGPU) to emit a single wide load/store instruction (`ld.global.b32`,
// `st.shared.b32`, etc.).
mlir::IntegerType GetPackedSubByteIntegerType(mlir::VectorType vector_type,
                                              mlir::Builder& b) {
  if (vector_type.isScalable()) {
    return nullptr;
  }
  std::optional<int> sub_byte_width =
      GetSubByteBitWidth(vector_type.getElementType());
  if (!sub_byte_width) {
    return nullptr;
  }
  int64_t total_bits = vector_type.getNumElements() * (*sub_byte_width);
  if (total_bits < kBitsPerByte || total_bits > kMaxPackedTransferBits ||
      !absl::has_single_bit(static_cast<uint64_t>(total_bits))) {
    return nullptr;
  }
  return b.getIntegerType(total_bits);
}

// Resolves the byte alignment to attach to a lowered vector transfer `load` or
// `store`, using provenance/index alignment inference when available and
// falling back to the target DataLayout's preferred element alignment.
uint64_t ResolveTransferAlignment(Value base, ValueRange indices,
                                  Type element_type, Operation* op) {
  if (std::optional<uint64_t> alignment =
          GetTransferAlignment(base, indices, element_type, op)) {
    return *alignment;
  }
  // TODO(willfroom): We should really just use getTypeABIAlignment here,
  // but this requires passing through the full data layout for the target
  // machine.
  return mlir::DataLayout::closest(op).getTypePreferredAlignment(element_type);
}

struct RewriteTransferRead : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op,
      mlir::PatternRewriter& rewriter) const override {
    assert(IsSupportedTransfer(op));

    auto source =
        mlir::dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(op.getBase());
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mlir::VectorType vector_type = op.getVectorType();
    mlir::Type element_type = vector_type.getElementType();
    ml::GEPOp gep = CreateTransferGep(source, op.getIndices(), element_type, b);

    mlir::LLVMTypeConverter converter(b.getContext());
    Type llvm_vector_type = converter.convertType(vector_type);
    mlir::IntegerType packed_int_type =
        GetPackedSubByteIntegerType(vector_type, b);
    Type load_type = packed_int_type ? Type{packed_int_type} : llvm_vector_type;

    auto load = ml::LoadOp::create(b, load_type, gep);
    load.setAlignment(ResolveTransferAlignment(op.getBase(), op.getIndices(),
                                               element_type, op));

    Value loaded = load.getResult();
    if (packed_int_type) {
      loaded = ml::BitcastOp::create(b, llvm_vector_type, loaded);
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            loaded);
    return success();
  }
};

class RewriteTensorInsert : public OpRewritePattern<mlir::tensor::InsertOp> {
 public:
  RewriteTensorInsert(mlir::MLIRContext* context, const DeviceSpec& device_spec)
      : OpRewritePattern<mlir::tensor::InsertOp>(context),
        device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value dest = GetDestinationBuffer(op.getDest());
    if (!dest) {
      return failure();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    auto linear_index = GetLinearIndex(op.getIndices(), b);
    auto scalar_value = op.getScalar();

    // For i4 we store 2 values into one byte. This needs special handling here.
    Type element_type = tensor_dest.getType().getElementType();
    std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type);
    if (sub_byte_width) {
      Value sub_byte_shift = nullptr;
      std::tie(linear_index, sub_byte_shift) =
          GetSubByteIndex(linear_index, *sub_byte_width, b);

      Type ty = b.getI8Type();
      if (scalar_value.getType() != b.getIntegerType(*sub_byte_width)) {
        scalar_value = arith::BitcastOp::create(
            b, b.getIntegerType(*sub_byte_width), scalar_value);
      }
      scalar_value = mlir::arith::ExtUIOp::create(b, ty, scalar_value);

      // Mask of 1s in the sub-byte position & 0s elsewhere.
      Value mask = mlir::arith::ShLIOp::create(
          b,
          mlir::arith::ConstantIntOp::create(b, ty, (1 << *sub_byte_width) - 1),
          sub_byte_shift);
      // Mask of 0s in the sub-byte position & 1s elsewhere.
      Value inverse_mask = mlir::arith::XOrIOp::create(
          b, mlir::arith::ConstantIntOp::create(b, ty, 0xff), mask);

      Value shifted_value =
          mlir::arith::ShLIOp::create(b, scalar_value, sub_byte_shift);

      if (device_spec_.IsCpu()) {
        // On CPU, workgroup tiles are byte-aligned (see `GetWorkDimensions`),
        // so distinct threads never access the same byte.
        auto gep = CreateGep(tensor_dest, linear_index, b);
        Value current_value = ml::LoadOp::create(b, ty, gep);
        Value masked_out_current_value =
            mlir::arith::AndIOp::create(b, current_value, inverse_mask);
        Value new_value = mlir::arith::OrIOp::create(
            b, masked_out_current_value, shifted_value);
        auto store_op = ml::StoreOp::create(b, new_value, gep);
        if (auto alias_scope_attr = op->getAttrOfType<mlir::ArrayAttr>(
                ml::LLVMDialect::getAliasScopesAttrName())) {
          store_op.setAliasScopesAttr(alias_scope_attr);
        }
        if (auto no_alias_attr = op->getAttrOfType<mlir::ArrayAttr>(
                ml::LLVMDialect::getNoAliasAttrName())) {
          store_op.setNoaliasScopesAttr(no_alias_attr);
        }
        op.replaceAllUsesWith(op.getDest());
      } else {
        // We need to use directly op.getDest() as input, otherwise the
        // following rewrite might remove the only user of it.
        tensor_dest = op.getDest();
        // Technically we should halve the number of elements when going to i8
        // element type, but it doesn't really matter because we only actually
        // use the element type. Indexing is done by linear index, and GEP ops
        // don't care about the number of elements. The tensor types will
        // disappear completely after the LowerTensors pass.
        Type tensor_ty = tensor_dest.getType().clone(ty);
        auto tensor_dest_i8 =
            UnrealizedConversionCastOp::create(b, tensor_ty, tensor_dest)
                .getResult(0);

        // We need AtomicRMWOp because it can happen that different threads try
        // to access the same memory location.
        SmallVector<Value, 1> rmw_indices;
        if (tensor_dest.getType().getRank() > 0) {
          rmw_indices.push_back(
              arith::IndexCastUIOp::create(b, b.getIndexType(), linear_index));
        }
        auto atomic_rmw = AtomicRMWOp::create(b, tensor_dest_i8, rmw_indices);
        mlir::ImplicitLocOpBuilder body_builder(atomic_rmw.getLoc(),
                                                atomic_rmw.getBodyBuilder());
        Value current_value = atomic_rmw.getCurrentValue();
        Value masked_out_current_value = mlir::arith::AndIOp::create(
            body_builder, current_value, inverse_mask);
        Value new_value = mlir::arith::OrIOp::create(
            body_builder, masked_out_current_value, shifted_value);
        xla::YieldOp::create(body_builder, new_value);
        Value casted_result =
            UnrealizedConversionCastOp::create(b, tensor_dest.getType(),
                                               atomic_rmw.getResult())
                .getResult(0);
        op.replaceAllUsesWith(casted_result);
      }
    } else {
      auto gep = CreateGep(tensor_dest, linear_index, b);
      mlir::LLVMTypeConverter converter(getContext());
      auto llvm_type = converter.convertType(scalar_value.getType());
      scalar_value =
          scalar_value.getType().isIntOrFloat()
              ? arith::BitcastOp::create(b, llvm_type, scalar_value)
              : UnrealizedConversionCastOp::create(b, llvm_type, scalar_value)
                    .getResult(0);
      auto store_op = ml::StoreOp::create(b, scalar_value, gep);
      if (auto alias_scope_attr = op->getAttrOfType<mlir::ArrayAttr>(
              ml::LLVMDialect::getAliasScopesAttrName())) {
        store_op.setAliasScopesAttr(alias_scope_attr);
      }
      if (auto no_alias_attr = op->getAttrOfType<mlir::ArrayAttr>(
              ml::LLVMDialect::getNoAliasAttrName())) {
        store_op.setNoaliasScopesAttr(no_alias_attr);
      }
      op.replaceAllUsesWith(op.getDest());
    }

    op.erase();
    return success();
  }

 private:
  const DeviceSpec& device_spec_;
};

struct RewriteTransferWrite : OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp op,
      mlir::PatternRewriter& rewriter) const override {
    assert(IsSupportedTransfer(op));
    Value dest = GetDestinationBuffer(op.getBase());

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    mlir::VectorType vector_type = op.getVectorType();
    mlir::Type element_type = vector_type.getElementType();
    ml::GEPOp gep =
        CreateTransferGep(tensor_dest, op.getIndices(), element_type, b);

    mlir::LLVMTypeConverter converter(getContext());
    Type llvm_vector_type = converter.convertType(vector_type);
    Value store_value =
        UnrealizedConversionCastOp::create(b, llvm_vector_type, op.getVector())
            .getResult(0);
    if (mlir::IntegerType packed_int_type =
            GetPackedSubByteIntegerType(vector_type, b)) {
      store_value = ml::BitcastOp::create(b, packed_int_type, store_value);
    }

    auto store = ml::StoreOp::create(b, store_value, gep);
    store.setAlignment(
        ResolveTransferAlignment(dest, op.getIndices(), element_type, op));

    rewriter.replaceOp(op, mlir::ValueRange{op.getBase()});
    return success();
  }
};

Value UnwrapTensorCasts(Value value) {
  while (value) {
    value = GetDestinationBuffer(value);
    if (!value) {
      return nullptr;
    }
    auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast.getNumOperands() != 1 ||
        !mlir::isa<mlir::RankedTensorType>(cast.getOperand(0).getType())) {
      break;
    }
    value = cast.getOperand(0);
  }
  return value;
}

int64_t GetCopySliceAlignment(Value base, ValueRange indices,
                              int64_t element_size_bytes,
                              int64_t element_abi_alignment) {
  Value unwrapped = UnwrapTensorCasts(base);
  if (!unwrapped) {
    return element_abi_alignment;
  }
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(unwrapped);
  if (!arg) {
    return element_abi_alignment;
  }
  auto func = mlir::dyn_cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func) {
    return element_abi_alignment;
  }
  auto align_attr = func.getArgAttrOfType<mlir::IntegerAttr>(
      arg.getArgNumber(), ml::LLVMDialect::getAlignAttrName());
  if (!align_attr) {
    return element_abi_alignment;
  }
  int64_t base_align = align_attr.getInt();
  if (base_align <= 1 || !llvm::isPowerOf2_64(base_align)) {
    return element_abi_alignment;
  }
  if (indices.empty()) {
    return base_align;
  }
  if (indices.size() == 1) {
    llvm::APInt const_idx;
    if (mlir::matchPattern(indices[0], mlir::m_ConstantInt(&const_idx)) &&
        const_idx.isNonNegative() && const_idx.getActiveBits() <= 60) {
      uint64_t offset_bytes =
          const_idx.getZExtValue() * static_cast<uint64_t>(element_size_bytes);
      return static_cast<int64_t>(
          llvm::MinAlign(static_cast<uint64_t>(base_align), offset_bytes));
    }
  }
  return static_cast<int64_t>(
      llvm::MinAlign(static_cast<uint64_t>(base_align),
                     static_cast<uint64_t>(element_size_bytes)));
}

struct RewriteCopySlice : mlir::OpRewritePattern<CopySliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      CopySliceOp op, mlir::PatternRewriter& rewriter) const override {
    Value src = GetDestinationBuffer(op.getSrc());
    Value dest = GetDestinationBuffer(op.getDest());
    if (!src || !dest) {
      return failure();
    }

    auto tensor_src = mlir::cast<TypedValue<mlir::RankedTensorType>>(src);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    Type element_type = tensor_dest.getType().getElementType();
    if (GetSubByteBitWidth(element_type)) {
      return op->emitOpError("sub-byte copy_slice is not supported");
    }

    int64_t num_elements = op.getNumElements();
    if (num_elements > 0) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      Value src_linear_index = GetLinearIndex(op.getSrcIndices(), b);
      Value dst_linear_index = GetLinearIndex(op.getDestIndices(), b);
      auto src_gep = CreateGep(tensor_src, src_linear_index, b);
      auto dst_gep = CreateGep(tensor_dest, dst_linear_index, b);

      mlir::DataLayout data_layout = mlir::DataLayout::closest(op);
      mlir::LLVMTypeConverter converter(getContext());
      Type llvm_element_type = converter.convertType(element_type);
      int64_t element_size_bytes = static_cast<int64_t>(
          data_layout.getTypeSize(llvm_element_type).getFixedValue());
      int64_t element_abi_alignment = static_cast<int64_t>(
          data_layout.getTypeABIAlignment(llvm_element_type));
      int64_t src_align =
          GetCopySliceAlignment(op.getSrc(), op.getSrcIndices(),
                                element_size_bytes, element_abi_alignment);
      int64_t dst_align =
          GetCopySliceAlignment(op.getDest(), op.getDestIndices(),
                                element_size_bytes, element_abi_alignment);

      auto make_align_dict = [&](int64_t align) {
        if (align <= 1) {
          return b.getDictionaryAttr({});
        }
        return b.getDictionaryAttr(b.getNamedAttr(
            ml::LLVMDialect::getAlignAttrName(), b.getI64IntegerAttr(align)));
      };
      mlir::ArrayAttr arg_attrs =
          (dst_align > 1 || src_align > 1)
              ? b.getArrayAttr({make_align_dict(dst_align),
                                make_align_dict(src_align),
                                b.getDictionaryAttr({})})
              : nullptr;

      Value byte_count = ml::ConstantOp::create(
          b, b.getI64Type(), num_elements * element_size_bytes);
      ml::MemcpyOp::create(b, dst_gep, src_gep, byte_count,
                           /*isVolatile=*/false, /*tbaa_struct=*/nullptr,
                           /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
                           /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
                           arg_attrs, /*res_attrs=*/nullptr);
    }

    op.replaceAllUsesWith(op.getDest());
    op.erase();
    return success();
  }
};

struct RewriteCall : OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::func::CallOp op, mlir::PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperandTypes(), [](Type ty) {
          return mlir::isa<mlir::RankedTensorType>(ty);
        })) {
      return rewriter.notifyMatchFailure(op, "the call has no input tensors");
    }

    auto ptr_ty = mlir::LLVM::LLVMPointerType::get(op.getContext());
    llvm::SmallVector<Value, 4> new_operands;
    new_operands.reserve(op.getNumOperands());
    llvm::SmallVector<Type, 4> new_result_types;
    for (const auto&& [index, arg] : llvm::enumerate(op.getOperands())) {
      if (mlir::isa<mlir::RankedTensorType>(arg.getType())) {
        new_operands.push_back(mlir::UnrealizedConversionCastOp::create(
                                   rewriter, op.getLoc(), ptr_ty, arg)
                                   .getResult(0));
      } else {
        new_operands.push_back(arg);
      }
    }
    for (const auto result_type : op.getResultTypes()) {
      if (!mlir::isa<mlir::RankedTensorType>(result_type)) {
        new_result_types.push_back(result_type);
      }
    }
    auto new_call = mlir::func::CallOp::create(
        rewriter, op.getLoc(), op.getCallee(), new_result_types, new_operands);

    if (new_call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, new_call.getResults());
    }
    return success();
  }
};

ml::GlobalOp CreateGlobalOp(mlir::Attribute value,
                            const std::string& name_prefix,
                            mlir::ShapedType shaped_ty, mlir::ModuleOp module,
                            bool is_constant, int addr_space,
                            mlir::ImplicitLocOpBuilder& b) {
  if (auto elements = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(value)) {
    // The lowering to LLVM only works for 1d tensors or those with trailing
    // unit dimensions.
    value = elements.reshape(mlir::RankedTensorType::get(
        {elements.getNumElements()}, elements.getElementType()));
  }

  Type element_type = shaped_ty.getElementType();
  int64_t num_elements = shaped_ty.getNumElements();
  // Needed to support complex element type.
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  if (element_type.isIntOrFloat() &&
      element_type.getIntOrFloatBitWidth() == 4) {
    num_elements = CeilOfRatio<int64_t>(num_elements, 2);
    llvm_element_type = b.getI8Type();
    if (value) {
      auto unpacked_data =
          mlir::cast<mlir::DenseElementsAttr>(value).getRawData();
      std::vector<char> packed_data(num_elements);
      absl::Span<char> packed_data_span = absl::MakeSpan(packed_data);
      PackIntN(4, unpacked_data, packed_data_span);
      value = mlir::DenseElementsAttr::getFromRawBuffer(
          mlir::RankedTensorType::get({num_elements}, llvm_element_type),
          packed_data);
    }
  }
  auto array_ty = ml::LLVMArrayType::get(llvm_element_type, num_elements);
  std::string name;
  int index = 0;
  do {
    name = absl::StrCat(name_prefix, index);
    ++index;
  } while (module.lookupSymbol(name));
  b.setInsertionPointToStart(module.getBody());
  return ml::GlobalOp::create(b, array_ty, is_constant,
                              /*linkage=*/ml::Linkage::Private, name, value,
                              /*alignment=*/0, addr_space);
}

struct RewriteAllocateShared : OpRewritePattern<gpu::AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      gpu::AllocateSharedOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = mlir::cast<mlir::ShapedType>(op.getResult().getType());
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto global =
        CreateGlobalOp(mlir::Attribute{}, "shared_", shaped_ty, module,
                       /*is_constant=*/false, kGPUSharedMemoryAddrSpace, b);
    // Shared memory exists only on GPU, where an atomic update of an element
    // narrower than 32 bits (every insert of an element narrower than a byte,
    // and any narrow AtomicRMWOp) becomes a 32 bit CAS on the containing word,
    // found with ptrmask. The masked pointer must stay inside this object, so
    // the allocation is word aligned and padded to a whole number of words.
    // This also keeps the CAS from touching bytes of a neighboring allocation.
    // Wider types keep their natural alignment.
    Type element_type = shaped_ty.getElementType();
    if (element_type.isIntOrFloat() &&
        element_type.getIntOrFloatBitWidth() < 32) {
      auto array_ty = mlir::cast<ml::LLVMArrayType>(global.getGlobalType());
      int64_t elem_bytes = CeilOfRatio<int64_t>(
          array_ty.getElementType().getIntOrFloatBitWidth(), 8);
      int64_t padded =
          RoundUpTo<int64_t>(array_ty.getNumElements() * elem_bytes, 4) /
          elem_bytes;
      global.setGlobalTypeAttr(mlir::TypeAttr::get(
          ml::LLVMArrayType::get(array_ty.getElementType(), padded)));
      global.setAlignment(4);
    }

    rewriter.setInsertionPoint(op);
    auto addr = ml::AddressOfOp::create(rewriter, op.getLoc(), global);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        ml::AddrSpaceCastOp::create(rewriter, op.getLoc(),
                                    ml::LLVMPointerType::get(op.getContext()),
                                    addr)
            .getResult());
    return success();
  }
};

struct RewriteNonScalarConstants : OpRewritePattern<mlir::arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (mlir::isa<mlir::VectorType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "the op is a vector constant");
    }
    auto shaped_ty = mlir::dyn_cast<mlir::ShapedType>(op.getValue().getType());
    // We only need to rewrite non-scalar constants.
    if (!shaped_ty || shaped_ty.getNumElements() < 2) {
      return rewriter.notifyMatchFailure(
          op, "the op is an effective scalar constant");
    }

    constexpr int kGPUGlobalMemoryAddrSpace = 0;
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto global =
        CreateGlobalOp(op.getValue(), "global_cst_", shaped_ty, module,
                       /*is_constant=*/true, kGPUGlobalMemoryAddrSpace, b);

    rewriter.setInsertionPoint(op);
    auto addr = ml::AddressOfOp::create(rewriter, op.getLoc(), global);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        ml::AddrSpaceCastOp::create(rewriter, op.getLoc(),
                                    ml::LLVMPointerType::get(op.getContext()),
                                    addr)
            .getResult());
    return success();
  }
};

struct RewriteSyncThreads : OpRewritePattern<gpu::SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      gpu::SyncThreadsOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::gpu::BarrierOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

// TODO(jreiffers): Generalize this to support index switches with some used
// results and upstream it as a canonicalization pattern.
struct RemoveUnusedIndexSwitchResults : OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::IndexSwitchOp op, mlir::PatternRewriter& rewriter) const override {
    if (op->getNumResults() == 0 || !op->use_empty()) {
      return rewriter.notifyMatchFailure(op, "the op has users");
    }

    auto new_op = scf::IndexSwitchOp::create(rewriter, op.getLoc(),
                                             mlir::TypeRange{}, op.getArg(),
                                             op.getCases(), op.getNumCases());
    for (int i = 0; i < op->getNumRegions(); ++i) {
      auto& old_region = op->getRegion(i);
      auto& new_region = new_op->getRegion(i);
      rewriter.mergeBlocks(&old_region.getBlocks().front(),
                           &new_region.emplaceBlock());
      auto yield_op = new_region.getBlocks().front().getTerminator();
      rewriter.modifyOpInPlace(yield_op, [&]() { yield_op->setOperands({}); });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

llvm::StringRef GetAtomicSyncScope(const DeviceSpec& device_spec) {
  if (device_spec.IsAmdGpu()) {
    return llvm::StringRef("agent-one-as");
  }
  return llvm::StringRef();
}

void SetAmdAtomicRMWAttrs(mlir::OpBuilder& b, ml::AtomicRMWOp rmw_op,
                          const DeviceSpec& device_spec) {
  if (!device_spec.IsAmdGpu()) {
    return;
  }
  mlir::UnitAttr unit_attr = b.getUnitAttr();
  rmw_op->setDiscardableAttr(
      mlir::ROCDL::ROCDLDialect::NoRemoteMemoryAttrHelper::getNameStr(),
      unit_attr);
  rmw_op->setDiscardableAttr(
      mlir::ROCDL::ROCDLDialect::NoFineGrainedMemoryAttrHelper::getNameStr(),
      unit_attr);
}

bool IsAtomicIntegral(Type element_type, const DeviceSpec& device_spec) {
  if (!element_type.isSignlessInteger()) {
    return false;
  }
  unsigned element_bitwidth = element_type.getIntOrFloatBitWidth();
  if (device_spec.IsCpu()) {
    return element_bitwidth == 8 || element_bitwidth == 16 ||
           element_bitwidth == 32 || element_bitwidth == 64;
  }
  return element_bitwidth == 32 || element_bitwidth == 64;
}

Value CreateBitcast(mlir::ImplicitLocOpBuilder& b, Value value, Type ty) {
  Type src_ty = value.getType();
  if (src_ty == ty) {
    return value;
  }
  if (src_ty.isIntOrFloat() && ty.isIntOrFloat()) {
    return arith::BitcastOp::create(b, ty, value);
  }
  if (auto src_complex = mlir::dyn_cast<mlir::ComplexType>(src_ty)) {
    Value re = mlir::complex::ReOp::create(b, value);
    Value im = mlir::complex::ImOp::create(b, value);
    auto vec_ty = mlir::VectorType::get({2}, src_complex.getElementType());
    value = vector::FromElementsOp::create(b, vec_ty, ValueRange{re, im});
    src_ty = vec_ty;
  }
  if (auto dst_complex = mlir::dyn_cast<mlir::ComplexType>(ty)) {
    auto vec_ty = mlir::VectorType::get({2}, dst_complex.getElementType());
    Value vec = CreateBitcast(b, value, vec_ty);
    Value re = vector::ExtractOp::create(b, vec, 0);
    Value im = vector::ExtractOp::create(b, vec, 1);
    return mlir::complex::CreateOp::create(b, dst_complex, re, im);
  }
  if (auto src_vec = mlir::dyn_cast<mlir::VectorType>(src_ty)) {
    if (!ml::isCompatibleType(src_vec)) {
      Type int_elem =
          b.getIntegerType(src_vec.getElementType().getIntOrFloatBitWidth());
      src_vec = mlir::VectorType::get(src_vec.getShape(), int_elem);
      value = arith::BitcastOp::create(b, src_vec, value);
    }
  } else if (!ml::isCompatibleType(src_ty)) {
    value = arith::BitcastOp::create(
        b, b.getIntegerType(src_ty.getIntOrFloatBitWidth()), value);
  }
  if (value.getType() == ty) {
    return value;
  }
  if (auto dst_vec = mlir::dyn_cast<mlir::VectorType>(ty)) {
    if (!ml::isCompatibleType(dst_vec)) {
      Type int_elem =
          b.getIntegerType(dst_vec.getElementType().getIntOrFloatBitWidth());
      auto compat_vec = mlir::VectorType::get(dst_vec.getShape(), int_elem);
      Value compat_val = ml::BitcastOp::create(b, compat_vec, value);
      return arith::BitcastOp::create(b, dst_vec, compat_val);
    }
  } else if (!ml::isCompatibleType(ty)) {
    Value int_val = ml::BitcastOp::create(
        b, b.getIntegerType(ty.getIntOrFloatBitWidth()), value);
    return arith::BitcastOp::create(b, ty, int_val);
  }
  return ml::BitcastOp::create(b, ty, value);
}

int64_t GetAtomicTypeBitWidth(Type ty) {
  if (auto complex_ty = mlir::dyn_cast<mlir::ComplexType>(ty)) {
    return static_cast<int64_t>(
               complex_ty.getElementType().getIntOrFloatBitWidth()) *
           2;
  }
  if (auto vec_ty = mlir::dyn_cast<mlir::VectorType>(ty)) {
    return vec_ty.getNumElements() *
           static_cast<int64_t>(
               vec_ty.getElementType().getIntOrFloatBitWidth());
  }
  if (ty.isIntOrFloat()) {
    return ty.getIntOrFloatBitWidth();
  }
  return 0;
}

// Returns the byte alignment proven for the address of element `indices` of
// `dest`, or 1 when none can be proven. Atomics address the traced destination
// buffer (see `ResolveAddressTensor`), so its alignment is the one that counts.
int64_t GetGuaranteedByteAlignment(TypedValue<mlir::RankedTensorType> dest,
                                   ValueRange indices, Operation* op) {
  Value buffer = TraceDestinationBuffer(dest);
  if (!buffer) {
    return 1;
  }
  return static_cast<int64_t>(
      GetTransferAlignment(buffer, indices, dest.getType().getElementType(), op)
          .value_or(1));
}

struct AtomicWordAccess {
  Type result_ty;
  int64_t result_size;
  bool small_type;
  Type atomic_ty;
  Value addr;
  Value shift;
  Value mask;
};

// Prepares the aligned atomic word address, bit shift, and preservation mask
// used when lowering narrow or complex types via atomicCAS.
//
// On GPUs, atomicCAS operates on 32 bit and 64 bit words, so elements narrower
// than 32 bits mask their pointer down to a 4 byte boundary via `ptrmask`
// (tensor buffers are 4 byte aligned and padded to a multiple of 4 bytes).
// On CPUs, 8 bit and 16 bit atomicCAS operations are supported natively and
// buffers are not guaranteed to have a size of 4N, so only types narrower than
// a byte widen to i8 without rounding the byte address down.
AtomicWordAccess PrepareAtomicWordAccess(
    mlir::Operation* op, TypedValue<mlir::RankedTensorType> tensor,
    ValueRange indices, Type result_ty, const DeviceSpec& device_spec,
    mlir::PatternRewriter& rewriter) {
  Location loc = op->getLoc();
  int64_t result_size = GetAtomicTypeBitWidth(result_ty);

  int64_t min_atomic_size = device_spec.IsCpu() ? 8 : 32;
  bool small_type = result_size < min_atomic_size;
  Type atomic_ty = mlir::IntegerType::get(
      op->getContext(),
      static_cast<unsigned>(small_type ? min_atomic_size : result_size));

  mlir::ImplicitLocOpBuilder b(loc, rewriter);
  Value linear_index = GetLinearIndex(indices, b);
  Value sub_byte_shift;
  std::optional<int> sub_byte_bit_width = GetSubByteBitWidth(result_ty);
  if (sub_byte_bit_width) {
    std::tie(linear_index, sub_byte_shift) =
        GetSubByteIndex(linear_index, *sub_byte_bit_width, b);
  }
  Value addr = CreateGep(ResolveAddressTensor(tensor, b), linear_index, b);
  Value shift;
  Value mask;
  if (small_type) {
    if (min_atomic_size > 8) {
      Type addr_int_ty = rewriter.getI64Type();
      Value addr_int = ml::PtrToIntOp::create(rewriter, loc, addr_int_ty, addr);
      Value addr_offset = ml::AndOp::create(
          rewriter, loc, addr_int,
          ml::ConstantOp::create(rewriter, loc, addr_int_ty, 3));
      // ptrmask instead of a negative GEP: LLVM would reassociate the GEP into
      // [base + imm], and ptxas for sm_6x drops the low bits of that immediate
      // on shared atomics, producing a misaligned CAS.
      addr = ml::PtrMaskOp::create(
          rewriter, loc, addr,
          ml::ConstantOp::create(rewriter, loc, addr_int_ty, -4));

      Value offset = ml::TruncOp::create(rewriter, loc, atomic_ty, addr_offset);
      shift = ml::MulOp::create(
          rewriter, loc, offset,
          ml::ConstantOp::create(rewriter, loc, offset.getType(), 8));
      if (sub_byte_bit_width) {
        sub_byte_shift =
            ml::ZExtOp::create(rewriter, loc, shift.getType(), sub_byte_shift);
        shift = ml::AddOp::create(rewriter, loc, shift, sub_byte_shift);
      }
    } else {
      shift = sub_byte_bit_width
                  ? sub_byte_shift
                  : ml::ConstantOp::create(rewriter, loc, atomic_ty, 0);
    }

    // Unpacked types narrower than a byte (such as i1 or 6 bit floats) occupy a
    // full byte per element in memory, so clear all 8 bits of that byte.
    int64_t storage_bit_size =
        (!sub_byte_bit_width && result_size < 8) ? 8 : result_size;
    Value bits_long = ml::ConstantOp::create(rewriter, loc, atomic_ty, -1);
    Value bits_short = ml::ConstantOp::create(
        rewriter, loc, atomic_ty,
        llvm::APInt::getLowBitsSet(atomic_ty.getIntOrFloatBitWidth(),
                                   static_cast<unsigned>(storage_bit_size)));
    mask =
        ml::XOrOp::create(rewriter, loc, bits_long,
                          ml::ShlOp::create(rewriter, loc, bits_short, shift));
  }
  return AtomicWordAccess{result_ty, result_size, small_type, atomic_ty,
                          addr,      shift,       mask};
}

bool CanLowerScalarOrComplexAsAtomicCAS(Operation* op,
                                        TypedValue<mlir::RankedTensorType> dest,
                                        ValueRange indices, Type result_ty,
                                        const DeviceSpec& device_spec) {
  if (mlir::isa<mlir::VectorType>(result_ty)) {
    return false;
  }
  int64_t result_size = GetAtomicTypeBitWidth(result_ty);
  if (result_size <= 0 ||
      (result_size != 6 && !llvm::isPowerOf2_64(result_size))) {
    return false;
  }
  if (result_size <= 64) {
    // A complex element only has the ABI alignment of its component type, but
    // the atomic word that holds it must be aligned to the size of the word.
    return !mlir::isa<mlir::ComplexType>(result_ty) ||
           GetGuaranteedByteAlignment(dest, indices, op) >= result_size / 8;
  }
  if (result_size == 128) {
    // CPU targets without a native 128 bit CAS call
    // __atomic_compare_exchange_16, which the XLA:CPU JIT does not provide,
    // and DeviceSpec carries no CPU features to tell such targets apart.
    return device_spec.IsNvidiaGpu() &&
           device_spec.gpu().cuda_compute_capability().IsAtLeastHopper() &&
           GetGuaranteedByteAlignment(dest, indices, op) >= 16;
  }
  return false;
}

class RewriteAtomicStore : public OpRewritePattern<AtomicStoreOp> {
 public:
  RewriteAtomicStore(mlir::MLIRContext* context, const DeviceSpec& device_spec)
      : OpRewritePattern<AtomicStoreOp>(context), device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      AtomicStoreOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIndices().size() > 1 || op.getDest() == op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only acyclic 0D and 1D tensors are supported");
    }
    if (!TraceDestinationBuffer(op.getDest())) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot trace the destination buffer");
    }
    Location loc = op.getLoc();
    Value value = op.getValue();
    Type element_type = value.getType();
    llvm::StringRef sync_scope = GetAtomicSyncScope(device_spec_);
    // AMDGPU SIMemoryLegalizer only applies agent scope L1 cache bypass when
    // ordering is at least monotonic.
    ml::AtomicOrdering direct_ordering = device_spec_.IsAmdGpu()
                                             ? ml::AtomicOrdering::monotonic
                                             : ml::AtomicOrdering::unordered;

    if (auto complex_ty = mlir::dyn_cast<mlir::ComplexType>(element_type)) {
      // The 64 bit store needs 8 byte alignment, but complex<f32> only has the
      // 4 byte ABI alignment of f32.
      if (complex_ty.getElementType().isF32() &&
          GetGuaranteedByteAlignment(op.getDest(), op.getIndices(), op) >= 8) {
        mlir::ImplicitLocOpBuilder b(loc, rewriter);
        Value addr = CreateAtomicGep(op.getDest(), op.getIndices(), b);
        Value store_value = CreateBitcast(b, value, b.getI64Type());
        ml::StoreOp::create(rewriter, loc, store_value, addr,
                            /*alignment=*/8,
                            /*isVolatile=*/false, /*isNonTemporal=*/false,
                            /*isInvariantGroup=*/false, direct_ordering,
                            sync_scope);
        rewriter.replaceOp(op, op.getDest());
        return success();
      }
    } else if (device_spec_.IsCpu()) {
      if (element_type.isIntOrFloat()) {
        unsigned bit_width = element_type.getIntOrFloatBitWidth();
        if (bit_width == 8 || bit_width == 16 || bit_width == 32 ||
            bit_width == 64) {
          mlir::ImplicitLocOpBuilder b(loc, rewriter);
          Value addr = CreateAtomicGep(op.getDest(), op.getIndices(), b);
          Value store_value = value;
          bool is_llvm_atomic_type =
              element_type.isSignlessInteger(bit_width) ||
              element_type.isF16() || element_type.isBF16() ||
              element_type.isF32() || element_type.isF64();
          if (!is_llvm_atomic_type) {
            store_value = arith::BitcastOp::create(
                b, b.getIntegerType(bit_width), store_value);
          }
          ml::StoreOp::create(rewriter, loc, store_value, addr,
                              /*alignment=*/bit_width / 8,
                              /*isVolatile=*/false, /*isNonTemporal=*/false,
                              /*isInvariantGroup=*/false, direct_ordering,
                              sync_scope);
          rewriter.replaceOp(op, op.getDest());
          return success();
        }
      }
    } else if (element_type.isF32() || element_type.isF64() ||
               IsAtomicIntegral(element_type, device_spec_)) {
      mlir::ImplicitLocOpBuilder b(loc, rewriter);
      Value addr = CreateAtomicGep(op.getDest(), op.getIndices(), b);
      Value store_value = value;
      // LLVM's SPIRV backend only declares Int64Atomics on OpAtomicStore for
      // 64 bit integer operands, not 64 bit floats.
      if (device_spec_.IsIntelGpu() && element_type.isF64()) {
        store_value = arith::BitcastOp::create(b, b.getI64Type(), store_value);
      }
      ml::StoreOp::create(
          rewriter, loc, store_value, addr,
          /*alignment=*/element_type.getIntOrFloatBitWidth() / 8,
          /*isVolatile=*/false, /*isNonTemporal=*/false,
          /*isInvariantGroup=*/false, direct_ordering, sync_scope);
      rewriter.replaceOp(op, op.getDest());
      return success();
    }

    if (!CanLowerScalarOrComplexAsAtomicCAS(op, op.getDest(), op.getIndices(),
                                            element_type, device_spec_)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported atomic word size for target atomicCAS");
    }
    AtomicWordAccess access =
        PrepareAtomicWordAccess(op, op.getDest(), op.getIndices(), element_type,
                                device_spec_, rewriter);
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    Value shifted_value;
    Value direct_new_value;
    if (access.small_type) {
      Value cast_value = ml::ZExtOp::create(
          b, access.atomic_ty,
          CreateBitcast(b, value,
                        rewriter.getIntegerType(
                            static_cast<unsigned>(access.result_size))));
      shifted_value = ml::ShlOp::create(b, cast_value, access.shift);
    } else {
      direct_new_value = CreateBitcast(b, value, access.atomic_ty);
    }

    Value initial =
        ml::LoadOp::create(rewriter, loc, access.atomic_ty, access.addr);
    scf::WhileOp::create(
        rewriter, loc, TypeRange{access.atomic_ty}, ValueRange{initial},
        [&](mlir::OpBuilder& builder, Location loop_loc, ValueRange values) {
          mlir::ImplicitLocOpBuilder loop_b(loop_loc, builder);
          Value old_value = values[0];
          Value new_value =
              access.small_type
                  ? ml::OrOp::create(
                        loop_b,
                        ml::AndOp::create(loop_b, old_value, access.mask),
                        shifted_value)
                        .getResult()
                  : direct_new_value;
          Value cmpxchg = ml::AtomicCmpXchgOp::create(
              loop_b, loop_loc, access.addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::monotonic,
              /*failure_ordering=*/ml::AtomicOrdering::monotonic, sync_scope);
          Value next = ml::ExtractValueOp::create(loop_b, cmpxchg, 0);
          Value ok = ml::ExtractValueOp::create(loop_b, cmpxchg, 1);
          Value low_bit = ml::ConstantOp::create(
              loop_b, loop_b.getOneAttr(loop_b.getI1Type()));
          Value not_ok = ml::XOrOp::create(loop_b, ok, low_bit);
          scf::ConditionOp::create(loop_b, not_ok, ValueRange{next});
        },
        [](mlir::OpBuilder& builder, Location loop_loc, ValueRange values) {
          scf::YieldOp::create(builder, loop_loc, values);
        });
    rewriter.replaceOp(op, op.getDest());
    return success();
  }

 private:
  const DeviceSpec& device_spec_;
};

bool CanLowerVectorAsAtomicCAS(AtomicRMWOp op, mlir::VectorType vector_type,
                               const DeviceSpec& device_spec) {
  if (vector_type.getRank() != 1 || vector_type.isScalable()) {
    return false;
  }
  Type elem_ty = vector_type.getElementType();
  if (!elem_ty.isIntOrFloat() || GetSubByteBitWidth(elem_ty).has_value()) {
    return false;
  }
  int64_t elem_bits = elem_ty.getIntOrFloatBitWidth();
  if (elem_bits < 8 || elem_bits % 8 != 0) {
    return false;
  }
  int64_t total_bits = vector_type.getNumElements() * elem_bits;
  bool valid_word_size = total_bits == 32 || total_bits == 64 ||
                         (device_spec.IsCpu() && total_bits == 16);
  if (!valid_word_size) {
    return false;
  }
  return GetGuaranteedByteAlignment(op.getInput(), op.getIndices(), op) >=
         (total_bits / 8);
}

class RewriteAtomicRMW : public OpRewritePattern<AtomicRMWOp> {
 public:
  RewriteAtomicRMW(mlir::MLIRContext* context, const DeviceSpec& device_spec)
      : OpRewritePattern<AtomicRMWOp>(context), device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIndices().size() > 1 || op.getInput() == op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only acyclic 0D and 1D tensors are supported");
    }
    if (!TraceDestinationBuffer(op.getInput())) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot trace the destination buffer");
    }
    auto modifier_parameters = GetAtomicModifierParameters(op);
    if (modifier_parameters.has_value() &&
        succeeded(
            rewriteAsDirectAtomicRMW(op, *modifier_parameters, rewriter))) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    if (auto vector_type =
            mlir::dyn_cast<mlir::VectorType>(op.getCurrentValue().getType())) {
      if (succeeded(scalarizeVectorAtomicRMW(op, vector_type, rewriter))) {
        return success();
      }
      if (!CanLowerVectorAsAtomicCAS(op, vector_type, device_spec_)) {
        return rewriter.notifyMatchFailure(
            op, "cannot lower cross lane vector atomic_rmw via atomicCAS");
      }
    } else if (!CanLowerScalarOrComplexAsAtomicCAS(
                   op, op.getInput(), op.getIndices(),
                   op.getCurrentValue().getType(), device_spec_)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported atomic word size for target atomicCAS");
    }
    rewriteAsAtomicCAS(op, rewriter);
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

 private:
  LogicalResult scalarizeVectorAtomicRMW(
      AtomicRMWOp op, mlir::VectorType vector_type,
      mlir::PatternRewriter& rewriter) const {
    if (vector_type.getRank() != 1 || vector_type.isScalable() ||
        op.getIndices().size() != 1) {
      return failure();
    }
    int64_t num_elements = vector_type.getNumElements();
    mlir::Block* body = op.getBody();
    Value yielded = body->getTerminator()->getOperand(0);
    bool is_trivially_removable_body =
        llvm::all_of(body->without_terminator(), [](mlir::Operation& body_op) {
          return mlir::wouldOpBeTriviallyDead(&body_op);
        });
    bool is_invariant_overwrite =
        is_trivially_removable_body && op.isDefinedOutsideOfLoop(yielded);
    if (!is_invariant_overwrite) {
      auto is_matching_vec_or_scalar = [&](Type t) {
        if (auto vt = mlir::dyn_cast<mlir::VectorType>(t)) {
          return vt.getRank() == 1 && !vt.isScalable() &&
                 vt.getNumElements() == num_elements;
        }
        return true;
      };
      bool is_elementwise_body = llvm::all_of(
          body->without_terminator(), [&](mlir::Operation& body_op) {
            if (body_op.use_empty()) {
              return mlir::wouldOpBeTriviallyDead(&body_op);
            }
            bool allowed_elementwise_op =
                (mlir::isa<vector::FMAOp>(body_op) &&
                 ml::isCompatibleType(mlir::getElementTypeOrSelf(
                     body_op.getResult(0).getType()))) ||
                (mlir::OpTrait::hasElementwiseMappableTraits(&body_op) &&
                 body_op.getDialect() != nullptr &&
                 (mlir::isa<arith::ArithDialect>(body_op.getDialect()) ||
                  body_op.getDialect()->getNamespace() == "math"));
            if (!allowed_elementwise_op || body_op.getNumRegions() != 0 ||
                body_op.getNumResults() != 1) {
              return false;
            }
            auto res_vec = mlir::dyn_cast<mlir::VectorType>(
                body_op.getResult(0).getType());
            if (!res_vec || !is_matching_vec_or_scalar(res_vec)) {
              return false;
            }
            return llvm::all_of(body_op.getOperandTypes(),
                                is_matching_vec_or_scalar);
          });
      if (!is_elementwise_body) {
        return failure();
      }
    }

    Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    Value current_tensor = op.getInput();
    SmallVector<Value, 4> indices(op.getIndices().begin(),
                                  op.getIndices().end());
    Value base_minor_index = indices.back();
    for (int64_t i = 0; i < num_elements; ++i) {
      if (i > 0) {
        indices.back() = arith::AddIOp::create(
            b, base_minor_index, arith::ConstantIndexOp::create(b, i));
      }
      auto extract_lane = [&](Value vec) -> Value {
        mlir::DenseElementsAttr dense_attr;
        if (mlir::matchPattern(vec, mlir::m_Constant(&dense_attr))) {
          return arith::ConstantOp::create(
              b, dense_attr.getValues<mlir::TypedAttr>()[i]);
        }
        return vector::ExtractOp::create(b, vec, i);
      };
      if (is_invariant_overwrite) {
        Value elem = extract_lane(yielded);
        current_tensor =
            AtomicStoreOp::create(b, elem, current_tensor, indices).getResult();
        continue;
      }

      mlir::IRMapping lane_map;
      for (mlir::Operation& body_op : body->without_terminator()) {
        if (body_op.use_empty()) {
          continue;
        }
        for (Value operand : body_op.getOperands()) {
          if (op.isDefinedOutsideOfLoop(operand) &&
              mlir::isa<mlir::VectorType>(operand.getType()) &&
              !lane_map.contains(operand)) {
            lane_map.map(operand, extract_lane(operand));
          }
        }
      }

      auto scalar_rmw = AtomicRMWOp::create(b, current_tensor, indices);
      lane_map.map(op.getCurrentValue(), scalar_rmw.getCurrentValue());
      mlir::OpBuilder body_b(scalar_rmw.getBody(), scalar_rmw.getBody()->end(),
                             rewriter.getListener());
      for (mlir::Operation& body_op : body->without_terminator()) {
        if (body_op.use_empty()) {
          continue;
        }
        SmallVector<Value, 3> mapped_operands;
        mapped_operands.reserve(body_op.getNumOperands());
        for (Value operand : body_op.getOperands()) {
          mapped_operands.push_back(lane_map.lookupOrDefault(operand));
        }
        if (mlir::isa<vector::FMAOp>(body_op)) {
          auto fma =
              ml::FMAOp::create(body_b, body_op.getLoc(), mapped_operands[0],
                                mapped_operands[1], mapped_operands[2]);
          lane_map.map(body_op.getResult(0), fma.getResult());
          continue;
        }
        Type scalar_res_ty =
            mlir::getElementTypeOrSelf(body_op.getResult(0).getType());
        mlir::Operation* cloned = mlir::Operation::create(
            body_op.getLoc(), body_op.getName(), {scalar_res_ty},
            mapped_operands, body_op.getDiscardableAttrDictionary(),
            body_op.getPropertiesStorage(), /*successors=*/{},
            /*numRegions=*/0);
        body_b.insert(cloned);
        lane_map.map(body_op.getResult(0), cloned->getResult(0));
      }
      xla::YieldOp::create(body_b, loc, lane_map.lookupOrDefault(yielded));
      current_tensor = scalar_rmw.getResult();
    }
    rewriter.replaceOp(op, current_tensor);
    return success();
  }

  // Certain computations, such as float addition and integer maximization, can
  // be implemented directly with an LLVM atomic instruction. Emits that
  // instruction and returns success when supported, or returns failure
  // otherwise.
  LogicalResult rewriteAsDirectAtomicRMW(
      AtomicRMWOp op, const AtomicModifierParameters& modifier_parameters,
      mlir::PatternRewriter& rewriter) const {
    Type element_type = op.getCurrentValue().getType();
    Location loc = op.getLoc();
    llvm::StringRef sync_scope = GetAtomicSyncScope(device_spec_);
    mlir::ImplicitLocOpBuilder b(loc, rewriter);

    auto materialize_addr_and_modifier = [&]() -> std::pair<Value, Value> {
      Value addr = CreateAtomicGep(op.getInput(), op.getIndices(), b);
      Value modifier_arg = MaterializeAtomicModifier(
          rewriter, loc, modifier_parameters.modifier);
      return {addr, modifier_arg};
    };

    switch (modifier_parameters.kind) {
      case AtomicRMWKind::kAdd:
      case AtomicRMWKind::kSub:
      case AtomicRMWKind::kMaxS:
      case AtomicRMWKind::kMinS:
      case AtomicRMWKind::kMaxU:
      case AtomicRMWKind::kMinU:
      case AtomicRMWKind::kAnd:
      case AtomicRMWKind::kOr:
      case AtomicRMWKind::kXor: {
        if (!IsAtomicIntegral(element_type, device_spec_)) {
          return failure();
        }
        auto [addr, modifier_arg] = materialize_addr_and_modifier();
        ml::AtomicBinOp atomic_bin_op;
        switch (modifier_parameters.kind) {
          case AtomicRMWKind::kAdd:
            atomic_bin_op = ml::AtomicBinOp::add;
            break;
          case AtomicRMWKind::kSub:
            atomic_bin_op = ml::AtomicBinOp::sub;
            break;
          case AtomicRMWKind::kMaxS:
            atomic_bin_op = ml::AtomicBinOp::max;
            break;
          case AtomicRMWKind::kMinS:
            atomic_bin_op = ml::AtomicBinOp::min;
            break;
          case AtomicRMWKind::kMaxU:
            atomic_bin_op = ml::AtomicBinOp::umax;
            break;
          case AtomicRMWKind::kMinU:
            atomic_bin_op = ml::AtomicBinOp::umin;
            break;
          case AtomicRMWKind::kAnd:
            atomic_bin_op = ml::AtomicBinOp::_and;
            break;
          case AtomicRMWKind::kOr:
            atomic_bin_op = ml::AtomicBinOp::_or;
            break;
          case AtomicRMWKind::kXor:
            atomic_bin_op = ml::AtomicBinOp::_xor;
            break;
          default:
            llvm_unreachable("unexpected integer atomic kind");
        }
        auto rmw_op = ml::AtomicRMWOp::create(
            rewriter, loc, atomic_bin_op, addr, modifier_arg,
            ml::AtomicOrdering::monotonic, sync_scope);
        SetAmdAtomicRMWAttrs(rewriter, rmw_op, device_spec_);
        return success();
      }
      case AtomicRMWKind::kFAdd:
      case AtomicRMWKind::kFSub: {
        if (device_spec_.IsCpu()) {
          return failure();
        }
        bool negated = (modifier_parameters.kind == AtomicRMWKind::kFSub);
        if (device_spec_.IsAmdGpu()) {
          return emitAMDAtomicFAdd(
              loc, op, modifier_parameters.modifier, element_type,
              device_spec_.gpu().rocm_compute_capability(), rewriter, negated);
        }
        if (device_spec_.IsIntelGpu()) {
          return emitSPIRVAtomicFAdd(loc, op, modifier_parameters.modifier,
                                     element_type, rewriter, negated);
        }
        return emitNVidiaAtomicFAdd(
            loc, op, modifier_parameters.modifier, element_type,
            device_spec_.gpu().cuda_compute_capability(), rewriter, negated);
      }
      case AtomicRMWKind::kFMax: {
        if (device_spec_.IsCpu() || !element_type.isF32()) {
          return failure();
        }
        auto [addr, modifier_arg] = materialize_addr_and_modifier();
        rewriteAtomicFMaxAsIntAtomics(loc, modifier_arg, addr, sync_scope,
                                      rewriter);
        return success();
      }
      case AtomicRMWKind::kFMin:
        return failure();
    }
    llvm_unreachable("unhandled AtomicRMWKind");
  }

  LogicalResult emitNVidiaAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      Type element_type,
      const se::CudaComputeCapability& cuda_compute_capability, OpBuilder& b,
      bool negated) const {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(element_type)) {
      return emitNvidiaVectorizedAtomicFAdd(
          loc, op, modifier, vector_type, cuda_compute_capability, b, negated);
    }
    // "atom.add.f64 requires sm_60 or higher."
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
    bool is_supported_f16_atomic =
        element_type.isF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kVolta);
    bool is_supported_bf16_atomic =
        element_type.isBF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kHopper);
    bool is_supported_f64_atomic =
        element_type.isF64() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kPascal);
    if (!element_type.isF32() && !is_supported_f16_atomic &&
        !is_supported_bf16_atomic && !is_supported_f64_atomic) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr = CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    auto rmw_op =
        ml::AtomicRMWOp::create(b, loc, ml::AtomicBinOp::fadd, addr,
                                modifier_arg, ml::AtomicOrdering::monotonic);
    rmw_op.setIgnoreDenormalMode(true);
    return success();
  }

  LogicalResult emitNvidiaVectorizedAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      mlir::VectorType vector_type,
      const se::CudaComputeCapability& cuda_compute_capability, OpBuilder& b,
      bool negated) const {
    // Hopper supports 2 element and 4 element vectors of f32 in global memory.
    if (!(vector_type.getElementType().isF32() &&
          (vector_type.getNumElements() == 2 ||
           vector_type.getNumElements() == 4) &&
          cuda_compute_capability.IsAtLeast(
              se::CudaComputeCapability::kHopper)) ||
        IsSharedMemoryBuffer(op.getInput())) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr = CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    // TODO(https://github.com/llvm/llvm-project/issues/122760): Switch to
    // AtomicRMWOp once the bug is fixed. Currently vectorized AtomicRMWOp
    // lowers incorrectly to PTX due to a bug in NVPTX.  We use inline asm to
    // work around this.
    auto asmDialectAttr =
        ml::AsmDialectAttr::get(b.getContext(), ml::AsmDialect::AD_ATT);
    std::string asm_string;
    std::string constraints;
    if (vector_type.getNumElements() == 2) {
      asm_string = "atom.global.v2.f32.add {$0, $1}, [$2], {$3, $4};";
      constraints = "=f,=f,l,f,f";
    } else if (vector_type.getNumElements() == 4) {
      asm_string =
          "atom.global.v4.f32.add {$0, $1, $2, $3}, [$4], {$5, $6, $7, $8};";
      constraints = "=f,=f,=f,=f,l,f,f,f,f";
    }
    SmallVector<Value> asm_operands{addr};
    for (int i = 0; i < vector_type.getNumElements(); ++i) {
      asm_operands.push_back(vector::ExtractOp::create(
          b, loc, modifier_arg,
          arith::ConstantIndexOp::create(b, loc, i).getResult()));
    }
    SmallVector<Type> outputTypes(vector_type.getNumElements(),
                                  vector_type.getElementType());
    auto outputType =
        ml::LLVMStructType::getLiteral(b.getContext(), outputTypes);
    ml::InlineAsmOp::create(b, loc, outputType, asm_operands, asm_string,
                            constraints,
                            /*has_side_effects=*/true,
                            /*is_align_stack=*/true, ml::TailCallKind::None,
                            /*convergent=*/false, asmDialectAttr,
                            /*operand_attrs=*/mlir::ArrayAttr());
    return success();
  }

  LogicalResult emitAMDAtomicFAdd(
      Location loc, AtomicRMWOp op, mlir::OpFoldResult modifier,
      Type element_type,
      const se::RocmComputeCapability& rocm_compute_capability, OpBuilder& b,
      bool negated) const {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(element_type)) {
      bool supported_v2f16 =
          vector_type.getElementType().isF16() &&
          rocm_compute_capability.has_packed_fp16_atomics_support();
      bool supported_v2bf16 =
          vector_type.getElementType().isBF16() &&
          rocm_compute_capability.has_packed_bf16_atomics_support();
      if (vector_type.getNumElements() != 2 ||
          (!supported_v2f16 && !supported_v2bf16) ||
          IsSharedMemoryBuffer(op.getInput()) ||
          GetGuaranteedByteAlignment(op.getInput(), op.getIndices(), op) < 4) {
        return failure();
      }
    } else if (!element_type.isF32() && !element_type.isF16() &&
               !element_type.isBF16() && !element_type.isF64()) {
      return failure();
    } else if ((element_type.isF16() || element_type.isBF16()) &&
               IsSharedMemoryBuffer(op.getInput())) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr = CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    if ((element_type.isF16() &&
         rocm_compute_capability.has_packed_fp16_atomics_support()) ||
        (element_type.isBF16() &&
         rocm_compute_capability.has_packed_bf16_atomics_support())) {
      auto packed_type = mlir::VectorType::get({2}, element_type);
      auto i64_type = b.getI64Type();
      auto i32_type = b.getI32Type();
      auto i16_type = b.getI16Type();
      Value addr_int = ml::PtrToIntOp::create(b, loc, i64_type, addr);
      Value addr_masked = ml::AndOp::create(
          b, loc, addr_int, ml::ConstantOp::create(b, loc, i64_type, -4));

      Value offset = ml::AndOp::create(
          b, loc, ml::TruncOp::create(b, loc, i32_type, addr_int),
          ml::ConstantOp::create(b, loc, i32_type, 2));

      Value shift = ml::MulOp::create(
          b, loc, offset, ml::ConstantOp::create(b, loc, i32_type, 8));

      Value modifier_int =
          ml::BitcastOp::create(b, loc, i16_type, modifier_arg);

      Value modifier_masked = ml::ShlOp::create(
          b, loc, ml::ZExtOp::create(b, loc, i32_type, modifier_int), shift);

      constexpr int kGlobalMemory = 1;
      addr = ml::IntToPtrOp::create(
          b, loc, ml::LLVMPointerType::get(b.getContext(), kGlobalMemory),
          addr_masked);

      modifier_arg =
          ml::BitcastOp::create(b, loc, packed_type, modifier_masked);
    }

    auto rmw_op = ml::AtomicRMWOp::create(
        b, loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
        ml::AtomicOrdering::monotonic, GetAtomicSyncScope(device_spec_));
    rmw_op.setIgnoreDenormalMode(true);
    SetAmdAtomicRMWAttrs(b, rmw_op, device_spec_);

    return success();
  }

  LogicalResult emitSPIRVAtomicFAdd(Location loc, AtomicRMWOp op,
                                    mlir::OpFoldResult modifier,
                                    Type element_type, OpBuilder& b,
                                    bool negated) const {
    if (mlir::isa<mlir::VectorType>(element_type)) {
      return failure();
    }
    // TODO(intel-tf): Expand support to 16 bit floating precisions on supported
    // platforms.
    if (!element_type.isF32() && !element_type.isF64()) {
      return failure();
    }
    mlir::ImplicitLocOpBuilder implicit_b(loc, b);
    Value addr = CreateAtomicGep(op.getInput(), op.getIndices(), implicit_b);
    Value modifier_arg = MaterializeAtomicModifier(b, loc, modifier);
    if (negated) {
      modifier_arg = arith::NegFOp::create(b, loc, modifier_arg);
    }

    auto rmw_op =
        ml::AtomicRMWOp::create(b, loc, ml::AtomicBinOp::fadd, addr,
                                modifier_arg, ml::AtomicOrdering::monotonic);
    rmw_op.setIgnoreDenormalMode(true);
    return success();
  }

  void rewriteAtomicFMaxAsIntAtomics(Location loc, Value modifier_arg,
                                     Value addr, llvm::StringRef sync_scope,
                                     OpBuilder& b) const {
    Type element_type = modifier_arg.getType();
    CHECK(element_type.isF32());
    // Evaluating floating max using integer atomics has the limitation of not
    // propagating -NaNs. To handle this, we check if the update value is -NaN
    // and convert it to a positive one by dropping the sign-bit.
    Value current = ml::LoadOp::create(b, loc, element_type, addr);

    Value current_is_nan =
        ml::FCmpOp::create(b, loc, ml::FCmpPredicate::uno, current, current);
    auto is_current_nan =
        scf::IfOp::create(b, loc, /*resultTypes=*/TypeRange{}, current_is_nan,
                          /*addThenBlock=*/true, /*addElseBlock=*/true);
    auto if_current_nan_then_builder =
        OpBuilder::atBlockEnd(is_current_nan.thenBlock(), b.getListener());
    scf::YieldOp::create(if_current_nan_then_builder, loc);

    auto if_current_nan_else_builder =
        OpBuilder::atBlockEnd(is_current_nan.elseBlock(), b.getListener());
    Value is_modifier_nan =
        ml::FCmpOp::create(if_current_nan_else_builder, loc,
                           ml::FCmpPredicate::uno, modifier_arg, modifier_arg);
    auto f32_nan = mlir::APFloat::getNaN(mlir::APFloat::IEEEsingle());
    Value nan = ml::ConstantOp::create(if_current_nan_else_builder, loc,
                                       b.getF32Type(), f32_nan);
    Value no_negative_nan_source = ml::SelectOp::create(
        if_current_nan_else_builder, loc, is_modifier_nan, nan, modifier_arg);
    Value current_less_than_modifier = ml::FCmpOp::create(
        if_current_nan_else_builder, loc, ml::FCmpPredicate::ult, current,
        no_negative_nan_source);

    // This check allows us to skip the atomic update all-together at the
    // expense of reading the value in memory for every update. Evaluated
    // against Waymo's benchmarks, adding the check achieves better overall
    // performance.
    auto if_need_update = scf::IfOp::create(if_current_nan_else_builder, loc,
                                            /*resultTypes=*/TypeRange{},
                                            current_less_than_modifier,
                                            /*withElseRegion=*/true,
                                            /*addElseBlock=*/false);
    scf::YieldOp::create(if_current_nan_else_builder, loc);

    auto then_builder =
        OpBuilder::atBlockEnd(if_need_update.thenBlock(), b.getListener());
    Value source_float_as_int = arith::BitcastOp::create(
        then_builder, loc, then_builder.getI32Type(), no_negative_nan_source);
    Value c0 = ml::ConstantOp::create(then_builder, loc, b.getI32Type(), 0);
    Value is_not_negative = ml::ICmpOp::create(
        then_builder, loc, ml::ICmpPredicate::sge, source_float_as_int, c0);
    scf::IfOp::create(
        then_builder, loc, is_not_negative,
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMax((int *)address, __float_as_int(val))
          auto rmw_op = ml::AtomicRMWOp::create(
              nested_b, loc, ml::AtomicBinOp::max, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          SetAmdAtomicRMWAttrs(nested_b, rmw_op, device_spec_);
          scf::YieldOp::create(nested_b, nested_loc);
        },
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMin((unsigned int *)address, __float_as_uint(val))
          auto rmw_op = ml::AtomicRMWOp::create(
              nested_b, loc, ml::AtomicBinOp::umin, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          SetAmdAtomicRMWAttrs(nested_b, rmw_op, device_spec_);
          scf::YieldOp::create(nested_b, nested_loc);
        });
    scf::YieldOp::create(then_builder, loc);
  }

  // Implements atomic binary operations using atomicCAS: loads the containing
  // word, applies the region computation to the extracted element, and retries
  // cmpxchg until it succeeds.
  void rewriteAsAtomicCAS(AtomicRMWOp op,
                          mlir::PatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    AtomicWordAccess access = PrepareAtomicWordAccess(
        op, op.getInput(), op.getIndices(), op.getCurrentValue().getType(),
        device_spec_, rewriter);
    llvm::StringRef sync_scope = GetAtomicSyncScope(device_spec_);

    // Load initial atomic value and create the loop.
    Value initial =
        ml::LoadOp::create(rewriter, loc, access.atomic_ty, access.addr);
    scf::WhileOp::create(
        rewriter, loc, TypeRange{access.atomic_ty}, ValueRange{initial},
        [&](mlir::OpBuilder& builder, Location loop_loc, ValueRange values) {
          mlir::ImplicitLocOpBuilder b(loop_loc, builder);
          Value old_value = values[0];

          // Convert atomic value to input value.
          Value input_value;
          if (access.small_type) {
            Value short_value = ml::TruncOp::create(
                b, b.getIntegerType(static_cast<unsigned>(access.result_size)),
                ml::LShrOp::create(b, old_value, access.shift));
            input_value = CreateBitcast(b, short_value, access.result_ty);
          } else {
            input_value = CreateBitcast(b, old_value, access.result_ty);
          }

          // Perform computation on the loaded input value.
          rewriter.mergeBlocks(&op.getComputation().front(), b.getBlock(),
                               {input_value});
          auto yield_op = b.getBlock()->getTerminator();
          Value result = yield_op->getOperand(0);
          rewriter.eraseOp(yield_op);

          // Convert resulting value to atomic value.
          Value new_value;
          if (access.small_type) {
            Value cast_value = ml::ZExtOp::create(
                b, access.atomic_ty,
                CreateBitcast(b, result,
                              rewriter.getIntegerType(
                                  static_cast<unsigned>(access.result_size))));
            new_value = ml::OrOp::create(
                b, ml::AndOp::create(b, old_value, access.mask),
                ml::ShlOp::create(b, cast_value, access.shift));
          } else {
            new_value = CreateBitcast(b, result, access.atomic_ty);
          }

          // Try saving the result atomically, retry if failed.
          Value cmpxchg = ml::AtomicCmpXchgOp::create(
              b, loop_loc, access.addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::monotonic,
              /*failure_ordering=*/ml::AtomicOrdering::monotonic, sync_scope);
          Value next = ml::ExtractValueOp::create(b, cmpxchg, 0);
          Value ok = ml::ExtractValueOp::create(b, cmpxchg, 1);
          Value low_bit =
              ml::ConstantOp::create(b, b.getOneAttr(b.getI1Type()));
          Value not_ok = ml::XOrOp::create(b, ok, low_bit);
          scf::ConditionOp::create(b, not_ok, ValueRange{next});
        },
        [](mlir::OpBuilder& b, Location loop_loc, ValueRange values) {
          scf::YieldOp::create(b, loop_loc, values);
        });
  }

  const DeviceSpec& device_spec_;
};

struct DynamicDimMetadataInfo {
  Value tensor_ptr;
  int64_t offset_bytes;
  std::optional<int> sub_byte_width;
};

std::optional<DynamicDimMetadataInfo> GetDynamicDimMetadataInfo(
    mlir::ImplicitLocOpBuilder& b, mlir::Operation* op, Value tensor,
    int64_t dim) {
  Value base_tensor = GetDestinationBuffer(tensor);
  if (!base_tensor) {
    return std::nullopt;
  }
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());

  Type element_type = tensor_type.getElementType();
  int64_t num_elements = tensor_type.getNumElements();
  std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type);
  if (sub_byte_width) {
    element_type = b.getI8Type();
    // Elements are packed.
    num_elements = CeilOfRatio<int64_t>(num_elements, 8 / *sub_byte_width);
  }

  // The offset of the dim size from the start of the buffer. The dynamic dim
  // sizes are stored after the tensor data as a tail-allocated metadata of
  // s32 type.
  mlir::LLVMTypeConverter converter(op->getContext());
  Type llvm_element_type = converter.convertType(element_type);
  int64_t element_size_bytes =
      static_cast<int64_t>(mlir::DataLayout::closest(op)
                               .getTypeSize(llvm_element_type)
                               .getFixedValue());
  int64_t offset_bytes =
      num_elements * element_size_bytes + dim * sizeof(int32_t);

  auto ptr_type = ml::LLVMPointerType::get(b.getContext());
  Value tensor_ptr =
      UnrealizedConversionCastOp::create(b, ptr_type, base_tensor).getResult(0);
  return DynamicDimMetadataInfo{tensor_ptr, offset_bytes, sub_byte_width};
}

Value AddByteOffsetToPointer(mlir::ImplicitLocOpBuilder& b, Value ptr,
                             int64_t offset_bytes) {
  if (offset_bytes == 0) {
    return ptr;
  }
  auto ptr_type = ml::LLVMPointerType::get(b.getContext());
  Value addr_int = ml::PtrToIntOp::create(b, b.getI64Type(), ptr);
  Value addr_offset = ml::ConstantOp::create(b, b.getI64Type(), offset_bytes);
  Value offset_addr_int = ml::AddOp::create(b, addr_int, addr_offset);
  return ml::IntToPtrOp::create(b, ptr_type, offset_addr_int);
}

class RewriteGetDynamicDimSize : public OpRewritePattern<GetDynamicDimSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      GetDynamicDimSizeOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    std::optional<DynamicDimMetadataInfo> info = GetDynamicDimMetadataInfo(
        b, op, op.getTensor(), static_cast<int64_t>(op.getDim()));
    if (!info) {
      return failure();
    }
    Value metadata_addr =
        AddByteOffsetToPointer(b, info->tensor_ptr, info->offset_bytes);
    int64_t alignment = info->offset_bytes % 4 == 0 ? 4 : 1;
    rewriter.replaceOpWithNewOp<ml::LoadOp>(op, b.getI32Type(), metadata_addr,
                                            alignment);
    return success();
  }
};

class RewriteSetDynamicDimSize : public OpRewritePattern<SetDynamicDimSizeOp> {
 public:
  RewriteSetDynamicDimSize(mlir::MLIRContext* context,
                           const DeviceSpec& device_spec)
      : OpRewritePattern<SetDynamicDimSizeOp>(context),
        device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      SetDynamicDimSizeOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    int64_t dim = static_cast<int64_t>(op.getDim());
    std::optional<DynamicDimMetadataInfo> info =
        GetDynamicDimMetadataInfo(b, op, op.getTensor(), dim);
    if (!info) {
      return failure();
    }

    int64_t misalign = info->offset_bytes % 4;
    if (!device_spec_.IsCpu() && info->sub_byte_width && dim == 0 &&
        misalign != 0) {
      // On GPU, sub-byte writes use 32-bit cmpxchg on the enclosing aligned
      // word. When dim 0 starts mid-word, update the overlapping aligned word
      // with cmpxchg so concurrent tail writes cannot clobber the low bytes of
      // the size, then write only the remaining bytes in the next word.
      Type i32_type = b.getI32Type();
      int64_t aligned_offset = info->offset_bytes - misalign;
      int low_bits = static_cast<int>(misalign * 8);
      uint32_t low_mask_val = (uint32_t{1} << low_bits) - 1;

      Value aligned_addr =
          AddByteOffsetToPointer(b, info->tensor_ptr, aligned_offset);
      Value low_mask = ml::ConstantOp::create(
          b, i32_type, static_cast<int32_t>(low_mask_val));
      Value low_shift = ml::ConstantOp::create(b, i32_type, low_bits);
      Value shifted_size = ml::ShlOp::create(b, op.getSize(), low_shift);
      Value initial = ml::LoadOp::create(b, i32_type, aligned_addr);

      llvm::StringRef sync_scope = device_spec_.IsAmdGpu()
                                       ? llvm::StringRef("agent-one-as")
                                       : llvm::StringRef();
      scf::WhileOp::create(
          b, TypeRange{i32_type}, ValueRange{initial},
          [&](mlir::OpBuilder& builder, Location loc, ValueRange values) {
            mlir::ImplicitLocOpBuilder loop_b(loc, builder);
            Value old_value = values[0];
            Value preserved = ml::AndOp::create(loop_b, old_value, low_mask);
            Value new_value = ml::OrOp::create(loop_b, preserved, shifted_size);
            Value cmpxchg = ml::AtomicCmpXchgOp::create(
                loop_b, loc, aligned_addr, old_value, new_value,
                /*success_ordering=*/ml::AtomicOrdering::monotonic,
                /*failure_ordering=*/ml::AtomicOrdering::monotonic, sync_scope);
            Value next = ml::ExtractValueOp::create(loop_b, cmpxchg, 0);
            Value ok = ml::ExtractValueOp::create(loop_b, cmpxchg, 1);
            Value low_bit = ml::ConstantOp::create(
                loop_b, loop_b.getOneAttr(loop_b.getI1Type()));
            Value not_ok = ml::XOrOp::create(loop_b, ok, low_bit);
            scf::ConditionOp::create(loop_b, not_ok, ValueRange{next});
          },
          [&](mlir::OpBuilder& builder, Location loc, ValueRange values) {
            scf::YieldOp::create(builder, loc, values);
          });

      Value tail_addr =
          AddByteOffsetToPointer(b, info->tensor_ptr, aligned_offset + 4);
      int high_shift_bits = 32 - low_bits;
      Value high_shifted = ml::LShrOp::create(
          b, op.getSize(),
          ml::ConstantOp::create(b, i32_type, high_shift_bits));
      if (misalign == 1) {
        Value byte_val = ml::TruncOp::create(b, b.getI8Type(), high_shifted);
        ml::StoreOp::create(b, byte_val, tail_addr, /*alignment=*/1);
      } else if (misalign == 2) {
        Value i16_val = ml::TruncOp::create(b, b.getI16Type(), high_shifted);
        ml::StoreOp::create(b, i16_val, tail_addr, /*alignment=*/2);
      } else {
        Value low_i16 = ml::TruncOp::create(b, b.getI16Type(), high_shifted);
        ml::StoreOp::create(b, low_i16, tail_addr, /*alignment=*/2);
        Value top_byte = ml::TruncOp::create(
            b, b.getI8Type(),
            ml::LShrOp::create(b, high_shifted,
                               ml::ConstantOp::create(b, i32_type, 16)));
        Value top_addr =
            AddByteOffsetToPointer(b, info->tensor_ptr, aligned_offset + 6);
        ml::StoreOp::create(b, top_byte, top_addr, /*alignment=*/1);
      }
    } else {
      Value metadata_addr =
          AddByteOffsetToPointer(b, info->tensor_ptr, info->offset_bytes);
      int64_t alignment = misalign == 0 ? 4 : 1;
      ml::StoreOp::create(b, op.getSize(), metadata_addr, alignment);
    }

    rewriter.replaceOp(op, op.getTensor());
    return success();
  }

 private:
  const DeviceSpec& device_spec_;
};

// Uses an explicit region/block worklist rather than `Operation::walk` to avoid
// instantiating `llvm::early_inc_iterator_impl` (-Wshadow-field in clang-tidy).
bool ReportUnloweredAtomics(mlir::Operation* root) {
  SmallVector<mlir::Operation*, 16> worklist{root};
  bool has_unlowered_atomic = false;
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (mlir::isa<AtomicRMWOp, AtomicStoreOp>(op)) {
      op->emitOpError("failed to lower to LLVM atomic operation");
      has_unlowered_atomic = true;
    }
    for (mlir::Region& region : op->getRegions()) {
      for (mlir::Block& block : region) {
        for (mlir::Operation& child : block) {
          worklist.push_back(&child);
        }
      }
    }
  }
  return has_unlowered_atomic;
}

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  LowerTensorsPass() = default;

  explicit LowerTensorsPass(const LowerTensorsPassOptions& options)
      : LowerTensorsPassBase(options) {}

  explicit LowerTensorsPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (target_type_ == "gpu" && !gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      absl::StatusOr<se::DeviceDescription> device_description =
          se::DeviceDescription::FromProto(device_info);
      CHECK_OK(device_description.status());
      *device_spec_.mutable_type() = *device_description;
    } else if (target_type_ == "cpu") {
      CHECK(gpu_device_info_.empty());
      *device_spec_.mutable_type() = CpuDeviceSpec{};
    }

    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet tensor_patterns(mlir_context);

    AtomicRMWOp::getCanonicalizationPatterns(tensor_patterns, mlir_context);
    tensor_patterns.add<RewriteAtomicRMW, RewriteAtomicStore,
                        RewriteSetDynamicDimSize, RewriteTensorInsert>(
        mlir_context, device_spec_);
    tensor_patterns
        .add<RewriteAllocateShared, RewriteCopySlice, RewriteGetDynamicDimSize,
             RewriteNonScalarConstants, RewriteSyncThreads,
             RewriteTensorExtract, RewriteTransferRead, RewriteTransferWrite>(
            mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(tensor_patterns)))) {
      signalPassFailure();
      return;
    }

    if (ReportUnloweredAtomics(getOperation())) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet function_patterns(mlir_context);
    function_patterns.add<RewriteFunctionSignatures>(mlir_context,
                                                     device_spec_);
    function_patterns
        .add<RewriteCall, RemoveUnusedIndexSwitchResults, RewriteFor>(
            mlir_context);
    scf::ForOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    scf::IfOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(function_patterns)))) {
      signalPassFailure();
      return;
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> createLowerTensorsPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerTensorsPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
