/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "deallocation/IR/deallocation_ops.h"

#include <optional>

#include "deallocation/IR/deallocation_dialect.cc.inc"
#include "deallocation/utils/util.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace deallocation {

void DeallocationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "deallocation/IR/deallocation_ops.cc.inc"
#undef GET_OP_LIST
      >();
}

namespace {

LogicalResult retainNoOp(RetainOp op, PatternRewriter& rewriter) {
  if (op.getAllocs().size() != 1 || op.getAllocs() != op.getRetained()) {
    return failure();
  }
  rewriter.replaceOp(op, op.getAllocs());
  return success();
}

enum AllocNullability : uint32_t {
  UNDEFINED = 0,
  ALWAYS_NULL = 1,
  NEVER_NULL = 2,
  SOMETIMES_NULL = 3
};

AllocNullability operator|=(AllocNullability& lhs, AllocNullability rhs) {
  return lhs = static_cast<AllocNullability>(static_cast<uint32_t>(lhs) | rhs);
}

// Returns the nullability of `v`. `pending` contains a set of `Values` we're
// already considering in the computation of some value's nullability. It is
// assumed that we will eventually take the maximum (logical or) of all
// nullability values in this set.
AllocNullability getAllocNullabilityImpl(Value v,
                                         llvm::DenseSet<Value>& pending) {
  if (llvm::isa_and_present<memref::AllocOp>(v.getDefiningOp())) {
    return NEVER_NULL;
  }

  if (llvm::isa_and_present<deallocation::NullOp>(v.getDefiningOp())) {
    return ALWAYS_NULL;
  }

  if (auto retain =
          llvm::dyn_cast_or_null<deallocation::RetainOp>(v.getDefiningOp())) {
    // We start with ALWAYS_NULL because a retain without any allocs is null.
    // Also, because a retain with a non-null alloc can be null (otherwise, this
    // would have been cleaned up by `retainNoOp`).
    AllocNullability nullability = ALWAYS_NULL;
    for (auto alloc : retain.getAllocs()) {
      if (pending.insert(alloc).second) {
        nullability |= getAllocNullabilityImpl(alloc, pending);
      }
      if (nullability == SOMETIMES_NULL) break;
    }
    return nullability;
  }

  if (llvm::isa_and_present<memref::SubViewOp, memref::CastOp,
                            memref::ExpandShapeOp, memref::CollapseShapeOp,
                            memref::ReshapeOp, memref::ViewOp,
                            memref::ReinterpretCastOp, memref::TransposeOp>(
          v.getDefiningOp())) {
    return getAllocNullabilityImpl(v.getDefiningOp()->getOperand(0), pending);
  }

  // Returns the nullability of an operand in each of the region's predecessors.
  auto getPredecessorNullability =
      [&](RegionBranchOpInterface rbi,
          std::optional<int64_t> successorRegionIndex,
          int64_t successorArgIndex) {
        AllocNullability nullability = UNDEFINED;
        for (const auto& pred :
             getPredecessorRegions(rbi, successorRegionIndex)) {
          Value operand = pred.getPredecessorOperand(successorArgIndex);
          // It is safe to skip values that are already being considered higher
          // up in the call stack, because we end up taking the maximum of all
          // nullability values.
          if (pending.insert(operand).second) {
            nullability |= getAllocNullabilityImpl(operand, pending);
          }
          if (nullability == SOMETIMES_NULL) break;
        }
        return nullability;
      };

  // If `v` is a block argument, check all incoming edges.
  if (auto bbarg = v.dyn_cast<BlockArgument>()) {
    if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(
            bbarg.getParentRegion()->getParentOp())) {
      return getPredecessorNullability(
          rbi, bbarg.getParentRegion()->getRegionNumber(),
          bbarg.getArgNumber());
    }
  }

  if (auto rbi =
          llvm::dyn_cast_or_null<RegionBranchOpInterface>(v.getDefiningOp())) {
    return getPredecessorNullability(rbi, std::nullopt,
                                     llvm::cast<OpResult>(v).getResultNumber());
  }

  // Something we don't understand.
  return AllocNullability::SOMETIMES_NULL;
}

bool allocIsNonNull(Value v) {
  llvm::DenseSet<Value> pendingChecks;
  return getAllocNullabilityImpl(v, pendingChecks) == NEVER_NULL;
}

bool allocIsNull(Value v) {
  llvm::DenseSet<Value> pendingChecks;
  return getAllocNullabilityImpl(v, pendingChecks) == ALWAYS_NULL;
}

LogicalResult retainIsDealloc(RetainOp op, PatternRewriter& rewriter) {
  if (!op.getRetained().empty() || op.getAllocs().size() != 1 ||
      !allocIsNonNull(op.getAllocs()[0])) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, op.getAllocs()[0]);
  return success();
}

LogicalResult retainIsNull(RetainOp op, PatternRewriter& rewriter) {
  // If all allocs are null, the result is null and there is nothing to
  // deallocate.
  if (!llvm::all_of(op.getAllocs(), allocIsNull)) {
    return failure();
  }

  auto nulls = llvm::to_vector(
      llvm::map_range(TypeRange{op.getRetained()}, [&](Type ty) -> Value {
        return rewriter.create<NullOp>(op.getLoc(), getUnrankedMemrefType(ty));
      }));
  rewriter.replaceOp(op, nulls);
  return success();
}

LogicalResult splitRetain(RetainOp op, PatternRewriter& rewriter) {
  if (!op.getRetained().empty() || op.getAllocs().size() <= 1) {
    return failure();
  }
  for (Value alloc : op.getAllocs()) {
    rewriter.create<deallocation::RetainOp>(op.getLoc(), TypeRange{},
                                            ValueRange{}, ValueRange{alloc});
  }
  op.erase();
  return success();
}

}  // namespace

void RetainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext*) {
  results.add(retainNoOp, 2);
  results.add(retainIsDealloc, 2);
  results.add(splitRetain, 2);
  // Run the above analyses first. They make retainIsNull cheaper.
  results.add(retainIsNull, 1);
}

LogicalResult RetainOp::verify() {
  Type elemTy = getElementTypeOrSelf(getOperandTypes().front());
  if (!llvm::all_of(
          getOperandTypes(),
          [&](Type it) { return getElementTypeOrSelf(it) == elemTy; }) ||
      !llvm::all_of(getResultTypes(), [&](Type it) {
        return getElementTypeOrSelf(it) == elemTy;
      })) {
    return emitOpError()
           << "expected homogeneous operand and result element type";
  }
  return success();
}

}  // namespace deallocation
}  // namespace mlir

#define GET_OP_CLASSES
#include "deallocation/IR/deallocation_ops.cc.inc"
