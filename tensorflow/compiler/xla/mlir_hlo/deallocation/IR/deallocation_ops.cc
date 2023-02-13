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
#include <vector>

#include "deallocation/IR/deallocation_dialect.cc.inc"
#include "deallocation/utils/util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

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

LogicalResult retainOfNothing(RetainOp op, PatternRewriter& rewriter) {
  if (!op.getAllocs().empty()) {
    return failure();
  }

  auto nulls = llvm::to_vector(
      llvm::map_range(TypeRange{op.getRetained()}, [&](Type ty) -> Value {
        return rewriter.create<NullOp>(op.getLoc(), ty);
      }));
  rewriter.replaceOp(op, nulls);
  return success();
}

LogicalResult retainNoOp(RetainOp op, PatternRewriter& rewriter) {
  if (op.getAllocs().size() != 1 || op.getAllocs() != op.getRetained()) {
    return failure();
  }
  rewriter.replaceOp(op, op.getAllocs());
  return success();
}

bool allocIsNonNullImpl(Value v, llvm::DenseSet<Value>& pending) {
  if (llvm::isa_and_present<memref::AllocOp>(v.getDefiningOp())) {
    return true;
  }

  // If v is a block argument, check all incoming edges.
  if (auto bbarg = v.dyn_cast<BlockArgument>()) {
    if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(
            bbarg.getParentRegion()->getParentOp())) {
      for (auto pred : getPredecessorRegions(
               rbi, bbarg.getParentRegion()->getRegionNumber())) {
        Value operand = pred.getPredecessorOperand(bbarg.getArgNumber());

        if (pending.insert(operand).second &&
            !allocIsNonNullImpl(operand, pending)) {
          return false;
        }
      }
      return true;
    }
  }

  if (auto op =
          llvm::dyn_cast_or_null<RegionBranchOpInterface>(v.getDefiningOp())) {
    unsigned resultNumber = llvm::cast<OpResult>(v).getResultNumber();
    for (const auto& exit : getPredecessorRegions(op, std::nullopt)) {
      if (!allocIsNonNullImpl(exit.getPredecessorOperand(resultNumber),
                              pending))
        return false;
    }
    return true;
  }

  return false;
}

bool allocIsNonNull(Value v) {
  llvm::DenseSet<Value> pendingChecks;
  return allocIsNonNullImpl(v, pendingChecks);
}

LogicalResult retainIsDealloc(RetainOp op, PatternRewriter& rewriter) {
  if (!op.getRetained().empty()) {
    return failure();
  }
  bool deletedSome = false;
  for (auto i : llvm::reverse(llvm::seq<size_t>(0, op.getAllocs().size()))) {
    if (allocIsNonNull(op.getAllocs()[i])) {
      rewriter.setInsertionPoint(op);
      rewriter.create<memref::DeallocOp>(op.getLoc(), op.getAllocs()[i]);
      op.getAllocsMutable().erase(i);
      deletedSome = true;
    }
  }
  return success(deletedSome);
}

}  // namespace

void RetainOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext*) {
  results.add(retainOfNothing);
  results.add(retainNoOp);
  results.add(retainIsDealloc);
}

}  // namespace deallocation
}  // namespace mlir

#define GET_OP_CLASSES
#include "deallocation/IR/deallocation_ops.cc.inc"
