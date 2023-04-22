/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the LMHLO dialect.

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <unordered_set>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc"
#include "mlir-hlo/utils/lhlo_utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace lmhlo {

LmhloDialect::LmhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<LmhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AbsOp op) {
  auto operand_type = getElementTypeOrSelf(op.input().getType());
  auto output_type = getElementTypeOrSelf(op.output().getType());
  if (auto complex_type = operand_type.dyn_cast<ComplexType>()) {
    if (complex_type.getElementType() != output_type) {
      return op.emitOpError(
          "requires output type to be the same as the element type of the "
          "input");
    }
    return success();
  }
  if (operand_type != output_type)
    return op.emitOpError("requires all operands to have the same type");
  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

// TODO(jurahul): Add verification for output shape.
static LogicalResult Verify(AllGatherOp op) {
  return VerifyReplicaGroups(op, /*is_uniform_sized=*/true);
}

// TODO(jurahul): Add verification for output shape.
static LogicalResult Verify(AllToAllOp op) {
  return VerifyReplicaGroups(op, /*is_uniform_sized=*/true);
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AllReduceOp op) { return VerifyAllReduce(op); }

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ReduceScatterOp op) {
  if (failed(VerifyReplicaGroups(op, /*is_uniform_sized=*/true)))
    return failure();
  if (failed(mlir::hlo::VerifyReduceScatter(
          op, /*operand_types=*/op.operands().getTypes(),
          /*result_types=*/op.results().getTypes(),
          /*scatter_dimension=*/op.scatter_dimension())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// CaseOp
//===----------------------------------------------------------------------===//

void CaseOp::getSuccessorRegions(Optional<unsigned> index,
                                 ArrayRef<Attribute> operands,
                                 SmallVectorImpl<RegionSuccessor>& regions) {
  // If the predecessor is the CaseOp, branch to all other branches.
  if (!index.hasValue()) {
    for (auto& branch : branches())
      regions.push_back(RegionSuccessor(&branch, branch.getArguments()));
  }
  // If the predecessor is one of the branches, branch back to the parent
  // operation.
  regions.push_back(RegionSuccessor());
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CollectivePermuteOp op) {
  return mlir::hlo::VerifyCollectivePermuteSourceTargetPairs(
      op, op.source_target_pairs());
}

//===----------------------------------------------------------------------===//
// ConstOp.
//===----------------------------------------------------------------------===//

/// An lho.constant on an memref that is locally allocated and with no other
/// users (other than dealloc's) can be erased.
// TODO: This can be generalized to an arbitrary op by making use of memory
// effects (write memory effect).
struct EraseConstOp : public OpRewritePattern<ConstOp> {
  using OpRewritePattern<ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOp op,
                                PatternRewriter& rewriter) const override {
    Value memref = op.output();
    if (!memref.getDefiningOp<memref::AllocOp>()) {
      return failure();
    }

    // Check that all uses of the memref are either DeallocOps or this op.
    for (Operation* user : memref.getUsers())
      if (user != op && !isa<memref::DeallocOp>(user)) return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

void ConstOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                          MLIRContext* context) {
  results.insert<EraseConstOp>(context);
}

//===----------------------------------------------------------------------===//
// CustomCallOp.
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CustomCallOp op) {
  if (op.target_arg_mapping()) {
    CustomCallTargetArgMapping mapping = *op.target_arg_mapping();
    auto verify_mapping = [&](int64_t target_num, size_t op_num,
                              ArrayAttr mapping,
                              StringRef kind) -> LogicalResult {
      if (target_num < op_num)
        return op.emitOpError("number of target " + kind + " (")
               << target_num << ") cannot be less than the number of " << kind
               << "(" << op_num << ") for the operation";

      if (mapping.size() != op_num)
        return op.emitOpError("number of entries in the mapping for " + kind +
                              " (")
               << mapping.size() << ") should match the number of " << kind
               << " for the operation (" << op_num << ")";

      std::unordered_set<int64_t> entries;
      // Each entry in the mapping should be < target_num and an entry cannot
      // appear more than once.
      for (Attribute entry : mapping) {
        int64_t int_entry = entry.cast<IntegerAttr>().getInt();
        // ODS verification will ensure that these entries are integers.
        if (!entries.insert(int_entry).second)
          return op.emitOpError("entry ")
                 << int_entry
                 << " cannot appear more than once in the mapping for " << kind;
        if (int_entry < 0 || int_entry >= target_num)
          return op.emitOpError(
                     "entries in mapping for " + kind +
                     " must be >= 0 and less than target's number of " + kind +
                     " (")
                 << target_num << ")";
      }
      return success();
    };
    if (failed(verify_mapping(mapping.num_args().getInt(), op.args().size(),
                              mapping.args_to_target_args(), "args")) ||
        failed(verify_mapping(mapping.num_results().getInt(),
                              op.output().size(),
                              mapping.results_to_target_results(), "results")))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

// Removes `lmhlo.copy` inside ReduceOp body.
//
// TODO(b/183920887): Remove this pattern as soon as bufferization is fixed.
struct RemoveCopyInReduceBody : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp reduce,
                                PatternRewriter& rewriter) const override {
    // Find the only `lmhlo.copy` in the body of `reduce`.
    CopyOp the_only_copy;
    for (auto& op : reduce.body().front()) {
      if (auto copy = dyn_cast<lmhlo::CopyOp>(op)) {
        if (the_only_copy == nullptr) {
          the_only_copy = copy;
        } else {
          the_only_copy = nullptr;
          break;
        }
      }
    }
    if (!the_only_copy) return failure();

    auto new_reduce = rewriter.cloneWithoutRegions(reduce);
    Block* new_block =
        rewriter.createBlock(&new_reduce.body(), new_reduce.body().end(),
                             reduce.body().front().getArgumentTypes());

    mlir::BlockAndValueMapping bvm;
    for (auto item : llvm::zip(reduce.body().front().getArguments(),
                               new_block->getArguments())) {
      bvm.map(std::get<0>(item), std::get<1>(item));
    }
    bvm.map(the_only_copy.operand(), bvm.lookup(the_only_copy.output()));

    rewriter.setInsertionPointToStart(new_block);
    for (auto& op : reduce.body().front()) {
      if (llvm::isa<lmhlo::CopyOp>(op) || llvm::isa<memref::DeallocOp>(op) ||
          llvm::isa<memref::AllocOp>(op))
        continue;
      rewriter.clone(op, bvm);
    }
    rewriter.eraseOp(reduce);
    return success();
  }
};

void ReduceOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                           MLIRContext* context) {
  results.insert<RemoveCopyInReduceBody>(context);
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp.
//===----------------------------------------------------------------------===//

// For reduce-window, all `inputs` need to have compatible shapes.
static LogicalResult Verify(ReduceWindowOp op) {
  if (failed(verifyCompatibleShapes(op.inputs().getTypes())))
    return op.emitOpError() << "requires same shape for all operands";
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void WhileOp::getSuccessorRegions(Optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor>& regions) {
  // If the predecessor is the WhileOp or the body region, branch into the
  // cond region.
  if (!index.hasValue() || index.getValue() == 1) {
    regions.push_back(RegionSuccessor(&cond(), cond().getArguments()));
    return;
  }
  // If the predecessor is the cond region, we can branch to the body region
  // or back to the parent operation.
  regions.push_back(RegionSuccessor(&body(), body().getArguments()));
  regions.push_back(RegionSuccessor());
}

Region& WhileOp::getLoopBody() { return body(); }

bool WhileOp::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

LogicalResult WhileOp::moveOutOfLoop(ArrayRef<Operation*> ops) {
  for (auto op : ops) op->moveBefore(*this);
  return success();
}

// suppress warning.

using mlir::hlo::parseWindowAttributes;
using mlir::hlo::printWindowAttributes;

}  // namespace lmhlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc"

namespace mlir {
namespace lmhlo {

// TODO(cheshire): Support folding, reuse code from hlo_ops.cc.

void FusionOp::build(OpBuilder& builder, OperationState& result,
                     ArrayRef<NamedAttribute> attributes) {
  result.addAttributes(attributes);
  Region* bodyRegion = result.addRegion();
  FusionOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void FusionOp::getSuccessorRegions(Optional<unsigned> index,
                                   ArrayRef<Attribute> operands,
                                   SmallVectorImpl<RegionSuccessor>& regions) {
  // If the predecessor is the fusion region, jump back to the parent op.
  if (index.hasValue()) {
    assert(index.getValue() == 0 && "expected fusion region");
    regions.push_back(RegionSuccessor());
  } else {
    // If the predecessor is the FusionOp, branch into the region.
    regions.push_back(RegionSuccessor(&region(), region().getArguments()));
  }
}

}  // namespace lmhlo
}  // namespace mlir
