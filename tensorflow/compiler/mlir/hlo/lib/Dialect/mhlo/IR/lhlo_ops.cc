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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
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

// Verifies replica groups attached to collective communication operations.
// If the attribute is not empty, it must be a rank 2 tensor, and each replica
// should appear exactly once. If `is_uniform_sized` is true, then we also check
// that each group is of the same size. If the operation has
// `use_global_device_id` set, then replica group cannot be empty.
template <typename OpT>
LogicalResult VerifyReplicaGroups(OpT op, bool is_uniform_sized) {
  DenseIntElementsAttr attr = op.replica_groups();
  auto replica_group_type = attr.getType().dyn_cast<RankedTensorType>();
  if (!replica_group_type || replica_group_type.getRank() != 2 ||
      !replica_group_type.getElementType().isInteger(/*width=*/64))
    return op.emitOpError(
        "replica groups should be a rank 2 tensor of 64 bit integers");

  if (replica_group_type.getShape().equals(ArrayRef<int64_t>{0, 0}))
    return success();

  int64_t max_replica_id_seen = 0;
  llvm::SmallSet<int64_t, 8> replica_seen;
  for (int64_t id : attr.getValues<int64_t>()) {
    if (is_uniform_sized && id == -1) {
      return op.emitOpError("Invalid replica id -1");
    }
    if (id != -1) {
      if (!replica_seen.insert(id).second) {
        return op.emitOpError("replica id #") << id << " seen more than once";
      }
      max_replica_id_seen = std::max(max_replica_id_seen, id);
    }
  }

  for (int64_t id = 0; id <= max_replica_id_seen; id++) {
    if (!replica_seen.contains(id)) {
      return op.emitOpError("replica id #")
             << id << " not seen in replica groups";
    }
  }
  return success();
}

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

static LogicalResult Verify(AllReduceOp op) {
  if (failed(VerifyReplicaGroups(op, /*is_uniform_sized=*/false)))
    return failure();

  // AllReduce had variadic operands and results that have the same size.
  // Each memeber of the operand should have the same type as the corresponding
  // member of the result.
  for (auto it : llvm::enumerate(
           llvm::zip(op.operands().getTypes(), op.results().getTypes()))) {
    Type operandType = std::get<0>(it.value());
    Type resultType = std::get<1>(it.value());
    if (operandType != resultType)
      return op.emitOpError("requires operand #")
             << it.index() << " (type: " << operandType << ") and result #"
             << it.index() << " (type: " << resultType << ") to have same type";
  }
  return success();
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
    if (!memref.getDefiningOp<AllocOp>()) {
      return failure();
    }

    // Check that all uses of the memref are either DeallocOps or this op.
    for (Operation* user : memref.getUsers())
      if (user != op && !isa<DeallocOp>(user)) return failure();

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

}  // namespace lmhlo
}  // namespace mlir
