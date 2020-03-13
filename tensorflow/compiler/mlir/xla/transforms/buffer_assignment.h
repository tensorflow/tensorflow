/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_

#include "mlir/Analysis/Dominance.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Builders.h"   // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace xla {
namespace detail {

/// A specialized dominator analysis that provided access to some private
/// methods.
///
/// TODO(dfki): merge this functionality into the underlying MLIR core dominator
/// analysis.
template <bool IsPostDom>
class BufferAssignmentDominators
    : public mlir::detail::DominanceInfoBase<IsPostDom> {
 public:
  using super = mlir::detail::DominanceInfoBase<IsPostDom>;
  using super::super;

  /// Finds the nearest common dominator block for the two given blocks first
  /// and second. If there is no common dominator this function will return
  /// nullptr.
  Block* findNearestCommonDominator(Block* first, Block* second) const {
    assert(first->getParent() == second->getParent() & "Invalid region");
    return super::dominanceInfos.find(first->getParent())
        ->second->findNearestCommonDominator(first, second);
  }

  /// Return the dominance node from the region containing block A.
  DominanceInfoNode* getNode(Block* a) const {
    return super::dominanceInfos.find(a->getParent())->second->getNode(a);
  }
};
}  // namespace detail

/// Stores a proper alloc position to place dialect-specific alloc operations.
struct BufferInsertPosition {
 public:
  /// Creates a new positions tuple including alloc and dealloc positions.
  BufferInsertPosition(Operation* allocPosition);

  /// Returns the alloc position before which the alloc operation has to be
  /// inserted.
  Operation* getAllocPosition() const { return allocPosition; }

  /// Inserts a new dialect-specific alloc operation that will be constructed in
  /// the right place using the arguments provided.
  template <typename AllocOpT, typename... Args>
  AllocOpT insertAlloc(Value value, Args... args) const {
    OpBuilder allocBuilder(value.getDefiningOp());
    allocBuilder.setInsertionPoint(allocPosition);
    return allocBuilder.create<AllocOpT>(args...);
  }

 private:
  Operation* allocPosition;
};

/// Prepares a buffer assignment phase. It can place (user-defined) alloc
/// nodes. This simplifies the integration of the actual buffer-assignment
/// pass. Sample usage:
///   BufferAssignmentLegalizer baHelper(value);
///   -> determine alloc positions
///   auto allocPosition = baHelper.computeAllocPosition(value);
///   -> place alloc
///   allocBuilder.setInsertionPoint(positions.getAllocPosition());
///   <create alloc>
///   alternatively:
///   -> place alloc
///   baHelper.insertAlloc<AllocOp>(...);
/// Note: this class is intended to be used during legalization. In order
/// to move alloc and dealloc nodes into the right places you can use the
/// createBufferAssignmentPass() function.
class BufferAssignmentLegalizer {
 public:
  /// Creates a new assignment builder.
  explicit BufferAssignmentLegalizer(Operation* op);

  /// Returns the operation this analysis was constructed from.
  Operation* getOperation() const { return operation; }

  /// Computes the actual position to place allocs for the given value.
  BufferInsertPosition computeAllocPosition(Value value) const;

 private:
  /// The operation this analysis was constructed from.
  Operation* operation;

  /// The dominator analysis to place allocs in the appropriate blocks.
  detail::BufferAssignmentDominators<false> dominators;
};

}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
