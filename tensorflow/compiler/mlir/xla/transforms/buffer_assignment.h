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
};

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferAssignment
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferAssignmentAliasAnalysis {
 public:
  using ValueSetT = SmallPtrSet<Value, 16>;

 public:
  /// Constructs a new alias analysis using the op provided.
  BufferAssignmentAliasAnalysis(Operation* op);

  /// Finds all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const;

 private:
  /// Recursively determines alias information for the given value. It stores
  /// all newly found potential aliases in the given result set.
  void resolveRecursive(Value value, ValueSetT& result) const;

  /// Initializes the internal mappings.
  void build(MutableArrayRef<Region> regions);

 private:
  /// Maps values to all immediate aliases this value can have.
  llvm::DenseMap<Value, ValueSetT> aliases;
};
}  // namespace detail

/// Stores proper alloc and dealloc positions to place dialect-specific alloc
/// and dealloc operations.
struct BufferAssignmentPositions {
 public:
  /// Creates a new positions tuple including alloc and dealloc positions.
  BufferAssignmentPositions(Operation* allocPosition,
                            Operation* deallocPosition);

  /// Returns the alloc position before which the alloc operation has to be
  /// inserted.
  Operation* getAllocPosition() const { return allocPosition; }

  /// Returns the dealloc position after which the dealloc operation has to be
  /// inserted.
  Operation* getDeallocPosition() const { return deallocPosition; }

  /// Inserts a new dialect-specific alloc operation that will be constructed in
  /// the right place using the arguments provided.
  template <typename AllocOpT, typename... Args>
  AllocOpT insertAlloc(Value value, Args... args) const {
    OpBuilder allocBuilder(value.getDefiningOp());
    allocBuilder.setInsertionPoint(allocPosition);
    return allocBuilder.create<AllocOpT>(args...);
  }

  /// Inserts a new dialect-specific dealloc operation that will be constructed
  /// in the right place using the arguments provided.
  template <typename DeallocOpT, typename... Args>
  DeallocOpT insertDealloc(Value value, Args... args) const {
    OpBuilder deallocBuilder(value.getDefiningOp());
    deallocBuilder.setInsertionPointAfter(deallocPosition);
    return deallocBuilder.create<DeallocOpT>(args...);
  }

 private:
  Operation* allocPosition;
  Operation* deallocPosition;
};

class BufferAssignment {
 public:
  /// Creates a new BufferAssignment analysis that computes liveness of values
  /// (including their aliases) accross block boundaries to place allocs and
  /// deallocs.
  BufferAssignment(Operation* op);

  /// Returns the operation this analysis was constructed from.
  Operation* getOperation() const { return operation; }

  /// Computes the actual positions to place allocs and deallocs for the given
  /// value.
  BufferAssignmentPositions computeAllocAndDeallocPositions(Value value) const;

  /// Dumps the buffer assignment information in a human readable format.
  void dump() const;

  /// Dumps the buffer assignment information to the given stream.
  void print(raw_ostream& os) const;

 private:
  /// The operation this analysis was constructed from.
  Operation* operation;

  /// The underlying liveness analysis to compute fine grained information about
  /// alloc and dealloc positions.
  Liveness liveness;

  /// The dominator analysis to place allocs in the appropriate blocks.
  detail::BufferAssignmentDominators<false> dominators;

  /// The post dominator analysis to place deallocs in the appropriate blocks.
  detail::BufferAssignmentDominators<true> postDominators;

  /// The internal alias analysis to ensure that allocs and deallocs take all
  /// their potential aliases into account.
  detail::BufferAssignmentAliasAnalysis aliases;
};

}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
