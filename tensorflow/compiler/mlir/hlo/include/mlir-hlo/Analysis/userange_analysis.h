/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H_

#include <vector>

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

/// Represents an analysis for computing the useranges of all alloc values
/// inside a given function operation. The analysis uses liveness information to
/// compute intervals starting at the first and ending with the last use of
/// every alloc value.
class UserangeAnalysis {
 public:
  /// A typedef declaration of an UseInterval, which represents an interval as a
  /// pair of begin to end.
  using UseInterval = std::pair<size_t, size_t>;
  using IntervalVector = SmallVector<UseInterval, 8>;

  UserangeAnalysis(Operation *op, const BufferPlacementAllocs &allocs,
                   const BufferViewFlowAnalysis &aliases);

  /// Returns the index of the first operation that uses the given value.
  /// Returns an empty Optional if the value has no uses.
  llvm::Optional<size_t> getFirstUseIndex(Value value) const {
    auto &intervals = useIntervalMap.find(value)->second;
    return intervals.empty() ? llvm::None
                             : llvm::Optional<size_t>(intervals.begin()->first);
  }

  /// Checks if the use intervals of the given values interfere.
  bool rangesInterfere(Value itemA, Value itemB) const;

  /// Merges the userange of itemB into the userange of itemA.
  /// Note: This assumes that there is no interference between the two
  /// ranges.
  void unionRanges(Value itemA, Value itemB);

  /// Dumps the liveness information to the given stream.
  void dump(raw_ostream &os);

 private:
  using ValueSetT = BufferViewFlowAnalysis::ValueSetT;
  using OperationListT = Liveness::OperationListT;

  /// Builds an IntervalVector corresponding to the given OperationList.
  IntervalVector computeInterval(Value value,
                                 const Liveness::OperationListT &operationList);

  /// Checks each operand of the operation for its memory effects and separates
  /// them into read and write. Operands with read or write effects are added
  /// to the opReadWriteMap.
  void gatherMemoryEffects(Operation *op);

  /// Computes the ID for the operation. If the operation contains operands
  /// which have read effects, the returning ID will be odd.
  size_t computeID(Value v, Operation *op) const;

  /// Merge two IntervalVectors into a new IntervalVector. Return a pair with
  /// the resulting IntervalVector and a boolean if there were interferences
  /// during merging.
  std::pair<IntervalVector, bool> intervalMerge(
      const IntervalVector &intervalA, const IntervalVector &intervalB) const;

  /// Performs an interval union of the interval vectors from the given values.
  /// Returns an empty Optional if there is an interval interference.
  bool intervalUnion(Value itemA, Value itemB) const;

  /// Performs an interval subtraction => A = A - B.
  /// Note: This assumes that all intervals of b are included in some interval
  ///       of a.
  void intervalSubtract(IntervalVector &a, const IntervalVector &b) const;

  /// Maps each Operation to a unique ID according to the program sequence.
  DenseMap<Operation *, size_t> operationIds;

  /// Maps a value to its use range interval.
  DenseMap<Value, IntervalVector> useIntervalMap;

  /// Maps an Operation to a pair of read and write Operands.
  DenseMap<Operation *, std::pair<SmallPtrSet<Value, 2>, SmallPtrSet<Value, 2>>>
      opReadWriteMap;

  /// Maps aliasValues to their use ranges. This is necessary to prevent
  /// recomputations of the use range intervals of the aliases.
  DenseMap<Value, OperationListT> aliasUseranges;

  /// Cache the alias lists for all values to avoid recomputation.
  BufferViewFlowAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness liveness;
};

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H_
