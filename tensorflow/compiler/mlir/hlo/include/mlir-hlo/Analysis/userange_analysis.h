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

/// Representation on an Interval for the Userange.
struct UseInterval {
  using Vector = SmallVector<UseInterval, 8>;

public:
  UseInterval();
  UseInterval(size_t left, size_t right) : left(left), right(right) {}

  /// A check if the given UseInterval overlaps with this UseInterval.
  bool isOverlapping(const UseInterval &other) const {
    return left <= other.right && right >= other.left;
  }

  /// A check if the given UseInterval is contigious with this UseInterval in
  /// terms of the doubled Ids.
  /// For example: (0, 2) and (4, 6) are contigious where (0, 2) and (5, 6) are
  ///              not.
  bool isContigious(const UseInterval &other) const {
    return left <= other.right + 2 && right + 2 >= other.left;
  }

  /// A check if the given element is inside this UseInterval.
  bool contains(size_t element) const {
    return left <= element && right >= element;
  }

  /// Merges this UseInterval with the given UseInterval by updating left and
  /// right border.
  bool mergeWith(const UseInterval &other) {
    if (!isContigious(other))
      return false;
    left = std::min(left, other.left);
    right = std::max(right, other.right);
    return true;
  }

  /// Performs an interval subtraction => A = A - B.
  /// Note: This assumes that all intervals of b are included in some interval
  ///       of a.
  static void intervalSubtract(Vector &a, const Vector &b);

  bool operator<(const UseInterval &other) const { return right < other.left; }

  bool operator>(const UseInterval &other) const { return left > other.right; }

  bool operator==(const UseInterval &other) const {
    return left == other.left && right == other.right;
  }

  /// The left border of this UseInterval.
  size_t left;

  /// The right border of this UseInterval.
  size_t right;
};

/// Represents an analysis for computing the useranges of all alloc values
/// inside a given function operation. The analysis uses liveness information to
/// compute intervals starting at the first and ending with the last use of
/// every alloc value.
class UserangeAnalysis {
public:
  /// A typedef declaration of an UseInterval, which represents an interval as a
  /// pair of begin to end.
  // using UseInterval = std::pair<size_t, size_t>;
  using UsePosition = std::pair<size_t, Operation *>;
  using UsePositionList = std::vector<UsePosition>;

  UserangeAnalysis(Operation *op, const BufferPlacementAllocs &allocs,
                   const BufferViewFlowAnalysis &aliases);

  /// Returns the index of the first operation that uses the given value.
  /// Returns an empty Optional if the value has no uses.
  llvm::Optional<size_t> getFirstUseIndex(Value value) const {
    auto &intervals = useIntervalMap.find(value)->second;
    return intervals.empty() ? llvm::None
                             : llvm::Optional<size_t>(intervals.begin()->left);
  }

  /// Returns the UseInterval Vector of the given Value.
  llvm::Optional<const UseInterval::Vector *>
  getUserangeInterval(Value value) const {
    auto intervals = useIntervalMap.find(value);
    if (intervals == useIntervalMap.end())
      return llvm::None;
    return &intervals->second;
  }

  /// Returns the UsePositionList of the given Value.
  llvm::Optional<const UsePositionList *>
  getUserangePositions(Value value) const {
    auto usePosition = usePositionMap.find(value);
    if (usePosition == usePositionMap.end() || usePosition->second.empty())
      return llvm::None;
    return &usePosition->second;
  }

  /// Returns the operation assosiated with a given Id.
  Operation *getOperation(size_t id) const {
    return operationList[unwrapId(id)];
  };

  /// Computes the ID for the operation. If the operation contains operands
  /// which have read effects, the returning ID will be odd.
  size_t computeId(Value v, Operation *op) const;

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

  /// Builds an UseInterval::Vector corresponding to the given OperationList.
  UseInterval::Vector
  computeInterval(Value value, const Liveness::OperationListT &operationList);

  /// Checks each operand of the operation for its memory effects and separates
  /// them into read and write. Operands with read or write effects are added
  /// to the opReadWriteMap.
  void gatherMemoryEffects(Operation *op);

  /// Computes the doubled Id back to the OperationId.
  size_t unwrapId(size_t id) const;

  /// Merge two UseInterval::Vector into a new UseInterval::Vector. Return a
  /// pair with the resulting UseInterval::Vector and a boolean if there were
  /// interferences during merging.
  std::pair<UseInterval::Vector, bool>
  intervalMerge(const UseInterval::Vector &intervalA,
                const UseInterval::Vector &intervalB) const;

  /// Performs an interval union of the interval vectors from the given values.
  /// Returns an empty Optional if there is an interval interference.
  bool intervalUnion(Value itemA, Value itemB) const;

  /// Maps each Operation to a unique ID according to the program sequence.
  DenseMap<Operation *, size_t> operationIds;

  /// Maps each ID to an Operation.
  std::vector<Operation *> operationList;

  /// Maps a value to its use range interval.
  DenseMap<Value, UseInterval::Vector> useIntervalMap;

  /// Maps an Operation to a pair of read and write Operands.
  DenseMap<Operation *, std::pair<SmallPtrSet<Value, 2>, SmallPtrSet<Value, 2>>>
      opReadWriteMap;

  /// Maps aliasValues to their use ranges. This is necessary to prevent
  /// recomputations of the use range intervals of the aliases.
  DenseMap<Value, OperationListT> aliasUseranges;

  /// Maps a value to a vector with the corresponding OperationID and Operation.
  /// This map contains all uses of an Value and their use range position.
  DenseMap<Value, UsePositionList> usePositionMap;

  /// Cache the alias lists for all values to avoid recomputation.
  BufferViewFlowAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness liveness;
};

} // namespace mlir

#endif // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H_
