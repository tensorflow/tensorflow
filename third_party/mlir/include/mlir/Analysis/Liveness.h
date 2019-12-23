//===- Liveness.h - Liveness analysis for MLIR ------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains an analysis for computing liveness information from a
// given top-level operation. The current version of the analysis uses a
// traditional algorithm to resolve detailed live-range information about all
// values within the specified regions. It is also possible to query liveness
// information on block level.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LIVENESS_H
#define MLIR_ANALYSIS_LIVENESS_H

#include <vector>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

class Block;
class LivenessBlockInfo;
class Operation;
class Region;
class Value;

// TODO(riverriddle) Remove this after Value is value-typed.
using ValuePtr = Value *;

/// Represents an analysis for computing liveness information from a
/// given top-level operation. The analysis iterates over all associated
/// regions that are attached to the given top-level operation. It
/// computes liveness information for every value and block that are
/// included in the mentioned regions. It relies on a fixpoint iteration
/// to compute all live-in and live-out values of all included blocks.
/// Sample usage:
///   Liveness liveness(topLevelOp);
///   auto &allInValues = liveness.getLiveIn(block);
///   auto &allOutValues = liveness.getLiveOut(block);
///   auto allOperationsInWhichValueIsLive = liveness.resolveLiveness(value);
///   bool lastUse = liveness.isLastUse(value, operation);
class Liveness {
public:
  using OperationListT = std::vector<Operation *>;
  using BlockMapT = DenseMap<Block *, LivenessBlockInfo>;
  using ValueSetT = SmallPtrSet<ValuePtr, 16>;

public:
  /// Creates a new Liveness analysis that computes liveness
  /// information for all associated regions.
  Liveness(Operation *op);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Gets liveness info (if any) for the given value.
  /// This includes all operations in which the given value is live.
  /// Note that the operations in this list are not ordered and the current
  /// implementation is computationally expensive (as it iterates over all
  /// blocks in which the given value is live).
  OperationListT resolveLiveness(ValuePtr value) const;

  /// Gets liveness info (if any) for the block.
  const LivenessBlockInfo *getLiveness(Block *block) const;

  /// Returns a reference to a set containing live-in values (unordered).
  const ValueSetT &getLiveIn(Block *block) const;

  /// Returns a reference to a set containing live-out values (unordered).
  const ValueSetT &getLiveOut(Block *block) const;

  /// Returns true if the given operation represent the last use of the
  /// given value.
  bool isLastUse(ValuePtr value, Operation *operation) const;

  /// Dumps the liveness information in a human readable format.
  void dump() const;

  /// Dumps the liveness information to the given stream.
  void print(raw_ostream &os) const;

private:
  /// Initializes the internal mappings.
  void build(MutableArrayRef<Region> regions);

private:
  /// The operation this analysis was constructed from.
  Operation *operation;

  /// Maps blocks to internal liveness information.
  BlockMapT blockMapping;
};

/// This class represents liveness information on block level.
class LivenessBlockInfo {
public:
  /// A typedef declaration of a value set.
  using ValueSetT = Liveness::ValueSetT;

public:
  /// Returns the underlying block.
  Block *getBlock() const { return block; }

  /// Returns all values that are live at the beginning
  /// of the block (unordered).
  const ValueSetT &in() const { return inValues; }

  /// Returns all values that are live at the end
  /// of the block (unordered).
  const ValueSetT &out() const { return outValues; }

  /// Returns true if the given value is in the live-in set.
  bool isLiveIn(ValuePtr value) const;

  /// Returns true if the given value is in the live-out set.
  bool isLiveOut(ValuePtr value) const;

  /// Gets the start operation for the given value. This is the first operation
  /// the given value is considered to be live. This could either be the start
  /// operation of the current block (in case the value is live-in) or the
  /// operation that defines the given value (must be referenced in this block).
  Operation *getStartOperation(ValuePtr value) const;

  /// Gets the end operation for the given value using the start operation
  /// provided (must be referenced in this block).
  Operation *getEndOperation(ValuePtr value, Operation *startOperation) const;

private:
  /// The underlying block.
  Block *block;

  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  friend class Liveness;
};

} // end namespace mlir

#endif // MLIR_ANALYSIS_LIVENESS_H
