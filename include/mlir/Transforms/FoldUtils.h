//===- FoldUtils.h - Operation Fold Utilities -------------------*- C++ -*-===//
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
// This header file declares various operation folding utilities. These
// utilities are intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_FOLDUTILS_H
#define MLIR_TRANSFORMS_FOLDUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Function;
class Operation;

/// A helper class for folding operations, and unifying duplicated constants
/// generated along the way.
///
/// To make sure constants properly dominate all their uses, constants are
/// moved to the beginning of the entry block of the function when tracked by
/// this class.
class FoldHelper {
public:
  /// Constructs an instance for managing constants in the given function `f`.
  /// Constants tracked by this instance will be moved to the entry block of
  /// `f`. The insertion always happens at the very top of the entry block.
  ///
  /// This instance does not proactively walk the operations inside `f`;
  /// instead, users must invoke the following methods to manually handle each
  /// operation of interest.
  FoldHelper(Function *f);

  /// Tries to perform folding on the given `op`, including unifying
  /// deduplicated constants. If successful, calls `preReplaceAction` (if
  /// provided) by passing in `op`, then replaces `op`'s uses with folded
  /// results, and returns success. If the op was completely folded it is
  /// erased.
  LogicalResult
  tryToFold(Operation *op,
            std::function<void(Operation *)> preReplaceAction = {});

  /// Notifies that the given constant `op` should be remove from this
  /// FoldHelper's internal bookkeeping.
  ///
  /// Note: this method must be called if a constant op is to be deleted
  /// externally to this FoldHelper. `op` must be a constant op.
  void notifyRemoval(Operation *op);

private:
  /// Tries to deduplicate the given constant and returns success if that can be
  /// done. This moves the given constant to the top of the entry block if it
  /// is first seen. If there is already an existing constant that is the same,
  /// this does *not* erases the given constant.
  LogicalResult tryToUnify(Operation *op);

  /// Moves the given constant `op` to entry block to guarantee dominance.
  void moveConstantToEntryBlock(Operation *op);

  /// The function where we are managing constant.
  Function *function;

  /// This map keeps track of uniqued constants.
  DenseMap<std::pair<Attribute, Type>, Operation *> uniquedConstants;
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_FOLDUTILS_H
