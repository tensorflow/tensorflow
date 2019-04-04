//===- ConstantFoldUtils.h - Constant Fold Utilities ------------*- C++ -*-===//
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
// This header file declares various constant fold utilities. These utilities
// are intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CONSTANT_UTILS_H
#define MLIR_TRANSFORMS_CONSTANT_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Function;
class Operation;

/// A helper class for constant folding operations, and unifying duplicated
/// constants along the way.
///
/// To make sure constants' proper dominance of all their uses, constants are
/// moved to the beginning of the entry block of the function when tracked by
/// this class.
class ConstantFoldHelper {
public:
  /// Constructs an instance for managing constants in the given function `f`.
  /// Constants tracked by this instance will be moved to the entry block of
  /// `f`. If `insertAtHead` is true, the insertion always happen at the very
  /// top of the entry block; otherwise, the insertion happens after the last
  /// one of consecutive constant ops at the beginning of the entry block.
  ///
  /// This instance does not proactively walk the operations inside `f`;
  /// instead, users must invoke the following methods to manually handle each
  /// operation of interest.
  ConstantFoldHelper(Function *f, bool insertAtHead = true);

  /// Tries to perform constant folding on the given `op`, including unifying
  /// deplicated constants. If successful, calls `preReplaceAction` (if
  /// provided) by passing in `op`, then replaces `op`'s uses with folded
  /// constants, and returns true.
  ///
  /// Note: `op` will *not* be erased to avoid invalidating potential walkers in
  /// the caller.
  bool
  tryToConstantFold(Operation *op,
                    std::function<void(Operation *)> preReplaceAction = {});

  /// Notifies that the given constant `op` should be remove from this
  /// ConstantFoldHelper's internal bookkeeping.
  ///
  /// Note: this method must be called if a constant op is to be deleted
  /// externally to this ConstantFoldHelper. `op` must be a constant op.
  void notifyRemoval(Operation *op);

private:
  /// Tries to deduplicate the given constant and returns true if that can be
  /// done. This moves the given constant to the top of the entry block if it
  /// is first seen. If there is already an existing constant that is the same,
  /// this does *not* erases the given constant.
  bool tryToUnify(Operation *op);

  /// Moves the given constant `op` to entry block to guarantee dominance.
  void moveConstantToEntryBlock(Operation *op);

  /// The function where we are managing constant.
  Function *function;

  /// Whether to always insert constants at the very top of the entry block.
  bool isInsertAtHead;

  /// This map keeps track of uniqued constants.
  DenseMap<std::pair<Attribute, Type>, Operation *> uniquedConstants;
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_CONSTANT_UTILS_H
