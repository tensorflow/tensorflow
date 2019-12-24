//===- Visitors.cpp - MLIR Visitor Utilties -------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Visitors.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

/// Walk all of the operations nested under and including the given operations.
void detail::walkOperations(Operation *op,
                            function_ref<void(Operation *op)> callback) {
  // TODO(b/140235992) This walk should be iterative over the operations.
  for (auto &region : op->getRegions())
    for (auto &block : region)
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        walkOperations(&nestedOp, callback);

  callback(op);
}

/// Walk all of the operations nested under and including the given operations.
/// This methods walks operations until an interrupt signal is received.
WalkResult
detail::walkOperations(Operation *op,
                       function_ref<WalkResult(Operation *op)> callback) {
  // TODO(b/140235992) This walk should be iterative over the operations.
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        if (walkOperations(&nestedOp, callback).wasInterrupted())
          return WalkResult::interrupt();
    }
  }
  return callback(op);
}
