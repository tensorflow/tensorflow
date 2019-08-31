//===- Visitors.cpp - MLIR Visitor Utilties -------------------------------===//
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
