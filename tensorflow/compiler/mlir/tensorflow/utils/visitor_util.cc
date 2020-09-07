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

#include "tensorflow/compiler/mlir/tensorflow/utils/visitor_util.h"

#include "mlir/IR/Operation.h"  // from @llvm-project

namespace tensorflow {

WalkStage::WalkStage(mlir::Operation *op)
    : num_regions_(op->getNumRegions()), next_region_(0) {}

namespace detail {

/// Walk all of the operations nested under and including the given operations.
void WalkOperations(mlir::Operation *op, VoidCallback callback) {
  WalkStage stage(op);

  for (auto &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    callback(op, stage);
    stage.Advance();

    for (auto &block : region)
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        WalkOperations(&nestedOp, callback);
  }

  // Invoke callback after all regions have been visited.
  callback(op, stage);
}

/// Walk all of the operations nested under and including the given operations.
/// This methods walks operations until an interrupt signal is received.
mlir::WalkResult WalkOperations(mlir::Operation *op,
                                InterruptCallback callback) {
  WalkStage stage(op);

  for (auto &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    if (callback(op, stage).wasInterrupted())
      return mlir::WalkResult::interrupt();

    stage.Advance();

    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        if (WalkOperations(&nestedOp, callback).wasInterrupted())
          return mlir::WalkResult::interrupt();
    }
  }
  return callback(op, stage);
}

}  // namespace detail
}  // namespace tensorflow
