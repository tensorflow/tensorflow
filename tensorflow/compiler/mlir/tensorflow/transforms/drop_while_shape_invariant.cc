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

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

namespace {

constexpr char kShapeInvariantAttr[] = "shape_invariant";

// Drop `shape_invariant` attribute from tf.While and tf.WhileRegion op. This
// would allow shape inference pass to further refine operand/result shapes of
// these ops. This is only safe to do when compiling to XLA.
class DropWhileShapeInvariantPass
    : public PassWrapper<DropWhileShapeInvariantPass, FunctionPass> {
  void runOnFunction() override;
};

void DropWhileShapeInvariantPass::runOnFunction() {
  getFunction().walk([](Operation* op) {
    if (llvm::isa<WhileOp, WhileRegionOp>(op))
      op->removeAttr(kShapeInvariantAttr);
  });
}

static PassRegistration<DropWhileShapeInvariantPass> pass(
    "tf-drop-while-shape-invariant",
    "Drop `shape_invariant` attrbute from While/WhileRegion ops.");

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateDropWhileShapeInvariantPass() {
  return std::make_unique<DropWhileShapeInvariantPass>();
}

}  // namespace TF
}  // namespace mlir
