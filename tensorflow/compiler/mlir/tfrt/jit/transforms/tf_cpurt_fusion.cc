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

#include <utility>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

// -------------------------------------------------------------------------- //
// Fuse Linalg generic operations on Tensors.
// -------------------------------------------------------------------------- //

using mlir::dyn_cast;
using mlir::isa;

using mlir::AffineMap;
using mlir::Operation;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::OwningRewritePatternList;

namespace linalg = mlir::linalg;

// Returns true if `op` is a linalg generic operation that only does the
// broadcast of the input.
static bool IsBroadcast(Operation *op) {
  // Operation must be a generic linalg operation.
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic) return false;

  // All iterators must be parallel.
  if (generic.getNumParallelLoops() != generic.getNumLoops()) return false;

  // The body must simple forward input to the output.
  if (!isa<linalg::YieldOp>(generic.getBody()->front())) return false;

  // Operation must have single input and output.
  if (generic.getNumInputs() != 1 || generic.getNumOutputs() != 1) return false;

  // Check the input operand indexing map.
  OpOperand *operand = generic.getInputOperand(0);
  AffineMap indexing_map = generic.getTiedIndexingMap(operand);

  if (!indexing_map.isProjectedPermutation() ||
      indexing_map.getNumDims() == indexing_map.getNumResults())
    return false;

  // We found a generic linalg operation that is a simple broadcast.
  return true;
}

// Decide if the producer operation should be fused into the consumer.
static bool ControlElementwiseOpsFusion(const OpResult &producer_result,
                                        OpOperand &) {
  // TODO(ezhulenev): This is a very simplistic heuristic, we need something
  // better to decide when fusion is beneficial.

  // Always fuse broadcasts into the consumer.
  if (IsBroadcast(producer_result.getOwner())) return true;

  // If producer result has multiple users do not fuse it into the consumer.
  if (!llvm::hasSingleElement(producer_result.getUsers())) return false;

  return true;
}

struct FusionPass : public FusionBase<FusionPass> {
  void runOnFunction() override {
    Operation *op = getOperation();

    OwningRewritePatternList patterns(op->getContext());
    linalg::populateElementwiseOpsFusionPatterns(
        patterns,
        linalg::LinalgElementwiseFusionOptions()
            .setControlElementwiseOpsFusionFn(ControlElementwiseOpsFusion));

    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};

std::unique_ptr<mlir::FunctionPass> CreateFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace tensorflow
