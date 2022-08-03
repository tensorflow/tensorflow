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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

// -------------------------------------------------------------------------- //
// Fuse Linalg generic operations on Tensors.
// -------------------------------------------------------------------------- //

using mlir::dyn_cast;
using mlir::isa;

using mlir::AffineMap;
using mlir::MLIRContext;
using mlir::Operation;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::RewritePatternSet;

namespace linalg = mlir::linalg;
namespace tensor = mlir::tensor;

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
  if (!producer_result.hasOneUse()) return false;

  return true;
}

// Check if the reshape operation is only expansion into/collapsing of
// unit-dimension.
template <typename TensorReshapeOp>
static bool IsUnitDimExpansionOnly(TensorReshapeOp reshape_op) {
  constexpr bool is_expanding =
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value;
  llvm::ArrayRef<int64_t> expanded_shape =
      (is_expanding ? reshape_op.getResultType().getShape()
                    : reshape_op.getSrcType().getShape());
  for (auto &indices : reshape_op.getReassociationIndices()) {
    unsigned num_unit_dims = 0;
    for (int64_t position : indices)
      if (expanded_shape[position] == 1) num_unit_dims++;
    if (num_unit_dims != indices.size() - 1) return false;
  }
  return true;
}

// Control function to skip unit dim reshape when fusing reshapes by expansion.
static bool SkipUnitDimReshape(const OpResult &producer, OpOperand &consumer) {
  // If producer result has multiple users do not fuse it into the consumer.
  if (!producer.hasOneUse()) return false;

  if (auto producer_collapse_op =
          dyn_cast<tensor::CollapseShapeOp>(producer.getOwner())) {
    return !IsUnitDimExpansionOnly(producer_collapse_op);
  }
  if (auto consumer_expand_op =
          dyn_cast<tensor::ExpandShapeOp>(consumer.getOwner())) {
    return !IsUnitDimExpansionOnly(consumer_expand_op);
  }
  return true;
}

struct FusionPass : public FusionBase<FusionPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(op->getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                 ControlElementwiseOpsFusion);

    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns,
                                                      SkipUnitDimReshape);

    linalg::populateConstantFoldLinalgOperations(patterns,
                                                 ControlElementwiseOpsFusion);

    mlir::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    context->getLoadedDialect<linalg::LinalgDialect>()
        ->getCanonicalizationPatterns(patterns);
    // Use TopDownTraversal for compile time reasons.
    mlir::GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns),
                                       grc);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace tensorflow
