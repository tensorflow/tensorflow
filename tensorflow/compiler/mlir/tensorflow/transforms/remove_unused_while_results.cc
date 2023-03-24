/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_REMOVEUNUSEDWHILERESULTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Removes unused results and related ops from while loops.
struct RemoveUnusedWhileResultsPass
    : public impl::RemoveUnusedWhileResultsPassBase<
          RemoveUnusedWhileResultsPass> {
  void runOnOperation() override;
};

// Prunes result defining op if possible, returns true if pruning was done.
bool TryPruneResultDefiningOp(TF::WhileRegionOp while_op, OpResult result) {
  // Don't prune if result is used.
  if (!result.use_empty()) return false;

  Block& body_block = while_op.getBody().front();
  Block& cond_block = while_op.getCond().front();
  Operation* body_yield_op = body_block.getTerminator();

  // The body yield operand, body block argument, condition block argument, and
  // result all correspond to each other (see definition of `WhileRegionOp`).
  int idx = result.getResultNumber();
  Value body_yield_operand = body_yield_op->getOperand(idx);
  Value body_block_argument = body_block.getArgument(idx);
  Value cond_block_argument = cond_block.getArgument(idx);
  // Consider the op that defines the unused result as a candidate for pruning.
  Operation* candidate_op = body_yield_operand.getDefiningOp();
  if (candidate_op == nullptr) return false;

  // Don't prune if candidate op might have side effects.
  if (isa_and_nonnull<TF::TensorFlowDialect>(candidate_op->getDialect())) {
    if (TF::TensorFlowDialect::CanHaveSideEffects(candidate_op)) {
      return false;
    }
  } else if (!isMemoryEffectFree(candidate_op)) {
    return false;
  }

  // Don't prune if the body block argument has any other user.
  for (Operation* op : body_block_argument.getUsers()) {
    if (op != candidate_op) return false;
  }

  // Don't prune if the condition block argument has any user.
  if (!cond_block_argument.use_empty()) return false;

  // Don't prune if `body_yield_operand` has more than one use (that would mean
  // it feeds into another op apart from `Yield`).
  if (!body_yield_operand.hasOneUse()) return false;

  // Don't prune if any other result of the candidate op is used.
  for (Value candidate_result : candidate_op->getResults()) {
    if (candidate_result == body_yield_operand) continue;
    if (!candidate_result.use_empty()) return false;
  }

  // Now we know that it is safe to erase the candidate op along with `result`
  // and the corresponding operand. Here we only erase the op and replace its
  // result usage with the corresponding block argument, the result and operand
  // will be removed later in a canonicalization pattern.
  VLOG(4) << "Pruning following op:\n" << debugString(*candidate_op);
  body_yield_operand.replaceAllUsesWith(body_block_argument);
  candidate_op->erase();
  return true;
}

void RemoveUnusedWhileResultsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  // Try to prune defining ops of unused results.
  func.walk([&](TF::WhileRegionOp while_op) {
    for (OpResult result : while_op.getResults()) {
      TryPruneResultDefiningOp(while_op, result);
    }
  });
  // Now eliminate passthrough operands/results with existing canonicalization
  // pattern.
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);
  TF::WhileRegionOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateRemoveUnusedWhileResultsPass() {
  return std::make_unique<RemoveUnusedWhileResultsPass>();
}

}  // namespace TF
}  // namespace mlir
