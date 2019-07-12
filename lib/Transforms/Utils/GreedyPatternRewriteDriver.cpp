//===- GreedyPatternRewriteDriver.cpp - A greedy rewriter -----------------===//
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
// This file implements mlir::applyPatternsGreedily.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "pattern-matcher"

static llvm::cl::opt<unsigned> maxPatternMatchIterations(
    "mlir-max-pattern-match-iterations",
    llvm::cl::desc(
        "Max number of iterations scanning the functions for pattern match"),
    llvm::cl::init(10));

namespace {

/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public PatternRewriter {
public:
  explicit GreedyPatternRewriteDriver(MLIRContext *ctx,
                                      OwningRewritePatternList &&patterns)
      : PatternRewriter(ctx), matcher(std::move(patterns)) {
    worklist.reserve(64);
  }

  /// Perform the rewrites. Return true if the rewrite converges in
  /// `maxIterations`.
  bool simplifyFunction(Region *region, int maxIterations);

  void addToWorklist(Operation *op) {
    // Check to see if the worklist already contains this op.
    if (worklistMap.count(op))
      return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();

    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.  If not, this is
  /// a no-op.
  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
    }
  }

  // These are hooks implemented for PatternRewriter.
protected:
  // Implement the hook for creating operations, and make sure that newly
  // created ops are added to the worklist for processing.
  Operation *createOperation(const OperationState &state) override {
    auto *result = OpBuilder::createOperation(state);
    addToWorklist(result);
    return result;
  }

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    addToWorklist(op->getOperands());
    removeFromWorklist(op);
    folder.notifyRemoval(op);
  }

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override {
    for (auto *result : op->getResults())
      for (auto *user : result->getUsers())
        addToWorklist(user);
  }

private:
  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  template <typename Operands> void addToWorklist(Operands &&operands) {
    for (Value *operand : operands) {
      // If the use count of this operand is now < 2, we re-add the defining
      // operation to the worklist.
      // TODO(riverriddle) This is based on the fact that zero use operations
      // may be deleted, and that single use values often have more
      // canonicalization opportunities.
      if (!operand->use_empty() && !operand->hasOneUse())
        continue;
      if (auto *defInst = operand->getDefiningOp())
        addToWorklist(defInst);
    }
  }

  /// The low-level pattern matcher.
  RewritePatternMatcher matcher;

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased from
  /// the function, even if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  OperationFolder folder;
};
} // end anonymous namespace

/// Perform the rewrites.
bool GreedyPatternRewriteDriver::simplifyFunction(Region *region,
                                                  int maxIterations) {
  // Add the given operation to the worklist.
  auto collectOps = [this](Operation *op) { addToWorklist(op); };

  bool changed = false;
  int i = 0;
  do {
    // Add all operations to the worklist.
    region->walk(collectOps);

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value *, 8> originalOperands, resultValues;

    changed = false;
    while (!worklist.empty()) {
      auto *op = popFromWorklist();

      // Nulls get added to the worklist when operations are removed, ignore
      // them.
      if (op == nullptr)
        continue;

      // If the operation has no side effects, and no users, then it is
      // trivially dead - remove it.
      if (op->hasNoSideEffect() && op->use_empty()) {
        // Be careful to update bookkeeping in OperationFolder to keep
        // consistency if this is a constant op.
        folder.notifyRemoval(op);
        op->erase();
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto collectOperandsAndUses = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto *result : op->getResults())
          for (auto *operand : result->getUsers())
            addToWorklist(operand);
      };

      // Try to fold this op.
      if (succeeded(folder.tryToFold(op, collectOps, collectOperandsAndUses))) {
        changed |= true;
        continue;
      }

      // Make sure that any new operations are inserted at this point.
      setInsertionPoint(op);

      // Try to match one of the canonicalization patterns. The rewriter is
      // automatically notified of any necessary changes, so there is nothing
      // else to do here.
      changed |= matcher.matchAndRewrite(op, *this);
    }
  } while (changed && ++i < maxIterations);
  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

/// Rewrite the specified function by repeatedly applying the highest benefit
/// patterns in a greedy work-list driven manner. Return true if no more
/// patterns can be matched in the result function.
///
bool mlir::applyPatternsGreedily(FuncOp fn,
                                 OwningRewritePatternList &&patterns) {
  GreedyPatternRewriteDriver driver(fn.getContext(), std::move(patterns));
  bool converged =
      driver.simplifyFunction(&fn.getBody(), maxPatternMatchIterations);
  LLVM_DEBUG(if (!converged) {
    llvm::dbgs()
        << "The pattern rewrite doesn't converge after scanning the function "
        << maxPatternMatchIterations << " times";
  });
  return converged;
}
