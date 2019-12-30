//===- GreedyPatternRewriteDriver.cpp - A greedy rewriter -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir::applyPatternsGreedily.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "pattern-matcher"

static llvm::cl::opt<unsigned> maxPatternMatchIterations(
    "mlir-max-pattern-match-iterations",
    llvm::cl::desc("Max number of iterations scanning for pattern match"),
    llvm::cl::init(10));

namespace {

/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public PatternRewriter {
public:
  explicit GreedyPatternRewriteDriver(MLIRContext *ctx,
                                      const OwningRewritePatternList &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    worklist.reserve(64);
  }

  /// Perform the rewrites. Return true if the rewrite converges in
  /// `maxIterations`.
  bool simplify(MutableArrayRef<Region> regions, int maxIterations);

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
      worklistMap.erase(it);
    }
  }

  // These are hooks implemented for PatternRewriter.
protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  Operation *insert(Operation *op) override {
    addToWorklist(op);
    return OpBuilder::insert(op);
  }

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    addToWorklist(op->getOperands());
    op->walk([this](Operation *operation) {
      removeFromWorklist(operation);
      folder.notifyRemoval(operation);
    });
  }

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override {
    for (auto result : op->getResults())
      for (auto *user : result->getUsers())
        addToWorklist(user);
  }

private:
  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  template <typename Operands> void addToWorklist(Operands &&operands) {
    for (Value operand : operands) {
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
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  OperationFolder folder;
};
} // end anonymous namespace

/// Perform the rewrites.
bool GreedyPatternRewriteDriver::simplify(MutableArrayRef<Region> regions,
                                          int maxIterations) {
  // Add the given operation to the worklist.
  auto collectOps = [this](Operation *op) { addToWorklist(op); };

  bool changed = false;
  int i = 0;
  do {
    // Add all nested operations to the worklist.
    for (auto &region : regions)
      region.walk(collectOps);

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

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
        // Be careful to update bookkeeping.
        notifyOperationRemoved(op);
        op->erase();
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list. Also remove `op` and nested ops from worklist.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto preReplaceAction = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto result : op->getResults())
          for (auto *operand : result->getUsers())
            addToWorklist(operand);

        notifyOperationRemoved(op);
      };

      // Try to fold this op.
      if (succeeded(folder.tryToFold(op, collectOps, preReplaceAction))) {
        changed |= true;
        continue;
      }

      // Make sure that any new operations are inserted at this point.
      setInsertionPoint(op);

      // Try to match one of the patterns. The rewriter is automatically
      // notified of any necessary changes, so there is nothing else to do here.
      changed |= matcher.matchAndRewrite(op, *this);
    }

    // After applying patterns, make sure that the CFG of each of the regions is
    // kept up to date.
    changed |= succeeded(simplifyRegions(regions));
  } while (changed && ++i < maxIterations);
  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner. Return true if no more patterns can be matched in
/// the result operation regions.
/// Note: This does not apply patterns to the top-level operation itself.
///
bool mlir::applyPatternsGreedily(Operation *op,
                                 const OwningRewritePatternList &patterns) {
  return applyPatternsGreedily(op->getRegions(), patterns);
}

/// Rewrite the given regions, which must be isolated from above.
bool mlir::applyPatternsGreedily(MutableArrayRef<Region> regions,
                                 const OwningRewritePatternList &patterns) {
  if (regions.empty())
    return true;

  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  auto regionIsIsolated = [](Region &region) {
    return region.getParentOp()->isKnownIsolatedFromAbove();
  };
  (void)regionIsIsolated;
  assert(llvm::all_of(regions, regionIsIsolated) &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Start the pattern driver.
  GreedyPatternRewriteDriver driver(regions[0].getContext(), patterns);
  bool converged = driver.simplify(regions, maxPatternMatchIterations);
  LLVM_DEBUG(if (!converged) {
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << maxPatternMatchIterations << " times";
  });
  return converged;
}
