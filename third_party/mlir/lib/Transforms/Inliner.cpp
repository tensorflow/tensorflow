//===- Inliner.cpp - Pass to inline function calls ------------------------===//
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
// This file implements a basic inlining algorithm that operates bottom up over
// the Strongly Connect Components(SCCs) of the CallGraph. This enables a more
// incremental propagation of inlining decisions from the leafs to the roots of
// the callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "inlining"

using namespace mlir;

static llvm::cl::opt<bool> disableCanonicalization(
    "mlir-disable-inline-simplify",
    llvm::cl::desc("Disable running simplifications during inlining"),
    llvm::cl::ReallyHidden, llvm::cl::init(false));

static llvm::cl::opt<unsigned> maxInliningIterations(
    "mlir-max-inline-iterations",
    llvm::cl::desc("Maximum number of iterations when inlining within an SCC"),
    llvm::cl::ReallyHidden, llvm::cl::init(4));

//===----------------------------------------------------------------------===//
// CallGraph traversal
//===----------------------------------------------------------------------===//

/// Run a given transformation over the SCCs of the callgraph in a bottom up
/// traversal.
static void runTransformOnCGSCCs(
    const CallGraph &cg,
    function_ref<void(ArrayRef<CallGraphNode *>)> sccTransformer) {
  std::vector<CallGraphNode *> currentSCCVec;
  auto cgi = llvm::scc_begin(&cg);
  while (!cgi.isAtEnd()) {
    // Copy the current SCC and increment so that the transformer can modify the
    // SCC without invalidating our iterator.
    currentSCCVec = *cgi;
    ++cgi;
    sccTransformer(currentSCCVec);
  }
}

namespace {
/// This struct represents a resolved call to a given callgraph node. Given that
/// the call does not actually contain a direct reference to the
/// Region(CallGraphNode) that it is dispatching to, we need to resolve them
/// explicitly.
struct ResolvedCall {
  ResolvedCall(CallOpInterface call, CallGraphNode *targetNode)
      : call(call), targetNode(targetNode) {}
  CallOpInterface call;
  CallGraphNode *targetNode;
};
} // end anonymous namespace

/// Collect all of the callable operations within the given range of blocks. If
/// `traverseNestedCGNodes` is true, this will also collect call operations
/// inside of nested callgraph nodes.
static void collectCallOps(iterator_range<Region::iterator> blocks,
                           CallGraph &cg, SmallVectorImpl<ResolvedCall> &calls,
                           bool traverseNestedCGNodes) {
  SmallVector<Block *, 8> worklist;
  auto addToWorklist = [&](iterator_range<Region::iterator> blocks) {
    for (Block &block : blocks)
      worklist.push_back(&block);
  };

  addToWorklist(blocks);
  while (!worklist.empty()) {
    for (Operation &op : *worklist.pop_back_val()) {
      if (auto call = dyn_cast<CallOpInterface>(op)) {
        CallGraphNode *node =
            cg.resolveCallable(call.getCallableForCallee(), &op);
        if (!node->isExternal())
          calls.emplace_back(call, node);
        continue;
      }

      // If this is not a call, traverse the nested regions. If
      // `traverseNestedCGNodes` is false, then don't traverse nested call graph
      // regions.
      for (auto &nestedRegion : op.getRegions())
        if (traverseNestedCGNodes || !cg.lookupNode(&nestedRegion))
          addToWorklist(nestedRegion);
    }
  }
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a specialization of the main inlining interface.
struct Inliner : public InlinerInterface {
  Inliner(MLIRContext *context, CallGraph &cg)
      : InlinerInterface(context), cg(cg) {}

  /// Process a set of blocks that have been inlined. This callback is invoked
  /// *before* inlined terminator operations have been processed.
  void
  processInlinedBlocks(iterator_range<Region::iterator> inlinedBlocks) final {
    collectCallOps(inlinedBlocks, cg, calls, /*traverseNestedCGNodes=*/true);
  }

  /// The current set of call instructions to consider for inlining.
  SmallVector<ResolvedCall, 8> calls;

  /// The callgraph being operated on.
  CallGraph &cg;
};
} // namespace

/// Returns true if the given call should be inlined.
static bool shouldInline(ResolvedCall &resolvedCall) {
  // Don't allow inlining terminator calls. We currently don't support this
  // case.
  if (resolvedCall.call.getOperation()->isKnownTerminator())
    return false;

  // Don't allow inlining if the target is an ancestor of the call. This
  // prevents inlining recursively.
  if (resolvedCall.targetNode->getCallableRegion()->isAncestor(
          resolvedCall.call.getParentRegion()))
    return false;

  // Otherwise, inline.
  return true;
}

/// Attempt to inline calls within the given scc. This function returns
/// success if any calls were inlined, failure otherwise.
static LogicalResult inlineCallsInSCC(Inliner &inliner,
                                      ArrayRef<CallGraphNode *> currentSCC) {
  CallGraph &cg = inliner.cg;
  auto &calls = inliner.calls;

  // Collect all of the direct calls within the nodes of the current SCC. We
  // don't traverse nested callgraph nodes, because they are handled separately
  // likely within a different SCC.
  for (auto *node : currentSCC) {
    if (!node->isExternal())
      collectCallOps(*node->getCallableRegion(), cg, calls,
                     /*traverseNestedCGNodes=*/false);
  }
  if (calls.empty())
    return failure();

  // Try to inline each of the call operations. Don't cache the end iterator
  // here as more calls may be added during inlining.
  bool inlinedAnyCalls = false;
  for (unsigned i = 0; i != calls.size(); ++i) {
    ResolvedCall &it = calls[i];
    LLVM_DEBUG({
      llvm::dbgs() << "* Considering inlining call: ";
      it.call.dump();
    });
    if (!shouldInline(it))
      continue;

    CallOpInterface call = it.call;
    Region *targetRegion = it.targetNode->getCallableRegion();
    LogicalResult inlineResult = inlineCall(
        inliner, call, cast<CallableOpInterface>(targetRegion->getParentOp()),
        targetRegion);
    if (failed(inlineResult))
      continue;

    // If the inlining was successful, then erase the call.
    call.erase();
    inlinedAnyCalls = true;
  }
  calls.clear();
  return success(inlinedAnyCalls);
}

/// Canonicalize the nodes within the given SCC with the given set of
/// canonicalization patterns.
static void canonicalizeSCC(CallGraph &cg, ArrayRef<CallGraphNode *> currentSCC,
                            MLIRContext *context,
                            const OwningRewritePatternList &canonPatterns) {
  // Collect the sets of nodes to canonicalize.
  SmallVector<CallGraphNode *, 4> nodesToCanonicalize;
  for (auto *node : currentSCC) {
    // Don't canonicalize the external node, it has no valid callable region.
    if (node->isExternal())
      continue;

    // Don't canonicalize nodes with children. Nodes with children
    // require special handling as we may remove the node during
    // canonicalization. In the future, we should be able to handle this
    // case with proper node deletion tracking.
    if (node->hasChildren())
      continue;

    // We also won't apply canonicalizations for nodes that are not
    // isolated. This avoids potentially mutating the regions of nodes defined
    // above, this is also a stipulation of the 'applyPatternsGreedily' driver.
    auto *region = node->getCallableRegion();
    if (!region->getParentOp()->isKnownIsolatedFromAbove())
      continue;
    nodesToCanonicalize.push_back(node);
  }
  if (nodesToCanonicalize.empty())
    return;

  // Canonicalize each of the nodes within the SCC in parallel.
  // NOTE: This is simple now, because we don't enable canonicalizing nodes
  // within children. When we remove this restriction, this logic will need to
  // be reworked.
  ParallelDiagnosticHandler canonicalizationHandler(context);
  llvm::parallel::for_each_n(
      llvm::parallel::par, /*Begin=*/size_t(0),
      /*End=*/nodesToCanonicalize.size(), [&](size_t index) {
        // Set the order for this thread so that diagnostics will be properly
        // ordered.
        canonicalizationHandler.setOrderIDForThread(index);

        // Apply the canonicalization patterns to this region.
        auto *node = nodesToCanonicalize[index];
        applyPatternsGreedily(*node->getCallableRegion(), canonPatterns);

        // Make sure to reset the order ID for the diagnostic handler, as this
        // thread may be used in a different context.
        canonicalizationHandler.eraseOrderIDForThread();
      });
}

/// Attempt to inline calls within the given scc, and run canonicalizations with
/// the given patterns, until a fixed point is reached. This allows for the
/// inlining of newly devirtualized calls.
static void inlineSCC(Inliner &inliner, ArrayRef<CallGraphNode *> currentSCC,
                      MLIRContext *context,
                      const OwningRewritePatternList &canonPatterns) {
  // If we successfully inlined any calls, run some simplifications on the
  // nodes of the scc. Continue attempting to inline until we reach a fixed
  // point, or a maximum iteration count. We canonicalize here as it may
  // devirtualize new calls, as well as give us a better cost model.
  unsigned iterationCount = 0;
  while (succeeded(inlineCallsInSCC(inliner, currentSCC))) {
    // If we aren't allowing simplifications or the max iteration count was
    // reached, then bail out early.
    if (disableCanonicalization || ++iterationCount >= maxInliningIterations)
      break;
    canonicalizeSCC(inliner.cg, currentSCC, context, canonPatterns);
  }
}

//===----------------------------------------------------------------------===//
// InlinerPass
//===----------------------------------------------------------------------===//

// TODO(riverriddle) This pass should currently only be used for basic testing
// of inlining functionality.
namespace {
struct InlinerPass : public OperationPass<InlinerPass> {
  void runOnOperation() override {
    CallGraph &cg = getAnalysis<CallGraph>();
    auto *context = &getContext();

    // Collect a set of canonicalization patterns to use when simplifying
    // callable regions within an SCC.
    OwningRewritePatternList canonPatterns;
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(canonPatterns, context);

    // Run the inline transform in post-order over the SCCs in the callgraph.
    Inliner inliner(context, cg);
    runTransformOnCGSCCs(cg, [&](ArrayRef<CallGraphNode *> scc) {
      inlineSCC(inliner, scc, context, canonPatterns);
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}

static PassRegistration<InlinerPass> pass("inline", "Inline function calls");
