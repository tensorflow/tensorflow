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
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CallGraph traversal
//===----------------------------------------------------------------------===//

/// Run a given transformation over the SCCs of the callgraph in a bottom up
/// traversal.
static void runTransformOnCGSCCs(
    const CallGraph &cg,
    function_ref<void(ArrayRef<CallGraphNode *>)> sccTransformer) {
  for (auto cgi = llvm::scc_begin(&cg); !cgi.isAtEnd(); ++cgi)
    sccTransformer(*cgi);
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
static void collectCallOps(llvm::iterator_range<Region::iterator> blocks,
                           CallGraph &cg, SmallVectorImpl<ResolvedCall> &calls,
                           bool traverseNestedCGNodes) {
  SmallVector<Block *, 8> worklist;
  auto addToWorklist = [&](llvm::iterator_range<Region::iterator> blocks) {
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
  void processInlinedBlocks(
      llvm::iterator_range<Region::iterator> inlinedBlocks) final {
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

/// Attempt to inline calls within the given scc.
static void inlineCallsInSCC(Inliner &inliner,
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
    return;

  // Try to inline each of the call operations. Don't cache the end iterator
  // here as more calls may be added during inlining.
  for (unsigned i = 0; i != calls.size(); ++i) {
    ResolvedCall &it = calls[i];
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
  }
  calls.clear();
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
    Inliner inliner(&getContext(), cg);

    // Run the inline transform in post-order over the SCCs in the callgraph.
    runTransformOnCGSCCs(cg, [&](ArrayRef<CallGraphNode *> scc) {
      inlineCallsInSCC(inliner, scc);
    });
  }
};
} // end anonymous namespace

static PassRegistration<InlinerPass> pass("inline", "Inline function calls");
