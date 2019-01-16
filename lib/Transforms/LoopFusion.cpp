//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
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
// This file implements loop fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-fusion"

using llvm::SetVector;

using namespace mlir;

namespace {

/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.

// TODO(andydavis) Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
// TODO(andydavis) Extend this pass to check for fusion preventing dependences,
// and add support for more general loop fusion algorithms.

struct LoopFusion : public FunctionPass {
  LoopFusion() : FunctionPass(&LoopFusion::passID) {}

  PassResult runOnFunction(Function *f) override;
  static char passID;
};

} // end anonymous namespace

char LoopFusion::passID = 0;

FunctionPass *mlir::createLoopFusionPass() { return new LoopFusion; }

// FusionCandidate encapsulates source and destination memref access within
// loop nests which are candidates for loop fusion.
struct FusionCandidate {
  // Load or store access within src loop nest to be fused into dst loop nest.
  MemRefAccess srcAccess;
  // Load or store access within dst loop nest.
  MemRefAccess dstAccess;
  explicit FusionCandidate(OperationInst *src, OperationInst *dst)
      : srcAccess(MemRefAccess(src)), dstAccess(MemRefAccess(dst)) {}
};

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not an IfInst was encountered in the loop nest.
class LoopNestStateCollector : public InstWalker<LoopNestStateCollector> {
public:
  SmallVector<ForInst *, 4> forInsts;
  SmallVector<OperationInst *, 4> loadOpInsts;
  SmallVector<OperationInst *, 4> storeOpInsts;
  bool hasIfInst = false;

  void visitForInst(ForInst *forInst) { forInsts.push_back(forInst); }

  void visitIfInst(IfInst *ifInst) { hasIfInst = true; }

  void visitOperationInst(OperationInst *opInst) {
    if (opInst->isa<LoadOp>())
      loadOpInsts.push_back(opInst);
    if (opInst->isa<StoreOp>())
      storeOpInsts.push_back(opInst);
  }
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level instructions in a Function which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO(andydavis) Add a depth parameter to dependence graph construction.
struct MemRefDependenceGraph {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statment which is (or contains) loads/stores.
    Instruction *inst;
    // List of load operations.
    SmallVector<OperationInst *, 4> loads;
    // List of store op insts.
    SmallVector<OperationInst *, 4> stores;
    Node(unsigned id, Instruction *inst) : id(id), inst(inst) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value *memref) {
      unsigned loadOpCount = 0;
      for (auto *loadOpInst : loads) {
        if (memref == loadOpInst->cast<LoadOp>()->getMemRef())
          ++loadOpCount;
      }
      return loadOpCount;
    }

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value *memref) {
      unsigned storeOpCount = 0;
      for (auto *storeOpInst : stores) {
        if (memref == storeOpInst->cast<StoreOp>()->getMemRef())
          ++storeOpCount;
      }
      return storeOpCount;
    }
  };

  // Edge represents a memref data dependece between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    unsigned id;
    // The memref on which this edge represents a dependence.
    Value *memref;
  };

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;

  MemRefDependenceGraph() {}

  // Initializes the dependence graph based on operations in 'f'.
  // Returns true on success, false otherwise.
  bool init(Function *f);

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id) {
    auto it = nodes.find(id);
    assert(it != nodes.end());
    return &it->second;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'memref'.
  void addEdge(unsigned srcId, unsigned dstId, Value *memref) {
    outEdges[srcId].push_back({dstId, memref});
    inEdges[dstId].push_back({srcId, memref});
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'memref'.
  void removeEdge(unsigned srcId, unsigned dstId, Value *memref) {
    assert(inEdges.count(dstId) > 0);
    assert(outEdges.count(srcId) > 0);
    // Remove 'srcId' from 'inEdges[dstId]'.
    for (auto it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
      if ((*it).id == srcId && (*it).memref == memref) {
        inEdges[dstId].erase(it);
        break;
      }
    }
    // Remove 'dstId' from 'outEdges[srcId]'.
    for (auto it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
      if ((*it).id == dstId && (*it).memref == memref) {
        outEdges[srcId].erase(it);
        break;
      }
    }
  }

  // Returns the input edge count for node 'id' and 'memref'.
  unsigned getInEdgeCount(unsigned id, Value *memref) {
    unsigned inEdgeCount = 0;
    if (inEdges.count(id) > 0)
      for (auto &inEdge : inEdges[id])
        if (inEdge.memref == memref)
          ++inEdgeCount;
    return inEdgeCount;
  }

  // Returns the output edge count for node 'id' and 'memref'.
  unsigned getOutEdgeCount(unsigned id, Value *memref) {
    unsigned outEdgeCount = 0;
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        if (outEdge.memref == memref)
          ++outEdgeCount;
    return outEdgeCount;
  }

  // Returns the min node id of all output edges from node 'id'.
  unsigned getMinOutEdgeNodeId(unsigned id) {
    unsigned minId = std::numeric_limits<unsigned>::max();
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        minId = std::min(minId, outEdge.id);
    return minId;
  }

  // Updates edge mappings from node 'srcId' to node 'dstId' and removes
  // state associated with node 'srcId'.
  void updateEdgesAndRemoveSrcNode(unsigned srcId, unsigned dstId) {
    // For each edge in 'inEdges[srcId]': add new edge remaping to 'dstId'.
    if (inEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
      for (auto &inEdge : oldInEdges) {
        // Remove edge from 'inEdge.id' to 'srcId'.
        removeEdge(inEdge.id, srcId, inEdge.memref);
        // Add edge from 'inEdge.id' to 'dstId'.
        addEdge(inEdge.id, dstId, inEdge.memref);
      }
    }
    // For each edge in 'outEdges[srcId]': add new edge remaping to 'dstId'.
    if (outEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
      for (auto &outEdge : oldOutEdges) {
        // Remove edge from 'srcId' to 'outEdge.id'.
        removeEdge(srcId, outEdge.id, outEdge.memref);
        // Add edge from 'dstId' to 'outEdge.id' (if 'outEdge.id' != 'dstId').
        if (outEdge.id != dstId)
          addEdge(dstId, outEdge.id, outEdge.memref);
      }
    }
    // Remove 'srcId' from graph state.
    inEdges.erase(srcId);
    outEdges.erase(srcId);
    nodes.erase(srcId);
  }

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<OperationInst *> &loads,
                 const SmallVectorImpl<OperationInst *> &stores) {
    Node *node = getNode(id);
    for (auto *loadOpInst : loads)
      node->loads.push_back(loadOpInst);
    for (auto *storeOpInst : stores)
      node->stores.push_back(storeOpInst);
  }

  void print(raw_ostream &os) const {
    os << "\nMemRefDependenceGraph\n";
    os << "\nNodes:\n";
    for (auto &idAndNode : nodes) {
      os << "Node: " << idAndNode.first << "\n";
      auto it = inEdges.find(idAndNode.first);
      if (it != inEdges.end()) {
        for (const auto &e : it->second)
          os << "  InEdge: " << e.id << " " << e.memref << "\n";
      }
      it = outEdges.find(idAndNode.first);
      if (it != outEdges.end()) {
        for (const auto &e : it->second)
          os << "  OutEdge: " << e.id << " " << e.memref << "\n";
      }
    }
  }
  void dump() const { print(llvm::errs()); }
};

// Intializes the data dependence graph by walking instructions in 'f'.
// Assigns each node in the graph a node id based on program order in 'f'.
// TODO(andydavis) Add support for taking a Block arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init(Function *f) {
  unsigned id = 0;
  DenseMap<Value *, SetVector<unsigned>> memrefAccesses;

  // TODO: support multi-block functions.
  if (f->getBlocks().size() != 1)
    return false;

  for (auto &inst : f->front()) {
    if (auto *forInst = dyn_cast<ForInst>(&inst)) {
      // Create graph node 'id' to represent top-level 'forInst' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.walkForInst(forInst);
      // Return false if IfInsts are found (not currently supported).
      if (collector.hasIfInst)
        return false;
      Node node(id++, &inst);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto *memref = opInst->cast<LoadOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto *memref = opInst->cast<StoreOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      nodes.insert({node.id, node});
    }
    if (auto *opInst = dyn_cast<OperationInst>(&inst)) {
      if (auto loadOp = opInst->dyn_cast<LoadOp>()) {
        // Create graph node for top-level load op.
        Node node(id++, &inst);
        node.loads.push_back(opInst);
        auto *memref = opInst->cast<LoadOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
        nodes.insert({node.id, node});
      }
      if (auto storeOp = opInst->dyn_cast<StoreOp>()) {
        // Create graph node for top-level store op.
        Node node(id++, &inst);
        node.stores.push_back(opInst);
        auto *memref = opInst->cast<StoreOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
        nodes.insert({node.id, node});
      }
    }
    // Return false if IfInsts are found (not currently supported).
    if (isa<IfInst>(&inst))
      return false;
  }

  // Walk memref access lists and add graph edges between dependent nodes.
  for (auto &memrefAndList : memrefAccesses) {
    unsigned n = memrefAndList.second.size();
    for (unsigned i = 0; i < n; ++i) {
      unsigned srcId = memrefAndList.second[i];
      bool srcHasStore =
          getNode(srcId)->getStoreOpCount(memrefAndList.first) > 0;
      for (unsigned j = i + 1; j < n; ++j) {
        unsigned dstId = memrefAndList.second[j];
        bool dstHasStore =
            getNode(dstId)->getStoreOpCount(memrefAndList.first) > 0;
        if (srcHasStore || dstHasStore)
          addEdge(srcId, dstId, memrefAndList.first);
      }
    }
  }
  return true;
}

namespace {

// LoopNestStats aggregates various per-loop statistics (eg. loop trip count
// and operation count) for a loop nest up until the innermost loop body.
struct LoopNestStats {
  // Map from ForInst to immediate child ForInsts in its loop body.
  DenseMap<ForInst *, SmallVector<ForInst *, 2>> loopMap;
  // Map from ForInst to count of operations in its loop body.
  DenseMap<ForInst *, uint64_t> opCountMap;
  // Map from ForInst to its constant trip count.
  DenseMap<ForInst *, uint64_t> tripCountMap;
};

// LoopNestStatsCollector walks a single loop nest and gathers per-loop
// trip count and operation count statistics and records them in 'stats'.
class LoopNestStatsCollector : public InstWalker<LoopNestStatsCollector> {
public:
  LoopNestStats *stats;
  bool hasLoopWithNonConstTripCount = false;

  LoopNestStatsCollector(LoopNestStats *stats) : stats(stats) {}

  void visitForInst(ForInst *forInst) {
    auto *parentInst = forInst->getParentInst();
    if (parentInst != nullptr) {
      assert(isa<ForInst>(parentInst) && "Expected parent ForInst");
      // Add mapping to 'forInst' from its parent ForInst.
      stats->loopMap[cast<ForInst>(parentInst)].push_back(forInst);
    }
    // Record the number of op instructions in the body of 'forInst'.
    unsigned count = 0;
    stats->opCountMap[forInst] = 0;
    for (auto &inst : *forInst->getBody()) {
      if (isa<OperationInst>(&inst))
        ++count;
    }
    stats->opCountMap[forInst] = count;
    // Record trip count for 'forInst'. Set flag if trip count is not constant.
    Optional<uint64_t> maybeConstTripCount = getConstantTripCount(*forInst);
    if (!maybeConstTripCount.hasValue()) {
      hasLoopWithNonConstTripCount = true;
      return;
    }
    stats->tripCountMap[forInst] = maybeConstTripCount.getValue();
  }
};

// Computes the total cost of the loop nest rooted at 'forInst'.
// Currently, the total cost is computed by counting the total operation
// instance count (i.e. total number of operations in the loop bodyloop
// operation count * loop trip count) for the entire loop nest.
// If 'tripCountOverrideMap' is non-null, overrides the trip count for loops
// specified in the map when computing the total op instance count.
// NOTE: this is used to compute the cost of computation slices, which are
// sliced along the iteration dimension, and thus reduce the trip count.
// If 'computeCostMap' is non-null, the total op count for forInsts specified
// in the map is increased (not overridden) by adding the op count from the
// map to the existing op count for the for loop. This is done before
// multiplying by the loop's trip count, and is used to model the cost of
// inserting a sliced loop nest of known cost into the loop's body.
// NOTE: this is used to compute the cost of fusing a slice of some loop nest
// within another loop.
static uint64_t
getComputeCost(ForInst *forInst, LoopNestStats *stats,
               DenseMap<ForInst *, uint64_t> *tripCountOverrideMap,
               DenseMap<ForInst *, uint64_t> *computeCostMap) {
  // 'opCount' is the total number operations in one iteration of 'forInst' body
  uint64_t opCount = stats->opCountMap[forInst];
  if (stats->loopMap.count(forInst) > 0) {
    for (auto *childForInst : stats->loopMap[forInst]) {
      opCount += getComputeCost(childForInst, stats, tripCountOverrideMap,
                                computeCostMap);
    }
  }
  // Add in additional op instances from slice (if specified in map).
  if (computeCostMap != nullptr) {
    auto it = computeCostMap->find(forInst);
    if (it != computeCostMap->end()) {
      opCount += it->second;
    }
  }
  // Override trip count (if specified in map).
  uint64_t tripCount = stats->tripCountMap[forInst];
  if (tripCountOverrideMap != nullptr) {
    auto it = tripCountOverrideMap->find(forInst);
    if (it != tripCountOverrideMap->end()) {
      tripCount = it->second;
    }
  }
  // Returns the total number of dynamic instances of operations in loop body.
  return tripCount * opCount;
}

} // end anonymous namespace

// Builds a map 'tripCountMap' from ForInst to constant trip count for loop
// nest surrounding 'srcAccess' utilizing slice loop bounds in 'sliceState'.
// Returns true on success, false otherwise (if a non-constant trip count
// was encountered).
// TODO(andydavis) Make this work with non-unit step loops.
static bool
buildSliceTripCountMap(MemRefAccess *srcAccess,
                       ComputationSliceState *sliceState,
                       DenseMap<ForInst *, uint64_t> *tripCountMap) {
  SmallVector<ForInst *, 4> srcLoopIVs;
  getLoopIVs(*srcAccess->opInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();
  // Populate map from ForInst -> trip count
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    AffineMap lbMap = sliceState->lbs[i];
    AffineMap ubMap = sliceState->ubs[i];
    if (lbMap == AffineMap::Null() || ubMap == AffineMap::Null()) {
      // The iteration of src loop IV 'i' was not sliced. Use full loop bounds.
      if (srcLoopIVs[i]->hasConstantLowerBound() &&
          srcLoopIVs[i]->hasConstantUpperBound()) {
        (*tripCountMap)[srcLoopIVs[i]] =
            srcLoopIVs[i]->getConstantUpperBound() -
            srcLoopIVs[i]->getConstantLowerBound();
        continue;
      }
      return false;
    }
    // TODO(andydavis) Merge this code with 'mlir::getTripCountExpr'.
    // ub_expr - lb_expr
    AffineExpr lbExpr(lbMap.getResult(0));
    AffineExpr ubExpr(ubMap.getResult(0));
    auto loopSpanExpr = simplifyAffineExpr(
        ubExpr - lbExpr, std::max(lbMap.getNumDims(), ubMap.getNumDims()),
        std::max(lbMap.getNumSymbols(), ubMap.getNumSymbols()));
    auto cExpr = loopSpanExpr.dyn_cast<AffineConstantExpr>();
    if (!cExpr)
      return false;
    (*tripCountMap)[srcLoopIVs[i]] = cExpr.getValue();
  }
  return true;
}

// Returns the maximum loop depth within the source loop nest at which a
// sliced loop bound is detected in 'sliceState'.
static unsigned getMaxSrcLoopDepth(unsigned srcLoopDepthLimit,
                                   ComputationSliceState *sliceState) {
  unsigned maxSrcPos = 0;
  for (unsigned i = 0; i < srcLoopDepthLimit; ++i) {
    if (sliceState->lbs[i] != AffineMap::Null() &&
        sliceState->ubs[i] != AffineMap::Null()) {
      maxSrcPos = std::max(maxSrcPos, i);
    }
  }
  return maxSrcPos + 1;
}

// Returns the minimum loop depth within the destination loop nest at which the
// computation slice can be inserted (based on the destination loop IVs that
// the source slice actually depends on / is a function of).
static unsigned getMinDstLoopDepth(unsigned srcLoopDepth,
                                   ComputationSliceState *sliceState) {
  // Record in 'maxDstLoopDepth' the largest position (+1) of a dst loop nest
  // IV, which is used in a sliced loop bound in the src loop nest.
  unsigned maxDstLoopDepth = 0;
  for (unsigned i = 0; i < srcLoopDepth; ++i) {
    if (AffineMap lbMap = sliceState->lbs[i]) {
      lbMap.walkExprs([&](AffineExpr expr) {
        if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
          maxDstLoopDepth =
              std::max(maxDstLoopDepth, dimExpr.getPosition() + 1);
        }
      });
    }
    if (AffineMap ubMap = sliceState->ubs[i]) {
      ubMap.walkExprs([&](AffineExpr expr) {
        if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
          maxDstLoopDepth =
              std::max(maxDstLoopDepth, dimExpr.getPosition() + 1);
        }
      });
    }
  }
  return maxDstLoopDepth;
}

// Checks the profitability of fusion candidate 'candidate'. Returns true if it
// profitable to fuse the candidate loop nests. Returns false otherwise.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'candidate.srcAccess'. This
//    computation slice of the loop nest surrounding 'candidate.srcAccess' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'candidate.dstAccess'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of src/dst loop depth, attempting to fuse
//    the biggest compution slice (max src loop depth) at the maximal dst loop
//    depth (closest to the load) to minimize reuse distance and opportunity for
//    subsequent load/store forwarding.
//    NOTE: 'srcLoopDepth' refers to the loop depth within the source loop nest
//    at which we slice the loops bounds (all src loops below this depth will
//    utilize full loop bounds).
//    NOTE: 'dstLoopDepth' refers the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the source loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fused an unsliced
//    loop (within the src computation slice) at a depth which results in
//    execessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
static bool isFusionProfitable(FusionCandidate *candidate,
                               ComputationSliceState *sliceState,
                               unsigned *srcLoopDepth, unsigned *dstLoopDepth) {
  // Compute backward computation slice state: src IV bounds w.r.t dst IVs, etc.
  if (!mlir::getBackwardComputationSliceState(
          candidate->srcAccess, candidate->dstAccess, sliceState)) {
    return false;
  }

  // Build trip count map for src loops with sliced loop bounds in 'sliceState'.
  DenseMap<ForInst *, uint64_t> sliceTripCountMap;
  if (!buildSliceTripCountMap(&candidate->srcAccess, sliceState,
                              &sliceTripCountMap))
    return false;

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<ForInst *, 4> srcLoopIVs;
  getLoopIVs(*candidate->srcAccess.opInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  LoopNestStatsCollector srcStatsCollector(&srcLoopNestStats);
  srcStatsCollector.walk(srcLoopIVs[0]);
  // Currently only constant trip count loop nests are supported.
  if (srcStatsCollector.hasLoopWithNonConstTripCount)
    return false;

  // Compute cost of dst loop nest.
  SmallVector<ForInst *, 4> dstLoopIVs;
  getLoopIVs(*candidate->dstAccess.opInst, &dstLoopIVs);
  unsigned numDstLoopIVs = dstLoopIVs.size();

  LoopNestStats dstLoopNestStats;
  LoopNestStatsCollector dstStatsCollector(&dstLoopNestStats);
  dstStatsCollector.walk(dstLoopIVs[0]);
  // Currently only constant trip count loop nests are supported.
  if (dstStatsCollector.hasLoopWithNonConstTripCount)
    return false;

  // Search for min cost values for 'srcLoopDepth' and 'dstLoopDepth'.
  // This search is O(n^2) where 'n' is very small (eg. six).
  // TODO(andydavis) Consider a solution where we just iteration through
  // dstLoopDepth possibilities and project out IVs we do not need (remove
  // dependence on 'srcLoopDepth'.
  DenseMap<ForInst *, uint64_t> tripCountMap;
  DenseMap<ForInst *, uint64_t> computeCostMap;
  unsigned maxSrcLoopDepth = getMaxSrcLoopDepth(numSrcLoopIVs, sliceState);
  unsigned minFusedLoopNestComputeCost = std::numeric_limits<unsigned>::max();
  unsigned bestSrcLoopDepth;
  unsigned bestDstLoopDepth;
  for (unsigned i = maxSrcLoopDepth; i >= 1; --i) {
    // Compute minDstLoopDepth based on dst loop IVs used in slice loop bounds.
    unsigned minDstLoopDepth = getMinDstLoopDepth(i, sliceState);
    assert(minDstLoopDepth <= numDstLoopIVs);
    if (minDstLoopDepth == 0) {
      // TODO(andydavis) Support inserting computation slices at top-level.
      continue;
    }
    // Copy elements from slice trip count map up to src loop depth 'i'.
    tripCountMap.clear();
    for (unsigned k = 0; k < i; ++k) {
      auto *forInst = srcLoopIVs[k];
      auto it = sliceTripCountMap.find(forInst);
      if (it != sliceTripCountMap.end()) {
        tripCountMap[forInst] = it->second;
      }
    }
    // Compute op instance count for the src loop nest with iteration slicing.
    uint64_t sliceComputeCost =
        getComputeCost(srcLoopIVs[0], &srcLoopNestStats, &tripCountMap,
                       /*computeCostMap=*/nullptr);

    for (unsigned j = numDstLoopIVs; j >= minDstLoopDepth; --j) {
      // Compute cost of fusion for these values of 'i' and 'j'.
      computeCostMap.clear();
      computeCostMap[dstLoopIVs[j - 1]] = sliceComputeCost;
      uint64_t fusedLoopNestComputeCost =
          getComputeCost(dstLoopIVs[0], &dstLoopNestStats,
                         /*tripCountOverrideMap=*/nullptr, &computeCostMap);
      if (fusedLoopNestComputeCost < minFusedLoopNestComputeCost) {
        minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
        bestSrcLoopDepth = i;
        bestDstLoopDepth = j;
      }
    }
  }

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], &srcLoopNestStats,
                                            /*tripCountOverrideMap=*/nullptr,
                                            /*computeCostMap=*/nullptr);
  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstLoopIVs[0], &dstLoopNestStats,
                                            /*tripCountOverrideMap=*/nullptr,
                                            /*computeCostMap=*/nullptr);

  LLVM_DEBUG(llvm::dbgs() << "LoopFusion statistics "
                          << " bestSrcLoopDepth: " << bestSrcLoopDepth
                          << " bestDstLoopDepth: " << bestDstLoopDepth
                          << " srcLoopNestCost: " << srcLoopNestCost
                          << " dstLoopNestCost: " << dstLoopNestCost
                          << " minFusedLoopNestComputeCost: "
                          << minFusedLoopNestComputeCost << "\n");

  // Do not fuse if fused loop would increase the total cost of the computation.
  // TODO(andydavis) Use locality/reduction in slice memref size/opportunity
  // for load/store forwarding in cost model.
  if (minFusedLoopNestComputeCost > srcLoopNestCost + dstLoopNestCost)
    return false;
  // Set src/dstLoopDepth based on best values from search.
  *srcLoopDepth = bestSrcLoopDepth;
  *dstLoopDepth = bestDstLoopDepth;
  // Update 'sliceState' bounds based on computed 'srcLoopDepth':
  // *) Canonicalize affine map now that 'srcLoopDepth' has been chosen.
  // *) Replace slice bound maps at depth > 'srcLoopDepth' withAffineMap::Null()
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    if (i < bestSrcLoopDepth) {
      if (sliceState->lbs[i] != AffineMap::Null()) {
        canonicalizeMapAndOperands(&sliceState->lbs[i],
                                   &sliceState->lbOperands[i]);
      }
      if (sliceState->ubs[i] != AffineMap::Null()) {
        canonicalizeMapAndOperands(&sliceState->ubs[i],
                                   &sliceState->ubOperands[i]);
      }
    } else {
      sliceState->lbs[i] = AffineMap::Null();
      sliceState->ubs[i] = AffineMap::Null();
    }
  }
  return true;
}

// GreedyFusion greedily fuses loop nests which have a producer/consumer
// relationship on a memref, with the goal of improving locality. Currently,
// this the producer/consumer relationship is required to be unique in the
// Function (there are TODOs to relax this constraint in the future).
//
// The steps of the algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop a ForInst of the worklist. This 'dstForInst' will be a candidate
//      destination ForInst into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstForInst' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Lookup dependent loop nests at earlier positions in the Function
//         which have a single store op to the same memref.
//      *) Check if dependences would be violated by the fusion. For example,
//         the src loop nest may load from memrefs which are different than
//         the producer-consumer memref between src and dest loop nests.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         just before the dst load op user.
//      *) Add the newly fused load/store operation instructions to the state,
//         and also add newly fuse load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// Given a graph where top-level instructions are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO to fix this.
//
// TODO(andydavis) Experiment with other fusion policies.
// TODO(andydavis) Add support for fusing for input reuse (perhaps by
// constructing a graph with edges which represent loads from the same memref
// in two different loop nestst.
struct GreedyFusion {
public:
  MemRefDependenceGraph *mdg;
  SmallVector<unsigned, 4> worklist;

  GreedyFusion(MemRefDependenceGraph *mdg) : mdg(mdg) {
    // Initialize worklist with nodes from 'mdg'.
    worklist.resize(mdg->nodes.size());
    std::iota(worklist.begin(), worklist.end(), 0);
  }

  void run() {
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<ForInst>(dstNode->inst))
        continue;

      SmallVector<OperationInst *, 4> loads = dstNode->loads;
      while (!loads.empty()) {
        auto *dstLoadOpInst = loads.pop_back_val();
        auto *memref = dstLoadOpInst->cast<LoadOp>()->getMemRef();
        // Skip 'dstLoadOpInst' if multiple loads to 'memref' in 'dstNode'.
        if (dstNode->getLoadOpCount(memref) != 1)
          continue;
        // Skip if no input edges along which to fuse.
        if (mdg->inEdges.count(dstId) == 0)
          continue;
        // Iterate through in edges for 'dstId'.
        for (auto &srcEdge : mdg->inEdges[dstId]) {
          // Skip 'srcEdge' if not for 'memref'.
          if (srcEdge.memref != memref)
            continue;
          auto *srcNode = mdg->getNode(srcEdge.id);
          // Skip if 'srcNode' is not a loop nest.
          if (!isa<ForInst>(srcNode->inst))
            continue;
          // Skip if 'srcNode' has more than one store to 'memref'.
          if (srcNode->getStoreOpCount(memref) != 1)
            continue;
          // Skip 'srcNode' if it has out edges on 'memref' other than 'dstId'.
          if (mdg->getOutEdgeCount(srcNode->id, memref) != 1)
            continue;
          // Skip 'srcNode' if it has in dependence edges. NOTE: This is overly
          // TODO(andydavis) Track dependence type with edges, and just check
          // for WAW dependence edge here.
          if (mdg->getInEdgeCount(srcNode->id, memref) != 0)
            continue;
          // Skip if 'srcNode' has out edges to other memrefs after 'dstId'.
          if (mdg->getMinOutEdgeNodeId(srcNode->id) != dstId)
            continue;
          // Get unique 'srcNode' store op.
          auto *srcStoreOpInst = srcNode->stores.front();
          // Build fusion candidate out of 'srcStoreOpInst' and 'dstLoadOpInst'.
          FusionCandidate candidate(srcStoreOpInst, dstLoadOpInst);
          // Check if fusion would be profitable.
          unsigned srcLoopDepth;
          unsigned dstLoopDepth;
          mlir::ComputationSliceState sliceState;
          if (!isFusionProfitable(&candidate, &sliceState, &srcLoopDepth,
                                  &dstLoopDepth))
            continue;
          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          auto *sliceLoopNest = mlir::insertBackwardComputationSlice(
              &candidate.srcAccess, &candidate.dstAccess, &sliceState,
              srcLoopDepth, dstLoopDepth);
          if (sliceLoopNest != nullptr) {
            // Remove edges between 'srcNode' and 'dstNode' and remove 'srcNode'
            mdg->updateEdgesAndRemoveSrcNode(srcNode->id, dstNode->id);
            // Record all load/store accesses in 'sliceLoopNest' at 'dstPos'.
            LoopNestStateCollector collector;
            collector.walkForInst(sliceLoopNest);
            mdg->addToNode(dstId, collector.loadOpInsts,
                           collector.storeOpInsts);
            // Add new load ops to current Node load op list 'loads' to
            // continue fusing based on new operands.
            for (auto *loadOpInst : collector.loadOpInsts)
              loads.push_back(loadOpInst);
            // Promote single iteration loops to single IV value.
            for (auto *forInst : collector.forInsts) {
              promoteIfSingleIteration(forInst);
            }
            // Remove old src loop nest.
            cast<ForInst>(srcNode->inst)->erase();
          }
        }
      }
    }
  }
};

} // end anonymous namespace

PassResult LoopFusion::runOnFunction(Function *f) {
  MemRefDependenceGraph g;
  if (g.init(f))
    GreedyFusion(&g).run();
  return success();
}

static PassRegistration<LoopFusion> pass("loop-fusion", "Fuse loop nests");
