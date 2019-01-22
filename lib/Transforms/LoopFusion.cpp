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
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-fusion"

using llvm::SetVector;

using namespace mlir;

static llvm::cl::opt<bool>
    clMaximalLoopFusion("fusion-maximal", llvm::cl::Hidden,
                        llvm::cl::desc("Enables maximal loop fusion."));

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

// TODO(b/117228571) Replace when this is modeled through side-effects/op traits
static bool isMemRefDereferencingOp(const OperationInst &op) {
  if (op.isa<LoadOp>() || op.isa<StoreOp>() || op.isa<DmaStartOp>() ||
      op.isa<DmaWaitOp>())
    return true;
  return false;
}
// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level instructions in a Function which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO(andydavis) Add a more flexible dependece graph representation.
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
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value *, unsigned> memrefEdgeCount;

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

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id) {
    // Remove each edge in 'inEdges[id]'.
    if (inEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[id];
      for (auto &inEdge : oldInEdges) {
        removeEdge(inEdge.id, id, inEdge.memref);
      }
    }
    // Remove each edge in 'outEdges[id]'.
    if (outEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[id];
      for (auto &outEdge : oldOutEdges) {
        removeEdge(id, outEdge.id, outEdge.memref);
      }
    }
    // Erase remaining node state.
    inEdges.erase(id);
    outEdges.erase(id);
    nodes.erase(id);
  }

  bool hasOutEdges(unsigned id) {
    return outEdges.count(id) > 0 && !outEdges[id].empty();
  }

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the function/block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id) {
    Node *node = getNode(id);
    for (auto *storeOpInst : node->stores) {
      auto *memref = storeOpInst->cast<StoreOp>()->getMemRef();
      auto *inst = memref->getDefiningInst();
      auto *opInst = dyn_cast_or_null<OperationInst>(inst);
      // Return false if 'memref' is a function argument.
      if (opInst == nullptr)
        return true;
      // Return false if any use of 'memref' escapes the function.
      for (auto &use : memref->getUses()) {
        auto *user = dyn_cast<OperationInst>(use.getOwner());
        if (!user || !isMemRefDereferencingOp(*user))
          return true;
      }
    }
    return false;
  }

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' for
  // 'memref'. Returns false otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value *memref) {
    if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
      return false;
    }
    bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
      return edge.id == dstId && edge.memref == memref;
    });
    bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
      return edge.id == srcId && edge.memref == memref;
    });
    return hasOutEdge && hasInEdge;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'memref'.
  void addEdge(unsigned srcId, unsigned dstId, Value *memref) {
    if (!hasEdge(srcId, dstId, memref)) {
      outEdges[srcId].push_back({dstId, memref});
      inEdges[dstId].push_back({srcId, memref});
      memrefEdgeCount[memref]++;
    }
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'memref'.
  void removeEdge(unsigned srcId, unsigned dstId, Value *memref) {
    assert(inEdges.count(dstId) > 0);
    assert(outEdges.count(srcId) > 0);
    assert(memrefEdgeCount.count(memref) > 0);
    memrefEdgeCount[memref]--;
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

  // Returns the min node id across all outgoing edges from node 'id', skipping
  // edges with 'memrefToSkip'.
  unsigned getMinOutEdgeNodeId(unsigned id, Value *memrefToSkip) {
    unsigned minId = std::numeric_limits<unsigned>::max();
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        if (outEdge.memref != memrefToSkip)
          minId = std::min(minId, outEdge.id);
    return minId;
  }

  // Updates edge mappings from node 'srcId' to node 'dstId'.
  void updateEdges(unsigned srcId, unsigned dstId) {
    // For each edge in 'inEdges[srcId]': add new edge remaping to 'dstId'.
    if (inEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
      for (auto &inEdge : oldInEdges) {
        // Add edge from 'inEdge.id' to 'dstId'.
        addEdge(inEdge.id, dstId, inEdge.memref);
      }
    }
    // For each edge in 'outEdges[srcId]': remove edge from 'srcId' to 'dstId'.
    if (outEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
      for (auto &outEdge : oldOutEdges) {
        // Remove any out edges from 'srcId' to 'dstId' across memrefs.
        if (outEdge.id == dstId)
          removeEdge(srcId, outEdge.id, outEdge.memref);
      }
    }
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

  void clearNodeLoadAndStores(unsigned id) {
    Node *node = getNode(id);
    node->loads.clear();
    node->stores.clear();
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
static uint64_t getComputeCost(
    ForInst *forInst, LoopNestStats *stats,
    llvm::SmallDenseMap<ForInst *, uint64_t, 8> *tripCountOverrideMap,
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

static Optional<uint64_t> getConstDifference(AffineMap lbMap, AffineMap ubMap) {
  assert(lbMap.getNumResults() == 1 && "expected single result bound map");
  assert(ubMap.getNumResults() == 1 && "expected single result bound map");
  assert(lbMap.getNumDims() == ubMap.getNumDims());
  assert(lbMap.getNumSymbols() == ubMap.getNumSymbols());
  // TODO(andydavis) Merge this code with 'mlir::getTripCountExpr'.
  // ub_expr - lb_expr
  AffineExpr lbExpr(lbMap.getResult(0));
  AffineExpr ubExpr(ubMap.getResult(0));
  auto loopSpanExpr = simplifyAffineExpr(ubExpr - lbExpr, lbMap.getNumDims(),
                                         lbMap.getNumSymbols());
  auto cExpr = loopSpanExpr.dyn_cast<AffineConstantExpr>();
  if (!cExpr)
    return None;
  return cExpr.getValue();
}

// Builds a map 'tripCountMap' from ForInst to constant trip count for loop
// nest surrounding 'srcAccess' utilizing slice loop bounds in 'sliceState'.
// Returns true on success, false otherwise (if a non-constant trip count
// was encountered).
// TODO(andydavis) Make this work with non-unit step loops.
static bool buildSliceTripCountMap(
    OperationInst *srcOpInst, ComputationSliceState *sliceState,
    llvm::SmallDenseMap<ForInst *, uint64_t, 8> *tripCountMap) {
  SmallVector<ForInst *, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
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
    Optional<uint64_t> tripCount = getConstDifference(lbMap, ubMap);
    if (!tripCount.hasValue())
      return false;
    (*tripCountMap)[srcLoopIVs[i]] = tripCount.getValue();
  }
  return true;
}

// Removes load operations from 'srcLoads' which operate on 'memref', and
// adds them to 'dstLoads'.
static void
moveLoadsAccessingMemrefTo(Value *memref,
                           SmallVectorImpl<OperationInst *> *srcLoads,
                           SmallVectorImpl<OperationInst *> *dstLoads) {
  dstLoads->clear();
  SmallVector<OperationInst *, 4> srcLoadsToKeep;
  for (auto *load : *srcLoads) {
    if (load->cast<LoadOp>()->getMemRef() == memref)
      dstLoads->push_back(load);
    else
      srcLoadsToKeep.push_back(load);
  }
  srcLoads->swap(srcLoadsToKeep);
}

// Returns the innermost common loop depth for the set of operations in 'ops'.
static unsigned getInnermostCommonLoopDepth(ArrayRef<OperationInst *> ops) {
  unsigned numOps = ops.size();
  assert(numOps > 0);

  std::vector<SmallVector<ForInst *, 4>> loops(numOps);
  unsigned loopDepthLimit = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < numOps; ++i) {
    getLoopIVs(*ops[i], &loops[i]);
    loopDepthLimit =
        std::min(loopDepthLimit, static_cast<unsigned>(loops[i].size()));
  }

  unsigned loopDepth = 0;
  for (unsigned d = 0; d < loopDepthLimit; ++d) {
    unsigned i;
    for (i = 1; i < numOps; ++i) {
      if (loops[i - 1][d] != loops[i][d]) {
        break;
      }
    }
    if (i != numOps)
      break;
    ++loopDepth;
  }
  return loopDepth;
}

// Returns the slice union of 'sliceStateA' and 'sliceStateB' in 'sliceStateB'
// using a rectangular bounding box.
// TODO(andydavis) This function assumes that lower bounds for 'sliceStateA'
// and 'sliceStateB' are aligned.
// Specifically, when taking the union of overlapping intervals, it assumes
// that both intervals start at zero. Support needs to be added to take into
// account interval start offset when computing the union.
// TODO(andydavis) Move this function to an analysis library.
static bool getSliceUnion(const ComputationSliceState &sliceStateA,
                          ComputationSliceState *sliceStateB) {
  assert(sliceStateA.lbs.size() == sliceStateB->lbs.size());
  assert(sliceStateA.ubs.size() == sliceStateB->ubs.size());

  for (unsigned i = 0, e = sliceStateA.lbs.size(); i < e; ++i) {
    AffineMap lbMapA = sliceStateA.lbs[i];
    AffineMap ubMapA = sliceStateA.ubs[i];
    if (lbMapA == AffineMap::Null()) {
      assert(ubMapA == AffineMap::Null());
      continue;
    }
    assert(ubMapA && "expected non-null ub map");

    AffineMap lbMapB = sliceStateB->lbs[i];
    AffineMap ubMapB = sliceStateB->ubs[i];
    if (lbMapB == AffineMap::Null()) {
      assert(ubMapB == AffineMap::Null());
      // Union 'sliceStateB' does not have a bound for 'i' so copy from A.
      sliceStateB->lbs[i] = lbMapA;
      sliceStateB->ubs[i] = ubMapA;
      continue;
    }

    // TODO(andydavis) Change this code to take the min across all lower bounds
    // and max across all upper bounds for each dimension. This code can for
    // cases where a unique min or max could not be statically determined.

    // Assumption: both lower bounds are the same.
    if (lbMapA != lbMapB)
      return false;

    // Add bound with the largest trip count to union.
    Optional<uint64_t> tripCountA = getConstDifference(lbMapA, ubMapA);
    Optional<uint64_t> tripCountB = getConstDifference(lbMapB, ubMapB);
    if (!tripCountA.hasValue() || !tripCountB.hasValue())
      return false;

    if (tripCountA.getValue() > tripCountB.getValue()) {
      sliceStateB->lbs[i] = lbMapA;
      sliceStateB->ubs[i] = ubMapA;
    }
  }
  return true;
}

// Creates and returns a private (single-user) memref for fused loop rooted
// at 'forInst', with (potentially reduced) memref size based on the
// MemRefRegion written to by 'srcStoreOpInst' at depth 'dstLoopDepth'.
// TODO(bondhugula): consider refactoring the common code from generateDma and
// this one.
static Value *createPrivateMemRef(ForInst *forInst,
                                  OperationInst *srcStoreOpInst,
                                  unsigned dstLoopDepth) {
  // Create builder to insert alloc op just before 'forInst'.
  FuncBuilder b(forInst);
  // Builder to create constants at the top level.
  FuncBuilder top(forInst->getFunction());
  // Create new memref type based on slice bounds.
  auto *oldMemRef = srcStoreOpInst->cast<StoreOp>()->getMemRef();
  auto oldMemRefType = oldMemRef->getType().cast<MemRefType>();
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region;
  getMemRefRegion(srcStoreOpInst, dstLoopDepth, &region);
  SmallVector<int, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  Optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors);
  assert(numElements.hasValue());

  const FlatAffineConstraints *cst = region.getConstraints();
  // 'outerIVs' holds the values that this memory region is symbolic/paramteric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value *, 8> outerIVs;
  cst->getIdValues(rank, cst->getNumIds(), &outerIVs);

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);
    offsets.push_back(offset);
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  auto newMemRefType = b.getMemRefType(newShape, oldMemRefType.getElementType(),
                                       {}, oldMemRefType.getMemorySpace());
  // Gather alloc operands for the dynamic dimensions of the memref.
  SmallVector<Value *, 4> allocOperands;
  unsigned dynamicDimCount = 0;
  for (auto dimSize : oldMemRefType.getShape()) {
    if (dimSize == -1)
      allocOperands.push_back(
          b.create<DimOp>(forInst->getLoc(), oldMemRef, dynamicDimCount++));
  }

  // Create new private memref for fused loop 'forInst'.
  Value *newMemRef =
      b.create<AllocOp>(forInst->getLoc(), newMemRefType, allocOperands);

  // Build an AffineMap to remap access functions based on lower bound offsets.
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  unsigned zeroOffsetCount = 0;
  for (unsigned i = 0; i < rank; i++) {
    if (auto constExpr = offsets[i].dyn_cast<AffineConstantExpr>())
      if (constExpr.getValue() == 0)
        ++zeroOffsetCount;
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }
  auto indexRemap =
      zeroOffsetCount == rank
          ? AffineMap::Null()
          : b.getAffineMap(outerIVs.size() + rank, 0, remapExprs, {});
  // Replace all users of 'oldMemRef' with 'newMemRef'.
  bool ret =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*domInstFilter=*/&*forInst->getBody()->begin());
  assert(ret && "replaceAllMemrefUsesWith should always succeed here");
  (void)ret;
  return newMemRef;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstOpInsts'.
// Returns true if it profitable to fuse the candidate loop nests. Returns
// false otherwise.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest compution slice at the maximal dst loop depth (closest to the
//    load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: If the dst loop nest includes multiple loads in 'dstOpInsts' for
//    the same memref as is written by 'srcOpInst', then the union of slice
//    loop bounds is used to compute the slice and associated slice cost.
//    NOTE: 'dstLoopDepth' refers the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    execessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
static bool isFusionProfitable(OperationInst *srcOpInst,
                               ArrayRef<OperationInst *> dstOpInsts,
                               ComputationSliceState *sliceState,
                               unsigned *dstLoopDepth) {
  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<ForInst *, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
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
  getLoopIVs(*dstOpInsts[0], &dstLoopIVs);

  LoopNestStats dstLoopNestStats;
  LoopNestStatsCollector dstStatsCollector(&dstLoopNestStats);
  dstStatsCollector.walk(dstLoopIVs[0]);
  // Currently only constant trip count loop nests are supported.
  if (dstStatsCollector.hasLoopWithNonConstTripCount)
    return false;

  // Compute the innermost common loop for ops in 'dstOpInst'.
  unsigned maxDstLoopDepth = getInnermostCommonLoopDepth(dstOpInsts);
  if (maxDstLoopDepth == 0)
    return false;

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxDstLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  unsigned minFusedLoopNestComputeCost = std::numeric_limits<unsigned>::max();
  unsigned bestDstLoopDepth;
  SmallVector<ComputationSliceState, 4> sliceStates;
  sliceStates.resize(maxDstLoopDepth);

  llvm::SmallDenseMap<ForInst *, uint64_t, 8> sliceTripCountMap;
  DenseMap<ForInst *, uint64_t> computeCostMap;
  for (unsigned i = maxDstLoopDepth; i >= 1; --i) {
    MemRefAccess srcAccess(srcOpInst);
    // Handle the common case of one dst load without a copy.
    if (!mlir::getBackwardComputationSliceState(
            srcAccess, MemRefAccess(dstOpInsts[0]), i, &sliceStates[i - 1]))
      return false;
    // Compute the union of slice bound of all ops in 'dstOpInsts'.
    for (int j = 1, e = dstOpInsts.size(); j < e; ++j) {
      MemRefAccess dstAccess(dstOpInsts[j]);
      ComputationSliceState tmpSliceState;
      if (!mlir::getBackwardComputationSliceState(srcAccess, dstAccess, i,
                                                  &tmpSliceState))
        return false;
      // Compute slice boun dunion of 'tmpSliceState' and 'sliceStates[i - 1]'.
      getSliceUnion(tmpSliceState, &sliceStates[i - 1]);
    }
    // Build trip count map for computation slice.
    sliceTripCountMap.clear();
    if (!buildSliceTripCountMap(srcOpInst, &sliceStates[i - 1],
                                &sliceTripCountMap))
      return false;

    // Compute op instance count for the src loop nest with iteration slicing.
    uint64_t sliceComputeCost =
        getComputeCost(srcLoopIVs[0], &srcLoopNestStats, &sliceTripCountMap,
                       /*computeCostMap=*/nullptr);

    // Compute cost of fusion for these values of 'i' and 'j'.
    computeCostMap.clear();
    computeCostMap[dstLoopIVs[i - 1]] = sliceComputeCost;
    uint64_t fusedLoopNestComputeCost =
        getComputeCost(dstLoopIVs[0], &dstLoopNestStats,
                       /*tripCountOverrideMap=*/nullptr, &computeCostMap);
    if (fusedLoopNestComputeCost < minFusedLoopNestComputeCost) {
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      bestDstLoopDepth = i;
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
                          << " bestDstLoopDepth: " << bestDstLoopDepth
                          << " srcLoopNestCost: " << srcLoopNestCost
                          << " dstLoopNestCost: " << dstLoopNestCost
                          << " minFusedLoopNestComputeCost: "
                          << minFusedLoopNestComputeCost << "\n");

  // Do not fuse if fused loop would increase the total cost of the computation,
  // unless 'clMaximalLoopFusion' flag is set.
  // TODO(andydavis) Use locality/reduction in slice memref size/opportunity
  // for load/store forwarding in cost model.
  if (!clMaximalLoopFusion &&
      minFusedLoopNestComputeCost > srcLoopNestCost + dstLoopNestCost)
    return false;
  // Update return parameter 'sliceState' with 'bestSliceState'.
  ComputationSliceState *bestSliceState = &sliceStates[bestDstLoopDepth - 1];
  sliceState->lbs = bestSliceState->lbs;
  sliceState->ubs = bestSliceState->ubs;
  sliceState->lbOperands = bestSliceState->lbOperands;
  sliceState->ubOperands = bestSliceState->ubOperands;
  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = bestDstLoopDepth;
  // Canonicalize slice bound affine maps.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    if (sliceState->lbs[i] != AffineMap::Null()) {
      canonicalizeMapAndOperands(&sliceState->lbs[i],
                                 &sliceState->lbOperands[i]);
    }
    if (sliceState->ubs[i] != AffineMap::Null()) {
      canonicalizeMapAndOperands(&sliceState->ubs[i],
                                 &sliceState->ubOperands[i]);
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
      SmallVector<OperationInst *, 4> dstLoadOpInsts;
      DenseSet<Value *> visitedMemrefs;
      while (!loads.empty()) {
        // Get memref of load on top of the stack.
        auto *memref = loads.back()->cast<LoadOp>()->getMemRef();
        if (visitedMemrefs.count(memref) > 0)
          continue;
        visitedMemrefs.insert(memref);
        // Move all loads in 'loads' accessing 'memref' to 'dstLoadOpInsts'.
        moveLoadsAccessingMemrefTo(memref, &loads, &dstLoadOpInsts);
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
          // Skip 'srcNode' if it has in dependence edges. NOTE: This is overly
          // TODO(andydavis) Track dependence type with edges, and just check
          // for WAW dependence edge here.
          if (mdg->getInEdgeCount(srcNode->id, memref) != 0)
            continue;
          // Skip if 'srcNode' has out edges to other memrefs after 'dstId'.
          if (mdg->getMinOutEdgeNodeId(srcNode->id, memref) < dstId)
            continue;
          // Get unique 'srcNode' store op.
          auto *srcStoreOpInst = srcNode->stores.front();
          // Check if fusion would be profitable.
          unsigned dstLoopDepth;
          mlir::ComputationSliceState sliceState;
          if (!isFusionProfitable(srcStoreOpInst, dstLoadOpInsts, &sliceState,
                                  &dstLoopDepth))
            continue;
          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          auto *sliceLoopNest = mlir::insertBackwardComputationSlice(
              srcStoreOpInst, dstLoadOpInsts[0], dstLoopDepth, &sliceState);
          if (sliceLoopNest != nullptr) {
            // Update edges between 'srcNode' and 'dstNode'.
            mdg->updateEdges(srcNode->id, dstNode->id);

            // Collect slice loop stats.
            LoopNestStateCollector sliceCollector;
            sliceCollector.walkForInst(sliceLoopNest);
            // Promote single iteration slice loops to single IV value.
            for (auto *forInst : sliceCollector.forInsts) {
              promoteIfSingleIteration(forInst);
            }

            // Create private memref for 'memref' in 'dstForInst'.
            auto *dstForInst = cast<ForInst>(dstNode->inst);
            SmallVector<OperationInst *, 4> storesForMemref;
            for (auto *storeOpInst : sliceCollector.storeOpInsts) {
              if (storeOpInst->cast<StoreOp>()->getMemRef() == memref)
                storesForMemref.push_back(storeOpInst);
            }
            assert(storesForMemref.size() == 1);
            auto *newMemRef = createPrivateMemRef(
                dstForInst, storesForMemref[0], dstLoopDepth);
            visitedMemrefs.insert(newMemRef);

            // Collect dst loop stats after memref privatizaton transformation.
            LoopNestStateCollector dstLoopCollector;
            dstLoopCollector.walkForInst(dstForInst);

            // Add new load ops to current Node load op list 'loads' to
            // continue fusing based on new operands.
            for (auto *loadOpInst : dstLoopCollector.loadOpInsts) {
              auto *loadMemRef = loadOpInst->cast<LoadOp>()->getMemRef();
              if (visitedMemrefs.count(loadMemRef) == 0)
                loads.push_back(loadOpInst);
            }

            // Clear and add back loads and stores
            mdg->clearNodeLoadAndStores(dstNode->id);
            mdg->addToNode(dstId, dstLoopCollector.loadOpInsts,
                           dstLoopCollector.storeOpInsts);
            // Remove old src loop nest if it no longer has outgoing dependence
            // edges, and it does not write to a memref which escapes the
            // function.
            if (!mdg->hasOutEdges(srcNode->id) &&
                !mdg->writesToLiveInOrEscapingMemrefs(srcNode->id)) {
              mdg->removeNode(srcNode->id);
              cast<ForInst>(srcNode->inst)->erase();
            }
          }
        }
      }
    }
    // Clean up any allocs with no users.
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto *memref = pair.first;
      // Skip if there exist other uses (return instruction or function calls).
      if (!memref->use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *inst = memref->getDefiningInst();
      auto *opInst = dyn_cast_or_null<OperationInst>(inst);
      if (opInst && opInst->isa<AllocOp>())
        opInst->erase();
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
