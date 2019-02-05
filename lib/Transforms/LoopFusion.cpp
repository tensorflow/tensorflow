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

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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
#include <iomanip>

#define DEBUG_TYPE "loop-fusion"

using llvm::SetVector;

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

/// Disables fusion profitability check and fuses if valid.
static llvm::cl::opt<bool>
    clMaximalLoopFusion("fusion-maximal", llvm::cl::Hidden,
                        llvm::cl::desc("Enables maximal loop fusion"),
                        llvm::cl::cat(clOptionsCategory));

/// A threshold in percent of additional computation allowed when fusing.
static llvm::cl::opt<double> clFusionAddlComputeTolerance(
    "fusion-compute-tolerance", llvm::cl::Hidden,
    llvm::cl::desc("Fractional increase in additional"
                   " computation tolerated while fusing"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clFusionFastMemorySpace(
    "fusion-fast-mem-space", llvm::cl::Hidden,
    llvm::cl::desc("Faster memory space number to promote fusion buffers to"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clFusionLocalBufThreshold(
    "fusion-local-buf-threshold", llvm::cl::Hidden,
    llvm::cl::desc("Threshold size (bytes) for promoting local buffers to fast "
                   "memory space"),
    llvm::cl::cat(clOptionsCategory));

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

  // Any local buffers smaller than this size will be created in
  // `fastMemorySpace` if provided.
  unsigned localBufSizeThreshold = 1024;
  Optional<unsigned> fastMemorySpace = None;

  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  constexpr static double kComputeToleranceThreshold = 0.30f;
};

} // end anonymous namespace

char LoopFusion::passID = 0;

FunctionPass *mlir::createLoopFusionPass() { return new LoopFusion; }

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not an IfInst was encountered in the loop nest.
struct LoopNestStateCollector {
  SmallVector<OpPointer<AffineForOp>, 4> forOps;
  SmallVector<Instruction *, 4> loadOpInsts;
  SmallVector<Instruction *, 4> storeOpInsts;
  bool hasNonForRegion = false;

  void collect(Instruction *instToWalk) {
    instToWalk->walk([&](Instruction *opInst) {
      if (opInst->isa<AffineForOp>())
        forOps.push_back(opInst->cast<AffineForOp>());
      else if (opInst->getNumBlockLists() != 0)
        hasNonForRegion = true;
      else if (opInst->isa<LoadOp>())
        loadOpInsts.push_back(opInst);
      else if (opInst->isa<StoreOp>())
        storeOpInsts.push_back(opInst);
    });
  }
};

// TODO(b/117228571) Replace when this is modeled through side-effects/op traits
static bool isMemRefDereferencingOp(const Instruction &op) {
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
    SmallVector<Instruction *, 4> loads;
    // List of store op insts.
    SmallVector<Instruction *, 4> stores;
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

  // Edge represents a data dependece between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant instruction defining a value which is used inside a loop
    // nest).
    Value *value;
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
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

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

  // Adds a node with 'inst' to the graph and returns its unique identifier.
  unsigned addNode(Instruction *inst) {
    Node node(nextNodeId++, inst);
    nodes.insert({node.id, node});
    return node.id;
  }

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id) {
    // Remove each edge in 'inEdges[id]'.
    if (inEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[id];
      for (auto &inEdge : oldInEdges) {
        removeEdge(inEdge.id, id, inEdge.value);
      }
    }
    // Remove each edge in 'outEdges[id]'.
    if (outEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[id];
      for (auto &outEdge : oldOutEdges) {
        removeEdge(id, outEdge.id, outEdge.value);
      }
    }
    // Erase remaining node state.
    inEdges.erase(id);
    outEdges.erase(id);
    nodes.erase(id);
  }

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the function/block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id) {
    Node *node = getNode(id);
    for (auto *storeOpInst : node->stores) {
      auto *memref = storeOpInst->cast<StoreOp>()->getMemRef();
      auto *inst = memref->getDefiningInst();
      // Return false if 'memref' is a block argument.
      if (!inst)
        return true;
      // Return false if any use of 'memref' escapes the function.
      for (auto &use : memref->getUses())
        if (!isMemRefDereferencingOp(*use.getOwner()))
          return true;
    }
    return false;
  }

  // Returns true if node 'id' can be removed from the graph. Returns false
  // otherwise. A node can be removed from the graph iff the following
  // conditions are met:
  // *) The node does not write to any memref which escapes (or is a
  //    function/block argument).
  // *) The node has no successors in the dependence graph.
  bool canRemoveNode(unsigned id) {
    if (writesToLiveInOrEscapingMemrefs(id))
      return false;
    Node *node = getNode(id);
    for (auto *storeOpInst : node->stores) {
      // Return false if there exist out edges from 'id' on 'memref'.
      if (getOutEdgeCount(id, storeOpInst->cast<StoreOp>()->getMemRef()) > 0)
        return false;
    }
    return true;
  }

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' for
  // 'value'. Returns false otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value *value) {
    if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
      return false;
    }
    bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
      return edge.id == dstId && edge.value == value;
    });
    bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
      return edge.id == srcId && edge.value == value;
    });
    return hasOutEdge && hasInEdge;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value *value) {
    if (!hasEdge(srcId, dstId, value)) {
      outEdges[srcId].push_back({dstId, value});
      inEdges[dstId].push_back({srcId, value});
      if (value->getType().isa<MemRefType>())
        memrefEdgeCount[value]++;
    }
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value *value) {
    assert(inEdges.count(dstId) > 0);
    assert(outEdges.count(srcId) > 0);
    if (value->getType().isa<MemRefType>()) {
      assert(memrefEdgeCount.count(value) > 0);
      memrefEdgeCount[value]--;
    }
    // Remove 'srcId' from 'inEdges[dstId]'.
    for (auto it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
      if ((*it).id == srcId && (*it).value == value) {
        inEdges[dstId].erase(it);
        break;
      }
    }
    // Remove 'dstId' from 'outEdges[srcId]'.
    for (auto it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
      if ((*it).id == dstId && (*it).value == value) {
        outEdges[srcId].erase(it);
        break;
      }
    }
  }

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref'.
  unsigned getIncomingMemRefAccesses(unsigned id, Value *memref) {
    unsigned inEdgeCount = 0;
    if (inEdges.count(id) > 0)
      for (auto &inEdge : inEdges[id])
        if (inEdge.value == memref) {
          Node *srcNode = getNode(inEdge.id);
          // Only count in edges from 'srcNode' if 'srcNode' accesses 'memref'
          if (srcNode->getLoadOpCount(memref) > 0 ||
              srcNode->getStoreOpCount(memref) > 0)
            ++inEdgeCount;
        }
    return inEdgeCount;
  }

  // Returns the output edge count for node 'id' and 'memref'.
  unsigned getOutEdgeCount(unsigned id, Value *memref) {
    unsigned outEdgeCount = 0;
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        if (outEdge.value == memref)
          ++outEdgeCount;
    return outEdgeCount;
  }

  // Computes and returns an insertion point instruction, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Instruction *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId,
                                              Value *memrefToSkip) {
    if (outEdges.count(srcId) == 0)
      return getNode(dstId)->inst;

    // Build set of insts in range (srcId, dstId) which depend on 'srcId'.
    SmallPtrSet<Instruction *, 2> srcDepInsts;
    for (auto &outEdge : outEdges[srcId])
      if (outEdge.id != dstId && outEdge.value != memrefToSkip)
        srcDepInsts.insert(getNode(outEdge.id)->inst);

    // Build set of insts in range (srcId, dstId) on which 'dstId' depends.
    SmallPtrSet<Instruction *, 2> dstDepInsts;
    for (auto &inEdge : inEdges[dstId])
      if (inEdge.id != srcId && inEdge.value != memrefToSkip)
        dstDepInsts.insert(getNode(inEdge.id)->inst);

    Instruction *srcNodeInst = getNode(srcId)->inst;
    Instruction *dstNodeInst = getNode(dstId)->inst;

    // Computing insertion point:
    // *) Walk all instruction positions in Block instruction list in the
    //    range (src, dst). For each instruction 'inst' visited in this search:
    //   *) Store in 'firstSrcDepPos' the first position where 'inst' has a
    //      dependence edge from 'srcNode'.
    //   *) Store in 'lastDstDepPost' the last position where 'inst' has a
    //      dependence edge to 'dstNode'.
    // *) Compare 'firstSrcDepPos' and 'lastDstDepPost' to determine the
    //    instruction insertion point (or return null pointer if no such
    //    insertion point exists: 'firstSrcDepPos' <= 'lastDstDepPos').
    SmallVector<Instruction *, 2> depInsts;
    Optional<unsigned> firstSrcDepPos;
    Optional<unsigned> lastDstDepPos;
    unsigned pos = 0;
    for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
         it != Block::iterator(dstNodeInst); ++it) {
      Instruction *inst = &(*it);
      if (srcDepInsts.count(inst) > 0 && firstSrcDepPos == None)
        firstSrcDepPos = pos;
      if (dstDepInsts.count(inst) > 0)
        lastDstDepPos = pos;
      depInsts.push_back(inst);
      ++pos;
    }

    if (firstSrcDepPos.hasValue()) {
      if (lastDstDepPos.hasValue()) {
        if (firstSrcDepPos.getValue() <= lastDstDepPos.getValue()) {
          // No valid insertion point exists which preserves dependences.
          return nullptr;
        }
      }
      // Return the insertion point at 'firstSrcDepPos'.
      return depInsts[firstSrcDepPos.getValue()];
    }
    // No dependence targets in range (or only dst deps in range), return
    // 'dstNodInst' insertion point.
    return dstNodeInst;
  }

  // Updates edge mappings from node 'srcId' to node 'dstId' after 'oldMemRef'
  // has been replaced in node at 'dstId' by a private memref.
  void updateEdges(unsigned srcId, unsigned dstId, Value *oldMemRef) {
    // For each edge in 'inEdges[srcId]': add new edge remaping to 'dstId'.
    if (inEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
      for (auto &inEdge : oldInEdges) {
        // Add edge from 'inEdge.id' to 'dstId' if not for 'oldMemRef'.
        if (inEdge.value != oldMemRef)
          addEdge(inEdge.id, dstId, inEdge.value);
      }
    }
    // For each edge in 'outEdges[srcId]': remove edge from 'srcId' to 'dstId'.
    if (outEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
      for (auto &outEdge : oldOutEdges) {
        // Remove any out edges from 'srcId' to 'dstId' across memrefs.
        if (outEdge.id == dstId)
          removeEdge(srcId, outEdge.id, outEdge.value);
      }
    }
    // Remove any edges in 'inEdges[dstId]' on 'oldMemRef' (which is being
    // replaced by a private memref). These edges could come from nodes
    // other than 'srcId' which were removed in the previous step.
    if (inEdges.count(dstId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[dstId];
      for (auto &inEdge : oldInEdges)
        if (inEdge.value == oldMemRef)
          removeEdge(inEdge.id, dstId, inEdge.value);
    }
  }

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Instruction *> &loads,
                 const SmallVectorImpl<Instruction *> &stores) {
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
          os << "  InEdge: " << e.id << " " << e.value << "\n";
      }
      it = outEdges.find(idAndNode.first);
      if (it != outEdges.end()) {
        for (const auto &e : it->second)
          os << "  OutEdge: " << e.id << " " << e.value << "\n";
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
  DenseMap<Value *, SetVector<unsigned>> memrefAccesses;

  // TODO: support multi-block functions.
  if (f->getBlocks().size() != 1)
    return false;

  DenseMap<Instruction *, unsigned> forToNodeMap;
  for (auto &inst : f->front()) {
    if (auto forOp = inst.dyn_cast<AffineForOp>()) {
      // Create graph node 'id' to represent top-level 'forOp' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.collect(&inst);
      // Return false if a non 'for' region was found (not currently supported).
      if (collector.hasNonForRegion)
        return false;
      Node node(nextNodeId++, &inst);
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
      forToNodeMap[&inst] = node.id;
      nodes.insert({node.id, node});
    } else if (auto loadOp = inst.dyn_cast<LoadOp>()) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, &inst);
      node.loads.push_back(&inst);
      auto *memref = inst.cast<LoadOp>()->getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (auto storeOp = inst.dyn_cast<StoreOp>()) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, &inst);
      node.stores.push_back(&inst);
      auto *memref = inst.cast<StoreOp>()->getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (inst.getNumBlockLists() != 0) {
      // Return false if another region is found (not currently supported).
      return false;
    } else if (inst.getNumResults() > 0 && !inst.use_empty()) {
      // Create graph node for top-level producer of SSA values, which
      // could be used by loop nest nodes.
      Node node(nextNodeId++, &inst);
      nodes.insert({node.id, node});
    }
  }

  // Add dependence edges between nodes which produce SSA values and their
  // users.
  for (auto &idAndNode : nodes) {
    const Node &node = idAndNode.second;
    if (!node.loads.empty() || !node.stores.empty())
      continue;
    auto *opInst = node.inst;
    for (auto *value : opInst->getResults()) {
      for (auto &use : value->getUses()) {
        SmallVector<OpPointer<AffineForOp>, 4> loops;
        getLoopIVs(*use.getOwner(), &loops);
        if (loops.empty())
          continue;
        assert(forToNodeMap.count(loops[0]->getInstruction()) > 0);
        unsigned userLoopNestId = forToNodeMap[loops[0]->getInstruction()];
        addEdge(node.id, userLoopNestId, value);
      }
    }
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
  // Map from AffineForOp to immediate child AffineForOps in its loop body.
  DenseMap<Instruction *, SmallVector<OpPointer<AffineForOp>, 2>> loopMap;
  // Map from AffineForOp to count of operations in its loop body.
  DenseMap<Instruction *, uint64_t> opCountMap;
  // Map from AffineForOp to its constant trip count.
  DenseMap<Instruction *, uint64_t> tripCountMap;
};

// LoopNestStatsCollector walks a single loop nest and gathers per-loop
// trip count and operation count statistics and records them in 'stats'.
struct LoopNestStatsCollector {
  LoopNestStats *stats;
  bool hasLoopWithNonConstTripCount = false;

  LoopNestStatsCollector(LoopNestStats *stats) : stats(stats) {}

  void collect(Instruction *inst) {
    inst->walk<AffineForOp>([&](OpPointer<AffineForOp> forOp) {
      auto *forInst = forOp->getInstruction();
      auto *parentInst = forOp->getInstruction()->getParentInst();
      if (parentInst != nullptr) {
        assert(parentInst->isa<AffineForOp>() && "Expected parent AffineForOp");
        // Add mapping to 'forOp' from its parent AffineForOp.
        stats->loopMap[parentInst].push_back(forOp);
      }

      // Record the number of op instructions in the body of 'forOp'.
      unsigned count = 0;
      stats->opCountMap[forInst] = 0;
      for (auto &inst : *forOp->getBody()) {
        if (!(inst.isa<AffineForOp>() || inst.isa<AffineIfOp>()))
          ++count;
      }
      stats->opCountMap[forInst] = count;
      // Record trip count for 'forOp'. Set flag if trip count is not
      // constant.
      Optional<uint64_t> maybeConstTripCount = getConstantTripCount(forOp);
      if (!maybeConstTripCount.hasValue()) {
        hasLoopWithNonConstTripCount = true;
        return;
      }
      stats->tripCountMap[forInst] = maybeConstTripCount.getValue();
    });
  }
};

// Computes the total cost of the loop nest rooted at 'forOp'.
// Currently, the total cost is computed by counting the total operation
// instance count (i.e. total number of operations in the loop bodyloop
// operation count * loop trip count) for the entire loop nest.
// If 'tripCountOverrideMap' is non-null, overrides the trip count for loops
// specified in the map when computing the total op instance count.
// NOTE: this is used to compute the cost of computation slices, which are
// sliced along the iteration dimension, and thus reduce the trip count.
// If 'computeCostMap' is non-null, the total op count for forOps specified
// in the map is increased (not overridden) by adding the op count from the
// map to the existing op count for the for loop. This is done before
// multiplying by the loop's trip count, and is used to model the cost of
// inserting a sliced loop nest of known cost into the loop's body.
// NOTE: this is used to compute the cost of fusing a slice of some loop nest
// within another loop.
static int64_t getComputeCost(
    Instruction *forInst, LoopNestStats *stats,
    llvm::SmallDenseMap<Instruction *, uint64_t, 8> *tripCountOverrideMap,
    DenseMap<Instruction *, int64_t> *computeCostMap) {
  // 'opCount' is the total number operations in one iteration of 'forOp' body
  int64_t opCount = stats->opCountMap[forInst];
  if (stats->loopMap.count(forInst) > 0) {
    for (auto childForOp : stats->loopMap[forInst]) {
      opCount += getComputeCost(childForOp->getInstruction(), stats,
                                tripCountOverrideMap, computeCostMap);
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
  int64_t tripCount = stats->tripCountMap[forInst];
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

// Builds a map 'tripCountMap' from AffineForOp to constant trip count for loop
// nest surrounding 'srcAccess' utilizing slice loop bounds in 'sliceState'.
// Returns true on success, false otherwise (if a non-constant trip count
// was encountered).
// TODO(andydavis) Make this work with non-unit step loops.
static bool buildSliceTripCountMap(
    Instruction *srcOpInst, ComputationSliceState *sliceState,
    llvm::SmallDenseMap<Instruction *, uint64_t, 8> *tripCountMap) {
  SmallVector<OpPointer<AffineForOp>, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();
  // Populate map from AffineForOp -> trip count
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    AffineMap lbMap = sliceState->lbs[i];
    AffineMap ubMap = sliceState->ubs[i];
    if (lbMap == AffineMap() || ubMap == AffineMap()) {
      // The iteration of src loop IV 'i' was not sliced. Use full loop bounds.
      if (srcLoopIVs[i]->hasConstantLowerBound() &&
          srcLoopIVs[i]->hasConstantUpperBound()) {
        (*tripCountMap)[srcLoopIVs[i]->getInstruction()] =
            srcLoopIVs[i]->getConstantUpperBound() -
            srcLoopIVs[i]->getConstantLowerBound();
        continue;
      }
      return false;
    }
    Optional<uint64_t> tripCount = getConstDifference(lbMap, ubMap);
    if (!tripCount.hasValue())
      return false;
    (*tripCountMap)[srcLoopIVs[i]->getInstruction()] = tripCount.getValue();
  }
  return true;
}

// Removes load operations from 'srcLoads' which operate on 'memref', and
// adds them to 'dstLoads'.
static void
moveLoadsAccessingMemrefTo(Value *memref,
                           SmallVectorImpl<Instruction *> *srcLoads,
                           SmallVectorImpl<Instruction *> *dstLoads) {
  dstLoads->clear();
  SmallVector<Instruction *, 4> srcLoadsToKeep;
  for (auto *load : *srcLoads) {
    if (load->cast<LoadOp>()->getMemRef() == memref)
      dstLoads->push_back(load);
    else
      srcLoadsToKeep.push_back(load);
  }
  srcLoads->swap(srcLoadsToKeep);
}

// Returns the innermost common loop depth for the set of operations in 'ops'.
static unsigned getInnermostCommonLoopDepth(ArrayRef<Instruction *> ops) {
  unsigned numOps = ops.size();
  assert(numOps > 0);

  std::vector<SmallVector<OpPointer<AffineForOp>, 4>> loops(numOps);
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
      if (loops[i - 1][d] != loops[i][d])
        break;
    }
    if (i != numOps)
      break;
    ++loopDepth;
  }
  return loopDepth;
}

// Returns the maximum loop depth at which no dependences between 'loadOpInsts'
// and 'storeOpInsts' are satisfied.
static unsigned getMaxLoopDepth(ArrayRef<Instruction *> loadOpInsts,
                                ArrayRef<Instruction *> storeOpInsts) {
  // Merge loads and stores into the same array.
  SmallVector<Instruction *, 2> ops(loadOpInsts.begin(), loadOpInsts.end());
  ops.append(storeOpInsts.begin(), storeOpInsts.end());

  // Compute the innermost common loop depth for loads and stores.
  unsigned loopDepth = getInnermostCommonLoopDepth(ops);

  // Return common loop depth for loads if there are no store ops.
  if (storeOpInsts.empty())
    return loopDepth;

  // Check dependences on all pairs of ops in 'ops' and store the minimum
  // loop depth at which a dependence is satisfied.
  for (unsigned i = 0, e = ops.size(); i < e; ++i) {
    auto *srcOpInst = ops[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = ops[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineConstraints dependenceConstraints;
        // TODO(andydavis) Cache dependence analysis results, check cache here.
        if (checkMemrefAccessDependence(srcAccess, dstAccess, d,
                                        &dependenceConstraints,
                                        /*dependenceComponents=*/nullptr)) {
          // Store minimum loop depth and break because we want the min 'd' at
          // which there is a dependence.
          loopDepth = std::min(loopDepth, d - 1);
          break;
        }
      }
    }
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
    if (lbMapA == AffineMap()) {
      assert(ubMapA == AffineMap());
      continue;
    }
    assert(ubMapA && "expected non-null ub map");

    AffineMap lbMapB = sliceStateB->lbs[i];
    AffineMap ubMapB = sliceStateB->ubs[i];
    if (lbMapB == AffineMap()) {
      assert(ubMapB == AffineMap());
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

//  TODO(mlir-team): improve/complete this when we have target data.
unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Creates and returns a private (single-user) memref for fused loop rooted
// at 'forOp', with (potentially reduced) memref size based on the
// MemRefRegion written to by 'srcStoreOpInst' at depth 'dstLoopDepth'.
// TODO(bondhugula): consider refactoring the common code from generateDma and
// this one.
static Value *createPrivateMemRef(OpPointer<AffineForOp> forOp,
                                  Instruction *srcStoreOpInst,
                                  unsigned dstLoopDepth,
                                  Optional<unsigned> fastMemorySpace,
                                  unsigned localBufSizeThreshold) {
  auto *forInst = forOp->getInstruction();

  // Create builder to insert alloc op just before 'forOp'.
  FuncBuilder b(forInst);
  // Builder to create constants at the top level.
  FuncBuilder top(forInst->getFunction());
  // Create new memref type based on slice bounds.
  auto *oldMemRef = srcStoreOpInst->cast<StoreOp>()->getMemRef();
  auto oldMemRefType = oldMemRef->getType().cast<MemRefType>();
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOpInst->getLoc());
  region.compute(srcStoreOpInst, dstLoopDepth);
  SmallVector<int64_t, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  Optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors);
  assert(numElements.hasValue() &&
         "non-constant number of elts in local buffer");

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
  uint64_t bufSize =
      getMemRefEltSizeInBytes(oldMemRefType) * numElements.getValue();
  unsigned newMemSpace;
  if (bufSize < localBufSizeThreshold && fastMemorySpace.hasValue()) {
    newMemSpace = fastMemorySpace.getValue();
  } else {
    newMemSpace = oldMemRefType.getMemorySpace();
  }
  auto newMemRefType = top.getMemRefType(
      newShape, oldMemRefType.getElementType(), {}, newMemSpace);
  // Gather alloc operands for the dynamic dimensions of the memref.
  SmallVector<Value *, 4> allocOperands;
  unsigned dynamicDimCount = 0;
  for (auto dimSize : oldMemRefType.getShape()) {
    if (dimSize == -1)
      allocOperands.push_back(
          top.create<DimOp>(forOp->getLoc(), oldMemRef, dynamicDimCount++));
  }

  // Create new private memref for fused loop 'forOp'.
  // TODO(andydavis) Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the function, because loop nests can be reordered
  // during the fusion pass.
  Value *newMemRef =
      top.create<AllocOp>(forOp->getLoc(), newMemRefType, allocOperands);

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
          ? AffineMap()
          : b.getAffineMap(outerIVs.size() + rank, 0, remapExprs, {});
  // Replace all users of 'oldMemRef' with 'newMemRef'.
  bool ret =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*domInstFilter=*/&*forOp->getBody()->begin());
  assert(ret && "replaceAllMemrefUsesWith should always succeed here");
  (void)ret;
  return newMemRef;
}

// Does the slice have a single iteration?
static uint64_t getSliceIterationCount(
    const llvm::SmallDenseMap<Instruction *, uint64_t, 8> &sliceTripCountMap) {
  uint64_t iterCount = 1;
  for (const auto &count : sliceTripCountMap) {
    iterCount *= count.second;
  }
  return iterCount;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstLoadOpInsts'.
// Returns true if it is profitable to fuse the candidate loop nests. Returns
// false otherwise. `dstLoopDepth` is set to the most profitable depth at which
// to materialize the source loop nest slice.
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
//    NOTE: If the dst loop nest includes multiple loads in 'dstLoadOpInsts' for
//    the same memref as is written by 'srcOpInst', then the union of slice
//    loop bounds is used to compute the slice and associated slice cost.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    execessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
static bool isFusionProfitable(Instruction *srcOpInst,
                               ArrayRef<Instruction *> dstLoadOpInsts,
                               ArrayRef<Instruction *> dstStoreOpInsts,
                               ComputationSliceState *sliceState,
                               unsigned *dstLoopDepth) {
  LLVM_DEBUG({
    llvm::dbgs() << "Checking whether fusion is profitable between:\n";
    llvm::dbgs() << " ";
    srcOpInst->dump();
    llvm::dbgs() << " and \n";
    for (auto dstOpInst : dstLoadOpInsts) {
      llvm::dbgs() << " ";
      dstOpInst->dump();
    };
  });

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<OpPointer<AffineForOp>, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  LoopNestStatsCollector srcStatsCollector(&srcLoopNestStats);
  srcStatsCollector.collect(srcLoopIVs[0]->getInstruction());
  // Currently only constant trip count loop nests are supported.
  if (srcStatsCollector.hasLoopWithNonConstTripCount)
    return false;

  // Compute cost of dst loop nest.
  SmallVector<OpPointer<AffineForOp>, 4> dstLoopIVs;
  getLoopIVs(*dstLoadOpInsts[0], &dstLoopIVs);

  LoopNestStats dstLoopNestStats;
  LoopNestStatsCollector dstStatsCollector(&dstLoopNestStats);
  dstStatsCollector.collect(dstLoopIVs[0]->getInstruction());
  // Currently only constant trip count loop nests are supported.
  if (dstStatsCollector.hasLoopWithNonConstTripCount)
    return false;

  // Compute the maximum loop depth at which we can can insert the src slice
  // and still satisfy dest loop nest dependences.
  unsigned maxDstLoopDepth = getMaxLoopDepth(dstLoadOpInsts, dstStoreOpInsts);
  if (maxDstLoopDepth == 0)
    return false;

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxDstLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  uint64_t maxStorageReduction = 0;
  Optional<uint64_t> sliceMemEstimate = None;

  SmallVector<ComputationSliceState, 4> sliceStates;
  sliceStates.resize(maxDstLoopDepth);
  // The best loop depth at which to materialize the slice.
  Optional<unsigned> bestDstLoopDepth = None;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost =
      getComputeCost(srcLoopIVs[0]->getInstruction(), &srcLoopNestStats,
                     /*tripCountOverrideMap=*/nullptr,
                     /*computeCostMap=*/nullptr);

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost =
      getComputeCost(dstLoopIVs[0]->getInstruction(), &dstLoopNestStats,
                     /*tripCountOverrideMap=*/nullptr,
                     /*computeCostMap=*/nullptr);

  llvm::SmallDenseMap<Instruction *, uint64_t, 8> sliceTripCountMap;
  DenseMap<Instruction *, int64_t> computeCostMap;
  for (unsigned i = maxDstLoopDepth; i >= 1; --i) {
    MemRefAccess srcAccess(srcOpInst);
    // Handle the common case of one dst load without a copy.
    if (!mlir::getBackwardComputationSliceState(
            srcAccess, MemRefAccess(dstLoadOpInsts[0]), i, &sliceStates[i - 1]))
      return false;
    // Compute the union of slice bound of all ops in 'dstLoadOpInsts'.
    for (int j = 1, e = dstLoadOpInsts.size(); j < e; ++j) {
      MemRefAccess dstAccess(dstLoadOpInsts[j]);
      ComputationSliceState tmpSliceState;
      if (!mlir::getBackwardComputationSliceState(srcAccess, dstAccess, i,
                                                  &tmpSliceState))
        return false;
      // Compute slice boun dunion of 'tmpSliceState' and 'sliceStates[i - 1]'.
      getSliceUnion(tmpSliceState, &sliceStates[i - 1]);
    }
    // Build trip count map for computation slice. We'll skip cases where the
    // trip count was non-constant.
    sliceTripCountMap.clear();
    if (!buildSliceTripCountMap(srcOpInst, &sliceStates[i - 1],
                                &sliceTripCountMap))
      continue;

    // Checks whether a store to load forwarding will happen.
    int64_t sliceIterationCount = getSliceIterationCount(sliceTripCountMap);
    assert(sliceIterationCount > 0);
    bool storeLoadFwdGuaranteed = (sliceIterationCount == 1);

    // Compute cost of fusion for this dest loop depth.

    computeCostMap.clear();

    // The store and loads to this memref will disappear.
    if (storeLoadFwdGuaranteed) {
      // A single store disappears: -1 for that.
      computeCostMap[srcLoopIVs[numSrcLoopIVs - 1]->getInstruction()] = -1;
      for (auto *loadOp : dstLoadOpInsts) {
        auto *parentInst = loadOp->getParentInst();
        if (parentInst && parentInst->isa<AffineForOp>())
          computeCostMap[parentInst] = -1;
      }
    }

    // Compute op instance count for the src loop nest with iteration slicing.
    int64_t sliceComputeCost =
        getComputeCost(srcLoopIVs[0]->getInstruction(), &srcLoopNestStats,
                       /*tripCountOverrideMap=*/&sliceTripCountMap,
                       /*computeCostMap=*/&computeCostMap);

    // Compute cost of fusion for this depth.
    computeCostMap[dstLoopIVs[i - 1]->getInstruction()] = sliceComputeCost;

    int64_t fusedLoopNestComputeCost =
        getComputeCost(dstLoopIVs[0]->getInstruction(), &dstLoopNestStats,
                       /*tripCountOverrideMap=*/nullptr, &computeCostMap);

    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;

    // TODO(bondhugula): This is an ugly approximation. Fix this by finding a
    // good way to calculate the footprint of the memref in the slice and
    // divide it by the total memory footprint of the fused computation.
    double storageReduction =
        static_cast<double>(srcLoopNestCost) / sliceIterationCount;

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::setprecision(2) << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   slice iteration count: " << sliceIterationCount << "\n";
      llvm::dbgs() << msg.str();
    });

    double computeToleranceThreshold =
        clFusionAddlComputeTolerance.getNumOccurrences() > 0
            ? clFusionAddlComputeTolerance
            : LoopFusion::kComputeToleranceThreshold;

    // TODO(b/123247369): This is a placeholder cost model.
    // Among all choices that add an acceptable amount of redundant computation
    // (as per computeToleranceThreshold), we will simply pick the one that
    // reduces the intermediary size the most.
    if ((storageReduction > maxStorageReduction) &&
        (clMaximalLoopFusion ||
         (additionalComputeFraction < computeToleranceThreshold))) {
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      // TODO(bondhugula,andydavis): find a good way to compute the memory
      // footprint of the materialized slice.
      // Approximating this to the compute cost of the slice. This could be an
      // under-approximation or an overapproximation, but in many cases
      // accurate.
      sliceMemEstimate = sliceIterationCount;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint. If
  // -maximal-fusion is set, fuse nevertheless.

  if (!clMaximalLoopFusion && !bestDstLoopDepth.hasValue()) {
    LLVM_DEBUG(llvm::dbgs()
               << "All fusion choices involve more than the threshold amount of"
                  "redundant computation; NOT fusing.\n");
    return false;
  }

  assert(bestDstLoopDepth.hasValue() &&
         "expected to have a value per logic above");

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = bestDstLoopDepth.getValue();

  LLVM_DEBUG(
      llvm::dbgs() << " LoopFusion fusion stats:"
                   << "\n  best loop depth: " << bestDstLoopDepth
                   << "\n  src loop nest compute cost: " << srcLoopNestCost
                   << "\n  dst loop nest compute cost: " << dstLoopNestCost
                   << "\n  fused loop nest compute cost: "
                   << minFusedLoopNestComputeCost << "\n");

  auto dstMemSize = getMemoryFootprintBytes(dstLoopIVs[0]);
  auto srcMemSize = getMemoryFootprintBytes(srcLoopIVs[0]);

  Optional<double> storageReduction = None;

  if (!clMaximalLoopFusion) {
    if (!dstMemSize.hasValue() || !srcMemSize.hasValue()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
      return false;
    }

    auto srcMemSizeVal = srcMemSize.getValue();
    auto dstMemSizeVal = dstMemSize.getValue();

    assert(sliceMemEstimate.hasValue() && "expected value");
    // This is an inaccurate estimate since sliceMemEstimate is isaccurate.
    auto fusedMem = dstMemSizeVal + sliceMemEstimate.getValue();

    LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                            << "   dst mem: " << dstMemSizeVal << "\n"
                            << "   fused mem: " << fusedMem << "\n"
                            << "   slice mem: " << sliceMemEstimate << "\n");

    if (fusedMem > srcMemSizeVal + dstMemSizeVal) {
      LLVM_DEBUG(llvm::dbgs() << "Fusion is not profitable; NOT fusing.\n");
      return false;
    }
    storageReduction =
        100.0 *
        (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal));
  }

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1);
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction.hasValue()
                ? std::to_string(storageReduction.getValue())
                : "<unknown>");
    msg << "% storage reduction.\n";
    llvm::dbgs() << msg.str();
  });

  // Update return parameter 'sliceState' with 'bestSliceState'.
  ComputationSliceState *bestSliceState = &sliceStates[*dstLoopDepth - 1];
  sliceState->lbs = bestSliceState->lbs;
  sliceState->ubs = bestSliceState->ubs;
  sliceState->lbOperands = bestSliceState->lbOperands;
  sliceState->ubOperands = bestSliceState->ubOperands;

  // Canonicalize slice bound affine maps.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    if (sliceState->lbs[i] != AffineMap()) {
      canonicalizeMapAndOperands(&sliceState->lbs[i],
                                 &sliceState->lbOperands[i]);
    }
    if (sliceState->ubs[i] != AffineMap()) {
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
//   *) Pop a AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
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
// in two different loop nests.
struct GreedyFusion {
public:
  MemRefDependenceGraph *mdg;
  SmallVector<unsigned, 4> worklist;

  GreedyFusion(MemRefDependenceGraph *mdg) : mdg(mdg) {
    // Initialize worklist with nodes from 'mdg'.
    worklist.resize(mdg->nodes.size());
    std::iota(worklist.begin(), worklist.end(), 0);
  }

  void run(unsigned localBufSizeThreshold, Optional<unsigned> fastMemorySpace) {
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!dstNode->inst->isa<AffineForOp>())
        continue;

      SmallVector<Instruction *, 4> loads = dstNode->loads;
      SmallVector<Instruction *, 4> dstLoadOpInsts;
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
        // Iterate through in edges for 'dstId' and src node id for any
        // edges on 'memref'.
        SmallVector<unsigned, 2> srcNodeIds;
        for (auto &srcEdge : mdg->inEdges[dstId]) {
          // Skip 'srcEdge' if not for 'memref'.
          if (srcEdge.value != memref)
            continue;
          srcNodeIds.push_back(srcEdge.id);
        }
        for (unsigned srcId : srcNodeIds) {
          // Skip if this node was removed (fused into another node).
          if (mdg->nodes.count(srcId) == 0)
            continue;
          // Get 'srcNode' from which to attempt fusion into 'dstNode'.
          auto *srcNode = mdg->getNode(srcId);
          // Skip if 'srcNode' is not a loop nest.
          if (!srcNode->inst->isa<AffineForOp>())
            continue;
          // Skip if 'srcNode' has more than one store to any memref.
          // TODO(andydavis) Support fusing multi-output src loop nests.
          if (srcNode->stores.size() != 1)
            continue;

          // Skip 'srcNode' if it has in edges on 'memref'.
          // TODO(andydavis) Track dependence type with edges, and just check
          // for WAW dependence edge here. Note that this check is overly
          // conservative and will be removed in the future.
          if (mdg->getIncomingMemRefAccesses(srcNode->id, memref) != 0)
            continue;

          // Skip if 'srcNode' writes to any live in or escaping memrefs.
          if (mdg->writesToLiveInOrEscapingMemrefs(srcNode->id))
            continue;

          // Compute an instruction list insertion point for the fused loop
          // nest which preserves dependences.
          Instruction *insertPointInst = mdg->getFusedLoopNestInsertionPoint(
              srcNode->id, dstNode->id, memref);
          if (insertPointInst == nullptr)
            continue;

          // Get unique 'srcNode' store op.
          auto *srcStoreOpInst = srcNode->stores.front();
          // Gather 'dstNode' store ops to 'memref'.
          SmallVector<Instruction *, 2> dstStoreOpInsts;
          for (auto *storeOpInst : dstNode->stores)
            if (storeOpInst->cast<StoreOp>()->getMemRef() == memref)
              dstStoreOpInsts.push_back(storeOpInst);

          unsigned bestDstLoopDepth;
          mlir::ComputationSliceState sliceState;
          // Check if fusion would be profitable.
          if (!isFusionProfitable(srcStoreOpInst, dstLoadOpInsts,
                                  dstStoreOpInsts, &sliceState,
                                  &bestDstLoopDepth))
            continue;

          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          auto sliceLoopNest = mlir::insertBackwardComputationSlice(
              srcStoreOpInst, dstLoadOpInsts[0], bestDstLoopDepth, &sliceState);
          if (sliceLoopNest != nullptr) {
            // Move 'dstAffineForOp' before 'insertPointInst' if needed.
            auto dstAffineForOp = dstNode->inst->cast<AffineForOp>();
            if (insertPointInst != dstAffineForOp->getInstruction()) {
              dstAffineForOp->getInstruction()->moveBefore(insertPointInst);
            }
            // Update edges between 'srcNode' and 'dstNode'.
            mdg->updateEdges(srcNode->id, dstNode->id, memref);

            // Collect slice loop stats.
            LoopNestStateCollector sliceCollector;
            sliceCollector.collect(sliceLoopNest->getInstruction());
            // Promote single iteration slice loops to single IV value.
            for (auto forOp : sliceCollector.forOps) {
              promoteIfSingleIteration(forOp);
            }
            // Create private memref for 'memref' in 'dstAffineForOp'.
            SmallVector<Instruction *, 4> storesForMemref;
            for (auto *storeOpInst : sliceCollector.storeOpInsts) {
              if (storeOpInst->cast<StoreOp>()->getMemRef() == memref)
                storesForMemref.push_back(storeOpInst);
            }
            assert(storesForMemref.size() == 1);
            auto *newMemRef = createPrivateMemRef(
                dstAffineForOp, storesForMemref[0], bestDstLoopDepth,
                fastMemorySpace, localBufSizeThreshold);
            visitedMemrefs.insert(newMemRef);
            // Create new node in dependence graph for 'newMemRef' alloc op.
            unsigned newMemRefNodeId =
                mdg->addNode(newMemRef->getDefiningInst());
            // Add edge from 'newMemRef' node to dstNode.
            mdg->addEdge(newMemRefNodeId, dstId, newMemRef);

            // Collect dst loop stats after memref privatizaton transformation.
            LoopNestStateCollector dstLoopCollector;
            dstLoopCollector.collect(dstAffineForOp->getInstruction());

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
            if (mdg->canRemoveNode(srcNode->id)) {
              mdg->removeNode(srcNode->id);
              srcNode->inst->erase();
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
      if (inst && inst->isa<AllocOp>())
        inst->erase();
    }
  }
};

} // end anonymous namespace

PassResult LoopFusion::runOnFunction(Function *f) {
  if (clFusionFastMemorySpace.getNumOccurrences() > 0) {
    fastMemorySpace = clFusionFastMemorySpace.getValue();
  }

  MemRefDependenceGraph g;
  if (g.init(f))
    GreedyFusion(&g).run(localBufSizeThreshold, fastMemorySpace);
  return success();
}

static PassRegistration<LoopFusion> pass("loop-fusion", "Fuse loop nests");
