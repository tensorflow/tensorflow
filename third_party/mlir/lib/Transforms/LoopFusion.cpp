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
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopFusionUtils.h"
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
#include <sstream>
#define DEBUG_TYPE "affine-loop-fusion"

using llvm::SetVector;

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

/// Disables fusion profitability check and fuses if valid. Ignore any
/// additional (redundant) computation tolerance threshold
/// that would have prevented fusion.
static llvm::cl::opt<bool>
    clMaximalLoopFusion("fusion-maximal",
                        llvm::cl::desc("Enables maximal loop fusion"),
                        llvm::cl::cat(clOptionsCategory));

/// A threshold in percent of additional computation allowed when fusing.
static llvm::cl::opt<double> clFusionAddlComputeTolerance(
    "fusion-compute-tolerance",
    llvm::cl::desc("Fractional increase in additional "
                   "computation tolerated while fusing"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clFusionFastMemorySpace(
    "fusion-fast-mem-space",
    llvm::cl::desc("Faster memory space number to promote fusion buffers to"),
    llvm::cl::cat(clOptionsCategory));

// A local buffer of size less than or equal to this size is automatically
// promoted to fast memory after producer-consumer fusion.
static llvm::cl::opt<unsigned long long> clFusionLocalBufThreshold(
    "fusion-local-buf-threshold",
    llvm::cl::desc("Threshold size (KiB) for promoting local buffers to fast "
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

struct LoopFusion : public FunctionPass<LoopFusion> {
  LoopFusion(unsigned fastMemorySpace = 0, uint64_t localBufSizeThreshold = 0,
             bool maximalFusion = false)
      : localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion) {}

  void runOnFunction() override;

  // Any local buffers smaller than this size (in bytes) will be created in
  // `fastMemorySpace` if provided.
  uint64_t localBufSizeThreshold;
  Optional<unsigned> fastMemorySpace = None;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;

  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  constexpr static double kComputeToleranceThreshold = 0.30f;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopFusionPass(unsigned fastMemorySpace,
                           uint64_t localBufSizeThreshold, bool maximalFusion) {
  return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
                                      maximalFusion);
}

// TODO(b/117228571) Replace when this is modeled through side-effects/op traits
static bool isMemRefDereferencingOp(Operation &op) {
  if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op) ||
      isa<AffineDmaStartOp>(op) || isa<AffineDmaWaitOp>(op))
    return true;
  return false;
}

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not an IfInst was encountered in the loop nest.
struct LoopNestStateCollector {
  SmallVector<AffineForOp, 4> forOps;
  SmallVector<Operation *, 4> loadOpInsts;
  SmallVector<Operation *, 4> storeOpInsts;
  bool hasNonForRegion = false;

  void collect(Operation *opToWalk) {
    opToWalk->walk([&](Operation *op) {
      if (isa<AffineForOp>(op))
        forOps.push_back(cast<AffineForOp>(op));
      else if (op->getNumRegions() != 0)
        hasNonForRegion = true;
      else if (isa<AffineLoadOp>(op))
        loadOpInsts.push_back(op);
      else if (isa<AffineStoreOp>(op))
        storeOpInsts.push_back(op);
    });
  }
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level operations in a FuncOp which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO(andydavis) Add a more flexible dependence graph representation.
// TODO(andydavis) Add a depth parameter to dependence graph construction.
struct MemRefDependenceGraph {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;
    // List of load operations.
    SmallVector<Operation *, 4> loads;
    // List of store op insts.
    SmallVector<Operation *, 4> stores;
    Node(unsigned id, Operation *op) : id(id), op(op) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value memref) {
      unsigned loadOpCount = 0;
      for (auto *loadOpInst : loads) {
        if (memref == cast<AffineLoadOp>(loadOpInst).getMemRef())
          ++loadOpCount;
      }
      return loadOpCount;
    }

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value memref) {
      unsigned storeOpCount = 0;
      for (auto *storeOpInst : stores) {
        if (memref == cast<AffineStoreOp>(storeOpInst).getMemRef())
          ++storeOpCount;
      }
      return storeOpCount;
    }

    // Returns all store ops in 'storeOps' which access 'memref'.
    void getStoreOpsForMemref(Value memref,
                              SmallVectorImpl<Operation *> *storeOps) {
      for (auto *storeOpInst : stores) {
        if (memref == cast<AffineStoreOp>(storeOpInst).getMemRef())
          storeOps->push_back(storeOpInst);
      }
    }

    // Returns all load ops in 'loadOps' which access 'memref'.
    void getLoadOpsForMemref(Value memref,
                             SmallVectorImpl<Operation *> *loadOps) {
      for (auto *loadOpInst : loads) {
        if (memref == cast<AffineLoadOp>(loadOpInst).getMemRef())
          loadOps->push_back(loadOpInst);
      }
    }

    // Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
    // has at least one load and store operation.
    void getLoadAndStoreMemrefSet(DenseSet<Value> *loadAndStoreMemrefSet) {
      llvm::SmallDenseSet<Value, 2> loadMemrefs;
      for (auto *loadOpInst : loads) {
        loadMemrefs.insert(cast<AffineLoadOp>(loadOpInst).getMemRef());
      }
      for (auto *storeOpInst : stores) {
        auto memref = cast<AffineStoreOp>(storeOpInst).getMemRef();
        if (loadMemrefs.count(memref) > 0)
          loadAndStoreMemrefSet->insert(memref);
      }
    }
  };

  // Edge represents a data dependence between nodes in the graph.
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
    // (e.g. a constant operation defining a value which is used inside a loop
    // nest).
    Value value;
  };

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  MemRefDependenceGraph() {}

  // Initializes the dependence graph based on operations in 'f'.
  // Returns true on success, false otherwise.
  bool init(FuncOp f);

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id) {
    auto it = nodes.find(id);
    assert(it != nodes.end());
    return &it->second;
  }

  // Returns the graph node for 'forOp'.
  Node *getForOpNode(AffineForOp forOp) {
    for (auto &idAndNode : nodes)
      if (idAndNode.second.op == forOp.getOperation())
        return &idAndNode.second;
    return nullptr;
  }

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op) {
    Node node(nextNodeId++, op);
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
      auto memref = cast<AffineStoreOp>(storeOpInst).getMemRef();
      auto *op = memref->getDefiningOp();
      // Return true if 'memref' is a block argument.
      if (!op)
        return true;
      // Return true if any use of 'memref' escapes the function.
      for (auto *user : memref->getUsers())
        if (!isMemRefDereferencingOp(*user))
          return true;
    }
    return false;
  }

  // Returns the unique AffineStoreOp in `node` that meets all the following:
  //   *) store is the only one that writes to a function-local memref live out
  //      of `node`,
  //   *) store is not the source of a self-dependence on `node`.
  // Otherwise, returns a null AffineStoreOp.
  AffineStoreOp getUniqueOutgoingStore(Node *node) {
    AffineStoreOp uniqueStore;

    // Return null if `node` doesn't have any outgoing edges.
    auto outEdgeIt = outEdges.find(node->id);
    if (outEdgeIt == outEdges.end())
      return nullptr;

    const auto &nodeOutEdges = outEdgeIt->second;
    for (auto *op : node->stores) {
      auto storeOp = cast<AffineStoreOp>(op);
      auto memref = storeOp.getMemRef();
      // Skip this store if there are no dependences on its memref. This means
      // that store either:
      // *) writes to a memref that is only read within the same loop nest
      //    (self-dependence edges are not represented in graph at the moment),
      // *) writes to a function live out memref (function parameter), or
      // *) is dead.
      if (llvm::all_of(nodeOutEdges, [=](const Edge &edge) {
            return (edge.value != memref);
          }))
        continue;

      if (uniqueStore)
        // Found multiple stores to function-local live-out memrefs.
        return nullptr;
      // Found first store to function-local live-out memref.
      uniqueStore = storeOp;
    }

    return uniqueStore;
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
      if (getOutEdgeCount(id, cast<AffineStoreOp>(storeOpInst).getMemRef()) > 0)
        return false;
    }
    return true;
  }

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr) {
    if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
      return false;
    }
    bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
      return edge.id == dstId && (!value || edge.value == value);
    });
    bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
      return edge.id == srcId && (!value || edge.value == value);
    });
    return hasOutEdge && hasInEdge;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value) {
    if (!hasEdge(srcId, dstId, value)) {
      outEdges[srcId].push_back({dstId, value});
      inEdges[dstId].push_back({srcId, value});
      if (value->getType().isa<MemRefType>())
        memrefEdgeCount[value]++;
    }
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value) {
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

  // Returns true if there is a path in the dependence graph from node 'srcId'
  // to node 'dstId'. Returns false otherwise.
  bool hasDependencePath(unsigned srcId, unsigned dstId) {
    // Worklist state is: <node-id, next-output-edge-index-to-visit>
    SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
    worklist.push_back({srcId, 0});
    // Run DFS traversal to see if 'dstId' is reachable from 'srcId'.
    while (!worklist.empty()) {
      auto &idAndIndex = worklist.back();
      // Return true if we have reached 'dstId'.
      if (idAndIndex.first == dstId)
        return true;
      // Pop and continue if node has no out edges, or if all out edges have
      // already been visited.
      if (outEdges.count(idAndIndex.first) == 0 ||
          idAndIndex.second == outEdges[idAndIndex.first].size()) {
        worklist.pop_back();
        continue;
      }
      // Get graph edge to traverse.
      Edge edge = outEdges[idAndIndex.first][idAndIndex.second];
      // Increment next output edge index for 'idAndIndex'.
      ++idAndIndex.second;
      // Add node at 'edge.id' to worklist.
      worklist.push_back({edge.id, 0});
    }
    return false;
  }

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref' with a store operation.
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref) {
    unsigned inEdgeCount = 0;
    if (inEdges.count(id) > 0)
      for (auto &inEdge : inEdges[id])
        if (inEdge.value == memref) {
          Node *srcNode = getNode(inEdge.id);
          // Only count in edges from 'srcNode' if 'srcNode' accesses 'memref'
          if (srcNode->getStoreOpCount(memref) > 0)
            ++inEdgeCount;
        }
    return inEdgeCount;
  }

  // Returns the output edge count for node 'id' and 'memref' (if non-null),
  // otherwise returns the total output edge count from node 'id'.
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr) {
    unsigned outEdgeCount = 0;
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        if (!memref || outEdge.value == memref)
          ++outEdgeCount;
    return outEdgeCount;
  }

  // Computes and returns an insertion point operation, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId) {
    if (outEdges.count(srcId) == 0)
      return getNode(dstId)->op;

    // Build set of insts in range (srcId, dstId) which depend on 'srcId'.
    SmallPtrSet<Operation *, 2> srcDepInsts;
    for (auto &outEdge : outEdges[srcId])
      if (outEdge.id != dstId)
        srcDepInsts.insert(getNode(outEdge.id)->op);

    // Build set of insts in range (srcId, dstId) on which 'dstId' depends.
    SmallPtrSet<Operation *, 2> dstDepInsts;
    for (auto &inEdge : inEdges[dstId])
      if (inEdge.id != srcId)
        dstDepInsts.insert(getNode(inEdge.id)->op);

    Operation *srcNodeInst = getNode(srcId)->op;
    Operation *dstNodeInst = getNode(dstId)->op;

    // Computing insertion point:
    // *) Walk all operation positions in Block operation list in the
    //    range (src, dst). For each operation 'op' visited in this search:
    //   *) Store in 'firstSrcDepPos' the first position where 'op' has a
    //      dependence edge from 'srcNode'.
    //   *) Store in 'lastDstDepPost' the last position where 'op' has a
    //      dependence edge to 'dstNode'.
    // *) Compare 'firstSrcDepPos' and 'lastDstDepPost' to determine the
    //    operation insertion point (or return null pointer if no such
    //    insertion point exists: 'firstSrcDepPos' <= 'lastDstDepPos').
    SmallVector<Operation *, 2> depInsts;
    Optional<unsigned> firstSrcDepPos;
    Optional<unsigned> lastDstDepPos;
    unsigned pos = 0;
    for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
         it != Block::iterator(dstNodeInst); ++it) {
      Operation *op = &(*it);
      if (srcDepInsts.count(op) > 0 && firstSrcDepPos == None)
        firstSrcDepPos = pos;
      if (dstDepInsts.count(op) > 0)
        lastDstDepPos = pos;
      depInsts.push_back(op);
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
  // has been replaced in node at 'dstId' by a private memref depending
  // on the value of 'createPrivateMemRef'.
  void updateEdges(unsigned srcId, unsigned dstId, Value oldMemRef,
                   bool createPrivateMemRef) {
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
    if (inEdges.count(dstId) > 0 && createPrivateMemRef) {
      SmallVector<Edge, 2> oldInEdges = inEdges[dstId];
      for (auto &inEdge : oldInEdges)
        if (inEdge.value == oldMemRef)
          removeEdge(inEdge.id, dstId, inEdge.value);
    }
  }

  // Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
  // of sibling node 'sidId' into node 'dstId'.
  void updateEdges(unsigned sibId, unsigned dstId) {
    // For each edge in 'inEdges[sibId]':
    // *) Add new edge from source node 'inEdge.id' to 'dstNode'.
    // *) Remove edge from source node 'inEdge.id' to 'sibNode'.
    if (inEdges.count(sibId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[sibId];
      for (auto &inEdge : oldInEdges) {
        addEdge(inEdge.id, dstId, inEdge.value);
        removeEdge(inEdge.id, sibId, inEdge.value);
      }
    }

    // For each edge in 'outEdges[sibId]' to node 'id'
    // *) Add new edge from 'dstId' to 'outEdge.id'.
    // *) Remove edge from 'sibId' to 'outEdge.id'.
    if (outEdges.count(sibId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[sibId];
      for (auto &outEdge : oldOutEdges) {
        addEdge(dstId, outEdge.id, outEdge.value);
        removeEdge(sibId, outEdge.id, outEdge.value);
      }
    }
  }

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores) {
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

  // Calls 'callback' for each input edge incident to node 'id' which carries a
  // memref dependence.
  void forEachMemRefInputEdge(unsigned id,
                              const std::function<void(Edge)> &callback) {
    if (inEdges.count(id) > 0)
      forEachMemRefEdge(inEdges[id], callback);
  }

  // Calls 'callback' for each output edge from node 'id' which carries a
  // memref dependence.
  void forEachMemRefOutputEdge(unsigned id,
                               const std::function<void(Edge)> &callback) {
    if (outEdges.count(id) > 0)
      forEachMemRefEdge(outEdges[id], callback);
  }

  // Calls 'callback' for each edge in 'edges' which carries a memref
  // dependence.
  void forEachMemRefEdge(ArrayRef<Edge> edges,
                         const std::function<void(Edge)> &callback) {
    for (auto &edge : edges) {
      // Skip if 'edge' is not a memref dependence edge.
      if (!edge.value->getType().isa<MemRefType>())
        continue;
      assert(nodes.count(edge.id) > 0);
      // Skip if 'edge.id' is not a loop nest.
      if (!isa<AffineForOp>(getNode(edge.id)->op))
        continue;
      // Visit current input edge 'edge'.
      callback(edge);
    }
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

} // end anonymous namespace

// Initializes the data dependence graph by walking operations in 'f'.
// Assigns each node in the graph a node id based on program order in 'f'.
// TODO(andydavis) Add support for taking a Block arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init(FuncOp f) {
  DenseMap<Value, SetVector<unsigned>> memrefAccesses;

  // TODO: support multi-block functions.
  if (f.getBlocks().size() != 1)
    return false;

  DenseMap<Operation *, unsigned> forToNodeMap;
  for (auto &op : f.front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      // Create graph node 'id' to represent top-level 'forOp' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.collect(&op);
      // Return false if a non 'affine.for' region was found (not currently
      // supported).
      if (collector.hasNonForRegion)
        return false;
      Node node(nextNodeId++, &op);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto memref = cast<AffineLoadOp>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<AffineStoreOp>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      forToNodeMap[&op] = node.id;
      nodes.insert({node.id, node});
    } else if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, &op);
      node.loads.push_back(&op);
      auto memref = cast<AffineLoadOp>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, &op);
      node.stores.push_back(&op);
      auto memref = cast<AffineStoreOp>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (op.getNumRegions() != 0) {
      // Return false if another region is found (not currently supported).
      return false;
    } else if (op.getNumResults() > 0 && !op.use_empty()) {
      // Create graph node for top-level producer of SSA values, which
      // could be used by loop nest nodes.
      Node node(nextNodeId++, &op);
      nodes.insert({node.id, node});
    }
  }

  // Add dependence edges between nodes which produce SSA values and their
  // users.
  for (auto &idAndNode : nodes) {
    const Node &node = idAndNode.second;
    if (!node.loads.empty() || !node.stores.empty())
      continue;
    auto *opInst = node.op;
    for (auto value : opInst->getResults()) {
      for (auto *user : value->getUsers()) {
        SmallVector<AffineForOp, 4> loops;
        getLoopIVs(*user, &loops);
        if (loops.empty())
          continue;
        assert(forToNodeMap.count(loops[0].getOperation()) > 0);
        unsigned userLoopNestId = forToNodeMap[loops[0].getOperation()];
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

// Removes load operations from 'srcLoads' which operate on 'memref', and
// adds them to 'dstLoads'.
static void moveLoadsAccessingMemrefTo(Value memref,
                                       SmallVectorImpl<Operation *> *srcLoads,
                                       SmallVectorImpl<Operation *> *dstLoads) {
  dstLoads->clear();
  SmallVector<Operation *, 4> srcLoadsToKeep;
  for (auto *load : *srcLoads) {
    if (cast<AffineLoadOp>(load).getMemRef() == memref)
      dstLoads->push_back(load);
    else
      srcLoadsToKeep.push_back(load);
  }
  srcLoads->swap(srcLoadsToKeep);
}

// Returns the innermost common loop depth for the set of operations in 'ops'.
static unsigned getInnermostCommonLoopDepth(ArrayRef<Operation *> ops) {
  unsigned numOps = ops.size();
  assert(numOps > 0);

  std::vector<SmallVector<AffineForOp, 4>> loops(numOps);
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
static unsigned getMaxLoopDepth(ArrayRef<Operation *> loadOpInsts,
                                ArrayRef<Operation *> storeOpInsts) {
  // Merge loads and stores into the same array.
  SmallVector<Operation *, 2> ops(loadOpInsts.begin(), loadOpInsts.end());
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
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            /*dependenceComponents=*/nullptr);
        if (hasDependence(result)) {
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

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op));
  AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(node->op));
  node->op = newRootForOp.getOperation();
}

//  TODO(mlir-team): improve/complete this when we have target data.
static unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
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
static Value createPrivateMemRef(AffineForOp forOp, Operation *srcStoreOpInst,
                                 unsigned dstLoopDepth,
                                 Optional<unsigned> fastMemorySpace,
                                 uint64_t localBufSizeThreshold) {
  auto *forInst = forOp.getOperation();

  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forInst);
  // Builder to create constants at the top level.
  OpBuilder top(forInst->getParentOfType<FuncOp>().getBody());
  // Create new memref type based on slice bounds.
  auto oldMemRef = cast<AffineStoreOp>(srcStoreOpInst).getMemRef();
  auto oldMemRefType = oldMemRef->getType().cast<MemRefType>();
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOpInst->getLoc());
  bool validRegion = succeeded(region.compute(srcStoreOpInst, dstLoopDepth));
  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
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
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
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
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.hasValue()) {
    newMemSpace = fastMemorySpace.getValue();
  } else {
    newMemSpace = oldMemRefType.getMemorySpace();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       {}, newMemSpace);
  // Gather alloc operands for the dynamic dimensions of the memref.
  SmallVector<Value, 4> allocOperands;
  unsigned dynamicDimCount = 0;
  for (auto dimSize : oldMemRefType.getShape()) {
    if (dimSize == -1)
      allocOperands.push_back(
          top.create<DimOp>(forOp.getLoc(), oldMemRef, dynamicDimCount++));
  }

  // Create new private memref for fused loop 'forOp'.
  // TODO(andydavis) Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the function, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef =
      top.create<AllocOp>(forOp.getLoc(), newMemRefType, allocOperands);

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
  auto indexRemap = zeroOffsetCount == rank
                        ? AffineMap()
                        : AffineMap::get(outerIVs.size() + rank, 0, remapExprs);
  // Replace all users of 'oldMemRef' with 'newMemRef'.
  LogicalResult res =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*symbolOperands=*/{},
                               /*domInstFilter=*/&*forOp.getBody()->begin());
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  return newMemRef;
}

// Checks if node 'srcId' can be safely fused into node 'dstId'. Node 'srcId'
// may write to multiple memrefs but it is required that only one of them,
// 'srcLiveOutStoreOp', has output edges.
// Returns true if 'dstNode's read/write region to 'memref' is a super set of
// 'srcNode's write region to 'memref' and 'srcId' has only one output edge.
// TODO(andydavis) Generalize this to handle more live in/out cases.
static bool canFuseSrcWhichWritesToLiveOut(unsigned srcId, unsigned dstId,
                                           AffineStoreOp srcLiveOutStoreOp,
                                           MemRefDependenceGraph *mdg) {
  assert(srcLiveOutStoreOp && "Expected a valid store op");
  auto *dstNode = mdg->getNode(dstId);
  Value memref = srcLiveOutStoreOp.getMemRef();
  // Return false if 'srcNode' has more than one output edge on 'memref'.
  if (mdg->getOutEdgeCount(srcId, memref) > 1)
    return false;

  // Compute MemRefRegion 'srcWriteRegion' for 'srcStoreOp' on 'memref'.
  MemRefRegion srcWriteRegion(srcLiveOutStoreOp.getLoc());
  if (failed(srcWriteRegion.compute(srcLiveOutStoreOp, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n.");
    return false;
  }
  SmallVector<int64_t, 4> srcShape;
  // Query 'srcWriteRegion' for 'srcShape' and 'srcNumElements'.
  // by 'srcStoreOp' at depth 'dstLoopDepth'.
  Optional<int64_t> srcNumElements =
      srcWriteRegion.getConstantBoundingSizeAndShape(&srcShape);
  if (!srcNumElements.hasValue())
    return false;

  // Compute MemRefRegion 'dstRegion' for 'dstStore/LoadOpInst' on 'memref'.
  // TODO(andydavis) Compute 'unionboundingbox' of all write regions (one for
  // each store op in 'dstStoreOps').
  SmallVector<Operation *, 2> dstStoreOps;
  dstNode->getStoreOpsForMemref(memref, &dstStoreOps);
  SmallVector<Operation *, 2> dstLoadOps;
  dstNode->getLoadOpsForMemref(memref, &dstLoadOps);

  auto *dstOpInst = dstStoreOps.empty() ? dstLoadOps[0] : dstStoreOps[0];
  MemRefRegion dstRegion(dstOpInst->getLoc());
  if (failed(dstRegion.compute(dstOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for dest operation\n.");
    return false;
  }
  SmallVector<int64_t, 4> dstShape;
  // Query 'dstRegion' for 'dstShape' and 'dstNumElements'.
  // by 'dstOpInst' at depth 'dstLoopDepth'.
  Optional<int64_t> dstNumElements =
      dstRegion.getConstantBoundingSizeAndShape(&dstShape);
  if (!dstNumElements.hasValue())
    return false;

  // Return false if write region is not a superset of 'srcNodes' write
  // region to 'memref'.
  // TODO(andydavis) Check the shape and lower bounds here too.
  if (srcNumElements != dstNumElements)
    return false;
  return true;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstLoadOpInsts'.
// The argument 'srcStoreOpInst' is used to calculate the storage reduction on
// the memref being produced and consumed, which is an input to the cost model.
// For producer-consumer fusion, 'srcStoreOpInst' will be the same as
// 'srcOpInst', as we are slicing w.r.t to that producer.
// For input-reuse fusion, 'srcOpInst' will be the src loop nest LoadOp which
// reads from the same memref as dst loop nest load ops, and 'srcStoreOpInst'
// will be the unique store op in the src node, which will be used to check
// that the write region is the same after input-reuse fusion.
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
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: If the dst loop nest includes multiple loads in 'dstLoadOpInsts' for
//    the same memref as is written by 'srcOpInst', then the union of slice
//    loop bounds is used to compute the slice and associated slice cost.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
static bool isFusionProfitable(Operation *srcOpInst, Operation *srcStoreOpInst,
                               ArrayRef<Operation *> dstLoadOpInsts,
                               ArrayRef<Operation *> dstStoreOpInsts,
                               ComputationSliceState *sliceState,
                               unsigned *dstLoopDepth, bool maximalFusion) {
  LLVM_DEBUG({
    llvm::dbgs() << "Checking whether fusion is profitable between:\n";
    llvm::dbgs() << " " << *srcOpInst << " and \n";
    for (auto dstOpInst : dstLoadOpInsts) {
      llvm::dbgs() << " " << *dstOpInst << "\n";
    };
  });

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcLoopIVs[0], &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*dstLoadOpInsts[0], &dstLoopIVs);

  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstLoopIVs[0], &dstLoopNestStats))
    return false;

  // Compute the maximum loop depth at which we can can insert the src slice
  // and still satisfy dest loop nest dependences, for producer-consumer fusion.
  unsigned maxDstLoopDepth =
      (srcOpInst == srcStoreOpInst)
          ? getMaxLoopDepth(dstLoadOpInsts, dstStoreOpInsts)
          : dstLoopIVs.size();
  if (maxDstLoopDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Can't fuse: maxDstLoopDepth == 0 .\n");
    return false;
  }

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxDstLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  double maxStorageReduction = 0.0;
  Optional<uint64_t> sliceMemEstimate = None;

  SmallVector<ComputationSliceState, 4> sliceStates;
  sliceStates.resize(maxDstLoopDepth);
  // The best loop depth at which to materialize the slice.
  Optional<unsigned> bestDstLoopDepth = None;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], srcLoopNestStats);

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOpInst->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n.");
    return false;
  }

  Optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.hasValue())
    return false;
  int64_t srcWriteRegionSizeBytes = maybeSrcWriteRegionSizeBytes.getValue();

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstLoopIVs[0], dstLoopNestStats);

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxDstLoopDepth; i >= 1; --i) {
    // Compute the union of slice bounds of all ops in 'dstLoadOpInsts'.
    if (failed(mlir::computeSliceUnion({srcOpInst}, dstLoadOpInsts,
                                       /*loopDepth=*/i,
                                       /*numCommonLoops=*/0,
                                       /*isBackwardSlice=*/true,
                                       &sliceStates[i - 1]))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "computeSliceUnion failed for loopDepth: " << i << "\n");
      continue;
    }

    int64_t fusedLoopNestComputeCost;
    if (!getFusionComputeCost(srcLoopIVs[0], srcLoopNestStats, dstLoopIVs[0],
                              dstLoopNestStats, &sliceStates[i - 1],
                              &fusedLoopNestComputeCost)) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost.\n.");
      continue;
    }

    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'sliceStates[i - 1]' were to be inserted into the dst loop
    // nest at loop depth 'i'
    MemRefRegion sliceWriteRegion(srcStoreOpInst->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0,
                                        &sliceStates[i - 1]))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to compute slice write region at loopDepth: " << i
                 << "\n");
      continue;
    }

    Optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize();
    if (!maybeSliceWriteRegionSizeBytes.hasValue() ||
        maybeSliceWriteRegionSizeBytes.getValue() == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get slice write region size at loopDepth: " << i
                 << "\n");
      continue;
    }
    int64_t sliceWriteRegionSizeBytes =
        maybeSliceWriteRegionSizeBytes.getValue();

    // If we are fusing for reuse, check that write regions remain the same.
    // TODO(andydavis) Write region check should check sizes and offsets in
    // each dimension, so that we are sure they are covering the same memref
    // region. Also, move this out to a isMemRefRegionSuperSet helper function.
    if (srcOpInst != srcStoreOpInst &&
        sliceWriteRegionSizeBytes != srcWriteRegionSizeBytes)
      continue;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes);

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes
          << "\n";
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
        (maximalFusion ||
         (additionalComputeFraction < computeToleranceThreshold))) {
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      sliceMemEstimate = sliceWriteRegionSizeBytes;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint. If
  // -maximal-fusion is set, fuse nevertheless.

  if (!maximalFusion && !bestDstLoopDepth.hasValue()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "All fusion choices involve more than the threshold amount of "
           "redundant computation; NOT fusing.\n");
    return false;
  }

  if (!bestDstLoopDepth.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "no fusion depth could be evaluated.\n");
    return false;
  }

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

  if (!maximalFusion) {
    if (!dstMemSize.hasValue() || !srcMemSize.hasValue()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
      return false;
    }

    auto srcMemSizeVal = srcMemSize.getValue();
    auto dstMemSizeVal = dstMemSize.getValue();

    assert(sliceMemEstimate.hasValue() && "expected value");
    auto fusedMem = dstMemSizeVal + sliceMemEstimate.getValue();

    LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                            << "   dst mem: " << dstMemSizeVal << "\n"
                            << "   fused mem: " << fusedMem << "\n"
                            << "   slice mem: " << sliceMemEstimate << "\n");

    if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) {
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
        << std::setprecision(2) << additionalComputeFraction
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

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO to fix this.
//
// TODO(andydavis) Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Set of graph nodes which are present on the worklist.
  llvm::SmallDenseSet<unsigned, 16> worklistSet;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  Optional<unsigned> fastMemorySpace;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               Optional<unsigned> fastMemorySpace, bool maximalFusion)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion) {}

  // Initializes 'worklist' with nodes from 'mdg'
  void init() {
    // TODO(andydavis) Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    worklistSet.clear();
    for (auto &idAndNode : mdg->nodes) {
      const Node &node = idAndNode.second;
      worklist.push_back(node.id);
      worklistSet.insert(node.id);
    }
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void run() {
    // TODO(andydavis) Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1);
    fuseSiblingNodes();
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      worklistSet.erase(dstId);

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
      // while preserving relative order. This can increase the maximum loop
      // depth at which we can fuse a slice of a producer loop nest into a
      // consumer loop nest.
      sinkSequentialLoops(dstNode);

      SmallVector<Operation *, 4> loads = dstNode->loads;
      SmallVector<Operation *, 4> dstLoadOpInsts;
      DenseSet<Value> visitedMemrefs;
      while (!loads.empty()) {
        // Get memref of load on top of the stack.
        auto memref = cast<AffineLoadOp>(loads.back()).getMemRef();
        if (visitedMemrefs.count(memref) > 0)
          continue;
        visitedMemrefs.insert(memref);
        // Move all loads in 'loads' accessing 'memref' to 'dstLoadOpInsts'.
        moveLoadsAccessingMemrefTo(memref, &loads, &dstLoadOpInsts);
        // Skip if no input edges along which to fuse.
        if (mdg->inEdges.count(dstId) == 0)
          continue;
        // Iterate through in-edges for 'dstId' and src node id for any
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
          if (!isa<AffineForOp>(srcNode->op))
            continue;
          // Skip if 'srcNode' has more than one live-out store to a
          // function-local memref.
          // TODO(andydavis) Support more generic multi-output src loop nests
          // fusion.
          auto srcStoreOp = mdg->getUniqueOutgoingStore(srcNode);
          if (!srcStoreOp) {
            // Get the src store op at the deepest loop depth.
            // We will use 'LoopFusionUtils::canFuseLoops' to check fusion
            // feasibility for loops with multiple stores.
            unsigned maxLoopDepth = 0;
            for (auto *op : srcNode->stores) {
              auto storeOp = cast<AffineStoreOp>(op);
              if (storeOp.getMemRef() != memref) {
                srcStoreOp = nullptr;
                break;
              }
              unsigned loopDepth = getNestingDepth(*storeOp);
              if (loopDepth > maxLoopDepth) {
                maxLoopDepth = loopDepth;
                srcStoreOp = storeOp;
              }
            }
            if (!srcStoreOp)
              continue;
          }

          // Unique outgoing store found must write to 'memref' since 'memref'
          // is the one that established the producer-consumer relationship
          // between 'srcNode' and 'dstNode'.
          assert(srcStoreOp.getMemRef() == memref &&
                 "Found store to unexpected memref");

          // Skip if 'srcNode' writes to any live in or escaping memrefs,
          // and cannot be fused.
          bool writesToLiveInOrOut =
              mdg->writesToLiveInOrEscapingMemrefs(srcNode->id);
          if (writesToLiveInOrOut &&
              !canFuseSrcWhichWritesToLiveOut(srcId, dstId, srcStoreOp, mdg))
            continue;

          // Don't create a private memref if 'writesToLiveInOrOut'.
          bool createPrivateMemref = !writesToLiveInOrOut;
          // Don't create a private memref if 'srcNode' has in edges on
          // 'memref', or if 'dstNode' has out edges on 'memref'.
          if (mdg->getIncomingMemRefAccesses(srcNode->id, memref) > 0 ||
              mdg->getOutEdgeCount(dstNode->id, memref) > 0) {
            createPrivateMemref = false;
          }

          // Skip if 'srcNode' out edge count on 'memref' > 'maxSrcUserCount'.
          if (mdg->getOutEdgeCount(srcNode->id, memref) > maxSrcUserCount)
            continue;

          // Compute an operation list insertion point for the fused loop
          // nest which preserves dependences.
          Operation *insertPointInst =
              mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
          if (insertPointInst == nullptr)
            continue;

          // Compute the innermost common loop depth for dstNode loads/stores.
          SmallVector<Operation *, 2> dstOps(dstNode->loads.begin(),
                                             dstNode->loads.end());
          dstOps.append(dstNode->stores.begin(), dstNode->stores.end());
          unsigned dstLoopDepthTest = getInnermostCommonLoopDepth(dstOps);
          // Check the feasibility of fusing src loop nest into dst loop nest
          // at loop depths in range [1, dstLoopDepthTest].
          // TODO(andydavis) Use slice union computation and union of memref
          // read/write regions to cost model and fusion.
          bool canFuse = false;
          for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
            ComputationSliceState sliceUnion;
            FusionResult result = mlir::canFuseLoops(
                cast<AffineForOp>(srcNode->op), cast<AffineForOp>(dstNode->op),
                /*dstLoopDepth=*/i, &sliceUnion);
            if (result.value == FusionResult::Success)
              canFuse = true;
          }

          // Skip if fusion is not feasible at all loop depths.
          if (!canFuse)
            continue;

          // Gather 'dstNode' store ops to 'memref'.
          SmallVector<Operation *, 2> dstStoreOpInsts;
          for (auto *storeOpInst : dstNode->stores)
            if (cast<AffineStoreOp>(storeOpInst).getMemRef() == memref)
              dstStoreOpInsts.push_back(storeOpInst);

          unsigned bestDstLoopDepth;
          mlir::ComputationSliceState sliceState;
          // Check if fusion would be profitable.
          if (!isFusionProfitable(srcStoreOp, srcStoreOp, dstLoadOpInsts,
                                  dstStoreOpInsts, &sliceState,
                                  &bestDstLoopDepth, maximalFusion))
            continue;

          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          auto sliceLoopNest = mlir::insertBackwardComputationSlice(
              srcStoreOp, dstLoadOpInsts[0], bestDstLoopDepth, &sliceState);
          if (sliceLoopNest) {
            LLVM_DEBUG(llvm::dbgs() << "\tslice loop nest:\n"
                                    << *sliceLoopNest.getOperation() << "\n");
            // Move 'dstAffineForOp' before 'insertPointInst' if needed.
            auto dstAffineForOp = cast<AffineForOp>(dstNode->op);
            if (insertPointInst != dstAffineForOp.getOperation()) {
              dstAffineForOp.getOperation()->moveBefore(insertPointInst);
            }
            // Update edges between 'srcNode' and 'dstNode'.
            mdg->updateEdges(srcNode->id, dstNode->id, memref,
                             createPrivateMemref);

            // Collect slice loop stats.
            LoopNestStateCollector sliceCollector;
            sliceCollector.collect(sliceLoopNest.getOperation());
            // Promote single iteration slice loops to single IV value.
            for (auto forOp : sliceCollector.forOps) {
              promoteIfSingleIteration(forOp);
            }
            if (createPrivateMemref) {
              // Create private memref for 'memref' in 'dstAffineForOp'.
              SmallVector<Operation *, 4> storesForMemref;
              for (auto *storeOpInst : sliceCollector.storeOpInsts) {
                if (cast<AffineStoreOp>(storeOpInst).getMemRef() == memref)
                  storesForMemref.push_back(storeOpInst);
              }
              // TODO(andydavis) Use union of memref write regions to compute
              // private memref footprint.
              auto newMemRef = createPrivateMemRef(
                  dstAffineForOp, storesForMemref[0], bestDstLoopDepth,
                  fastMemorySpace, localBufSizeThreshold);
              visitedMemrefs.insert(newMemRef);
              // Create new node in dependence graph for 'newMemRef' alloc op.
              unsigned newMemRefNodeId =
                  mdg->addNode(newMemRef->getDefiningOp());
              // Add edge from 'newMemRef' node to dstNode.
              mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
            }

            // Collect dst loop stats after memref privatization transformation.
            LoopNestStateCollector dstLoopCollector;
            dstLoopCollector.collect(dstAffineForOp.getOperation());

            // Add new load ops to current Node load op list 'loads' to
            // continue fusing based on new operands.
            for (auto *loadOpInst : dstLoopCollector.loadOpInsts) {
              auto loadMemRef = cast<AffineLoadOp>(loadOpInst).getMemRef();
              if (visitedMemrefs.count(loadMemRef) == 0)
                loads.push_back(loadOpInst);
            }

            // Clear and add back loads and stores.
            mdg->clearNodeLoadAndStores(dstNode->id);
            mdg->addToNode(dstId, dstLoopCollector.loadOpInsts,
                           dstLoopCollector.storeOpInsts);
            // Remove old src loop nest if it no longer has outgoing dependence
            // edges, and if it does not write to a memref which escapes the
            // function. If 'writesToLiveInOrOut' is true, then 'srcNode' has
            // been fused into 'dstNode' and write region of 'dstNode' covers
            // the write region of 'srcNode', and 'srcNode' has no other users
            // so it is safe to remove.
            if (writesToLiveInOrOut || mdg->canRemoveNode(srcNode->id)) {
              mdg->removeNode(srcNode->id);
              srcNode->op->erase();
            } else {
              // Add remaining users of 'oldMemRef' back on the worklist (if not
              // already there), as its replacement with a local/private memref
              // has reduced dependences on 'oldMemRef' which may have created
              // new fusion opportunities.
              if (mdg->outEdges.count(srcNode->id) > 0) {
                SmallVector<MemRefDependenceGraph::Edge, 2> oldOutEdges =
                    mdg->outEdges[srcNode->id];
                for (auto &outEdge : oldOutEdges) {
                  if (outEdge.value == memref &&
                      worklistSet.count(outEdge.id) == 0) {
                    worklist.push_back(outEdge.id);
                    worklistSet.insert(outEdge.id);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() {
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      worklistSet.erase(dstId);

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode);
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) {
    DenseSet<unsigned> visitedSibNodeIds;
    std::pair<unsigned, Value> idAndMemref;
    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) {
      unsigned sibId = idAndMemref.first;
      Value memref = idAndMemref.second;
      // TODO(andydavis) Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst =
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id);
      if (insertPointInst == nullptr)
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts);
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      assert(sibLoadOpInsts.size() == 1);
      Operation *sibLoadOpInst = sibLoadOpInsts[0];
      assert(!sibNode->stores.empty());
      // TODO(andydavis) Choose the store which postdominates all other stores.
      auto *sibStoreOpInst = sibNode->stores.back();

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts);

      // Gather 'dstNode' store ops to 'memref'.
      SmallVector<Operation *, 2> dstStoreOpInsts;
      dstNode->getStoreOpsForMemref(memref, &dstStoreOpInsts);

      unsigned bestDstLoopDepth;
      mlir::ComputationSliceState sliceState;

      // Check if fusion would be profitable.
      if (!isFusionProfitable(sibLoadOpInst, sibStoreOpInst, dstLoadOpInsts,
                              dstStoreOpInsts, &sliceState, &bestDstLoopDepth,
                              maximalFusion))
        continue;

      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      auto sliceLoopNest = mlir::insertBackwardComputationSlice(
          sibLoadOpInst, dstLoadOpInsts[0], bestDstLoopDepth, &sliceState);
      if (sliceLoopNest != nullptr) {
        auto dstForInst = cast<AffineForOp>(dstNode->op);
        // Update operation position of fused loop nest (if needed).
        if (insertPointInst != dstForInst.getOperation()) {
          dstForInst.getOperation()->moveBefore(insertPointInst);
        }
        // Update data dependence graph state post fusion.
        updateStateAfterSiblingFusion(sliceLoopNest, sibNode, dstNode);
      }
    }
  }

  // Searches function argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) {
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) {
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO(andydavis) Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1)
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id))
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet);
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) {
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false;

      // Check that all stores are to the same memref.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) {
        storeMemrefs.insert(cast<AffineStoreOp>(storeOpInst).getMemRef());
      }
      if (storeMemrefs.size() != 1)
        return false;
      return true;
    };

    // Search for siblings which load the same memref function argument.
    auto fn = dstNode->op->getParentOfType<FuncOp>();
    for (unsigned i = 0, e = fn.getNumArguments(); i != e; ++i) {
      for (auto *user : fn.getArgument(i)->getUsers()) {
        if (auto loadOp = dyn_cast<AffineLoadOp>(user)) {
          // Gather loops surrounding 'use'.
          SmallVector<AffineForOp, 4> loops;
          getLoopIVs(*user, &loops);
          // Skip 'use' if it is not within a loop nest.
          if (loops.empty())
            continue;
          Node *sibNode = mdg->getForOpNode(loops[0]);
          assert(sibNode != nullptr);
          // Skip 'use' if it not a sibling to 'dstNode'.
          if (sibNode->id == dstNode->id)
            continue;
          // Skip 'use' if it has been visited.
          if (visitedSibNodeIds->count(sibNode->id) > 0)
            continue;
          // Skip 'use' if it does not load from the same memref as 'dstNode'.
          auto memref = loadOp.getMemRef();
          if (dstNode->getLoadOpCount(memref) == 0)
            continue;
          // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
          if (canFuseWithSibNode(sibNode, memref)) {
            visitedSibNodeIds->insert(sibNode->id);
            idAndMemrefToFuse->first = sibNode->id;
            idAndMemrefToFuse->second = memref;
            return true;
          }
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge(
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence.
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0)
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) {
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge(
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0)
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value)
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op))
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) {
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) {
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false;
  }

  void updateStateAfterSiblingFusion(AffineForOp sliceLoopNest, Node *sibNode,
                                     Node *dstNode) {
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id);

    // Collect slice loop stats.
    LoopNestStateCollector sliceCollector;
    sliceCollector.collect(sliceLoopNest.getOperation());
    // Promote single iteration slice loops to single IV value.
    for (auto forOp : sliceCollector.forOps) {
      promoteIfSingleIteration(forOp);
    }

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollector dstLoopCollector;
    dstLoopCollector.collect(dstForInst.getOperation());
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts);
    // Remove old sibling loop nest if it no longer has outgoing dependence
    // edges, and it does not write to a memref which escapes the
    // function.
    if (mdg->getOutEdgeCount(sibNode->id) == 0) {
      mdg->removeNode(sibNode->id);
      sibNode->op->erase();
    }
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref->use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref->getDefiningOp();
      if (isa_and_nonnull<AllocOp>(op))
        op->erase();
    }
  }
};

} // end anonymous namespace

void LoopFusion::runOnFunction() {
  // Override if a command line argument was provided.
  if (clFusionFastMemorySpace.getNumOccurrences() > 0) {
    fastMemorySpace = clFusionFastMemorySpace.getValue();
  }

  // Override if a command line argument was provided.
  if (clFusionLocalBufThreshold.getNumOccurrences() > 0) {
    localBufSizeThreshold = clFusionLocalBufThreshold * 1024;
  }

  if (clMaximalLoopFusion.getNumOccurrences() > 0)
    maximalFusion = clMaximalLoopFusion;

  MemRefDependenceGraph g;
  if (g.init(getFunction()))
    GreedyFusion(&g, localBufSizeThreshold, fastMemorySpace, maximalFusion)
        .run();
}

static PassRegistration<LoopFusion> pass("affine-loop-fusion",
                                         "Fuse loop nests");
