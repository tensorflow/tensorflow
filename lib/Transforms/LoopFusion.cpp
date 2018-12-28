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
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using llvm::SetVector;

using namespace mlir;

// TODO(andydavis) These flags are global for the pass to be used for
// experimentation. Find a way to provide more fine grained control (i.e.
// depth per-loop nest, or depth per load/store op) for this pass utilizing a
// cost model.
static llvm::cl::opt<unsigned> clSrcLoopDepth(
    "src-loop-depth", llvm::cl::Hidden,
    llvm::cl::desc("Controls the depth of the source loop nest at which "
                   "to apply loop iteration slicing before fusion."));

static llvm::cl::opt<unsigned> clDstLoopDepth(
    "dst-loop-depth", llvm::cl::Hidden,
    llvm::cl::desc("Controls the depth of the destination loop nest at which "
                   "to fuse the source loop nest slice."));

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

  PassResult runOnMLFunction(MLFunction *f) override;
  static char passID;
};

} // end anonymous namespace

char LoopFusion::passID = 0;

FunctionPass *mlir::createLoopFusionPass() { return new LoopFusion; }

static void getSingleMemRefAccess(OperationInst *loadOrStoreOpStmt,
                                  MemRefAccess *access) {
  if (auto loadOp = loadOrStoreOpStmt->dyn_cast<LoadOp>()) {
    access->memref = loadOp->getMemRef();
    access->opStmt = loadOrStoreOpStmt;
    auto loadMemrefType = loadOp->getMemRefType();
    access->indices.reserve(loadMemrefType.getRank());
    for (auto *index : loadOp->getIndices()) {
      access->indices.push_back(index);
    }
  } else {
    assert(loadOrStoreOpStmt->isa<StoreOp>());
    auto storeOp = loadOrStoreOpStmt->dyn_cast<StoreOp>();
    access->opStmt = loadOrStoreOpStmt;
    access->memref = storeOp->getMemRef();
    auto storeMemrefType = storeOp->getMemRefType();
    access->indices.reserve(storeMemrefType.getRank());
    for (auto *index : storeOp->getIndices()) {
      access->indices.push_back(index);
    }
  }
}

// FusionCandidate encapsulates source and destination memref access within
// loop nests which are candidates for loop fusion.
struct FusionCandidate {
  // Load or store access within src loop nest to be fused into dst loop nest.
  MemRefAccess srcAccess;
  // Load or store access within dst loop nest.
  MemRefAccess dstAccess;
};

static FusionCandidate buildFusionCandidate(OperationInst *srcStoreOpStmt,
                                            OperationInst *dstLoadOpStmt) {
  FusionCandidate candidate;
  // Get store access for src loop nest.
  getSingleMemRefAccess(srcStoreOpStmt, &candidate.srcAccess);
  // Get load access for dst loop nest.
  getSingleMemRefAccess(dstLoadOpStmt, &candidate.dstAccess);
  return candidate;
}

// Returns the loop depth of the loop nest surrounding 'opStmt'.
static unsigned getLoopDepth(OperationInst *opStmt) {
  unsigned loopDepth = 0;
  auto *currStmt = opStmt->getParentStmt();
  ForStmt *currForStmt;
  while (currStmt && (currForStmt = dyn_cast<ForStmt>(currStmt))) {
    ++loopDepth;
    currStmt = currStmt->getParentStmt();
  }
  return loopDepth;
}

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not an IfStmt was encountered in the loop nest.
class LoopNestStateCollector : public StmtWalker<LoopNestStateCollector> {
public:
  SmallVector<ForStmt *, 4> forStmts;
  SmallVector<OperationInst *, 4> loadOpStmts;
  SmallVector<OperationInst *, 4> storeOpStmts;
  bool hasIfStmt = false;

  void visitForStmt(ForStmt *forStmt) { forStmts.push_back(forStmt); }

  void visitIfStmt(IfStmt *ifStmt) { hasIfStmt = true; }

  void visitOperationInst(OperationInst *opStmt) {
    if (opStmt->isa<LoadOp>())
      loadOpStmts.push_back(opStmt);
    if (opStmt->isa<StoreOp>())
      storeOpStmts.push_back(opStmt);
  }
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level statements in an MLFunction which contain load/store ops, and edges
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
    Statement *stmt;
    // List of load operations.
    SmallVector<OperationInst *, 4> loads;
    // List of store op stmts.
    SmallVector<OperationInst *, 4> stores;
    Node(unsigned id, Statement *stmt) : id(id), stmt(stmt) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value *memref) {
      unsigned loadOpCount = 0;
      for (auto *loadOpStmt : loads) {
        if (memref == loadOpStmt->cast<LoadOp>()->getMemRef())
          ++loadOpCount;
      }
      return loadOpCount;
    }

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value *memref) {
      unsigned storeOpCount = 0;
      for (auto *storeOpStmt : stores) {
        if (memref == storeOpStmt->cast<StoreOp>()->getMemRef())
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
  bool init(MLFunction *f);

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
    for (auto *loadOpStmt : loads)
      node->loads.push_back(loadOpStmt);
    for (auto *storeOpStmt : stores)
      node->stores.push_back(storeOpStmt);
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

// Intializes the data dependence graph by walking statements in 'f'.
// Assigns each node in the graph a node id based on program order in 'f'.
// TODO(andydavis) Add support for taking a StmtBlock arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init(MLFunction *f) {
  unsigned id = 0;
  DenseMap<Value *, SetVector<unsigned>> memrefAccesses;
  for (auto &stmt : *f->getBody()) {
    if (auto *forStmt = dyn_cast<ForStmt>(&stmt)) {
      // Create graph node 'id' to represent top-level 'forStmt' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.walkForStmt(forStmt);
      // Return false if IfStmts are found (not currently supported).
      if (collector.hasIfStmt)
        return false;
      Node node(id++, &stmt);
      for (auto *opStmt : collector.loadOpStmts) {
        node.loads.push_back(opStmt);
        auto *memref = opStmt->cast<LoadOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      for (auto *opStmt : collector.storeOpStmts) {
        node.stores.push_back(opStmt);
        auto *memref = opStmt->cast<StoreOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      nodes.insert({node.id, node});
    }
    if (auto *opStmt = dyn_cast<OperationInst>(&stmt)) {
      if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
        // Create graph node for top-level load op.
        Node node(id++, &stmt);
        node.loads.push_back(opStmt);
        auto *memref = opStmt->cast<LoadOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
        nodes.insert({node.id, node});
      }
      if (auto storeOp = opStmt->dyn_cast<StoreOp>()) {
        // Create graph node for top-level store op.
        Node node(id++, &stmt);
        node.stores.push_back(opStmt);
        auto *memref = opStmt->cast<StoreOp>()->getMemRef();
        memrefAccesses[memref].insert(node.id);
        nodes.insert({node.id, node});
      }
    }
    // Return false if IfStmts are found (not currently supported).
    if (isa<IfStmt>(&stmt))
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

// GreedyFusion greedily fuses loop nests which have a producer/consumer
// relationship on a memref, with the goal of improving locality. Currently,
// this the producer/consumer relationship is required to be unique in the
// MLFunction (there are TODOs to relax this constraint in the future).
//
// The steps of the algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop a ForStmt of the worklist. This 'dstForStmt' will be a candidate
//      destination ForStmt into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstForStmt' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Lookup dependent loop nests at earlier positions in the MLFunction
//         which have a single store op to the same memref.
//      *) Check if dependences would be violated by the fusion. For example,
//         the src loop nest may load from memrefs which are different than
//         the producer-consumer memref between src and dest loop nests.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         just before the dst load op user.
//      *) Add the newly fused load/store operation statements to the state,
//         and also add newly fuse load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// Given a graph where top-level statements are vertices in the set 'V' and
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
      if (!isa<ForStmt>(dstNode->stmt))
        continue;

      SmallVector<OperationInst *, 4> loads = dstNode->loads;
      while (!loads.empty()) {
        auto *dstLoadOpStmt = loads.pop_back_val();
        auto *memref = dstLoadOpStmt->cast<LoadOp>()->getMemRef();
        // Skip 'dstLoadOpStmt' if multiple loads to 'memref' in 'dstNode'.
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
          if (!isa<ForStmt>(srcNode->stmt))
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
          auto *srcStoreOpStmt = srcNode->stores.front();
          // Build fusion candidate out of 'srcStoreOpStmt' and 'dstLoadOpStmt'.
          FusionCandidate candidate =
              buildFusionCandidate(srcStoreOpStmt, dstLoadOpStmt);
          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          unsigned srcLoopDepth = clSrcLoopDepth.getNumOccurrences() > 0
                                      ? clSrcLoopDepth
                                      : getLoopDepth(srcStoreOpStmt);
          unsigned dstLoopDepth = clDstLoopDepth.getNumOccurrences() > 0
                                      ? clDstLoopDepth
                                      : getLoopDepth(dstLoadOpStmt);
          auto *sliceLoopNest = mlir::insertBackwardComputationSlice(
              &candidate.srcAccess, &candidate.dstAccess, srcLoopDepth,
              dstLoopDepth);
          if (sliceLoopNest != nullptr) {
            // Remove edges between 'srcNode' and 'dstNode' and remove 'srcNode'
            mdg->updateEdgesAndRemoveSrcNode(srcNode->id, dstNode->id);
            // Record all load/store accesses in 'sliceLoopNest' at 'dstPos'.
            LoopNestStateCollector collector;
            collector.walkForStmt(sliceLoopNest);
            mdg->addToNode(dstId, collector.loadOpStmts,
                           collector.storeOpStmts);
            // Add new load ops to current Node load op list 'loads' to
            // continue fusing based on new operands.
            for (auto *loadOpStmt : collector.loadOpStmts)
              loads.push_back(loadOpStmt);
            // Promote single iteration loops to single IV value.
            for (auto *forStmt : collector.forStmts) {
              promoteIfSingleIteration(forStmt);
            }
            // Remove old src loop nest.
            cast<ForStmt>(srcNode->stmt)->erase();
          }
        }
      }
    }
  }
};

} // end anonymous namespace

PassResult LoopFusion::runOnMLFunction(MLFunction *f) {
  MemRefDependenceGraph g;
  if (g.init(f))
    GreedyFusion(&g).run();
  return success();
}

static PassRegistration<LoopFusion> pass("loop-fusion", "Fuse loop nests");
