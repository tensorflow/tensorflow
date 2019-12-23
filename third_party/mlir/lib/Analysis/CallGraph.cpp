//===- CallGraph.cpp - CallGraph analysis for MLIR ------------------------===//
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
// This file contains interfaces and analyses for defining a nested callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/CallInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CallInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// Returns if this node refers to the indirect/external node.
bool CallGraphNode::isExternal() const { return !callableRegion; }

/// Return the callable region this node represents. This can only be called
/// on non-external nodes.
Region *CallGraphNode::getCallableRegion() const {
  assert(!isExternal() && "the external node has no callable region");
  return callableRegion;
}

/// Adds an reference edge to the given node. This is only valid on the
/// external node.
void CallGraphNode::addAbstractEdge(CallGraphNode *node) {
  assert(isExternal() && "abstract edges are only valid on external nodes");
  addEdge(node, Edge::Kind::Abstract);
}

/// Add an outgoing call edge from this node.
void CallGraphNode::addCallEdge(CallGraphNode *node) {
  addEdge(node, Edge::Kind::Call);
}

/// Adds a reference edge to the given child node.
void CallGraphNode::addChildEdge(CallGraphNode *child) {
  addEdge(child, Edge::Kind::Child);
}

/// Returns true if this node has any child edges.
bool CallGraphNode::hasChildren() const {
  return llvm::any_of(edges, [](const Edge &edge) { return edge.isChild(); });
}

/// Add an edge to 'node' with the given kind.
void CallGraphNode::addEdge(CallGraphNode *node, Edge::Kind kind) {
  edges.insert({node, kind});
}

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

/// Recursively compute the callgraph edges for the given operation. Computed
/// edges are placed into the given callgraph object.
static void computeCallGraph(Operation *op, CallGraph &cg,
                             CallGraphNode *parentNode);

/// Compute the set of callgraph nodes that are created by regions nested within
/// 'op'.
static void computeCallables(Operation *op, CallGraph &cg,
                             CallGraphNode *parentNode) {
  if (op->getNumRegions() == 0)
    return;
  if (auto callableOp = dyn_cast<CallableOpInterface>(op)) {
    SmallVector<Region *, 1> callables;
    callableOp.getCallableRegions(callables);
    for (auto *callableRegion : callables)
      cg.getOrAddNode(callableRegion, parentNode);
  }
}

/// Recursively compute the callgraph edges within the given region. Computed
/// edges are placed into the given callgraph object.
static void computeCallGraph(Region &region, CallGraph &cg,
                             CallGraphNode *parentNode) {
  // Iterate over the nested operations twice:
  /// One to fully create nodes in the for each callable region of a nested
  /// operation;
  for (auto &block : region)
    for (auto &nested : block)
      computeCallables(&nested, cg, parentNode);

  /// And another to recursively compute the callgraph.
  for (auto &block : region)
    for (auto &nested : block)
      computeCallGraph(&nested, cg, parentNode);
}

/// Recursively compute the callgraph edges for the given operation. Computed
/// edges are placed into the given callgraph object.
static void computeCallGraph(Operation *op, CallGraph &cg,
                             CallGraphNode *parentNode) {
  // Compute the callgraph nodes and edges for each of the nested operations.
  auto isCallable = isa<CallableOpInterface>(op);
  for (auto &region : op->getRegions()) {
    // Check to see if this region is a callable node, if so this is the parent
    // node of the nested region.
    CallGraphNode *nestedParentNode;
    if (!isCallable || !(nestedParentNode = cg.lookupNode(&region)))
      nestedParentNode = parentNode;
    computeCallGraph(region, cg, nestedParentNode);
  }

  // If there is no parent node, we ignore this operation. Even if this
  // operation was a call, there would be no callgraph node to attribute it to.
  if (!parentNode)
    return;

  // If this is a call operation, resolve the callee.
  if (auto call = dyn_cast<CallOpInterface>(op))
    parentNode->addCallEdge(
        cg.resolveCallable(call.getCallableForCallee(), op));
}

CallGraph::CallGraph(Operation *op) : externalNode(/*callableRegion=*/nullptr) {
  computeCallGraph(op, *this, /*parentNode=*/nullptr);
}

/// Get or add a call graph node for the given region.
CallGraphNode *CallGraph::getOrAddNode(Region *region,
                                       CallGraphNode *parentNode) {
  assert(region && isa<CallableOpInterface>(region->getParentOp()) &&
         "expected parent operation to be callable");
  std::unique_ptr<CallGraphNode> &node = nodes[region];
  if (!node) {
    node.reset(new CallGraphNode(region));

    // Add this node to the given parent node if necessary.
    if (parentNode)
      parentNode->addChildEdge(node.get());
    else
      // Otherwise, connect all callable nodes to the external node, this allows
      // for conservatively including all callable nodes within the graph.
      // FIXME(riverriddle) This isn't correct, this is only necessary for
      // callable nodes that *could* be called from external sources. This
      // requires extending the interface for callables to check if they may be
      // referenced externally.
      externalNode.addAbstractEdge(node.get());
  }
  return node.get();
}

/// Lookup a call graph node for the given region, or nullptr if none is
/// registered.
CallGraphNode *CallGraph::lookupNode(Region *region) const {
  auto it = nodes.find(region);
  return it == nodes.end() ? nullptr : it->second.get();
}

/// Resolve the callable for given callee to a node in the callgraph, or the
/// external node if a valid node was not resolved.
CallGraphNode *CallGraph::resolveCallable(CallInterfaceCallable callable,
                                          Operation *from) const {
  // Get the callee operation from the callable.
  Operation *callee;
  if (auto symbolRef = callable.dyn_cast<SymbolRefAttr>())
    // TODO(riverriddle) Support nested references.
    callee = SymbolTable::lookupNearestSymbolFrom(from,
                                                  symbolRef.getRootReference());
  else
    callee = callable.get<ValuePtr>()->getDefiningOp();

  // If the callee is non-null and is a valid callable object, try to get the
  // called region from it.
  if (callee && callee->getNumRegions()) {
    if (auto callableOp = dyn_cast_or_null<CallableOpInterface>(callee)) {
      if (auto *node = lookupNode(callableOp.getCallableRegion(callable)))
        return node;
    }
  }

  // If we don't have a valid direct region, this is an external call.
  return getExternalNode();
}

//===----------------------------------------------------------------------===//
// Printing

/// Dump the graph in a human readable format.
void CallGraph::dump() const { print(llvm::errs()); }
void CallGraph::print(raw_ostream &os) const {
  os << "// ---- CallGraph ----\n";

  // Functor used to output the name for the given node.
  auto emitNodeName = [&](const CallGraphNode *node) {
    if (node->isExternal()) {
      os << "<External-Node>";
      return;
    }

    auto *callableRegion = node->getCallableRegion();
    auto *parentOp = callableRegion->getParentOp();
    os << "'" << callableRegion->getParentOp()->getName() << "' - Region #"
       << callableRegion->getRegionNumber();
    if (auto attrs = parentOp->getAttrList().getDictionary())
      os << " : " << attrs;
  };

  for (auto &nodeIt : nodes) {
    const CallGraphNode *node = nodeIt.second.get();

    // Dump the header for this node.
    os << "// - Node : ";
    emitNodeName(node);
    os << "\n";

    // Emit each of the edges.
    for (auto &edge : *node) {
      os << "// -- ";
      if (edge.isCall())
        os << "Call";
      else if (edge.isChild())
        os << "Child";

      os << "-Edge : ";
      emitNodeName(edge.getTarget());
      os << "\n";
    }
    os << "//\n";
  }

  os << "// -- SCCs --\n";

  for (auto &scc : make_range(llvm::scc_begin(this), llvm::scc_end(this))) {
    os << "// - SCC : \n";
    for (auto &node : scc) {
      os << "// -- Node :";
      emitNodeName(node);
      os << "\n";
    }
    os << "\n";
  }

  os << "// -------------------\n";
}
