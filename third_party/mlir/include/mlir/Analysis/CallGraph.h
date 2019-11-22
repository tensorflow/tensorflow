//===- CallGraph.h - CallGraph analysis for MLIR ----------------*- C++ -*-===//
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
// This file contains an analysis for computing the multi-level callgraph from a
// given top-level operation. This nodes within this callgraph are defined by
// the `CallOpInterface` and `CallableOpInterface` operation interfaces defined
// in CallInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_CALLGRAPH_H
#define MLIR_ANALYSIS_CALLGRAPH_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
struct CallInterfaceCallable;
class Operation;
class Region;

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// This class represents a single callable in the callgraph. Aside from the
/// external node, each node represents a callable node in the graph and
/// contains a valid corresponding Region. The external node is a virtual node
/// used to represent external edges into, and out of, the callgraph.
class CallGraphNode {
public:
  /// This class represents a directed edge between two nodes in the callgraph.
  class Edge {
    enum class Kind {
      // An 'Abstract' edge represents an opaque, non-operation, reference
      // between this node and the target. Edges of this type are only valid
      // from the external node, as there is no valid connection to an operation
      // in the module.
      Abstract,

      // A 'Call' edge represents a direct reference to the target node via a
      // call-like operation within the callable region of this node.
      Call,

      // A 'Child' edge is used when the region of target node is defined inside
      // of the callable region of this node. This means that the region of this
      // node is an ancestor of the region for the target node. As such, this
      // edge cannot be used on the 'external' node.
      Child,
    };

  public:
    /// Returns if this edge represents an `Abstract` edge.
    bool isAbstract() const { return targetAndKind.getInt() == Kind::Abstract; }

    /// Returns if this edge represents a `Call` edge.
    bool isCall() const { return targetAndKind.getInt() == Kind::Call; }

    /// Returns if this edge represents a `Child` edge.
    bool isChild() const { return targetAndKind.getInt() == Kind::Child; }

    /// Returns the target node for this edge.
    CallGraphNode *getTarget() const { return targetAndKind.getPointer(); }

    bool operator==(const Edge &edge) const {
      return targetAndKind == edge.targetAndKind;
    }

  private:
    Edge(CallGraphNode *node, Kind kind) : targetAndKind(node, kind) {}
    explicit Edge(llvm::PointerIntPair<CallGraphNode *, 2, Kind> targetAndKind)
        : targetAndKind(targetAndKind) {}

    /// The target node of this edge, as well as the edge kind.
    llvm::PointerIntPair<CallGraphNode *, 2, Kind> targetAndKind;

    // Provide access to the constructor and Kind.
    friend class CallGraphNode;
  };

  /// Returns if this node is the external node.
  bool isExternal() const;

  /// Returns the callable region this node represents. This can only be called
  /// on non-external nodes.
  Region *getCallableRegion() const;

  /// Adds an abstract reference edge to the given node. An abstract edge does
  /// not come from any observable operations, so this is only valid on the
  /// external node.
  void addAbstractEdge(CallGraphNode *node);

  /// Add an outgoing call edge from this node.
  void addCallEdge(CallGraphNode *node);

  /// Adds a reference edge to the given child node.
  void addChildEdge(CallGraphNode *child);

  /// Iterator over the outgoing edges of this node.
  using iterator = SmallVectorImpl<Edge>::const_iterator;
  iterator begin() const { return edges.begin(); }
  iterator end() const { return edges.end(); }

  /// Returns true if this node has any child edges.
  bool hasChildren() const;

private:
  /// DenseMap info for callgraph edges.
  struct EdgeKeyInfo {
    using BaseInfo =
        DenseMapInfo<llvm::PointerIntPair<CallGraphNode *, 2, Edge::Kind>>;

    static Edge getEmptyKey() { return Edge(BaseInfo::getEmptyKey()); }
    static Edge getTombstoneKey() { return Edge(BaseInfo::getTombstoneKey()); }
    static unsigned getHashValue(const Edge &edge) {
      return BaseInfo::getHashValue(edge.targetAndKind);
    }
    static bool isEqual(const Edge &lhs, const Edge &rhs) { return lhs == rhs; }
  };

  CallGraphNode(Region *callableRegion) : callableRegion(callableRegion) {}

  /// Add an edge to 'node' with the given kind.
  void addEdge(CallGraphNode *node, Edge::Kind kind);

  /// The callable region defines the boundary of the call graph node. This is
  /// the region referenced by 'call' operations. This is at a per-region
  /// boundary as operations may define multiple callable regions.
  Region *callableRegion;

  /// A set of out-going edges from this node to other nodes in the graph.
  llvm::SetVector<Edge, SmallVector<Edge, 4>,
                  llvm::SmallDenseSet<Edge, 4, EdgeKeyInfo>>
      edges;

  // Provide access to private methods.
  friend class CallGraph;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

class CallGraph {
  using NodeMapT = llvm::MapVector<Region *, std::unique_ptr<CallGraphNode>>;

  /// This class represents an iterator over the internal call graph nodes. This
  /// class unwraps the map iterator to access the raw node.
  class NodeIterator final
      : public llvm::mapped_iterator<
            NodeMapT::const_iterator,
            CallGraphNode *(*)(const NodeMapT::value_type &)> {
    static CallGraphNode *unwrap(const NodeMapT::value_type &value) {
      return value.second.get();
    }

  public:
    /// Initializes the result type iterator to the specified result iterator.
    NodeIterator(NodeMapT::const_iterator it)
        : llvm::mapped_iterator<
              NodeMapT::const_iterator,
              CallGraphNode *(*)(const NodeMapT::value_type &)>(it, &unwrap) {}
  };

public:
  CallGraph(Operation *op);

  /// Get or add a call graph node for the given region. `parentNode`
  /// corresponds to the direct node in the callgraph that contains the parent
  /// operation of `region`, or nullptr if there is no parent node.
  CallGraphNode *getOrAddNode(Region *region, CallGraphNode *parentNode);

  /// Lookup a call graph node for the given region, or nullptr if none is
  /// registered.
  CallGraphNode *lookupNode(Region *region) const;

  /// Return the callgraph node representing the indirect-external callee.
  CallGraphNode *getExternalNode() const {
    return const_cast<CallGraphNode *>(&externalNode);
  }

  /// Resolve the callable for given callee to a node in the callgraph, or the
  /// external node if a valid node was not resolved. 'from' provides an anchor
  /// for symbol table lookups, and is only required if the callable is a symbol
  /// reference.
  CallGraphNode *resolveCallable(CallInterfaceCallable callable,
                                 Operation *from = nullptr) const;

  /// An iterator over the nodes of the graph.
  using iterator = NodeIterator;
  iterator begin() const { return nodes.begin(); }
  iterator end() const { return nodes.end(); }

  /// Dump the graph in a human readable format.
  void dump() const;
  void print(raw_ostream &os) const;

private:
  /// The set of nodes within the callgraph.
  NodeMapT nodes;

  /// A special node used to indicate an external edges.
  CallGraphNode externalNode;
};

} // end namespace mlir

namespace llvm {
// Provide graph traits for traversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<const mlir::CallGraphNode *> {
  using NodeRef = mlir::CallGraphNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  static NodeRef unwrap(const mlir::CallGraphNode::Edge &edge) {
    return edge.getTarget();
  }

  // ChildIteratorType/begin/end - Allow iteration over all nodes in the graph.
  using ChildIteratorType =
      mapped_iterator<mlir::CallGraphNode::iterator, decltype(&unwrap)>;
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->begin(), &unwrap};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->end(), &unwrap};
  }
};

template <>
struct GraphTraits<const mlir::CallGraph *>
    : public GraphTraits<const mlir::CallGraphNode *> {
  /// The entry node into the graph is the external node.
  static NodeRef getEntryNode(const mlir::CallGraph *cg) {
    return cg->getExternalNode();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  using nodes_iterator = mlir::CallGraph::iterator;
  static nodes_iterator nodes_begin(mlir::CallGraph *cg) { return cg->begin(); }
  static nodes_iterator nodes_end(mlir::CallGraph *cg) { return cg->end(); }
};
} // end namespace llvm

#endif // MLIR_ANALYSIS_CALLGRAPH_H
