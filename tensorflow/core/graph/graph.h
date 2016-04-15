/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A Graph describes a set of computations that are to be
// performed, as well as the dependencies between those
// computations. The basic model is a DAG (directed acyclic graph) with
// * internal nodes representing computational operations to be performed;
// * edges represent dependencies, indicating the target may only be
//   executed once the source has completed; and
// * predefined "source" (start) and "sink" (finish) nodes -- the source
//   should be the only node that doesn't depend on anything, and the sink
//   should be the only node that nothing depends on.
//
// Note: Node ids are intended to be relatively dense in the
// 0..max_id range, but there may be gaps since ids won't be reused.
//
// Note: Some dependencies between operations are due to one operation
// consuming the output of another. In fact operations can produce
// multiple outputs and consume multiple inputs, and some
// optimizations will care about which specific outputs are connected
// to which specific inputs.  We therefore represent data dependency
// between output O of layer A and input I of layer B using
// "input index" and "output index" labels per edge.

#ifndef TENSORFLOW_GRAPH_GRAPH_H_
#define TENSORFLOW_GRAPH_GRAPH_H_

#include <functional>
#include <string>
#include <vector>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Edge;
class EdgeSetTest;
class Graph;
class Node;

class NeighborIter;  // Declared below
class NodeIter;      // Declared below

class Node {
 public:
  string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const string& name() const { return props_->node_def_.name(); }
  const string& type_string() const { return props_->node_def_.op(); }
  const NodeDef& def() const { return props_->node_def_; }
  const OpDef& op_def() const { return *props_->op_def_; }

  // input and output types
  int num_inputs() const { return props_->input_types_.size(); }
  DataType input_type(int i) const { return props_->input_types_[i]; }
  const DataTypeVector& input_types() const { return props_->input_types_; }

  int num_outputs() const { return props_->output_types_.size(); }
  DataType output_type(int o) const { return props_->output_types_[o]; }
  const DataTypeVector& output_types() const { return props_->output_types_; }

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move device_name outside of Node into a NodeId->DeviceName
  // map.
  string assigned_device_name() const { return assigned_device_name_; }
  void set_assigned_device_name(const string& device_name) {
    assigned_device_name_ = device_name;
  }

  // Get the neighboring nodes via edges either in or out of this node.
  gtl::iterator_range<NeighborIter> in_nodes() const;
  gtl::iterator_range<NeighborIter> out_nodes() const;
  const EdgeSet& in_edges() const { return in_edges_; }
  const EdgeSet& out_edges() const { return out_edges_; }

  // Node type helpers.
  bool IsSource() const { return id() == 0; }
  bool IsSink() const { return id() == 1; }
  // Anything other than the special Source & Sink nodes.
  bool IsOp() const { return id() > 1; }

  // Node class helpers
  bool IsSwitch() const { return (class_ == NC_SWITCH); }
  bool IsMerge() const { return (class_ == NC_MERGE); }
  bool IsEnter() const { return (class_ == NC_ENTER); }
  bool IsExit() const { return (class_ == NC_EXIT); }
  bool IsNextIteration() const { return (class_ == NC_NEXT_ITERATION); }
  bool IsLoopCond() const { return (class_ == NC_LOOP_COND); }
  bool IsControlTrigger() const { return (class_ == NC_CONTROL_TRIGGER); }
  bool IsSend() const { return (class_ == NC_SEND); }
  bool IsRecv() const { return (class_ == NC_RECV); }
  bool IsConstant() const { return (class_ == NC_CONSTANT); }
  bool IsVariable() const { return (class_ == NC_VARIABLE); }
  bool IsIdentity() const { return (class_ == NC_IDENTITY); }
  bool IsGetSessionHandle() const { return (class_ == NC_GET_SESSION_HANDLE); }
  bool IsGetSessionTensor() const { return (class_ == NC_GET_SESSION_TENSOR); }
  bool IsDeleteSessionTensor() const {
    return (class_ == NC_DELETE_SESSION_TENSOR);
  }
  bool IsControlFlow() const {
    return (class_ != NC_OTHER) &&  // Fast path
           (IsSwitch() || IsMerge() || IsEnter() || IsExit() ||
            IsNextIteration());
  }

  template <typename T>
  void AddAttr(const string& name, const T& val) {
    MaybeCopyOnWrite();
    SetAttrValue(val, &((*props_->node_def_.mutable_attr())[name]));
  }

 private:
  friend class Graph;
  Node();
  ~Node();

  class Properties : public core::RefCounted {
   public:
    Properties(const OpDef* op_def, const NodeDef& node_def,
               const DataTypeSlice inputs, const DataTypeSlice outputs);

    const OpDef* op_def_;  // not owned
    NodeDef node_def_;
    const DataTypeVector input_types_;
    const DataTypeVector output_types_;

   private:
    // Destructor invoked when last reference goes away via Unref()
    virtual ~Properties();
    TF_DISALLOW_COPY_AND_ASSIGN(Properties);
  };

  Properties* properties() const { return props_; }

  // Initialize() adopts a reference to props, and so is suitable if props was
  // just allocated or you call props->Ref() to increment the reference
  // count for a props being held by another Node.
  void Initialize(int id, int cost_id, Properties* props);
  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();
  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  void MaybeCopyOnWrite();

  // A set of mutually exclusive classes for different kinds of nodes,
  // class_ is initialized in the Node::Initialize routine based on the
  // node's type_string().
  enum NodeClass {
    NC_UNINITIALIZED,
    NC_SWITCH,
    NC_MERGE,
    NC_ENTER,
    NC_EXIT,
    NC_NEXT_ITERATION,
    NC_LOOP_COND,
    NC_CONTROL_TRIGGER,
    NC_SEND,
    NC_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_OTHER  // Not a special kind of node
  };

  int id_;       // -1 until Initialize() is called
  int cost_id_;  // -1 if there is no corresponding cost accounting node
  NodeClass class_;

  EdgeSet in_edges_;
  EdgeSet out_edges_;

  Properties* props_;

  // Name of device assigned to perform this computation.
  string assigned_device_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(Node);
};

class Edge {
 public:
  Node* src() const { return src_; }
  Node* dst() const { return dst_; }
  int id() const { return id_; }

  // Return the number of the source output that produces the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int src_output() const { return src_output_; }

  // Return the number of the destination input that consumes the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int dst_input() const { return dst_input_; }

  // Return true iff this is an edge that indicates a control-flow
  // (as opposed to a data-flow) dependency.
  bool IsControlEdge() const;

 private:
  Edge() {}

  friend class EdgeSetTest;
  friend class Graph;
  Node* src_;
  Node* dst_;
  int id_;
  int src_output_;
  int dst_input_;
};

// Thread compatible but not thread safe.
class Graph {
 public:
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in registry.
  explicit Graph(const OpRegistryInterface* registry);
  ~Graph();

  static const int kControlSlot = -1;

  // The GraphDef version range of this graph (see graph.proto).
  const VersionDef& versions() const { return versions_; }
  void set_versions(const VersionDef& versions) { versions_ = versions; }

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  Node* AddNode(const NodeDef& node_def, Status* status);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  Node* CopyNode(Node* node);

  // Remove a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  void RemoveNode(Node* node);

  // Add an edge that connects the xth output of "source" to the yth input
  // of "dest".
  const Edge* AddEdge(Node* source, int x, Node* dest, int y);

  // Add a control-edge (no data flows along this edge) that
  // connects "source" to "dest".
  const Edge* AddControlEdge(Node* source, Node* dest) {
    return AddEdge(source, kControlSlot, dest, kControlSlot);
  }

  // Removes edge from the graph.
  // REQUIRES: The edge must exist.
  void RemoveEdge(const Edge* edge);

  // The number of live nodes in the graph.
  //
  // Because nodes can be removed from the graph, num_nodes() is often
  // smaller than num_node_ids(). If one needs to create an array of
  // nodes indexed by node ids, num_node_ids() should be used as the
  // array's size.
  int num_nodes() const { return num_nodes_; }

  // The number of live edges in the graph.
  //
  // Because edges can be removed from the graph, num_edges() is often
  // smaller than num_edge_ids(). If one needs to create an array of
  // edges indexed by edge ids, num_edge_ids() should be used as the
  // array's size.
  int num_edges() const { return edges().size(); }

  // Serialize to a GraphDef.
  void ToGraphDef(GraphDef* graph_def) const;

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  string NewName(StringPiece prefix);

  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  gtl::iterator_range<NodeIter> nodes() const;

  // Returns one more than the maximum id assigned to any node.
  int num_node_ids() const { return nodes_.size(); }

  // Returns the node associated with an id, or nullptr if no node
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  Node* FindNodeId(int id) const { return nodes_[id]; }

  // Returns one more than the maximum id assigned to any edge.
  int num_edge_ids() const { return edges_.size(); }

  // Returns the Edge associated with an id, or nullptr if no edge
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  const Edge* FindEdgeId(int id) const { return edges_[id]; }

  // Access to the set of all edges.  Example usage:
  //   for (const Edge* e : graph.edges()) { ... }
  const EdgeSet& edges() const { return edge_set_; }

  // The pre-defined nodes.
  enum { kSourceId = 0, kSinkId = 1 };
  Node* source_node() const { return FindNodeId(kSourceId); }
  Node* sink_node() const { return FindNodeId(kSinkId); }

  const OpRegistryInterface* op_registry() const { return ops_; }

  // TODO(josh11b): uint64 hash() const;

 private:
  bool IsValidNode(Node* node) const;
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  Node* AllocateNode(Node::Properties* props, const Node* cost_node);
  void ReleaseNode(Node* node);

  // Registry of all known ops.  Not owned.
  const OpRegistryInterface* const ops_;

  // GraphDef versions
  VersionDef versions_;

  // Allocator which will give us good locality.
  core::Arena arena_;

  // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
  // the node with that id was removed from the graph.
  std::vector<Node*> nodes_;

  // Number of nodes alive.
  int64 num_nodes_ = 0;

  // Map from edge ids to allocated edges.  edges_[id] may be nullptr if
  // the edge with that id was removed from the graph.
  std::vector<Edge*> edges_;

  // For ease of iteration, we currently just keep a set of all live
  // edges.  May want to optimize by removing this copy.
  EdgeSet edge_set_;

  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  std::vector<Edge*> free_edges_;

  // For generating unique names.
  int name_counter_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};

// TODO(josh11b): We may want to support keeping an index on various
// node/edge attributes in a graph, particularly node names.

// Helper routines

inline bool IsSwitch(const Node* node) { return node->IsSwitch(); }
inline bool IsMerge(const Node* node) { return node->IsMerge(); }
inline bool IsEnter(const Node* node) { return node->IsEnter(); }
inline bool IsExit(const Node* node) { return node->IsExit(); }
inline bool IsNextIteration(const Node* n) { return n->IsNextIteration(); }
inline bool IsLoopCond(const Node* node) { return node->IsLoopCond(); }
inline bool IsControlTrigger(const Node* n) { return n->IsControlTrigger(); }
inline bool IsSend(const Node* node) { return node->IsSend(); }
inline bool IsRecv(const Node* node) { return node->IsRecv(); }

// True for Nodes that mediate the transfer of values between processes.
inline bool IsTransferNode(const Node* n) { return IsSend(n) || IsRecv(n); }

inline bool IsConstant(const Node* node) { return node->IsConstant(); }
inline bool IsVariable(const Node* node) { return node->IsVariable(); }
inline bool IsIdentity(const Node* node) { return node->IsIdentity(); }

// Returns true iff 'n' is a control flow node.
inline bool IsControlFlow(const Node* n) { return n->IsControlFlow(); }

inline bool IsHostMemoryPreserving(const Node* node) {
  return IsIdentity(node) || IsControlFlow(node);
}

// Iterator for stepping through the nodes of a graph.
class NodeIter {
 public:
  NodeIter(const Graph* graph, int id);
  bool operator==(const NodeIter& rhs);
  bool operator!=(const NodeIter& rhs);
  void operator++();
  Node* operator*();
  Node* operator->();

 private:
  // Invariant: id_ == graph_->num_node_ids() || graph_->FindId(id_) != nullptr
  const Graph* graph_;
  int id_;
};

// Iterator for stepping through the neighbors of a node.
class NeighborIter {
 public:
  NeighborIter(EdgeSet::const_iterator iter, bool incoming);
  bool operator==(const NeighborIter& rhs);
  bool operator!=(const NeighborIter& rhs);
  void operator++();
  Node* operator*();
  Node* operator->();

 private:
  EdgeSet::const_iterator iter_;
  bool incoming_;
};

// IMPLEMENTATION DETAILS, PLEASE IGNORE

inline NodeIter::NodeIter(const Graph* graph, int id)
    : graph_(graph), id_(id) {}

inline bool NodeIter::operator==(const NodeIter& rhs) {
  DCHECK(graph_ == rhs.graph_);
  return id_ == rhs.id_;
}

inline bool NodeIter::operator!=(const NodeIter& rhs) {
  return !(*this == rhs);
}

inline void NodeIter::operator++() {
  while (1) {
    DCHECK_LE(id_, graph_->num_node_ids());
    ++id_;
    if (id_ >= graph_->num_node_ids() || graph_->FindNodeId(id_) != nullptr) {
      return;
    }
  }
}

inline Node* NodeIter::operator*() { return graph_->FindNodeId(id_); }

inline Node* NodeIter::operator->() { return graph_->FindNodeId(id_); }

inline NeighborIter::NeighborIter(EdgeSet::const_iterator iter, bool incoming)
    : iter_(iter), incoming_(incoming) {}

inline bool NeighborIter::operator==(const NeighborIter& rhs) {
  return iter_ == rhs.iter_ && incoming_ == rhs.incoming_;
}

inline bool NeighborIter::operator!=(const NeighborIter& rhs) {
  return !(*this == rhs);
}

inline void NeighborIter::operator++() { ++iter_; }

inline Node* NeighborIter::operator*() {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline Node* NeighborIter::operator->() {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline bool Edge::IsControlEdge() const {
  // Note that if either src_output_ or dst_input_ is kControlSlot,
  // so is the other one (AddEdge checks this).
  return src_output_ == Graph::kControlSlot;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_GRAPH_H_
