/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_H_

#include <functional>
#include <string>
#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
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
class GraphDef;
class Node;
class VersionDef;
class WhileContext;

class NeighborIter;    // Declared below
class NodeIter;        // Declared below
class NodeProperties;  // Defined in .cc

class Node {
 public:
  string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const string& name() const;
  const string& type_string() const;

  // def() provides the NodeDef the user supplied, but the specifics
  // of this Node may have changed due to placement, optimization, etc.
  // In particular:
  // * def().name() will match name();
  // * def().op() will match type_string() and op_def().name();
  // * def().input() is not reliable, use "in_edges()" below instead;
  // * def().device() is the "user's requested device" and may not match
  //   the actual assigned device, see assigned_device_name() below;
  // * def().attr() is authoritative.
  // TODO(irving): Replace with NodeInfo.
  const NodeDef& def() const;
  const OpDef& op_def() const;

  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32 i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32 o) const;
  const DataTypeVector& output_types() const;

  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.
  const string& requested_device() const;

  // This changes the user requested device but not necessarily the device that
  // on which the operation will run.
  void set_requested_device(const string& device);

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move assigned_device_name outside of Node into a
  // NodeId->DeviceName map.
  const string& assigned_device_name() const;
  void set_assigned_device_name(const string& device_name);
  bool has_assigned_device_name() const {
    return assigned_device_name_index_ > 0;
  }
  int assigned_device_name_index() const { return assigned_device_name_index_; }
  void set_assigned_device_name_index(int index);

  // Read only access to attributes
  AttrSlice attrs() const;

  // Inputs requested by the NodeDef.  For the actual inputs, use in_edges.
  const protobuf::RepeatedPtrField<string>& requested_inputs() const;

  // Get the neighboring nodes via edges either in or out of this node.  This
  // includes control edges.
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
  bool IsSwitch() const { return class_ == NC_SWITCH; }
  bool IsMerge() const { return class_ == NC_MERGE; }
  bool IsEnter() const { return class_ == NC_ENTER; }
  bool IsExit() const { return class_ == NC_EXIT; }
  bool IsNextIteration() const { return class_ == NC_NEXT_ITERATION; }
  bool IsLoopCond() const { return class_ == NC_LOOP_COND; }
  bool IsControlTrigger() const { return class_ == NC_CONTROL_TRIGGER; }
  bool IsSend() const { return class_ == NC_SEND || class_ == NC_HOST_SEND; }
  bool IsRecv() const { return class_ == NC_RECV || class_ == NC_HOST_RECV; }
  bool IsConstant() const { return class_ == NC_CONSTANT; }
  bool IsVariable() const { return class_ == NC_VARIABLE; }
  bool IsIdentity() const { return class_ == NC_IDENTITY; }
  bool IsGetSessionHandle() const { return class_ == NC_GET_SESSION_HANDLE; }
  bool IsGetSessionTensor() const { return class_ == NC_GET_SESSION_TENSOR; }
  bool IsDeleteSessionTensor() const {
    return class_ == NC_DELETE_SESSION_TENSOR;
  }
  bool IsControlFlow() const {
    return (class_ != NC_OTHER) &&  // Fast path
           (IsSwitch() || IsMerge() || IsEnter() || IsExit() ||
            IsNextIteration());
  }
  bool IsHostSend() const { return class_ == NC_HOST_SEND; }
  bool IsHostRecv() const { return class_ == NC_HOST_RECV; }
  bool IsScopedAllocator() const { return class_ == NC_SCOPED_ALLOCATOR; }
  bool IsCollective() const { return class_ == NC_COLLECTIVE; }

  bool IsMetadata() const { return class_ == NC_METADATA; }

  template <typename T>
  void AddAttr(const string& name, const T& val) {
    SetAttrValue(val, AddAttrHelper(name));
  }

  void ClearAttr(const string& name);

  // Returns into '*e' the edge connecting to the 'idx' input of this Node.
  Status input_edge(int idx, const Edge** e) const;

  // Returns into '*edges' the input data edges of this Node, indexed by input
  // number. Does not return control edges.
  Status input_edges(std::vector<const Edge*>* edges) const;

  // Returns into '*n' the node that has an output connected to the
  // 'idx' input of this Node.
  Status input_node(int idx, const Node** n) const;
  Status input_node(int idx, Node** n) const;

  WhileContext* while_ctx() const { return while_ctx_; }
  void set_while_ctx(WhileContext* while_ctx) {
    DCHECK(IsExit());
    DCHECK(while_ctx_ == nullptr);
    while_ctx_ = while_ctx;
  }

 private:
  friend class Graph;
  Node();

  NodeProperties* properties() const { return props_.get(); }

  void Initialize(int id, int cost_id, std::shared_ptr<NodeProperties> props);

  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();

  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  void MaybeCopyOnWrite();

  AttrValue* AddAttrHelper(const string& name);

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
    NC_HOST_SEND,
    NC_RECV,
    NC_HOST_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_METADATA,
    NC_SCOPED_ALLOCATOR,
    NC_COLLECTIVE,
    NC_OTHER  // Not a special kind of node
  };

  static const std::unordered_map<string, NodeClass>& kNodeClassTable;

  static NodeClass GetNodeClassForOp(const string& ts);

  int id_;       // -1 until Initialize() is called
  int cost_id_;  // -1 if there is no corresponding cost accounting node
  NodeClass class_;

  EdgeSet in_edges_;
  EdgeSet out_edges_;

  // NOTE(skyewm): inheriting from core::RefCounted may have a slight
  // performance benefit over using shared_ptr, at the cost of manual ref
  // counting
  std::shared_ptr<NodeProperties> props_;

  // Index within Graph::device_names_ of the name of device assigned
  // to perform this computation.
  int assigned_device_name_index_;

  // A back-pointer to the Graph that owns this node.  Currently, this exists
  // solely to allow Node::[set_]assigned_device_name() to work. However, if all
  // callers of Node::[set_]assigned_device_name() are modified to use the
  // equivalent methods defined directly on Graph, then we can remove this
  // field and reclaim that memory.
  Graph* graph_;

  // Set if this is an exit node of a while loop with an associated
  // WhileContext. Otherwise null. (This is only set for exit nodes because
  // they're the first nodes of a loop encountered while creating the gradient
  // graph. Exit nodes that are part of while loop gradient graphs will not have
  // this set.)
  WhileContext* while_ctx_;

  TF_DISALLOW_COPY_AND_ASSIGN(Node);
};

// Represents an input of a node, i.e., the `index`-th input to `node`.
struct InputTensor {
  const Node* node;
  int index;

  InputTensor(const Node* n, int i) : node(n), index(i) {}
  InputTensor() : node(nullptr), index(0) {}
};

// Represents an output of a node, i.e., the `index`-th output of `node`. Note
// that a single `OutputTensor` can correspond to multiple `Edge`s if the output
// is consumed by multiple destination nodes.
struct OutputTensor {
  const Node* node;
  int index;

  OutputTensor(const Node* n, int i) : node(n), index(i) {}
  OutputTensor() : node(nullptr), index(0) {}
};

class Edge {
 public:
  Node* src() const { return src_; }
  Node* dst() const { return dst_; }
  int id() const { return id_; }

  // Return the index of the source output that produces the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int src_output() const { return src_output_; }

  // Return the index of the destination input that consumes the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int dst_input() const { return dst_input_; }

  // Return true iff this is an edge that indicates a control-flow
  // (as opposed to a data-flow) dependency.
  bool IsControlEdge() const;

  string DebugString() const;

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

// Allows for iteration of the edges of a Graph, by iterating the underlying
// Graph.edges_ vector while skipping over null entries.
class GraphEdgesIterable {
 private:
  const std::vector<Edge*>& edges_;

 public:
  explicit GraphEdgesIterable(const std::vector<Edge*>& edges)
      : edges_(edges) {}

  typedef Edge* value_type;

  class const_iterator {
   private:
    // The underlying iterator.
    std::vector<value_type>::const_iterator iter_;

    // The end of the underlying iterator.
    std::vector<value_type>::const_iterator end_;

    // Advances iter_ until it reaches a non-null item, or reaches the end.
    void apply_filter() {
      while (iter_ != end_ && *iter_ == nullptr) {
        ++iter_;
      }
    }

   public:
    const_iterator(std::vector<value_type>::const_iterator iter,
                   std::vector<value_type>::const_iterator end)
        : iter_(iter), end_(end) {
      apply_filter();
    }

    bool operator==(const const_iterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const const_iterator& other) const {
      return iter_ != other.iter_;
    }

    // This is the prefix increment operator (++x), which is the operator
    // used by C++ range iteration (for (x : y) ...).  We intentionally do not
    // provide a postfix increment operator.
    const_iterator& operator++() {
      ++iter_;
      apply_filter();
      return *this;
    }

    value_type operator*() { return *iter_; }
  };

  const_iterator begin() {
    return const_iterator(edges_.begin(), edges_.end());
  }
  const_iterator end() { return const_iterator(edges_.end(), edges_.end()); }
};

// Thread compatible but not thread safe.
class Graph {
 public:
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in registry. `registry`s lifetime must be at
  // least that of the constructed graph's.
  explicit Graph(const OpRegistryInterface* registry);

  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in `flib_def`. Unlike the constructor taking
  // an OpRegistryInterface, this constructor copies the function definitions in
  // `flib_def` so its lifetime may be shorter than that of the graph's. The
  // OpRegistryInterface backing `flib_def` must still have the lifetime of the
  // graph though.
  explicit Graph(const FunctionLibraryDefinition& flib_def);

  ~Graph();

  static const int kControlSlot;

  // The GraphDef version range of this graph (see graph.proto).
  const VersionDef& versions() const;
  void set_versions(const VersionDef& versions);

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  Node* AddNode(const NodeDef& node_def, Status* status);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  Node* CopyNode(const Node* node);

  // Removes a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  void RemoveNode(Node* node);

  // Adds an edge that connects the xth output of `source` to the yth input of
  // `dest` and returns it. Does not update dest's NodeDef.
  const Edge* AddEdge(Node* source, int x, Node* dest, int y);

  // Adds a control edge (no data flows along this edge) that connects `source`
  // to `dest`. If `dest`s NodeDef is missing the corresponding control input,
  // adds the control input.
  //
  // If such a control edge already exists and `allow_duplicates` is false, no
  // edge is added and the function returns nullptr. Otherwise the edge is
  // unconditionally created and returned. The NodeDef is not updated if
  // `allow_duplicates` is true.
  // TODO(skyewm): // TODO(skyewm): allow_duplicates is needed only by
  // graph_partition.cc. Figure out if we can do away with it.
  const Edge* AddControlEdge(Node* source, Node* dest,
                             bool allow_duplicates = false);

  // Removes edge from the graph. Does not update the destination node's
  // NodeDef.
  // REQUIRES: The edge must exist.
  void RemoveEdge(const Edge* edge);

  // Removes control edge `edge` from the graph. Note that this also updates
  // the corresponding NodeDef to reflect the change.
  // REQUIRES: The control edge must exist.
  void RemoveControlEdge(const Edge* e);
  // Updates the input to a node.  The existing edge to `dst` is removed and an
  // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
  // is also updated.
  Status UpdateEdge(Node* new_src, int new_src_index, Node* dst, int dst_index);

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);

  // The number of live nodes in the graph.
  //
  // Because nodes can be removed from the graph, num_nodes() is often
  // smaller than num_node_ids(). If one needs to create an array of
  // nodes indexed by node ids, num_node_ids() should be used as the
  // array's size.
  int num_nodes() const { return num_nodes_; }

  // The number of live nodes in the graph, excluding the Source and Sink nodes.
  int num_op_nodes() const {
    DCHECK_GE(num_nodes_, 2);
    return num_nodes_ - 2;
  }

  // The number of live edges in the graph.
  //
  // Because edges can be removed from the graph, num_edges() is often
  // smaller than num_edge_ids(). If one needs to create an array of
  // edges indexed by edge ids, num_edge_ids() should be used as the
  // array's size.
  int num_edges() const { return num_edges_; }

  // Serialize the nodes starting at `from_node_id` to a GraphDef.
  void ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const;

  // Serialize to a GraphDef.
  void ToGraphDef(GraphDef* graph_def) const;

  // This version can be called from debugger to inspect the graph content.
  // Use the previous version outside debug context for efficiency reasons.
  //
  // Note: We do not expose a DebugString() API, since GraphDef.DebugString() is
  // not defined in some TensorFlow builds.
  GraphDef ToGraphDefDebug() const;

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  string NewName(StringPiece prefix);

  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  gtl::iterator_range<NodeIter> nodes() const;

  // Access to the list of all nodes, excluding the Source and Sink nodes.
  gtl::iterator_range<NodeIter> op_nodes() const;

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
  GraphEdgesIterable edges() const { return GraphEdgesIterable(edges_); }

  // The pre-defined nodes.
  enum { kSourceId = 0, kSinkId = 1 };
  Node* source_node() const { return FindNodeId(kSourceId); }
  Node* sink_node() const { return FindNodeId(kSinkId); }

  const OpRegistryInterface* op_registry() const { return &ops_; }
  const FunctionLibraryDefinition& flib_def() const { return ops_; }

  void CheckDeviceNameIndex(int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, static_cast<int>(device_names_.size()));
  }

  int InternDeviceName(const string& device_name);

  const string& get_assigned_device_name(const Node& node) const {
    return device_names_[node.assigned_device_name_index()];
  }

  void set_assigned_device_name_index(Node* node, int device_name_index) {
    CheckDeviceNameIndex(device_name_index);
    node->assigned_device_name_index_ = device_name_index;
  }

  void set_assigned_device_name(Node* node, const string& device_name) {
    node->assigned_device_name_index_ = InternDeviceName(device_name);
  }

  // Returns OK if `node` is non-null and belongs to this graph
  Status IsValidNode(const Node* node) const;

  // Returns OK if IsValidNode(`node`) and `idx` is less than
  // node->num_outputs()
  Status IsValidOutputTensor(const Node* node, int idx) const;

  // Returns OK if IsValidNode(`node`) and `idx` is less than
  // node->num_inputs()
  Status IsValidInputTensor(const Node* node, int idx) const;

  // Create and return a new WhileContext owned by this graph. This is called
  // when a new while loop is created. `frame_name` must be unique among
  // WhileContexts in this graph.
  Status AddWhileContext(StringPiece frame_name, std::vector<Node*> enter_nodes,
                         std::vector<Node*> exit_nodes,
                         OutputTensor cond_output,
                         std::vector<OutputTensor> body_inputs,
                         std::vector<OutputTensor> body_outputs,
                         WhileContext** result);

  // TODO(josh11b): uint64 hash() const;

 private:
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  //
  // Ownership of the returned Node is not transferred to caller.
  Node* AllocateNode(std::shared_ptr<NodeProperties> props,
                     const Node* cost_node);
  void ReleaseNode(Node* node);

  // Registry of all known ops, including functions.
  FunctionLibraryDefinition ops_;

  // GraphDef versions
  const std::unique_ptr<VersionDef> versions_;

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

  // The number of entries in edges_ that are not nullptr.
  int num_edges_ = 0;

  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  std::vector<Edge*> free_edges_;

  // For generating unique names.
  int name_counter_ = 0;

  // In most graphs, the number of unique values used for the
  // Node::assigned_device_name() property is quite small.  If the graph is
  // large, then this duplication of values can consume a significant amount of
  // memory.  Instead, we represent the same information using an interning
  // table, which consists of a vector of unique strings (device_names_), as
  // well a map (device_names_map_) from unique strings to indices within the
  // unique string table.
  //
  // The InternDeviceName() method handles adding a new entry into the table,
  // or locating the index of an existing entry.
  //
  // The fact that Node::assigned_device_name() is implemented using an
  // interning table is intentionally public.  This allows algorithms that
  // frequently access this field to do so efficiently, especially for the case
  // where the assigned_device_name of one Node is copied directly from that
  // of another Node.

  // A table of the unique assigned device names.  Indices do NOT correspond
  // to node IDs.  Index 0 is always the empty string.
  std::vector<string> device_names_;

  // Maps unique device names to indices within device_names_[i].
  std::unordered_map<string, int> device_names_map_;

  // All the while contexts owned by this graph, keyed by frame name,
  // corresponding to all the while loops contained in this graph (including
  // nested loops). The stored contexts are usually accessed via
  // AddWhileContext() or Node::while_ctx(), but this manages the lifetime.
  std::map<string, WhileContext> while_ctxs_;

  // Searches through edges_ for the Edge whose destination node and index
  // matches dst. An edge with destination `dst` must exist in the graph.
  const Edge* FindEdge(const Node* dst, int index);

  TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};

// TODO(josh11b): We may want to support keeping an index on various
// node/edge attributes in a graph, particularly node names.

// Helper routines

inline bool IsSource(const Node* node) { return node->IsSource(); }
inline bool IsSink(const Node* node) { return node->IsSink(); }
inline bool IsSwitch(const Node* node) { return node->IsSwitch(); }
inline bool IsMerge(const Node* node) { return node->IsMerge(); }
inline bool IsEnter(const Node* node) { return node->IsEnter(); }
inline bool IsExit(const Node* node) { return node->IsExit(); }
inline bool IsNextIteration(const Node* n) { return n->IsNextIteration(); }
inline bool IsLoopCond(const Node* node) { return node->IsLoopCond(); }
inline bool IsControlTrigger(const Node* n) { return n->IsControlTrigger(); }
inline bool IsSend(const Node* node) { return node->IsSend(); }
inline bool IsRecv(const Node* node) { return node->IsRecv(); }
inline bool IsHostSend(const Node* node) { return node->IsHostSend(); }
inline bool IsHostRecv(const Node* node) { return node->IsHostRecv(); }

// True for Nodes that mediate the transfer of values between processes.
inline bool IsTransferNode(const Node* n) { return IsSend(n) || IsRecv(n); }

inline bool IsConstant(const Node* node) { return node->IsConstant(); }
inline bool IsVariable(const Node* node) { return node->IsVariable(); }
inline bool IsIdentity(const Node* node) { return node->IsIdentity(); }

// Returns true iff 'n' is a control flow node.
inline bool IsControlFlow(const Node* n) { return n->IsControlFlow(); }

// Returns true if the node only depends on its input's metadata
// (shape).  Specifically, returns true for "Size", "Shape" and "Rank" ops.
inline bool IsMetadata(const Node* n) { return n->IsMetadata(); }

inline bool IsScopedAllocator(const Node* n) { return n->IsScopedAllocator(); }

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

inline gtl::iterator_range<NodeIter> Graph::nodes() const {
  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
}

inline gtl::iterator_range<NodeIter> Graph::op_nodes() const {
  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  //
  // The current implementation of Graph maintains the invariant that the
  // first two nodes are the source and sink nodes, and all other nodes are op
  // nodes. This method (op_nodes()) relies on this invariant.
  NodeIter begin(this, 0);
  NodeIter end(this, num_node_ids());
  if (begin != end) {
    ++begin;
  }
  if (begin != end) {
    ++begin;
  }
  return gtl::make_range(begin, end);
}

inline void Node::set_assigned_device_name_index(int index) {
  graph_->CheckDeviceNameIndex(index);
  assigned_device_name_index_ = index;
}

inline void Node::set_assigned_device_name(const string& device_name) {
  graph_->set_assigned_device_name(this, device_name);
}

inline const string& Node::assigned_device_name() const {
  return graph_->get_assigned_device_name(*this);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_H_
