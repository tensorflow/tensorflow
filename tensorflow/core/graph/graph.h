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
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Edge;
class EdgeSetTest;
class Graph;
class GraphDebugInfo;
class GraphDef;
class GraphTest;
class Node;
struct OutputTensor;
class VersionDef;
class WhileContext;

class NeighborIter;  // Declared below
class NodeIter;      // Declared below

// Indicates where the graph instance is originated from.
enum class ConstructionContext {
  kNotTracked,     // Not tracked.
  kDirectSession,  // From `tensorflow::DirectSession`, TF1 session API.
  kEagerRuntime,   // Registered from TF2 eager runtime.
};

class Node {
 public:
  std::string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const std::string& name() const;
  void set_name(std::string name);
  const std::string& type_string() const;

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

  NodeDef* mutable_def();

  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32_t i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32_t o) const;
  const DataTypeVector& output_types() const;

  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.
  const std::string& requested_device() const;

  // This changes the user requested device but not necessarily the device that
  // on which the operation will run.
  void set_requested_device(const std::string& device);

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move assigned_device_name outside of Node into a
  // NodeId->DeviceName map.
  const std::string& assigned_device_name() const;
  void set_assigned_device_name(const std::string& device_name);
  bool has_assigned_device_name() const {
    return assigned_device_name_index_ > 0;
  }
  int assigned_device_name_index() const { return assigned_device_name_index_; }
  void set_assigned_device_name_index(int index);

  // Sets 'original_node_names' field of this node's DebugInfo proto to
  // 'names'.
  void set_original_node_names(const std::vector<string>& names);
  void set_original_func_names(const std::vector<string>& names);

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
  bool IsFakeParam() const { return class_ == NC_FAKE_PARAM; }
  bool IsPartitionedCall() const { return class_ == NC_PARTITIONED_CALL; }

  // Returns true if this node is any kind of function call node.
  //
  // NOTE: "function call nodes" include partitioned call ops, symbolic gradient
  // ops, and ops whose type_string is the name of a function ("function ops").
  bool IsFunctionCall() const {
    return class_ == NC_PARTITIONED_CALL || class_ == NC_FUNCTION_OP ||
           class_ == NC_SYMBOLIC_GRADIENT;
  }

  bool IsIfNode() const { return class_ == NC_IF; }
  bool IsWhileNode() const { return class_ == NC_WHILE; }
  bool IsCaseNode() const { return class_ == NC_CASE; }
  // Is this node a function input
  bool IsArg() const { return class_ == NC_ARG; }
  // Is this node a function output
  bool IsRetval() const { return class_ == NC_RETVAL; }

  bool IsDistributedCommunication() const {
    return op_def().is_distributed_communication();
  }

  template <typename T>
  void AddAttr(const std::string& name, const T& val) {
    SetAttrValue(val, AddAttrHelper(name));
    UpdateProperties();
  }

  void AddAttr(const std::string& name, std::vector<string>&& val) {
    MoveAttrValue(std::move(val), AddAttrHelper(name));
    UpdateProperties();
  }

  void ClearAttr(const std::string& name);

  // Returns into '*e' the edge connecting to the 'idx' input of this Node.
  absl::Status input_edge(int idx, const Edge** e) const;

  // Returns into '*edges' the input data edges of this Node, indexed by input
  // number. Does not return control edges.
  absl::Status input_edges(std::vector<const Edge*>* edges) const;

  // Returns into '*n' the node that has an output connected to the
  // 'idx' input of this Node.
  absl::Status input_node(int idx, const Node** n) const;
  absl::Status input_node(int idx, Node** n) const;

  // Returns into '*t' the idx-th input tensor of this node, represented as the
  // output tensor of input_node(idx).
  absl::Status input_tensor(int idx, OutputTensor* t) const;

  WhileContext* while_ctx() const { return while_ctx_; }
  void set_while_ctx(WhileContext* while_ctx) {
    DCHECK(IsExit());
    DCHECK(while_ctx_ == nullptr);
    while_ctx_ = while_ctx;
  }

  std::shared_ptr<NodeProperties> properties() const { return props_; }

  // Sets the stack trace for the node. Assumes that getting and setting the
  // stack trace for a given node will not race.
  void SetStackTrace(const std::shared_ptr<AbstractStackTrace>& stack_trace) {
    stack_trace_ = stack_trace;
  }

  // Get the stack trace for when the node was instantiated.
  const std::shared_ptr<AbstractStackTrace>& GetStackTrace() const {
    return stack_trace_;
  }

  // Called after an attr has changed. Decides whether we need to update some
  // property of the node (stored in props_).
  void UpdateProperties();

  // Erases type information from the node.
  void ClearTypeInfo();

  // Update type information for a node with a list of inputs and/or outputs
  // described by its TYPE_ATTR_NAME attr when removing some of these. The keys
  // of INDEX_MAPPING are the indexes of the inputs/outputs that are not
  // removed. dtype information in the TYPE_ATTR_NAME attr is always updated.
  // Use UPDATE_FULL_TYPE=true when this changes the node's outputs to also
  // update the node's full type information (if present).
  absl::Status ShrinkTypeInfo(
      const absl::flat_hash_map<int, int>& index_mapping,
      const string& type_attr_name, bool update_full_type);

  // Called after an incident non-control edge has changed. Does nothing if not
  // all input edges are defined.
  void RunForwardTypeInference();

 private:
  // TODO(mdan): Drop this.
  friend class Graph;
  Node();

  // Stack trace for the user code for node instantiation. Can be shared across
  // multiple nodes (e.g. when inlining).
  std::shared_ptr<AbstractStackTrace> stack_trace_;

  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();

  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  void MaybeCopyOnWrite();

  AttrValue* AddAttrHelper(const std::string& name);

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
    NC_FAKE_PARAM,
    NC_PARTITIONED_CALL,
    NC_FUNCTION_OP,
    NC_SYMBOLIC_GRADIENT,
    NC_IF,
    NC_WHILE,
    NC_CASE,
    NC_ARG,
    NC_RETVAL,
    NC_OTHER  // Not a special kind of node
  };

  void Initialize(int id, int cost_id, std::shared_ptr<NodeProperties> props,
                  NodeClass node_class);

  static NodeClass GetNodeClassForOp(const std::string& ts);

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

  Node(const Node&) = delete;
  void operator=(const Node&) = delete;
};

// Stores debug information associated with the Node.
struct NodeDebugInfo {
  const std::string name;
  std::vector<string> original_node_names;
  std::vector<string> original_func_names;

  NodeDebugInfo(const Node& n);
  NodeDebugInfo(const NodeDef& ndef);
  NodeDebugInfo(absl::string_view node_name, bool has_experimental_debug_info,
                const NodeDef_ExperimentalDebugInfo& experimental_debug_info);
};

// Represents an input of a node, i.e., the `index`-th input to `node`.
struct InputTensor {
  Node* node;
  int index;

  InputTensor(Node* n, int i) : node(n), index(i) {}
  InputTensor() : node(nullptr), index(0) {}

  // Returns true if this InputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const InputTensor& other) const;

  // A hash function for InputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(InputTensor const& s) const;
  };
};

// Represents an output of a node, i.e., the `index`-th output of `node`. Note
// that a single `OutputTensor` can correspond to multiple `Edge`s if the output
// is consumed by multiple destination nodes.
struct OutputTensor {
  Node* node;
  int index;

  OutputTensor(Node* n, int i) : node(n), index(i) {}
  OutputTensor() : node(nullptr), index(0) {}

  // Returns true if this OutputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const OutputTensor& other) const;

  // A hash function for OutputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(OutputTensor const& s) const;
  };
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

  std::string DebugString() const;

 private:
  Edge() {}

  friend class EdgeSetTest;
  friend class GraphTest;
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
  // The graph can hold ops found in the registry. `ops`s lifetime must be at
  // least that of the constructed graph's.
  explicit Graph(const OpRegistryInterface* ops);

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

  // Clone the current graph into a new one.
  std::unique_ptr<Graph> Clone();

  static constexpr int kControlSlot = -1;

  // The GraphDef version range of this graph (see graph.proto).
  const VersionDef& versions() const;
  void set_versions(const VersionDef& versions);

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  Node* AddNode(NodeDef node_def, absl::Status* status);

  // Same as above, but using StatusOr. This method is always preferred.
  absl::StatusOr<Node*> AddNode(NodeDef node_def);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  Node* CopyNode(const Node* node);

  // Removes a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  void RemoveNode(Node* node);

  void Copy(const Graph& src);

  // Removes all nodes from this graph, including all edges from or to them.
  // No Node* references to the Graph are valid post.
  void Clear();

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
  // NodeDef. Does not update the full type information of the source node's
  // NodeDef. (See ShrinkTypeInfo for an example of updating full type
  // information when removing some outputs from a node.)
  // REQUIRES: The edge must exist.
  void RemoveEdge(const Edge* edge);

  // Removes control edge `edge` from the graph. Note that this also updates
  // the corresponding NodeDef to reflect the change.
  // REQUIRES: The control edge must exist.
  void RemoveControlEdge(const Edge* e);

  // Updates the input to a node.  The existing edge to `dst` is removed and an
  // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
  // is also updated.
  absl::Status UpdateEdge(Node* new_src, int new_src_index, Node* dst,
                          int dst_index);

  // Add an input to dst that comes from the "src_slot" output of the
  // node named by "src_name".
  static void AddInput(NodeDef* dst, absl::string_view src_name, int src_slot);

  // Like AddEdge but updates dst's NodeDef. Used to add an input edge to a
  // "While" op during gradient construction, see AddInputWhileHack in
  // python_api.h for more details.
  absl::Status AddWhileInputHack(Node* new_src, int new_src_index, Node* dst);

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name. This overload adds the function definitions with no stack traces.
  absl::Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);
  absl::Status AddFunctionLibrary(FunctionDefLibrary&& fdef_lib);

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  absl::Status AddFunctionLibrary(
      const FunctionDefLibrary& fdef_lib,
      const FunctionDefLibraryStackTraces& stack_traces);
  absl::Status AddFunctionLibrary(
      FunctionDefLibrary&& fdef_lib,
      const FunctionDefLibraryStackTraces& stack_traces);

  // Adds the function definition and its stacktraces to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  absl::Status AddFunctionDef(const FunctionDef& fdef,
                              const StackTracesMap& stack_traces);

  // Adds the gradient definition to this graph's op registry. Ignores duplicate
  // gradients of the same function, and returns a bad status if an imported
  // gradient differs from an existing gradient of the same function name.
  absl::Status AddGradientDef(const GradientDef& gdef);

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
  // `include_flib_def` indicates whether the function library will be populated
  // in the `graph_def`. `include_flib_def` should be usually set to true so
  // that the populated `graph_def` will be complete. Setting `include_flib_def`
  // to false would mean that the returned `graph_def` is incomplete and may
  // contain references to functions whose definition is not included. It can
  // make sense to do this in cases where the caller already has a copy of the
  // function library.
  // If `include_debug_info` is true, the `debug_info` field of the GraphDef
  // will be populated with stack traces from the nodes and the function
  // library. Note that if `include_debug_info` is true and `include_flib_def`
  // is false, then `debug_info` will contain stack traces for nodes in the
  // function library, which will not itself be included in the GraphDef.
  void ToGraphDefSubRange(GraphDef* graph_def, int from_node_id,
                          bool include_flib_def = true,
                          bool include_debug_info = false) const;

  // Serialize to a GraphDef. `include_flib_def` indicates whether the function
  // library will be populated in the `graph_def`. `include_flib_def` should be
  // usually set to true so that the populated `graph_def` will be complete.
  // Setting `include_flib_def` to false would mean that the returned
  // `graph_def` is incomplete and may contain references to functions whose
  // definition is not included. It can make sense to do this in cases where the
  // caller already has a copy of the function library.
  // If `include_debug_info` is true, the `debug_info` field of the GraphDef
  // will be populated with stack traces from the nodes and the function
  // library. Note that if `include_debug_info` is true and `include_flib_def`
  // is false, then `debug_info` will contain stack traces for nodes in the
  // function library, which will not itself be included in the GraphDef.
  void ToGraphDef(GraphDef* graph_def, bool include_flib_def = true,
                  bool include_debug_info = false) const;

  // This version can be called from debugger to inspect the graph content.
  // Use the previous version outside debug context for efficiency reasons.
  //
  // Note: We do not expose a DebugString() API, since GraphDef.DebugString() is
  // not defined in some TensorFlow builds.
  GraphDef ToGraphDefDebug() const;

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  std::string NewName(absl::string_view prefix);

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
  // with that id (the edge with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_edge_ids().
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

  FunctionLibraryDefinition* mutable_flib_def() { return &ops_; }

  void CheckDeviceNameIndex(int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, static_cast<int>(device_names_.size()));
  }

  int InternDeviceName(const std::string& device_name);

  const std::string& get_assigned_device_name(const Node& node) const {
    return device_names_[node.assigned_device_name_index()];
  }

  void set_assigned_device_name_index(Node* node, int device_name_index) {
    CheckDeviceNameIndex(device_name_index);
    node->assigned_device_name_index_ = device_name_index;
  }

  void set_assigned_device_name(Node* node, const std::string& device_name) {
    node->assigned_device_name_index_ = InternDeviceName(device_name);
  }

  // Returns OK if `node` is non-null and belongs to this graph
  absl::Status IsValidNode(const Node* node) const;

  // Returns OK if IsValidNode(`node`) and `idx` is a valid output.  Does not
  // accept control outputs.
  absl::Status IsValidOutputTensor(const Node* node, int idx) const;

  // Returns OK if IsValidNode(`node`) and `idx` a valid input.  Does not accept
  // control inputs.
  absl::Status IsValidInputTensor(const Node* node, int idx) const;

  // Create and return a new WhileContext owned by this graph. This is called
  // when a new while loop is created. `frame_name` must be unique among
  // WhileContexts in this graph.
  absl::Status AddWhileContext(absl::string_view frame_name,
                               std::vector<Node*> enter_nodes,
                               std::vector<Node*> exit_nodes,
                               OutputTensor cond_output,
                               std::vector<OutputTensor> body_inputs,
                               std::vector<OutputTensor> body_outputs,
                               WhileContext** result);

  // Builds a node name to node pointer index for all nodes in the graph.
  std::unordered_map<string, Node*> BuildNodeNameIndex() const;

  absl::optional<std::vector<bool>>& GetConstArgIndicesCache() const {
    return const_arg_indices_cache_;
  }

  // TODO(kkb): Add to the constructor when it becomes managable.
  // Sets the graph construction context.
  void SetConstructionContext(ConstructionContext construction_context) {
    construction_context_ = construction_context;
  }

  // TODO(kkb): Rename to `GetConstructionContext` once we're comfortable
  // making this stable and make it available widely.
  // Returns the graph construction context. It's `kUnknown` if not set.
  ConstructionContext GetConstructionContextInternal() const {
    return construction_context_;
  }

  // Set full type information for a node given its name.
  // Note that if this is called in a loop iterating over all the nodes
  // elsewhere it would be O(n^2) complexity. If this case was important in the
  // future, an alternative method could be added that takes in a flat_hash_map
  // of name: type and simply iterates through the graph once and annotates all
  // nodes.
  void SetNodeType(absl::string_view name, const FullTypeDef& type);

  // Get full type information for a node given its name.
  // Note that if this is called in a loop iterating over all the nodes
  // elsewhere it would be O(n^2) complexity. If this case was important in the
  // future, an alternative method could be added that takes in flat_hash_map of
  // name: type and simply iterates through the graph once and stores all the
  // information in the map.
  void NodeType(absl::string_view name, const FullTypeDef** result);

  // Builds a GraphDebugInfo from the functions and nodes in this graph. Stack
  // traces associated with function definitions will have a key of the form
  // <node_name> '@' <function_name>. Stack traces associated with other Nodes
  // will use the node name as the key.
  GraphDebugInfo BuildDebugInfo() const;

  // TODO(josh11b): uint64 hash() const;

 private:
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  //
  // Ownership of the returned Node is not transferred to caller.
  Node* AllocateNode(std::shared_ptr<NodeProperties> props,
                     const Node* cost_node, Node::NodeClass node_class);
  void ReleaseNode(Node* node);
  // Insert edge in free_edges_ for possible reuse.
  void RecycleEdge(const Edge* edge);
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
  int64_t num_nodes_ = 0;

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

  // Cache of the indices of the arguments which need to be constant for the XLA
  // compilation.
  mutable absl::optional<std::vector<bool>> const_arg_indices_cache_;

  // Indicates the context that this Graph instance is constructed.
  ConstructionContext construction_context_ = ConstructionContext::kNotTracked;

  Graph(const Graph&) = delete;
  void operator=(const Graph&) = delete;
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

inline bool IsDistributedCommunication(const Node* n) {
  return n->IsDistributedCommunication();
}

// NOTE: We declare Reference type of NodeIter and NeighborIter as Node* (see
// https://en.cppreference.com/w/cpp/iterator/iterator).

// Iterator for stepping through the nodes of a graph.
class NodeIter {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Node;
  using difference_type = std::ptrdiff_t;
  using pointer = Node*;
  using reference = Node*;

  NodeIter(const Graph* graph, int id);
  bool operator==(const NodeIter& rhs) const;
  bool operator!=(const NodeIter& rhs) const;
  void operator++();
  reference operator*() const;
  pointer operator->() const;

 private:
  // Invariant: id_ == graph_->num_node_ids() || graph_->FindId(id_) != nullptr
  const Graph* graph_;
  int id_;
};

// Iterator for stepping through the neighbors of a node.
class NeighborIter {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Node;
  using difference_type = std::ptrdiff_t;
  using pointer = Node*;
  using reference = Node*;

  NeighborIter(EdgeSet::const_iterator iter, bool incoming);
  bool operator==(const NeighborIter& rhs) const;
  bool operator!=(const NeighborIter& rhs) const;
  void operator++();
  reference operator*() const;
  pointer operator->() const;

 private:
  EdgeSet::const_iterator iter_;
  bool incoming_;
};

// IMPLEMENTATION DETAILS, PLEASE IGNORE

inline NodeIter::NodeIter(const Graph* graph, int id)
    : graph_(graph), id_(id) {}

inline bool NodeIter::operator==(const NodeIter& rhs) const {
  DCHECK(graph_ == rhs.graph_);
  return id_ == rhs.id_;
}

inline bool NodeIter::operator!=(const NodeIter& rhs) const {
  return !(*this == rhs);
}

inline void NodeIter::operator++() {
  while (true) {
    DCHECK_LE(id_, graph_->num_node_ids());
    ++id_;
    if (id_ >= graph_->num_node_ids() || graph_->FindNodeId(id_) != nullptr) {
      return;
    }
  }
}

inline Node* NodeIter::operator*() const { return graph_->FindNodeId(id_); }

inline Node* NodeIter::operator->() const { return graph_->FindNodeId(id_); }

inline NeighborIter::NeighborIter(EdgeSet::const_iterator iter, bool incoming)
    : iter_(iter), incoming_(incoming) {}

inline bool NeighborIter::operator==(const NeighborIter& rhs) const {
  return iter_ == rhs.iter_ && incoming_ == rhs.incoming_;
}

inline bool NeighborIter::operator!=(const NeighborIter& rhs) const {
  return !(*this == rhs);
}

inline void NeighborIter::operator++() { ++iter_; }

inline Node* NeighborIter::operator*() const {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline Node* NeighborIter::operator->() const {
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

inline void Node::set_assigned_device_name(const std::string& device_name) {
  graph_->set_assigned_device_name(this, device_name);
}

inline const std::string& Node::assigned_device_name() const {
  return graph_->get_assigned_device_name(*this);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_H_
