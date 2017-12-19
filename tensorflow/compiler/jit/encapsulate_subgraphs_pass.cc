/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/jit/graph_to_functiondef.h"
#include "tensorflow/compiler/jit/legacy_flags/encapsulate_subgraphs_pass_flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char* const kXlaCompiledKernelAttr = "_XlaCompiledKernel";
const char* const kXlaNumConstantArgsAttr = "_XlaNumConstantArgs";
const char* const kXlaNumResourceArgsAttr = "_XlaNumResourceArgs";

namespace {

bool AreAllParentsConst(const Node& n,
                        const gtl::FlatSet<const Node*>& runtime_const_nodes) {
  if (n.type_string() == "GuaranteeConst" || n.type_string() == "Const") {
    // If the current node is itself a cast-to-const, no need
    // to look at the incoming edges.
    return true;
  }

  bool all_parents_const = true;
  bool atleast_one_non_control_edge = false;
  for (const Edge* in : n.in_edges()) {
    atleast_one_non_control_edge =
        atleast_one_non_control_edge || !in->IsControlEdge();
    if (!in->IsControlEdge() && runtime_const_nodes.count(in->src()) == 0) {
      all_parents_const = false;
      break;
    }
  }
  return all_parents_const && atleast_one_non_control_edge;
}

void MarkGuaranteedConstants(
    const Graph& graph,
    const std::vector<std::pair<const Node*, Node*>>& src_arg_pairs) {
  gtl::FlatSet<const Node*> guaranteed_const_nodes;
  std::vector<const Node*> srcs;
  srcs.reserve(src_arg_pairs.size());
  for (const auto& src_arg : src_arg_pairs) {
    srcs.push_back(src_arg.first);
  }
  ReverseDFSFrom(graph, srcs, /*enter=*/nullptr,
                 /*leave=*/[&guaranteed_const_nodes](const Node* n) {
                   // TODO(vinuraja): Doesn't work in the presence of loops.
                   if (AreAllParentsConst(*n, guaranteed_const_nodes)) {
                     guaranteed_const_nodes.insert(n);
                   }
                 });

  for (auto& src_arg : src_arg_pairs) {
    if (guaranteed_const_nodes.count(src_arg.first) != 0) {
      VLOG(1) << "Guaranteed const found: " << src_arg.first->DebugString();
      src_arg.second->AddAttr("_is_guaranteed_constant", true);
    }
  }
}

// A node/slot pair.
// TODO(phawkins): is there a common definition of this?
struct NodeSlot {
  NodeSlot() : node(nullptr), slot(-1), dtype(DT_INVALID) {}
  NodeSlot(const Node* node, int slot)
      : node(node), slot(slot), dtype(DT_INVALID) {}
  NodeSlot(const Node* node, int slot, DataType dtype)
      : node(node), slot(slot), dtype(dtype) {}

  const Node* node;
  int slot;

  // Optional: used to record the destination type of a source NodeSlot in case
  // the source output is a Ref type that is cast to a Tensor at the
  // destination.
  DataType dtype;

  bool operator==(const NodeSlot& other) const {
    return node == other.node && slot == other.slot && dtype == other.dtype;
  }

  // Leave dtype out of the hash since there are never two NodeSlots with the
  // same node and slot and different dtypes.
  struct Hasher {
    uint64 operator()(NodeSlot const& s) const {
      return Hash64Combine(std::hash<const Node*>()(s.node),
                           std::hash<int>()(s.slot));
    }
  };

  struct PairHasher {
    uint64 operator()(std::pair<NodeSlot, NodeSlot> const& s) const {
      return Hash64Combine(Hasher()(s.first), Hasher()(s.second));
    }
  };
};

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";
static const char* const kSendToHostOp = "_XlaSendToHost";
static const char* const kRecvFromHostOp = "_XlaRecvFromHost";
static const char* const kSendFromHostOp = "_XlaSendFromHost";
static const char* const kRecvAtHostOp = "_XlaRecvAtHost";

class Encapsulator {
 public:
  Encapsulator(string group_attribute, string outside_compilation_attribute,
               Graph const* graph_in)
      : group_attribute_(std::move(group_attribute)),
        outside_compilation_attribute_(
            std::move(outside_compilation_attribute)),
        graph_in_(graph_in) {}

  // Find subgraphs marked with 'group_attribute', and build a new
  // subgraph, one for each value of 'group_attribute'.
  Status SplitIntoSubgraphs();

  // Build a FunctionDef for each subgraph, and add it 'library'. The values of
  // the 'group_attribute' annotations become the function names.
  // If 'reuse_existing_functions' is set, use an existing function with the
  // same name, if any.
  // If 'rewrite_subgraph_fn' is set, it is applied to each subgraph before
  // function conversion.
  Status BuildFunctionDefs(const RewriteSubgraphFn& rewrite_subgraph_fn,
                           bool reuse_existing_functions,
                           FunctionLibraryDefinition* library);

  // Write a copy of the input graph to 'graph_out', where the subgraphs are
  // replaced with calls to the new functions.
  Status BuildOutputGraph(bool parallel_checking, Graph* graph_out);

 private:
  // A subgraph of the input, all marked with a common 'group_attribute'
  // value. A subgraph may contain multiple `outside_compilation' clusters.
  //
  // In the following simple example, A, B, ..., E are nodes in the original
  // graph. The group attributes and outside_compilation attributes g and oc are
  // each shown as either 0 or empty.
  //
  //  A  -->  B  -->  C  -->  D  -->  E
  //  g:      g:0     g:0     g:0     g:
  //  oc:     oc:     oc:0    oc:     oc:
  //
  // The example is rewritten to two graphs; one on the host and one to be
  // compiled. The host graph is as follows. RAH is a RecvAtHost node receiving
  // input from the compiled cluster, and SFH is a SendFromHost node sending
  // input back to the compiled cluster. Dotted edges are control edges. A
  // 'sequencing' node S is inserted, and both RAH and SFH are connected via S
  // to E (and in general all nodes that depend on nodes in the compiled
  // cluster) to ensure that they are not pruned.
  //
  //  A  -->  Call  -->  E
  //                     ^
  //                     .
  //           ........> S
  //       ....          ^
  //     ..             .
  //  RAH -->  C  --> SFH
  //
  // The compiled cluster is as follows. STH is a SendToHost node which is the
  // source of a channel to the RAH node above. RFH is a RecvFromHost node which
  // is the destination of a channel from the SFH node above. There is a control
  // edge that ensures RFH follows STH, which is used in shape inference to
  // ensure that the shapes on the STH host channel are known before the RFH
  // channel is compiled.
  //
  //  Arg  --> B  --> STH  ..>  RFH  --> D --> Retval
  //
  // The channels STH/RAH and SFH/RFH each transmit a tuple, so there is at most
  // one RAH and SFH in each compiled cluster. This design is preferred over
  // adding separate Arg/Retval nodes for each transmitted value because it
  // simplifies the host code that would like to limit communication between
  // host and device and, e.g., raise only one interrupt per channel rather than
  // one per transmitted value.
  class Subgraph {
   public:
    // Creates a graph to build the subgraph in, if it doesn't already exist,
    // using the same op registry and versions as graph_in.
    Node* MakeNodeImage(const Graph* graph_in, Node* node);

    // Returns the graph the subgraph is being built in.
    Graph* GetGraph() const;

    // Builds a FunctionDef, and adds it to 'library'. The value of the
    // 'group_attribute' annotations becomes the function name.  If
    // 'reuse_existing_functions' is set, use an existing function with the same
    // name, if any.  If 'rewrite_subgraph_fn' is set, it is applied to the
    // subgraph before function conversion.
    Status BuildFunctionDef(const string& name_in,
                            const RewriteSubgraphFn& rewrite_subgraph_fn,
                            bool reuse_existing_functions,
                            FunctionLibraryDefinition* library);

    // Adds the function call node to graph_out.
    Status AddFunctionCallNode(
        const std::unordered_map<const Node*, Node*>& node_images,
        bool parallel_checking, Graph* graph_out);

    // Adds _RecvAtHost and _SendFromHost nodes, where needed, to graph_out.
    Status AddOutsideCompilationHostIONodes(
        const string& subgraph_name,
        const std::unordered_map<const Node*, Node*>& node_images,
        Graph* graph_out);

    // Returns the Node that inputs to the function should be wired up to.
    Node* GetCallNodeForInputs() const;

    // Returns the Node that outputs to the function should be wired up to.
    Node* GetCallNodeForOutputs() const;

    // Returns the index of the arg that the dst of edge should connect to.
    int GetArgIndexForEdge(const Edge* edge) const;

    // Returns the index of the result that the src of edge should connect to.
    int GetResultIndexForEdge(const Edge* edge) const;

    // Returns the RecvAtHost node for an outside_compilation subgraph.
    Node* GetRecvAtHostNode(
        const string& outside_compilation_subgraph_name) const;

    // Returns the output slot for the RecvAtHost node that corresponds to the
    // source of edge in an outside_compilation subgraph.
    int GetRecvAtHostSlot(const string& outside_compilation_subgraph_name,
                          const Edge* edge) const;

    // Returns the SendFromHost node for an outside_compilation subgraph.
    Node* GetSendFromHostNode(
        const string& outside_compilation_subgraph_name) const;

    // Returns the input slot for the SendFromHost node that corresponds to the
    // destination of edge in an outside_compilation subgraph.
    int GetSendFromHostSlot(const string& outside_compilation_subgraph_name,
                            const Edge* edge) const;

    // Creates an _Arg node for the src node of edge, and add its index to
    // args_by_src_, if none exists yet. Also adds its index to args_by_dst_,
    // and adds the edge within the subgraph from the _Arg node to the image of
    // the dst node.
    Status RecordArg(const Edge* edge,
                     const std::unordered_map<const Node*, Node*>& node_images,
                     std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

    // Creates a _Retval node for the src node of edge, and add it to results_,
    // if none exists yet. If a new _Retval node is created, also adds the edge
    // within the subgraph from the src to the _Retval node.
    Status RecordResult(
        const Edge* edge,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates an outside_compilation subgraph for outside_compilation_id if
    // none exists yet. Creates an entry for the src node of edge in the list of
    // inputs for the outside_compilation subgraph, if none exists yet.
    void RecordOutsideCompilationInputOrControl(
        const string& outside_compilation_id, const Edge* edge);

    // Creates an outside_compilation subgraph for outside_compilation_id if
    // none exists yet. Creates an entry for the src node of edge in the list of
    // outputs by src for the outside_compilation subgraph, if none exists
    // yet. Creates an entry for the dst node of edge in the list of outputs by
    // dst for the outside_compilation subgraph.
    void RecordOutsideCompilationOutputOrControl(
        const string& outside_compilation_id, const Edge* edge);

    // Adds the SendToHost nodes for each outside_compilation subgraph once the
    // edges have all been recorded via RecordOutsideCompilationInputOrControl.
    Status AddSendsToOutsideCompilation(
        const std::unordered_map<const Node*, Node*>& node_images);

    // Adds the RecvFromHost nodes for each outside_compilation subgraph once
    // the edges have all been recorded via
    // RecordOutsideCompilationOutputOrControl.
    Status AddRecvsFromOutsideCompilation(
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates the sequencer node if it doesn't exist, adding it to graph_out.
    Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);

    // If there is a sequencer node, adds a control edge from the sequencer to
    // all the downstream nodes of call_node_outputs.
    void ConnectSequencerToOutputs(Graph* graph_out);

   private:
    struct OutsideCompilationSubgraph {
      // Map from source (producer node/slot) tensors in the original graph to
      // input index (slot number in the SendToHost/RecvAtHost nodes that will
      // be created) for the outside_compilation subgraph.
      std::unordered_map<NodeSlot, int, NodeSlot::Hasher> inputs;

      // Set of nodes in the original graph that are the source of control edges
      // that cross from the containing compiled subgraph into the
      // outside_compilation subgraph. These are recorded by
      // RecordOutsideCompilationInputOrControl while walking all the subgraph
      // edges, and lifted control edges within the subgraph are added by
      // AddSendsToOutsideCompilation once the _SendToHost node has been
      // created. The matching control edge from _RecvAtHost to the
      // destination is added by CopyEdgeToOutputGraph.
      std::unordered_set<const Node*> control_inputs;

      // Maps from source (producer node/slot) and destination (consumer
      // node/slot) tensors in the original graph to output index (slot number
      // in the SendFromHost/RecvFromHost nodes that will be created) for the
      // outside_compilation subgraph.
      std::unordered_map<NodeSlot, int, NodeSlot::Hasher> outputs_by_src;
      std::unordered_map<NodeSlot, int, NodeSlot::Hasher> outputs_by_dst;

      // Set of nodes in the original graph that are the destination of control
      // edges that cross from the outside_compilation subgraph into the
      // containing compiled subgraph. These are recorded by
      // RecordOutsideCompilationOutputOrControl while walking all the subgraph
      // edges, and lifted control edges within the subgraph are added by
      // AddRecvsFromToOutsideCompilation once the _RecvFromHost node has been
      // created. The matching control edge from the source to _SendFromHost to
      // the destination is added by CopyEdgeToOutputGraph.
      std::unordered_set<const Node*> control_outputs;

      // _SendToHost node in the subgraph. Not owned.
      Node* send_to_host = nullptr;

      // _RecvAtHost node in the output graph. Not owned.
      Node* recv_at_host = nullptr;

      // _SendFromHost node in the output graph. Not owned.
      Node* send_from_host = nullptr;
    };

    // Builds a ParallelCheck op that compares the output of the original
    // subgraph with the encapsulated subgraph.
    Status BuildParallelCheckOp(
        const std::unordered_map<const Node*, Node*>& node_images,
        Graph* graph_out);

    // Builds a _RecvAtHost node producing all the inputs of an
    // outside_compilation subgraph and stores it in oc_subgraph.recv_at_host.
    Status AddRecvAtHostNode(const string& subgraph_name,
                             const string& oc_subgraph_name,
                             OutsideCompilationSubgraph* oc_subgraph,
                             Graph* graph_out);

    // Builds a _SendFromHost node consuming all the outputs of an
    // outside_compilation subgraph and stores it in oc_subgraph.send_from_host.
    Status AddSendFromHostNode(
        const std::unordered_map<const Node*, Node*>& node_images,
        const string& subgraph_name, const string& oc_subgraph_name,
        OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out);

    // The subgraph extracted from the input graph, suitable for being turned
    // into a FunctionDef. Inputs are fed by _Arg nodes, and outputs are
    // returned by _Retval nodes.
    std::unique_ptr<Graph> graph_;

    // Which device are these nodes on? Used to assign a device to the call
    // node.
    string device_;

    // NodeDef for the function call node.
    NodeDef call_node_def_;

    // Function call node(s) in the output graph. Not owned.
    // If parallel_checking is enabled, 'call_node_inputs' is the function call
    // node to which inputs should be fed, and 'call_node_outputs' is the
    // parallel check op from which outputs should be read. If parallel checking
    // is disabled, both point to the function call node.
    Node* call_node_inputs_;
    Node* call_node_outputs_;

    // Maps from source (producer node/slot) and destination
    // (consumer node/slot) tensors in the input graph to _Arg numbers in
    // the subgraph. The source map is one-to-one, whereas the dest map may be
    // many-to-one.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> args_by_src_;
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> args_by_dst_;

    // The _Arg nodes in the subgraph, in order by argument number.
    std::vector<Node*> args_;

    // Map from source tensor in the input graph to result #.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> results_;

    // The outside_compilation clusters in this subgraph.
    std::unordered_map<string, OutsideCompilationSubgraph>
        outside_compilation_subgraphs_;

    // NoOp node in the output graph that is sequenced after the call node and
    // used to prevent host-side outside_compilation sends and recvs from being
    // pruned.
    Node* sequencer_ = nullptr;
  };

  // Returns the key attribute and outside_compilation attribute associated
  // with a node in attr, and outside_compilation_attr, respectively. Sets
  // either result to the empty string if the respective attribute is not
  // found. Returns error status if there is an outside_compilation attribute
  // and no key attribute,
  Status GetFunctionNameAttr(Node const* node, string* attr,
                             string* outside_compilation_attr) const;

  // Copies edges local to a subgraph. Adds _Arg and _Retval nodes to
  // subgraphs for data edges that cross subgraph boundaries.
  Status CopySubgraphEdges(
      const std::unordered_map<const Node*, Node*>& node_images,
      std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

  // Copies all marked nodes to a subgraph. Does nothing for unmarked nodes,
  // or nodes marked outside_compilation.
  Status CopySubgraphNodes(std::unordered_map<const Node*, Node*>* node_images);

  // Copies all nodes that aren't in a compiled subgraph to the output graph.
  Status CopyNodesToOutputGraph(
      bool parallel_checking, Graph* graph_out,
      std::unordered_map<const Node*, Node*>* node_images);

  // Adds function call nodes for each compiled subgraph.
  Status AddFunctionCallNodes(
      const std::unordered_map<const Node*, Node*>& node_images,
      bool parallel_checking, Graph* graph_out);

  // Adds _RecvAtHost and _SendFromHost nodes, where needed, for all
  // outside_compilation subgraphs.
  Status AddOutsideCompilationHostIONodes(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Finds the image of an edge source in the output graph. If the edge crosses
  // a subgraph boundary it is the output of a call node, otherwise it is a node
  // in the output graph.
  Status FindOutputImageOfEdgeSrc(
      const string& src_func_id, const string& src_outside_compilation_id,
      const string& dst_func_id, const string& dst_outside_compilation_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_src_node, Node** src_image);

  // Finds an edge source slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the output of a call node or a
  // _RecvAtHost node, otherwise it is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeSrc(const string& src_func_id,
                              const string& src_outside_compilation_id,
                              const string& dst_func_id,
                              const string& dst_outside_compilation_id,
                              const Edge* edge);

  // Finds the image of an edge destination in the output graph. If the edge
  // crosses a subgraph boundary it is the input of a call node or a
  // _SendFromHost node, otherwise it is a node in the output graph.
  Status FindOutputImageOfEdgeDst(
      const string& src_func_id, const string& src_outside_compilation_id,
      const string& dst_func_id, const string& dst_outside_compilation_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_dst_node, Node** dst_image);

  // Finds an edge destination slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the input of a call node or a
  // _SendFromHost node, otherwise it is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeDst(const string& src_func_id,
                              const string& src_outside_compilation_id,
                              const string& dst_func_id,
                              const string& dst_outside_compilation_id,
                              const Edge* edge);

  // Copies a single edge to the output graph. The edge is either entirely
  // within the output graph, or crosses into or out of a compiled subgraph.
  Status CopyEdgeToOutputGraph(
      const Edge* edge, const string& src_func_id,
      const string& src_outside_compilation_id, const string& dst_func_id,
      const string& dst_outside_compilation_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      bool parallel_checking, Graph* graph_out,
      std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher>*
          edges_added);

  // Adds all edges to the output graph.
  Status AddEdgesToOutputGraph(
      const std::unordered_map<const Node*, Node*>& node_images,
      bool parallel_checking, Graph* graph_out);

  const string group_attribute_;
  const string outside_compilation_attribute_;
  const Graph* graph_in_;

  std::unordered_map<string, Subgraph> subgraphs_;

  TF_DISALLOW_COPY_AND_ASSIGN(Encapsulator);
};

Node* Encapsulator::Subgraph::GetCallNodeForInputs() const {
  return call_node_inputs_;
}

Node* Encapsulator::Subgraph::GetCallNodeForOutputs() const {
  return call_node_outputs_;
}

int Encapsulator::Subgraph::GetArgIndexForEdge(const Edge* edge) const {
  return args_by_dst_.at(NodeSlot(edge->dst(), edge->dst_input()));
}

int Encapsulator::Subgraph::GetResultIndexForEdge(const Edge* edge) const {
  return results_.at(NodeSlot(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetRecvAtHostNode(
    const string& outside_compilation_subgraph_name) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .recv_at_host;
}

int Encapsulator::Subgraph::GetRecvAtHostSlot(
    const string& outside_compilation_subgraph_name, const Edge* edge) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .inputs.at(NodeSlot(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetSendFromHostNode(
    const string& outside_compilation_subgraph_name) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .send_from_host;
}

int Encapsulator::Subgraph::GetSendFromHostSlot(
    const string& outside_compilation_subgraph_name, const Edge* edge) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .outputs_by_dst.at(NodeSlot(edge->dst(), edge->dst_input()));
}

Node* Encapsulator::Subgraph::MakeNodeImage(const Graph* graph_in, Node* node) {
  if (!graph_) {
    graph_.reset(new Graph(graph_in->op_registry()));
    graph_->set_versions(graph_in->versions());
  }

  if (device_.empty()) {
    device_ = node->assigned_device_name().empty()
                  ? node->requested_device()
                  : node->assigned_device_name();
  }

  return graph_->CopyNode(node);
}

Graph* Encapsulator::Subgraph::GetGraph() const { return graph_.get(); }

Status Encapsulator::Subgraph::RecordArg(
    const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  Node* src_node = edge->src();
  int src_slot = edge->src_output();
  std::unordered_map<NodeSlot, int, NodeSlot::Hasher>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) =
      args_by_src_.emplace(NodeSlot(src_node, src_slot), args_by_src_.size());
  int arg_index = iter->second;
  if (inserted) {
    NodeDef arg_def;
    NodeDefBuilder builder(
        strings::StrCat(src_node->name(), "_", src_slot, "_arg"), kArgOp);
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    builder.Attr("T", dtype);
    builder.Attr("index", arg_index);
    Status s = builder.Finalize(&arg_def);
    if (!s.ok()) return s;

    Node* arg = graph_->AddNode(arg_def, &s);
    if (!s.ok()) return s;

    src_arg_pairs->push_back({src_node, arg});
    args_.push_back(arg);
  }
  Node* dst_node = edge->dst();
  Node* dst_image = node_images.at(dst_node);
  int dst_slot = edge->dst_input();
  args_by_dst_[NodeSlot(dst_node, dst_slot)] = arg_index;
  graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
  return Status::OK();
}

Status Encapsulator::Subgraph::RecordResult(
    const Edge* edge,
    const std::unordered_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  int src_slot = edge->src_output();
  std::unordered_map<NodeSlot, int, NodeSlot::Hasher>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) =
      results_.emplace(NodeSlot(src_node, src_slot), results_.size());
  int ret_index = iter->second;
  if (inserted) {
    NodeDef ret_def;
    NodeDefBuilder builder(
        strings::StrCat(src_node->name(), "_", src_slot, "_retval"), kRetValOp);
    DataType dtype = src_node->output_type(src_slot);
    builder.Attr("T", dtype);
    builder.Attr("index", ret_index);
    builder.Input(src_image->name(), src_slot, dtype);
    Status s = builder.Finalize(&ret_def);
    if (!s.ok()) return s;
    Node* ret = graph_->AddNode(ret_def, &s);
    if (!s.ok()) return s;

    graph_->AddEdge(src_image, src_slot, ret, 0);
  }
  return Status::OK();
}

void Encapsulator::Subgraph::RecordOutsideCompilationInputOrControl(
    const string& outside_compilation_id, const Edge* edge) {
  auto iter = outside_compilation_subgraphs_
                  .emplace(outside_compilation_id, OutsideCompilationSubgraph())
                  .first;
  OutsideCompilationSubgraph& outside_subgraph = iter->second;
  if (edge->IsControlEdge()) {
    outside_subgraph.control_inputs.insert(edge->src());
  } else {
    int input_index = outside_subgraph.inputs.size();
    outside_subgraph.inputs.emplace(NodeSlot(edge->src(), edge->src_output()),
                                    input_index);
  }
}

void Encapsulator::Subgraph::RecordOutsideCompilationOutputOrControl(
    const string& outside_compilation_id, const Edge* edge) {
  auto subgraph_iter =
      outside_compilation_subgraphs_
          .emplace(outside_compilation_id, OutsideCompilationSubgraph())
          .first;
  OutsideCompilationSubgraph& outside_subgraph = subgraph_iter->second;
  if (edge->IsControlEdge()) {
    outside_subgraph.control_outputs.insert(edge->dst());
  } else {
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    auto output_iter =
        outside_subgraph.outputs_by_src
            .emplace(NodeSlot(edge->src(), edge->src_output(), dtype),
                     outside_subgraph.outputs_by_src.size())
            .first;
    int output_index = output_iter->second;
    outside_subgraph.outputs_by_dst[NodeSlot(edge->dst(), edge->dst_input())] =
        output_index;
  }
}

Status Encapsulator::Subgraph::AddSendsToOutsideCompilation(
    const std::unordered_map<const Node*, Node*>& node_images) {
  for (auto& oc_subgraph_iter : outside_compilation_subgraphs_) {
    const string& oc_subgraph_name = oc_subgraph_iter.first;
    OutsideCompilationSubgraph& oc_subgraph = oc_subgraph_iter.second;
    if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty()) {
      // Build a _SendToHost node sending all the args of the appropriate
      // types.
      std::vector<DataType> dtypes(oc_subgraph.inputs.size(), DT_INVALID);
      std::vector<NodeDefBuilder::NodeOut> inputs(oc_subgraph.inputs.size());

      for (const auto& input_src : oc_subgraph.inputs) {
        const Node* src_node = input_src.first.node;
        Node* src_image = node_images.at(src_node);
        int src_slot = input_src.first.slot;
        int input_index = input_src.second;

        DataType dtype = src_node->output_type(src_slot);
        dtypes[input_index] = dtype;
        inputs[input_index].Reset(src_image->name(), src_slot, dtype);
      }

      NodeDef send_def;
      NodeDefBuilder builder(
          strings::StrCat("outside_compilation_", oc_subgraph_name, "_send"),
          kSendToHostOp);
      builder.Attr("dtypes", dtypes);
      builder.Input(inputs);
      Status s = builder.Finalize(&send_def);
      if (!s.ok()) return s;

      oc_subgraph.send_to_host = graph_->AddNode(send_def, &s);
      if (!s.ok()) return s;

      // Connect the _SendToHost node to its producers in the subgraph.
      for (auto& input_src : oc_subgraph.inputs) {
        const Node* src_node = input_src.first.node;
        Node* src_image = node_images.at(src_node);
        int src_slot = input_src.first.slot;
        int input_index = input_src.second;
        graph_->AddEdge(src_image, src_slot, oc_subgraph.send_to_host,
                        input_index);
      }

      // Connect the _SendToHost node to its control edge producers in the
      // subgraph.
      for (const auto& src_node : oc_subgraph.control_inputs) {
        Node* src_image = node_images.at(src_node);
        graph_->AddControlEdge(src_image, oc_subgraph.send_to_host);
      }
    }
  }

  return Status::OK();
}

Status Encapsulator::Subgraph::AddRecvsFromOutsideCompilation(
    const std::unordered_map<const Node*, Node*>& node_images) {
  for (auto& oc_subgraph_iter : outside_compilation_subgraphs_) {
    const string& oc_subgraph_name = oc_subgraph_iter.first;
    OutsideCompilationSubgraph& oc_subgraph = oc_subgraph_iter.second;
    if (!oc_subgraph.outputs_by_src.empty() ||
        !oc_subgraph.control_outputs.empty()) {
      // Build a _RecvFromHost node producing all the outputs of the appropriate
      // types.
      std::vector<DataType> dtypes(oc_subgraph.outputs_by_src.size(),
                                   DT_INVALID);

      for (const auto& output : oc_subgraph.outputs_by_src) {
        DataType dtype = output.first.dtype;
        int output_index = output.second;
        dtypes[output_index] = dtype;
      }

      NodeDef recv_def;
      NodeDefBuilder builder(
          strings::StrCat("outside_compilation_", oc_subgraph_name, "_recv"),
          kRecvFromHostOp);
      builder.Attr("dtypes", dtypes);
      Status s = builder.Finalize(&recv_def);
      if (!s.ok()) return s;

      Node* recv = graph_->AddNode(recv_def, &s);
      if (!s.ok()) return s;

      // Connect the consumers in the subgraph to the _RecvFromHost node.
      for (const auto& output : oc_subgraph.outputs_by_dst) {
        const Node* dst_node = output.first.node;
        Node* dst_image = node_images.at(dst_node);
        int dst_slot = output.first.slot;
        int output_index = output.second;

        graph_->AddEdge(recv, output_index, dst_image, dst_slot);
      }

      // Connect the control edge consumers in the subgraph to the _RecvFromHost
      // node.
      for (const auto& dst_node : oc_subgraph.control_outputs) {
        Node* dst_image = node_images.at(dst_node);
        graph_->AddControlEdge(recv, dst_image);
      }

      // Add a control edge in the subgraph so that the _SendToHost node, if
      // any, is compiled before the _RecvFromHost node.
      if (oc_subgraph.send_to_host != nullptr) {
        graph_->AddControlEdge(oc_subgraph.send_to_host, recv);
      }
    }
  }

  return Status::OK();
}

Status Encapsulator::Subgraph::MakeSequencingNode(const string& subgraph_name,
                                                  Graph* graph_out) {
  if (sequencer_ == nullptr) {
    NodeDef seq_def;
    NodeDefBuilder builder(strings::StrCat(subgraph_name, "_sequencer"),
                           "NoOp");
    Status s = builder.Finalize(&seq_def);
    if (!s.ok()) return s;

    sequencer_ = graph_out->AddNode(seq_def, &s);
    if (!s.ok()) return s;
    sequencer_->set_assigned_device_name(device_);
  }
  return Status::OK();
}

void Encapsulator::Subgraph::ConnectSequencerToOutputs(Graph* graph_out) {
  if (sequencer_ != nullptr) {
    std::unordered_set<Node*> output_dependencies;
    for (Node* node : call_node_outputs_->out_nodes()) {
      output_dependencies.insert(node);
    }
    for (Node* node : output_dependencies) {
      graph_out->AddControlEdge(sequencer_, node);
    }
  }
}

Status Encapsulator::Subgraph::BuildFunctionDef(
    const string& name_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
    bool reuse_existing_functions, FunctionLibraryDefinition* library) {
  // name_in is copied here because name may be modified below if
  // rewrite_subgraph_fn is true.
  string name = name_in;
  call_node_def_.set_op(name);
  call_node_def_.set_name(name);
  call_node_def_.set_device(device_);

  if (rewrite_subgraph_fn) {
    // Initialize the input and output permutations to the identity.
    std::vector<int> input_permutation(args_by_src_.size());
    std::iota(input_permutation.begin(), input_permutation.end(), 0);
    std::vector<int> output_permutation(results_.size());
    std::iota(output_permutation.begin(), output_permutation.end(), 0);

    TF_RETURN_IF_ERROR(rewrite_subgraph_fn(
        &graph_, &input_permutation, &output_permutation, &call_node_def_));

    // Apply the input/output permutations to the 'args_by_...' and 'results_'
    // mappings, so when we build edges in BuildOutputGraph() we
    // connect them to the right input/output positions.
    if (input_permutation.size() != args_by_src_.size()) {
      return errors::InvalidArgument("Input permutation has incorrect size.");
    }
    if (output_permutation.size() != results_.size()) {
      return errors::InvalidArgument("Output permutation has incorrect size.");
    }
    for (auto& arg : args_by_src_) {
      arg.second = input_permutation[arg.second];
    }
    for (auto& arg : args_by_dst_) {
      arg.second = input_permutation[arg.second];
    }
    for (auto& result : results_) {
      result.second = output_permutation[result.second];
    }

    name = call_node_def_.op();
  }

  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));

  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Build function def " << name;
    dump_graph::DumpGraphToFile(
        strings::StrCat("encapsulate_fdef_graph_", name), *graph_, library);
    dump_graph::DumpFunctionDefToFile(
        strings::StrCat("encapsulate_fdef_", name), fdef);
  }

  if (!reuse_existing_functions || library->Find(name) == nullptr) {
    TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
  }
  return Status::OK();
}

Status Encapsulator::Subgraph::BuildParallelCheckOp(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  // Build an index mapping output positions to node/slot pairs in the
  // original graph.
  std::vector<NodeSlot> results_by_num(results_.size());
  for (const auto& entry : results_) {
    results_by_num[entry.second] = entry.first;
  }

  // Build a parallel check NodeDef.
  int num_results = results_by_num.size();
  std::vector<DataType> result_dtypes(num_results);
  std::vector<NodeDefBuilder::NodeOut> expected_outputs(num_results);
  std::vector<NodeDefBuilder::NodeOut> actual_outputs(num_results);
  for (int i = 0; i < num_results; ++i) {
    const NodeSlot& node_slot = results_by_num[i];
    result_dtypes[i] = node_slot.node->output_type(node_slot.slot);
    expected_outputs[i] =
        NodeDefBuilder::NodeOut(node_images.at(node_slot.node)->name(),
                                node_slot.slot, result_dtypes[i]);
    actual_outputs[i] =
        NodeDefBuilder::NodeOut(call_node_def_.name(), i, result_dtypes[i]);
  }
  // Assign the parallel check op to a CPU on the same task as the cluster it is
  // checking.
  string device, dummy;
  if (!DeviceNameUtils::SplitDeviceName(
          call_node_inputs_->assigned_device_name(), &device, &dummy)) {
    return errors::InvalidArgument("Could not parse device name");
  }
  strings::StrAppend(&device, "/cpu:0");

  NodeDef check_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder(graph_out->NewName(strings::StrCat(call_node_def_.name(),
                                                        "_parallel_check")),
                     "ParallelCheck")
          .Device(device)
          .Attr("T", result_dtypes)
          .Input(expected_outputs)
          .Input(actual_outputs)
          .Finalize(&check_def));

  Status s;
  Node* check_op = graph_out->AddNode(check_def, &s);
  if (!s.ok()) return s;
  check_op->set_assigned_device_name(device);

  // TODO(phawkins): it seems redundant to call AddEdge as well as
  // pass Inputs to the NodeDefBuilder, but I have been unable to find a
  // way to avoid it.
  for (int i = 0; i < num_results; ++i) {
    const NodeSlot& node_slot = results_by_num[i];
    graph_out->AddEdge(node_images.at(node_slot.node), node_slot.slot, check_op,
                       i);
    graph_out->AddEdge(call_node_inputs_, i, check_op, num_results + i);
  }

  call_node_outputs_ = check_op;
  return Status::OK();
}

Status Encapsulator::Subgraph::AddFunctionCallNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    bool parallel_checking, Graph* graph_out) {
  Status s;
  call_node_inputs_ = graph_out->AddNode(call_node_def_, &s);
  if (!s.ok()) return s;

  // Copy the assigned device and the key_annotation over.
  call_node_inputs_->set_assigned_device_name(device_);
  call_node_outputs_ = call_node_inputs_;

  if (parallel_checking) {
    TF_RETURN_IF_ERROR(BuildParallelCheckOp(node_images, graph_out));
  }
  return Status::OK();
}

Status Encapsulator::Subgraph::AddRecvAtHostNode(
    const string& subgraph_name, const string& oc_subgraph_name,
    OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out) {
  std::vector<DataType> dtypes(oc_subgraph->inputs.size(), DT_INVALID);

  for (const auto& input : oc_subgraph->inputs) {
    const Node* src_node = input.first.node;
    int src_slot = input.first.slot;
    int input_index = input.second;

    DataType dtype = src_node->output_type(src_slot);
    dtypes[input_index] = dtype;
  }

  NodeDef recv_def;
  NodeDefBuilder builder(strings::StrCat("outside_compilation_", subgraph_name,
                                         "_", oc_subgraph_name, "_recv"),
                         kRecvAtHostOp);
  builder.Attr("dtypes", dtypes);
  Status s = builder.Finalize(&recv_def);
  if (!s.ok()) return s;

  oc_subgraph->recv_at_host = graph_out->AddNode(recv_def, &s);
  if (!s.ok()) return s;
  oc_subgraph->recv_at_host->set_assigned_device_name(device_);

  // Add a control dependency forcing the RecvAtHost to run before the subgraph
  // completes. This has no effect on execution order but prevents the
  // RecvAtHost being pruned.
  TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
  graph_out->AddControlEdge(oc_subgraph->recv_at_host, sequencer_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddSendFromHostNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    const string& subgraph_name, const string& oc_subgraph_name,
    OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out) {
  std::vector<DataType> dtypes(oc_subgraph->outputs_by_src.size(), DT_INVALID);
  std::vector<NodeDefBuilder::NodeOut> inputs(
      oc_subgraph->outputs_by_src.size());

  for (const auto& output : oc_subgraph->outputs_by_src) {
    const Node* src_node = output.first.node;
    Node* src_image = node_images.at(src_node);
    int src_slot = output.first.slot;
    int output_index = output.second;

    DataType dtype = src_node->output_type(src_slot);
    dtypes[output_index] = dtype;
    inputs[output_index].Reset(src_image->name(), src_slot, dtype);
  }

  NodeDef send_def;
  NodeDefBuilder builder(strings::StrCat("outside_compilation_", subgraph_name,
                                         "_", oc_subgraph_name, "_send"),
                         kSendFromHostOp);
  builder.Attr("dtypes", dtypes);
  builder.Input(inputs);
  Status s = builder.Finalize(&send_def);
  if (!s.ok()) return s;

  oc_subgraph->send_from_host = graph_out->AddNode(send_def, &s);
  if (!s.ok()) return s;
  oc_subgraph->send_from_host->set_assigned_device_name(device_);

  // Add a control dependency forcing the SendFromHost to run before the
  // subgraph completes. This has no effect on execution order but prevents the
  // RecvAtHost being pruned.
  TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
  graph_out->AddControlEdge(oc_subgraph->send_from_host, sequencer_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddOutsideCompilationHostIONodes(
    const string& subgraph_name,
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& outside_compilation_subgraph_entry :
       outside_compilation_subgraphs_) {
    const string& oc_name = outside_compilation_subgraph_entry.first;
    OutsideCompilationSubgraph& oc_subgraph =
        outside_compilation_subgraph_entry.second;

    if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty()) {
      TF_RETURN_IF_ERROR(
          AddRecvAtHostNode(subgraph_name, oc_name, &oc_subgraph, graph_out));
    }

    if (!oc_subgraph.outputs_by_src.empty() ||
        !oc_subgraph.control_outputs.empty()) {
      TF_RETURN_IF_ERROR(AddSendFromHostNode(node_images, subgraph_name,
                                             oc_name, &oc_subgraph, graph_out));
    }
  }
  return Status::OK();
}

Status Encapsulator::GetFunctionNameAttr(
    Node const* node, string* attr, string* outside_compilation_attr) const {
  Status s = GetNodeAttr(node->attrs(), group_attribute_, attr);
  if (s.code() == error::Code::NOT_FOUND) {
    // Return empty attr if there's no group_attribute.
    attr->clear();
  } else {
    TF_RETURN_IF_ERROR(s);
  }
  bool has_group_attr = s.ok();
  s = GetNodeAttr(node->attrs(), outside_compilation_attribute_,
                  outside_compilation_attr);
  if (s.code() == error::Code::NOT_FOUND) {
    // Return empty attr if there's no outside_compilation attribute.
    outside_compilation_attr->clear();
  } else {
    TF_RETURN_IF_ERROR(s);
    if (!has_group_attr) {
      return errors::InvalidArgument(
          "Node ", node->name(), " has ", outside_compilation_attribute_,
          " attribute but no ", group_attribute_, " attribute.");
    }
  }
  return Status::OK();
}

bool IsInSubgraph(const string& func_id, const string& outside_compilation_id) {
  return !func_id.empty() && outside_compilation_id.empty();
}

Status Encapsulator::CopySubgraphNodes(
    std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    string outside_compilation_id;
    TF_RETURN_IF_ERROR(
        GetFunctionNameAttr(node, &func_id, &outside_compilation_id));
    if (!IsInSubgraph(func_id, outside_compilation_id)) continue;

    Subgraph& subgraph = subgraphs_[func_id];
    Node* image = subgraph.MakeNodeImage(graph_in_, node);
    image->ClearAttr(group_attribute_);
    (*node_images)[node] = image;
  }
  return Status::OK();
}

Status Encapsulator::CopySubgraphEdges(
    const std::unordered_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id;
    string src_outside_compilation_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id,
                                           &src_outside_compilation_id));
    string dst_func_id;
    string dst_outside_compilation_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id,
                                           &dst_outside_compilation_id));
    Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
    Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);

    // Copy edges that are local to a subgraph.
    if (IsInSubgraph(src_func_id, src_outside_compilation_id) &&
        IsInSubgraph(dst_func_id, dst_outside_compilation_id) &&
        src_func_id == dst_func_id) {
      Graph* g = subgraphs_[src_func_id].GetGraph();
      if (edge->IsControlEdge()) {
        g->AddControlEdge(src_image, dst_image);
      } else {
        g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
      }
      continue;
    }

    // Record 'src' as an output of its subgraph, if applicable.
    if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->src()->output_type(edge->src_output());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as results: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }

      Subgraph& src_subgraph = subgraphs_[src_func_id];
      if (src_func_id == dst_func_id) {
        // src is in the subgraph and dst is outside_compilation in the same
        // subgraph.
        src_subgraph.RecordOutsideCompilationInputOrControl(
            dst_outside_compilation_id, edge);
      } else {
        // Ignore control edges leaving the subgraph. We will lift them onto the
        // enclosing call operators in BuildOutputGraph().
        if (!edge->IsControlEdge()) {
          TF_RETURN_IF_ERROR(src_subgraph.RecordResult(edge, node_images));
        }
      }
    }

    // Record 'dst' as an input of its subgraph, if applicable.
    if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
      // Look at the type of the destination not the source, since Ref output
      // Tensors can be automatically cast to non-Ref Tensors at the
      // destination.
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as args: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }

      Subgraph& dst_subgraph = subgraphs_[dst_func_id];
      if (src_func_id == dst_func_id) {
        // dst is in the subgraph and src is outside_compilation in the same
        // subgraph.
        dst_subgraph.RecordOutsideCompilationOutputOrControl(
            src_outside_compilation_id, edge);
      } else {
        // Ignore control edges entering the subgraph. We will lift them onto
        // the enclosing call operators in BuildOutputGraph().
        if (!edge->IsControlEdge()) {
          TF_RETURN_IF_ERROR(
              dst_subgraph.RecordArg(edge, node_images, src_arg_pairs));
        }
      }
    }
  }
  return Status::OK();
}

Status Encapsulator::SplitIntoSubgraphs() {
  Status s;

  // Map from input graph nodes to subgraph nodes.
  std::unordered_map<const Node*, Node*> node_images;

  // Each entry of src_arg_pairs is a pair whose first element is a node in the
  // original graph that has an output edge in the subgraph, and whose second
  // element is the arg node in the subgraph that it sends to. The vector will
  // be filled in below in AddArgs.
  std::vector<std::pair<const Node*, Node*>> src_arg_pairs;

  TF_RETURN_IF_ERROR(CopySubgraphNodes(&node_images));
  TF_RETURN_IF_ERROR(CopySubgraphEdges(node_images, &src_arg_pairs));

  // For each subgraph, add the nodes that deal with inputs and outputs its
  // nested outside_compilation subgraphs. These could not be added earlier
  // during CopySubgraphEdges since we need to discover all the types of the
  // inputs and outputs for an outside_compilation subgraph before creating a
  // single input and output node for it.
  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    TF_RETURN_IF_ERROR(subgraph.AddSendsToOutsideCompilation(node_images));
    TF_RETURN_IF_ERROR(subgraph.AddRecvsFromOutsideCompilation(node_images));
  }

  MarkGuaranteedConstants(*graph_in_, src_arg_pairs);

  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    FixupSourceAndSinkEdges(subgraph.GetGraph());
  }

  return s;
}

Status Encapsulator::BuildFunctionDefs(
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
    FunctionLibraryDefinition* library) {
  for (auto& subgraph_entry : subgraphs_) {
    string name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;
    TF_RETURN_IF_ERROR(subgraph.BuildFunctionDef(
        name, rewrite_subgraph_fn, reuse_existing_functions, library));
  }
  return Status::OK();
}

Status Encapsulator::CopyNodesToOutputGraph(
    bool parallel_checking, Graph* graph_out,
    std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    string outside_compilation_id;
    TF_RETURN_IF_ERROR(
        GetFunctionNameAttr(node, &func_id, &outside_compilation_id));

    // Don't copy nodes that going to be encapsulated, unless parallel checking
    // is enabled.
    if (IsInSubgraph(func_id, outside_compilation_id) && !parallel_checking)
      continue;

    Node* image = graph_out->CopyNode(node);
    if (!outside_compilation_id.empty()) {
      if (parallel_checking) {
        return errors::InvalidArgument(
            "Parallel checking is not supported when outside_compilation "
            "clusters are present.");
      }
      image->ClearAttr(group_attribute_);
      image->ClearAttr(outside_compilation_attribute_);
    }
    (*node_images)[node] = image;
  }
  (*node_images)[graph_in_->source_node()] = graph_out->source_node();
  (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
  return Status::OK();
}

Status Encapsulator::AddFunctionCallNodes(
    const std::unordered_map<const Node*, Node*>& node_images,
    bool parallel_checking, Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    TF_RETURN_IF_ERROR(subgraph_entry.second.AddFunctionCallNode(
        node_images, parallel_checking, graph_out));
  }
  return Status::OK();
}

Status Encapsulator::AddOutsideCompilationHostIONodes(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    const string& subgraph_name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;
    TF_RETURN_IF_ERROR(subgraph.AddOutsideCompilationHostIONodes(
        subgraph_name, node_images, graph_out));
  }
  return Status::OK();
}

Status Encapsulator::FindOutputImageOfEdgeSrc(
    const string& src_func_id, const string& src_outside_compilation_id,
    const string& dst_func_id, const string& dst_outside_compilation_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_src_node, Node** src_image) {
  if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
    if (dst_func_id == src_func_id) {
      // The edge is from a subgraph to an outside_compilation cluster in the
      // same subgraph so use the appropriate _RecvAtHost node in the output
      // graph.
      TF_RET_CHECK(!dst_outside_compilation_id.empty());
      *src_image = subgraphs_.at(src_func_id)
                       .GetRecvAtHostNode(dst_outside_compilation_id);
    } else {
      // The edge is from a subgraph to a regular node in the output graph so
      // use the subgraph's call node output.
      *src_image = subgraphs_.at(src_func_id).GetCallNodeForOutputs();
    }
  } else {
    // The source of the edge is in the output graph so use the node image in
    // the output graph.
    *src_image = node_images.at(original_src_node);
  }
  return Status::OK();
}

int Encapsulator::FindOutputSlotOfEdgeSrc(
    const string& src_func_id, const string& src_outside_compilation_id,
    const string& dst_func_id, const string& dst_outside_compilation_id,
    const Edge* edge) {
  if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
    const Subgraph& src_subgraph = subgraphs_.at(src_func_id);
    if (src_func_id == dst_func_id) {
      // 'src' is in a subgraph and 'dst' is outside_compilation in the same
      // subgraph. Use the corresponding _RecvAtHost output instead.
      return src_subgraph.GetRecvAtHostSlot(dst_outside_compilation_id, edge);
    } else {
      // 'src' is in a subgraph and 'dst' is a regular node in the output
      // graph. Use the corresponding call output instead.
      return src_subgraph.GetResultIndexForEdge(edge);
    }
  } else {
    // The source of the edge is in the output graph so use the regular edge
    // slot.
    return edge->src_output();
  }
}

Status Encapsulator::FindOutputImageOfEdgeDst(
    const string& src_func_id, const string& src_outside_compilation_id,
    const string& dst_func_id, const string& dst_outside_compilation_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_dst_node, Node** dst_image) {
  if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
    if (src_func_id == dst_func_id) {
      // The edge is to a subgraph from an outside_compilation cluster in the
      // same subgraph so use the appropriate _SendFromHost node in the output
      // graph.
      TF_RET_CHECK(!src_outside_compilation_id.empty());
      *dst_image = subgraphs_.at(dst_func_id)
                       .GetSendFromHostNode(src_outside_compilation_id);
    } else {
      // The edge is to a subgraph from a regular node in the output graph so
      // use the subgraph's call node input.
      *dst_image = subgraphs_.at(dst_func_id).GetCallNodeForInputs();
    }
  } else {
    // The destination of the edge is in the output graph so use the node image
    // in the output graph.
    *dst_image = node_images.at(original_dst_node);
  }
  return Status::OK();
}

int Encapsulator::FindOutputSlotOfEdgeDst(
    const string& src_func_id, const string& src_outside_compilation_id,
    const string& dst_func_id, const string& dst_outside_compilation_id,
    const Edge* edge) {
  if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
    const Subgraph& dst_subgraph = subgraphs_.at(dst_func_id);
    if (dst_func_id == src_func_id) {
      // 'dst' is in a subgraph and 'src' is outside_compilation in the same
      // subgraph. Use the corresponding _SendFromHost input instead.
      return dst_subgraph.GetSendFromHostSlot(src_outside_compilation_id, edge);
    } else {
      // 'dst' is in a subgraph and 'src' is a regular node in the output
      // graph. Use the corresponding call input instead.
      return dst_subgraph.GetArgIndexForEdge(edge);
    }
  } else {
    // The destination of the edge is in the output graph so use the regular
    // edge slot.
    return edge->dst_input();
  }
}

Status Encapsulator::CopyEdgeToOutputGraph(
    const Edge* edge, const string& src_func_id,
    const string& src_outside_compilation_id, const string& dst_func_id,
    const string& dst_outside_compilation_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    bool parallel_checking, Graph* graph_out,
    std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher>*
        edges_added) {
  Node* src_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(
      src_func_id, src_outside_compilation_id, dst_func_id,
      dst_outside_compilation_id, node_images, edge->src(), &src_image));
  Node* dst_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(
      src_func_id, src_outside_compilation_id, dst_func_id,
      dst_outside_compilation_id, node_images, edge->dst(), &dst_image));

  // If this is a control edge then copy it and return. Lift control edges onto
  // the enclosing call operator.
  if (edge->IsControlEdge()) {
    // Add the control edge, if we have not already added it, using the images
    // determined above (potentially call operators or RecvAtHost/SendFromHost).
    if (edges_added->emplace(NodeSlot(src_image, -1), NodeSlot(dst_image, -1))
            .second) {
      graph_out->AddControlEdge(src_image, dst_image);
    }

    // If parallel checking is enabled, also add a control edge to the
    // corresponding parallel check op.
    if (parallel_checking) {
      graph_out->AddControlEdge(src_image, node_images.at(edge->dst()));
    }
    return Status::OK();
  }

  int src_output =
      FindOutputSlotOfEdgeSrc(src_func_id, src_outside_compilation_id,
                              dst_func_id, dst_outside_compilation_id, edge);

  int dst_input =
      FindOutputSlotOfEdgeDst(src_func_id, src_outside_compilation_id,
                              dst_func_id, dst_outside_compilation_id, edge);

  if (IsInSubgraph(dst_func_id, dst_outside_compilation_id) &&
      parallel_checking) {
    // If we are parallel checking, also feed the tensor as an input to the
    // corresponding parallel check subgraph.
    graph_out->AddEdge(src_image, src_output, node_images.at(edge->dst()),
                       edge->dst_input());
  }

  // Add the edge, if we have not already added it.
  if (edges_added
          ->emplace(NodeSlot(src_image, src_output),
                    NodeSlot(dst_image, dst_input))
          .second) {
    graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
  }
  return Status::OK();
}

Status Encapsulator::AddEdgesToOutputGraph(
    const std::unordered_map<const Node*, Node*>& node_images,
    bool parallel_checking, Graph* graph_out) {
  // Set of edges already added to the output graph, represented as (src, dst)
  // pairs. We use the set to deduplicate edges; multiple edges in the input
  // graph may map to one edge in the output graph.
  std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher>
      edges_added;

  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id;
    string src_outside_compilation_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id,
                                           &src_outside_compilation_id));
    string dst_func_id;
    string dst_outside_compilation_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id,
                                           &dst_outside_compilation_id));

    // Ignore edges that are strictly contained within one subgraph, unless
    // we are constructing parallel check graphs.
    if (IsInSubgraph(src_func_id, src_outside_compilation_id) &&
        IsInSubgraph(dst_func_id, dst_outside_compilation_id) &&
        src_func_id == dst_func_id) {
      if (parallel_checking) {
        Node* src_image = node_images.at(edge->src());
        Node* dst_image = node_images.at(edge->dst());
        if (edge->IsControlEdge()) {
          graph_out->AddControlEdge(src_image, dst_image);
        } else {
          graph_out->AddEdge(src_image, edge->src_output(), dst_image,
                             edge->dst_input());
        }
      }
      continue;
    }

    // We have an edge that crosses a cluster boundary or is entirely within the
    // unclustered graph.
    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(
        edge, src_func_id, src_outside_compilation_id, dst_func_id,
        dst_outside_compilation_id, node_images, parallel_checking, graph_out,
        &edges_added));
  }

  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    subgraph.ConnectSequencerToOutputs(graph_out);
  }

  return Status::OK();
}

Status Encapsulator::BuildOutputGraph(bool parallel_checking,
                                      Graph* graph_out) {
  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node*, Node*> node_images;

  TF_RETURN_IF_ERROR(
      CopyNodesToOutputGraph(parallel_checking, graph_out, &node_images));
  TF_RETURN_IF_ERROR(
      AddFunctionCallNodes(node_images, parallel_checking, graph_out));
  TF_RETURN_IF_ERROR(AddOutsideCompilationHostIONodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(
      AddEdgesToOutputGraph(node_images, parallel_checking, graph_out));

  return Status::OK();
}

}  // anonymous namespace

Status EncapsulateSubgraphsInFunctions(
    string group_attribute, string outside_compilation_attribute,
    const Graph& graph_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
    bool parallel_checking, bool reuse_existing_functions,
    std::unique_ptr<Graph>* graph_out, FunctionLibraryDefinition* library) {
  Status s;

  Encapsulator encapsulator(std::move(group_attribute),
                            std::move(outside_compilation_attribute),
                            &graph_in);
  TF_RETURN_IF_ERROR(encapsulator.SplitIntoSubgraphs());

  TF_RETURN_IF_ERROR(encapsulator.BuildFunctionDefs(
      rewrite_subgraph_fn, reuse_existing_functions, library));

  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  TF_RETURN_IF_ERROR(
      encapsulator.BuildOutputGraph(parallel_checking, out.get()));

  *graph_out = std::move(out);
  return Status::OK();
}

// Finds the types of the _Arg nodes, indexed by position.
static Status GetArgTypes(const Graph& graph, DataTypeVector* types) {
  for (Node* n : graph.op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (index < 0 || index >= types->size()) {
        return errors::InvalidArgument("Invalid argument number");
      }
      (*types)[index] = n->output_type(0);
    }
  }
  return Status::OK();
}

// Renumber the indices of _Arg nodes in a graph, according to
// 'permutation' that maps old indices to new indices.
static Status RenumberArguments(Graph* graph,
                                const std::vector<int>& permutation) {
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (index < 0 || index >= permutation.size()) {
        return errors::InvalidArgument("Invalid argument number");
      }
      n->AddAttr("index", permutation[index]);
    }
  }
  return Status::OK();
}

Status EncapsulateSubgraphsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "EncapsulateSubgraphsPass::Run";
  legacy_flags::EncapsulateSubgraphsPassFlags* flags =
      legacy_flags::GetEncapsulateSubgraphsPassFlags();
  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("before_encapsulate_subgraphs", **options.graph,
                                options.flib_def);
  }

  std::unique_ptr<Graph> graph_out;
  FunctionLibraryDefinition* const library = options.flib_def;

  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, options.session_options->env,
                                        TF_GRAPH_DEF_VERSION, library, opts));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  auto rewrite_subgraph = [flr](std::unique_ptr<Graph>* subgraph,
                                std::vector<int>* input_permutation,
                                std::vector<int>* output_permutation,
                                NodeDef* node) {
    // Optimize the subgraph.
    OptimizeGraph(flr, subgraph);

    const int num_args = input_permutation->size();
    std::vector<bool> const_args(num_args);
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(**subgraph, &const_args));

    DataTypeVector arg_types(num_args);
    TF_RETURN_IF_ERROR(GetArgTypes(**subgraph, &arg_types));

    // Compute a permutation of the arguments such that the constant arguments
    // are first.
    const int num_consts =
        std::count(const_args.begin(), const_args.end(), true);

    const int num_resources =
        std::count(arg_types.begin(), arg_types.end(), DT_RESOURCE);
    const int num_nonconsts = num_args - num_resources - num_consts;
    if (num_nonconsts < 0) {
      return errors::Internal("num_nonconsts should be >= 0, was ",
                              num_nonconsts);
    }

    int const_pos = 0;
    int arg_pos = num_consts;
    int resource_pos = num_consts + num_nonconsts;
    for (int i = 0; i < num_args; ++i) {
      if (const_args[i]) {
        if (arg_types[i] == DT_RESOURCE) {
          return errors::Internal(
              "Resource arguments cannot be constant (argument ", i, ")");
        }
        (*input_permutation)[i] = const_pos;
        ++const_pos;
      } else if (arg_types[i] == DT_RESOURCE) {
        (*input_permutation)[i] = resource_pos;
        ++resource_pos;
      } else {
        (*input_permutation)[i] = arg_pos;
        ++arg_pos;
      }
    }

    // Renumber argument nodes in the graph.
    TF_RETURN_IF_ERROR(RenumberArguments(subgraph->get(), *input_permutation));

    // TODO(phawkins): add a forward is-constant analysis, similarly split
    // outputs into host-memory constants and device-memory non-constants.

    AddNodeAttr(kXlaCompiledKernelAttr, true, node);
    AddNodeAttr(kXlaNumConstantArgsAttr, num_consts, node);
    AddNodeAttr(kXlaNumResourceArgsAttr, num_resources, node);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(EncapsulateSubgraphsInFunctions(
      kXlaClusterAttr, kXlaOutsideCompilationAttr, **options.graph,
      rewrite_subgraph, flags->tf_xla_parallel_checking,
      /*reuse_existing_functions=*/false, &graph_out, library));

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("after_encapsulate_subgraphs", *graph_out,
                                options.flib_def);
  }

  *options.graph = std::move(graph_out);
  return Status::OK();
}

bool IsXlaCompiledKernel(const Node& node) {
  bool is_compiled = false;
  bool has_compilation_attr =
      GetNodeAttr(node.attrs(), kXlaCompiledKernelAttr, &is_compiled).ok() &&
      is_compiled;
  return has_compilation_attr ? is_compiled : false;
}

}  // namespace tensorflow
