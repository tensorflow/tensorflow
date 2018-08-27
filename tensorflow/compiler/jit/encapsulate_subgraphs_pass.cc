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

#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char* const kXlaCompiledKernelAttr = "_XlaCompiledKernel";
const char* const kXlaNumConstantArgsAttr = "_XlaNumConstantArgs";
const char* const kXlaNumResourceArgsAttr = "_XlaNumResourceArgs";
const char* const kXlaHostTransferSequencerAttr =
    "_xla_host_transfer_sequencer";

namespace {

bool AreAllParentsGuaranteedConst(
    const Node& n, const gtl::FlatSet<const Node*>& runtime_const_nodes) {
  if (n.type_string() == "GuaranteeConst") {
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
                   if (AreAllParentsGuaranteedConst(*n,
                                                    guaranteed_const_nodes)) {
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

struct OutputInputTensorPairHasher {
  uint64 operator()(std::pair<OutputTensor, InputTensor> const& s) const {
    return Hash64Combine(OutputTensor::Hash()(s.first),
                         InputTensor::Hash()(s.second));
  }
};

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";
static const char* const kHostComputeOp = "XlaHostCompute";
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

  // Find dependencies between subgraphs and outside_compilation clusters that
  // only manifest via edges between outside_compilation clusters in the outer
  // (non-compiled) graph.
  Status FindClusterDependencies();

  // Find subgraphs marked with 'group_attribute', and build a new
  // subgraph, one for each value of 'group_attribute'.
  Status SplitIntoSubgraphs(FunctionLibraryDefinition* library);

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
  Status BuildOutputGraph(Graph* graph_out, FunctionLibraryDefinition* library);

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
  // The compiled cluster is as follows. HC is a HostCompute node which is the
  // source of a channel to the RAH node above and the destination of a channel
  // from the SFH node above.
  //
  //  Arg  --> B  --> HC  --> D --> Retval
  //
  // The channels HC/RAH and SFH/HC each transmit multiple tensors, so there is
  // at most one RAH and SFH in each outside_compilation cluster. This design is
  // preferred over adding separate Arg/Retval nodes for each transmitted value
  // because it allows optimizations to the host code that would like to limit
  // communication between host and device and, e.g., raise only one interrupt
  // per channel rather than one per transmitted value.
  //
  // The shapes of the outputs from the HC node in general cannot be determined
  // until the shapes of its inputs are known at compile time, since e.g.,
  // above, the shape of C's outputs aren't known until the shape of its inputs
  // are known. If the shapes of the HC's outputs can be determined during the
  // rewrite, they are stored in the node's 'shapes' attr. Otherwise a minimal
  // graph is stored in the shape_inference_graph attr. This graph can be used
  // when compiling the HC Op to determined the shape of the SFH inputs given
  // the shapes of any ancestor RAH outputs. If it can be determined that the
  // shape of the SFH inputs will not be inferrable even once the shapes of the
  // RAH outputs are known, an error is returned by the rewriter.
  //
  // Once edges between compiled and outside_compilation clusters have been
  // replaced by send/recv ops, some dependencies may no longer be apparent.
  // A clustering pass finds all the dependencies between HC nodes that are only
  // present as a result of edges between nodes in outside_compilation clusters.
  // Suppose there is a path from outside_compilation cluster C in subgraph S
  // to outside_compilation cluster D in subgraph T. If S != T then a control
  // edge is added from the call node for S to the call node for T, which
  // ensures that C will execute before D because S executes before T. If S==T
  // then a control dependency is added between the HC nodes for C and D in S,
  // and the HC node for C is added to an 'ancestors' attr in the HC node for D
  // so that during compilation of the HC node for D, an XLA control dependency
  // can be added to ensure C's SendToHost executes before D's RecvFromHost.
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
        Graph* graph_out);

    // Adds _RecvAtHost and _SendFromHost nodes, where needed, to graph_out.
    Status AddOutsideCompilationHostIONodes(
        const string& group_attribute, const string& subgraph_name,
        const string& outside_compilation_attribute,
        const std::unordered_map<const Node*, Node*>& node_images,
        Graph* graph_out);

    // Returns the names of all the outside_compilation subgraphs in this
    // Subgraph.
    void GetOutsideCompilationSubgraphNames(std::vector<string>* names) const;

    // Returns the Node that the inputs and outputs of the function should be
    // wired up to.
    Node* GetCallNode() const;

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

    // Records the fact that there is a path from a node in outside_compilation
    // cluster ancestor to node in cluster successor that does not go through
    // the subgraph.
    void RecordOutsideCompilationDependency(const string& successor,
                                            const string& ancestor);

    // Returns the mapping from outside_compilation cluster C to the set of
    // outside_compilation clusters that have a path to C entirely outside
    // compiled subgraphs.
    const std::unordered_map<string, std::unordered_set<string>>
    OutsideCompilationAncestorMap() const;

    // Adds the HostCompute nodes for each outside_compilation subgraph.
    Status AddHostComputes(
        const string& subgraph_name,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates the sequencer node if it doesn't exist, adding it to graph_out.
    Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);

    // If there is a sequencer node, adds a control edge from the sequencer to
    // the call node.
    void ConnectSequencerToCallNode(Graph* graph_out);

    Status AddShapeInferenceInfo(
        const string& subgraph_name,
        const string& outside_compilation_subgraph_name,
        const std::vector<TensorShapeProto>& shapes, Graph* inference_graph,
        FunctionLibraryDefinition* library);

    Status ReplaceFunctionDef(FunctionLibraryDefinition* library);

   private:
    struct OutsideCompilationSubgraph {
      // Map from source (producer node/slot) tensors in the original graph to
      // input index (slot number in the HostCompute/RecvAtHost nodes that will
      // be created) for the outside_compilation subgraph.
      std::unordered_map<OutputTensor, int, OutputTensor::Hash> inputs;

      // Set of nodes in the original graph that are the source of control edges
      // that cross from the containing compiled subgraph into the
      // outside_compilation subgraph. These are recorded by
      // RecordOutsideCompilationInputOrControl while walking all the subgraph
      // edges, and lifted control edges within the subgraph are added by
      // AddSendsToOutsideCompilation once the _HostCompute node has been
      // created. The matching control edge from _RecvAtHost to the
      // destination is added by CopyEdgeToOutputGraph.
      std::unordered_set<const Node*> control_inputs;

      // Maps from source (producer node/slot) and destination (consumer
      // node/slot) tensors in the original graph to output index (slot number
      // in the SendFromHost/HostCompute nodes that will be created) for the
      // outside_compilation subgraph.
      struct ArgNumAndType {
        int index;
        DataType dtype;

        ArgNumAndType(int i, DataType t) : index(i), dtype(t) {}
      };
      std::unordered_map<OutputTensor, ArgNumAndType, OutputTensor::Hash>
          outputs_by_src;
      std::unordered_map<InputTensor, int, InputTensor::Hash> outputs_by_dst;

      // Set of nodes in the original graph that are the destination of control
      // edges that cross from the outside_compilation subgraph into the
      // containing compiled subgraph. These are recorded by
      // RecordOutsideCompilationOutputOrControl while walking all the subgraph
      // edges, and lifted control edges within the subgraph are added by
      // AddRecvsFromToOutsideCompilation once the _HostCompute node has been
      // created. The matching control edge from the source to _SendFromHost to
      // the destination is added by CopyEdgeToOutputGraph.
      std::unordered_set<const Node*> control_outputs;

      // Name of the _HostCompute node in the subgraph.
      string host_compute_name;

      // _RecvAtHost node in the output graph. Not owned.
      Node* recv_at_host = nullptr;

      // _SendFromHost node in the output graph. Not owned.
      Node* send_from_host = nullptr;
    };

    // Creates an outside_compilation subgraph for outside_compilation_id if
    // none exists yet. Returns the (possible newly created) subgraph for
    // outside_compilation_id.
    OutsideCompilationSubgraph* LookupOrCreateOutsideCompilationSubgraph(
        const string& outside_compilation_id);

    // Builds a placeholder node used to provide the key input to a RecvAtHost
    // or SendFromHost node. This placeholder node will be removed by a later
    // pass.
    Status AddHostComputeKeyPlaceholder(OutsideCompilationSubgraph* oc_subgraph,
                                        Graph* graph_out);

    // Get the set of outside_compilation clusters and the dependency edges
    // between them.
    void GetActiveClusterDependencyGraph(
        std::unordered_set<string>* clusters,
        std::unordered_set<string>* has_successor,
        std::unordered_map<string, std::unordered_set<string>>* ancestors_map);

    // Builds a _RecvAtHost node producing all the inputs of an
    // outside_compilation subgraph and stores it in oc_subgraph.recv_at_host.
    Status AddRecvAtHostNode(const string& group_attribute,
                             const string& subgraph_name,
                             const string& outside_compilation_attribute,
                             const string& oc_subgraph_name,
                             OutsideCompilationSubgraph* oc_subgraph,
                             Graph* graph_out);

    // Builds a _SendFromHost node consuming all the outputs of an
    // outside_compilation subgraph and stores it in oc_subgraph.send_from_host.
    Status AddSendFromHostNode(
        const std::unordered_map<const Node*, Node*>& node_images,
        const string& group_attribute, const string& subgraph_name,
        const string& outside_compilation_attribute,
        const string& oc_subgraph_name, OutsideCompilationSubgraph* oc_subgraph,
        Graph* graph_out);

    // The subgraph extracted from the input graph, suitable for being turned
    // into a FunctionDef. Inputs are fed by _Arg nodes, and outputs are
    // returned by _Retval nodes.
    std::unique_ptr<Graph> graph_;

    // Which device are these nodes on? Used to assign a device to the call
    // node.
    string device_;

    // NodeDef for the function call node.
    NodeDef call_node_def_;

    // Name that is used for the call node. This may not be
    // call_node_def_.name() if the client supplies a rewrite lambda.
    string function_def_name_;

    // Placeholder node simulating the host compute key in the output graph.
    // Not owned.
    Node* host_compute_key_placeholder_ = nullptr;

    // Function call node in the output graph. Not owned.
    Node* call_node_;

    // Maps from source (producer node/slot) and destination
    // (consumer node/slot) tensors in the input graph to _Arg numbers in
    // the subgraph. The source map is one-to-one, whereas the dest map may be
    // many-to-one.
    std::unordered_map<OutputTensor, int, OutputTensor::Hash> args_by_src_;
    std::unordered_map<InputTensor, int, InputTensor::Hash> args_by_dst_;

    // The arguments to the subgraph, in order.
    std::vector<Node*> args_;

    // Map from source tensor in the input graph to result #.
    std::unordered_map<OutputTensor, int, OutputTensor::Hash> results_;

    // The outside_compilation clusters in this subgraph.
    std::unordered_map<string, OutsideCompilationSubgraph>
        outside_compilation_subgraphs_;
    // For each outside_compilation cluster C, the outside_compilation clusters
    // that have a path to C outside the compiled graph.
    std::unordered_map<string, std::unordered_set<string>>
        outside_compilation_ancestors_;
    // For each outside_compilation cluster C, the outside_compilation clusters
    // that have a path from C outside the compiled graph.
    std::unordered_map<string, std::unordered_set<string>>
        outside_compilation_successors_;

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
      Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images);

  // Adds function call nodes for each compiled subgraph.
  Status AddFunctionCallNodes(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

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
      Graph* graph_out,
      std::unordered_set<std::pair<OutputTensor, InputTensor>,
                         OutputInputTensorPairHasher>* edges_added);

  // Adds control dependencies between subgraph call nodes that have
  // dependencies via outside_compilation edges.
  Status AddCallNodeDependencies(Graph* graph_out);

  // Adds all edges to the output graph.
  Status AddEdgesToOutputGraph(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Constructs a minimal shape inference graph that can be used to determine
  // the shape of send_node at the time that the subgraph is compiled.
  // recv_at_host_nodes contains the names of all the recv_at_host nodes that
  // send_node might depend on. These recv_at_host nodes have shapes that are
  // not known during the rewrite pass, but will be known at compile time.
  //
  // If the shapes of all the inputs to send_node can be determined during the
  // rewrite pass, on exit graphdef_out is empty and the shapes are returned in
  // static_shape_out. Otherwise graphdef_out contains a graph that can be used
  // for shape inference at compile time, where all the source nodes of the
  // graph are either constants with known shapes, or nodes named in
  // recv_at_host_nodes.
  //
  // A non-OK status is returned if neither of the above conditions can be
  // satisfied, e.g., because send_node depends on a node that doesn't have a
  // registered shape inference function.
  Status DoStaticShapeInferenceForOutsideCompilationSend(
      const Graph& graph_in, const BackEdgeHelper& back_edge_helper,
      const ShapeRefiner& shape_refiner,
      const std::unordered_set<string>& recv_at_host_nodes, Node* send_node,
      FunctionLibraryDefinition* library,
      std::vector<TensorShapeProto>* static_shape_out,
      std::unique_ptr<Graph>* graph_out);

  // Makes a copy of graph containing only nodes that are ancestors of at least
  // one node in send_from_host_nodes and store it in pruned_graph. On exit
  // nodes_images contains a mapping from nodes in graph to nodes in
  // pruned_graph. All functions in the copied graph are inlined.
  Status MakePrunedGraphCopyAndInline(
      const Graph& graph, const std::vector<Node*>& sink_nodes,
      std::unique_ptr<Graph>* pruned_graph,
      std::unordered_map<const Node*, Node*>* node_images,
      FunctionLibraryDefinition* library);

  // Makes a copy of graph containing only nodes that are ancestors of a
  // send_from_host node in an outside_compilation subgraph, and store it in
  // pruned_graph. Also perform shape inference on the pruned graph, using
  // shape_refiner. On exit node_images contains a mapping from nodes in graph
  // to nodes in pruned_graph.
  Status MakeGraphForOutsideCompilationSends(
      const Graph& graph, std::unique_ptr<Graph>* pruned_graph,
      BackEdgeHelper* back_edge_helper, ShapeRefiner* shape_refiner,
      std::unordered_map<const Node*, Node*>* node_images,
      FunctionLibraryDefinition* library);

  // Performs static shape inference, as far as possible, for the send_from_host
  // nodes in each outside_compilation subgraph. Where it is not possible to
  // determine the shape statically, stores a serialized GraphDef in the
  // HostCompute 'shape_inference_graph' attr, to be used at compile time for
  // final inference. If the shapes are known statically they are stored in the
  // HostCompute 'shapes' attr.
  Status GetShapeInfoForOutsideCompilationSends(
      Graph* graph_out, FunctionLibraryDefinition* library);

  const string group_attribute_;
  const string outside_compilation_attribute_;
  const Graph* graph_in_;

  std::unordered_map<string, Subgraph> subgraphs_;
  // For each subgraph S the subgraphs S' such that there is a path in some
  // outside_compilation cluster C in S to some outside_compilation cluster C'
  // in S', that goes only through the uncompiled graph.
  std::unordered_map<string, std::unordered_set<string>> subgraph_ancestors_;

  TF_DISALLOW_COPY_AND_ASSIGN(Encapsulator);
};

namespace {

// Return in 'sorted' a topological sort of clusters according to the
// dependencies encoded in ancestors. clusters is the list of all clusters
// including clusters that are not present in the ancestors map. has_successors
// is the set of clusters that are ancestors of some other cluster.
void TopologicalClusterSort(
    const std::unordered_set<string>& clusters,
    const std::unordered_set<string>& has_successors,
    const std::unordered_map<string, std::unordered_set<string>>& ancestors,
    std::vector<string>* sorted) {
  // The nodes are placed in 'sorted' in topological order.
  sorted->clear();
  // We don't use the standard DFS because we are not operating on Node*
  // objects.
  struct Work {
    string cluster;
    bool leave;
  };
  std::set<string> visited;
  std::vector<Work> stack;
  // Seed the processing list with clusters that have no successors.
  for (const auto& cluster : clusters) {
    if (has_successors.find(cluster) == has_successors.end()) {
      stack.push_back({cluster, false});
    }
  }
  while (!stack.empty()) {
    const Work item = stack.back();
    stack.pop_back();
    if (item.leave) {
      sorted->push_back(item.cluster);
      continue;
    }

    if (visited.find(item.cluster) != visited.end()) continue;
    visited.insert(item.cluster);

    stack.push_back({item.cluster, true});
    const auto& iter = ancestors.find(item.cluster);
    if (iter != ancestors.end()) {
      for (const auto& ancestor : iter->second) {
        stack.push_back({ancestor, false});
      }
    }
  }
  CHECK(sorted->size() == clusters.size());
}

}  // namespace

Node* Encapsulator::Subgraph::GetCallNode() const { return call_node_; }

int Encapsulator::Subgraph::GetArgIndexForEdge(const Edge* edge) const {
  return args_by_dst_.at(InputTensor(edge->dst(), edge->dst_input()));
}

int Encapsulator::Subgraph::GetResultIndexForEdge(const Edge* edge) const {
  return results_.at(OutputTensor(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetRecvAtHostNode(
    const string& outside_compilation_subgraph_name) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .recv_at_host;
}

int Encapsulator::Subgraph::GetRecvAtHostSlot(
    const string& outside_compilation_subgraph_name, const Edge* edge) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .inputs.at(OutputTensor(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetSendFromHostNode(
    const string& outside_compilation_subgraph_name) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .send_from_host;
}

int Encapsulator::Subgraph::GetSendFromHostSlot(
    const string& outside_compilation_subgraph_name, const Edge* edge) const {
  return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
      .outputs_by_dst.at(InputTensor(edge->dst(), edge->dst_input()));
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
  std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) = args_by_src_.emplace(
      OutputTensor(src_node, src_slot), args_by_src_.size());
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
  args_by_dst_[InputTensor(dst_node, dst_slot)] = arg_index;
  graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
  return Status::OK();
}

Status Encapsulator::Subgraph::RecordResult(
    const Edge* edge,
    const std::unordered_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  int src_slot = edge->src_output();
  std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) =
      results_.emplace(OutputTensor(src_node, src_slot), results_.size());
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

Encapsulator::Subgraph::OutsideCompilationSubgraph*
Encapsulator::Subgraph::LookupOrCreateOutsideCompilationSubgraph(
    const string& outside_compilation_id) {
  auto iter = outside_compilation_subgraphs_
                  .emplace(outside_compilation_id, OutsideCompilationSubgraph())
                  .first;
  OutsideCompilationSubgraph* outside_subgraph = &iter->second;
  return outside_subgraph;
}

void Encapsulator::Subgraph::RecordOutsideCompilationInputOrControl(
    const string& outside_compilation_id, const Edge* edge) {
  OutsideCompilationSubgraph* outside_subgraph =
      LookupOrCreateOutsideCompilationSubgraph(outside_compilation_id);
  if (edge->IsControlEdge()) {
    outside_subgraph->control_inputs.insert(edge->src());
  } else {
    int input_index = outside_subgraph->inputs.size();
    outside_subgraph->inputs.emplace(
        OutputTensor(edge->src(), edge->src_output()), input_index);
  }
}

void Encapsulator::Subgraph::RecordOutsideCompilationOutputOrControl(
    const string& outside_compilation_id, const Edge* edge) {
  OutsideCompilationSubgraph* outside_subgraph =
      LookupOrCreateOutsideCompilationSubgraph(outside_compilation_id);
  if (edge->IsControlEdge()) {
    outside_subgraph->control_outputs.insert(edge->dst());
  } else {
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    auto output_iter =
        outside_subgraph->outputs_by_src
            .emplace(OutputTensor(edge->src(), edge->src_output()),
                     OutsideCompilationSubgraph::ArgNumAndType(
                         outside_subgraph->outputs_by_src.size(), dtype))
            .first;
    const int output_index = output_iter->second.index;
    outside_subgraph
        ->outputs_by_dst[InputTensor(edge->dst(), edge->dst_input())] =
        output_index;
  }
}

void Encapsulator::Subgraph::RecordOutsideCompilationDependency(
    const string& successor, const string& ancestor) {
  outside_compilation_ancestors_[successor].insert(ancestor);
  outside_compilation_successors_[ancestor].insert(successor);
}

const std::unordered_map<string, std::unordered_set<string>>
Encapsulator::Subgraph::OutsideCompilationAncestorMap() const {
  return outside_compilation_ancestors_;
}

void Encapsulator::Subgraph::GetActiveClusterDependencyGraph(
    std::unordered_set<string>* clusters,
    std::unordered_set<string>* has_successor,
    std::unordered_map<string, std::unordered_set<string>>* ancestors_map) {
  // During initial clustering the ancestor and successor datastructures may
  // have been built including oc_cluster names that never turned into subgraphs
  // because they had no edges into or out of the compiled cluster. Remove them
  // before proceeding to simplify the logic. Get the set of clusters that was
  // actually added, then remove references to the others.
  for (const auto& oc_subgraph : outside_compilation_subgraphs_) {
    clusters->insert(oc_subgraph.first);
  }
  for (const auto& cluster : outside_compilation_successors_) {
    if (clusters->find(cluster.first) != clusters->end()) {
      for (const auto& successor : cluster.second) {
        if (clusters->find(successor) != clusters->end()) {
          has_successor->insert(cluster.first);
          break;
        }
      }
    }
  }
  for (const auto& cluster : outside_compilation_ancestors_) {
    if (clusters->find(cluster.first) != clusters->end()) {
      std::unordered_set<string>& ancestors = (*ancestors_map)[cluster.first];
      for (const auto& ancestor : cluster.second) {
        if (clusters->find(ancestor) != clusters->end()) {
          ancestors.insert(ancestor);
        }
      }
    }
  }
}

Status Encapsulator::Subgraph::AddHostComputes(
    const string& subgraph_name,
    const std::unordered_map<const Node*, Node*>& node_images) {
  // Get the set of outside_compilation clusters and the dependency edges
  // between them.
  std::unordered_set<string> clusters;
  std::unordered_set<string> has_successor;
  std::unordered_map<string, std::unordered_set<string>> ancestors_map;
  GetActiveClusterDependencyGraph(&clusters, &has_successor, &ancestors_map);
  // Topologically sort the outside_compilation clusters according to their
  // dependency relation.
  std::vector<string> sorted_clusters;
  TopologicalClusterSort(clusters, has_successor, ancestors_map,
                         &sorted_clusters);

  // The host compute nodes added for each outside_compilation_cluster;
  std::unordered_map<string, Node*> host_compute_node;
  for (const string& oc_subgraph_name : sorted_clusters) {
    OutsideCompilationSubgraph& oc_subgraph =
        outside_compilation_subgraphs_[oc_subgraph_name];
    if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty() ||
        !oc_subgraph.outputs_by_src.empty() ||
        !oc_subgraph.control_outputs.empty()) {
      // Build a _HostCompute node.
      std::vector<NodeDefBuilder::NodeOut> inputs(oc_subgraph.inputs.size());
      std::vector<DataType> input_dtypes(oc_subgraph.inputs.size(), DT_INVALID);
      std::vector<DataType> output_dtypes(oc_subgraph.outputs_by_src.size(),
                                          DT_INVALID);

      for (const auto& input_src : oc_subgraph.inputs) {
        const Node* src_node = input_src.first.node;
        Node* src_image = node_images.at(src_node);
        int src_slot = input_src.first.index;
        int input_index = input_src.second;

        DataType dtype = src_node->output_type(src_slot);
        inputs[input_index].Reset(src_image->name(), src_slot, dtype);
        input_dtypes[input_index] = dtype;
      }
      for (const auto& output : oc_subgraph.outputs_by_src) {
        DataType dtype = output.second.dtype;
        int output_index = output.second.index;
        output_dtypes[output_index] = dtype;
      }

      std::vector<string> host_compute_ancestors;
      const auto iter = ancestors_map.find(oc_subgraph_name);
      if (iter != ancestors_map.end()) {
        for (const string& ancestor_cluster : iter->second) {
          host_compute_ancestors.push_back(
              outside_compilation_subgraphs_[ancestor_cluster]
                  .host_compute_name);
        }
      }

      NodeDef host_compute_def;
      NodeDefBuilder builder(strings::StrCat("outside_compilation_",
                                             oc_subgraph_name, "_host_compute"),
                             kHostComputeOp);
      builder.Input(inputs);
      builder.Attr("Tinputs", input_dtypes);
      builder.Attr("Toutputs", output_dtypes);
      builder.Attr("ancestors", host_compute_ancestors);
      builder.Attr("key",
                   strings::StrCat("host_compute_channel_", subgraph_name, "_",
                                   oc_subgraph_name));
      builder.Attr("_outside_compilation_subgraph", oc_subgraph_name);
      Status s = builder.Finalize(&host_compute_def);
      if (!s.ok()) return s;

      Node* host_compute = graph_->AddNode(host_compute_def, &s);
      if (!s.ok()) return s;
      host_compute_node[host_compute->name()] = host_compute;
      oc_subgraph.host_compute_name = host_compute->name();

      // Connect the _HostCompute node to its producers in the subgraph.
      for (auto& input_src : oc_subgraph.inputs) {
        const Node* src_node = input_src.first.node;
        Node* src_image = node_images.at(src_node);
        int src_slot = input_src.first.index;
        int input_index = input_src.second;
        graph_->AddEdge(src_image, src_slot, host_compute, input_index);
      }

      // Connect the _HostCompute node to its control edge producers in the
      // subgraph.
      for (const auto& src_node : oc_subgraph.control_inputs) {
        Node* src_image = node_images.at(src_node);
        graph_->AddControlEdge(src_image, host_compute);
      }

      // Connect the _HostCompute node to its ancestor host compute nodes.
      for (const auto& ancestor_name : host_compute_ancestors) {
        Node* ancestor = host_compute_node[ancestor_name];
        graph_->AddControlEdge(ancestor, host_compute);
      }

      // Connect the consumers in the subgraph to the _HostCompute node.
      for (const auto& output : oc_subgraph.outputs_by_dst) {
        const Node* dst_node = output.first.node;
        Node* dst_image = node_images.at(dst_node);
        int dst_slot = output.first.index;
        int output_index = output.second;

        graph_->AddEdge(host_compute, output_index, dst_image, dst_slot);
      }

      // Connect the control edge consumers in the subgraph to the _HostCompute
      // node.
      for (const auto& dst_node : oc_subgraph.control_outputs) {
        Node* dst_image = node_images.at(dst_node);
        graph_->AddControlEdge(host_compute, dst_image);
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
    builder.Attr(kXlaHostTransferSequencerAttr, subgraph_name);
    builder.Device(device_);
    Status s = builder.Finalize(&seq_def);
    if (!s.ok()) return s;

    sequencer_ = graph_out->AddNode(seq_def, &s);
    if (!s.ok()) return s;
  }
  return Status::OK();
}

void Encapsulator::Subgraph::ConnectSequencerToCallNode(Graph* graph_out) {
  if (sequencer_ != nullptr) {
    VLOG(2) << "ConnectSequencerToCallNode";
    graph_out->AddControlEdge(sequencer_, call_node_);
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
    std::vector<OutputTensor> arg_source_tensors(args_by_src_.size());
    for (const auto& arg : args_by_src_) {
      arg_source_tensors.at(arg.second) = arg.first;
    }
    // Initialize the input and output permutations to the identity.
    std::vector<int> input_permutation(args_by_src_.size());
    std::iota(input_permutation.begin(), input_permutation.end(), 0);
    std::vector<int> output_permutation(results_.size());
    std::iota(output_permutation.begin(), output_permutation.end(), 0);

    TF_RETURN_IF_ERROR(
        rewrite_subgraph_fn(arg_source_tensors, &graph_, &input_permutation,
                            &output_permutation, &call_node_def_));

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

  function_def_name_ = name;

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

Status Encapsulator::Subgraph::AddShapeInferenceInfo(
    const string& subgraph_name,
    const string& outside_compilation_subgraph_name,
    const std::vector<TensorShapeProto>& shapes, Graph* inference_graph,
    FunctionLibraryDefinition* library) {
  OutsideCompilationSubgraph& oc_subgraph =
      outside_compilation_subgraphs_.at(outside_compilation_subgraph_name);

  Node* host_compute = nullptr;
  for (Node* n : graph_->nodes()) {
    if (n->name() == oc_subgraph.host_compute_name) {
      host_compute = n;
      break;
    }
  }
  if (host_compute == nullptr) {
    return errors::InvalidArgument(
        "After rewriting subgraph ", outside_compilation_subgraph_name,
        " there is no HostCompute Op for outside compilation subgraph ",
        oc_subgraph.host_compute_name);
  }

  if (inference_graph == nullptr) {
    host_compute->AddAttr("shape_inference_graph", "");
    host_compute->AddAttr("shapes", shapes);
  } else {
    string inference_graph_name =
        strings::StrCat("_outside_compilation_shape_inference_", subgraph_name,
                        "_", outside_compilation_subgraph_name);
    FunctionDef fdef;
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*inference_graph, inference_graph_name, &fdef));
    host_compute->AddAttr("shape_inference_graph", inference_graph_name);
    host_compute->AddAttr("shapes", std::vector<TensorShapeProto>());
    // TODO(sibyl-Aix6ihai): Understand why there are multiple calls to Encapsulator.
    if (library->Find(inference_graph_name) == nullptr) {
      TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
    }
  }
  return Status::OK();
}

Status Encapsulator::Subgraph::ReplaceFunctionDef(
    FunctionLibraryDefinition* library) {
  const string& name = function_def_name_;

  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));

  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Replace function def " << name;
    dump_graph::DumpGraphToFile(
        strings::StrCat("replace_encapsulate_fdef_graph_", name), *graph_,
        library);
    dump_graph::DumpFunctionDefToFile(
        strings::StrCat("replace_encapsulate_fdef_", name), fdef);
  }

  TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  return Status::OK();
}

Status Encapsulator::Subgraph::AddFunctionCallNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  Status s;
  call_node_ = graph_out->AddNode(call_node_def_, &s);
  if (!s.ok()) return s;

  // Copy the assigned device and the key_annotation over.
  call_node_->set_assigned_device_name(device_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddHostComputeKeyPlaceholder(
    OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out) {
  TensorShapeProto shape_proto;
  TensorShape shape({2});
  shape.AsProto(&shape_proto);
  GraphDefBuilder::Options options(graph_out, /*status=*/nullptr);
  NodeDef key_def;
  NodeDefBuilder builder(
      strings::StrCat(call_node_def_.name(), "_key_placeholder"),
      "Placeholder");
  builder.Attr("dtype", DT_STRING);
  builder.Attr("shape", shape_proto);
  builder.Attr("_host_compute_call_node", call_node_def_.name());
  Status s = builder.Finalize(&key_def);
  if (!s.ok()) return s;

  host_compute_key_placeholder_ = graph_out->AddNode(key_def, &s);
  if (!s.ok()) return s;
  host_compute_key_placeholder_->set_assigned_device_name(device_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddRecvAtHostNode(
    const string& group_attribute, const string& subgraph_name,
    const string& outside_compilation_attribute, const string& oc_subgraph_name,
    OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out) {
  if (host_compute_key_placeholder_ == nullptr) {
    TF_RETURN_IF_ERROR(AddHostComputeKeyPlaceholder(oc_subgraph, graph_out));
  }

  std::vector<DataType> dtypes(oc_subgraph->inputs.size(), DT_INVALID);

  for (const auto& input : oc_subgraph->inputs) {
    const Node* src_node = input.first.node;
    int src_slot = input.first.index;
    int input_index = input.second;

    DataType dtype = src_node->output_type(src_slot);
    dtypes[input_index] = dtype;
  }

  NodeDef recv_def;
  NodeDefBuilder builder(strings::StrCat("outside_compilation_", subgraph_name,
                                         "_", oc_subgraph_name, "_recv"),
                         kRecvAtHostOp);
  builder.Device(device_);
  builder.Attr("Toutputs", dtypes);
  // The correct device_ordinal will be inserted during replication in a
  // subsequent rewrite.
  builder.Attr("device_ordinal", 0);
  builder.Attr("key", strings::StrCat("host_compute_channel_", subgraph_name,
                                      "_", oc_subgraph_name));
  builder.Attr(group_attribute, subgraph_name);
  builder.Attr(outside_compilation_attribute, oc_subgraph_name);
  builder.Input(host_compute_key_placeholder_->name(), 0, DT_STRING);
  Status s = builder.Finalize(&recv_def);
  if (!s.ok()) return s;

  oc_subgraph->recv_at_host = graph_out->AddNode(recv_def, &s);
  if (!s.ok()) return s;
  graph_out->AddEdge(host_compute_key_placeholder_, 0,
                     oc_subgraph->recv_at_host, 0);

  // Add a control dependency forcing the RecvAtHost to run before the subgraph
  // completes. This has no effect on execution order but prevents the
  // RecvAtHost being pruned.
  TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
  graph_out->AddControlEdge(oc_subgraph->recv_at_host, sequencer_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddSendFromHostNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    const string& group_attribute, const string& subgraph_name,
    const string& outside_compilation_attribute, const string& oc_subgraph_name,
    OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out) {
  if (host_compute_key_placeholder_ == nullptr) {
    TF_RETURN_IF_ERROR(AddHostComputeKeyPlaceholder(oc_subgraph, graph_out));
  }

  std::vector<DataType> dtypes(oc_subgraph->outputs_by_src.size(), DT_INVALID);
  std::vector<NodeDefBuilder::NodeOut> inputs(
      oc_subgraph->outputs_by_src.size());

  for (const auto& output : oc_subgraph->outputs_by_src) {
    const Node* src_node = output.first.node;
    Node* src_image = node_images.at(src_node);
    int src_slot = output.first.index;
    int output_index = output.second.index;

    DataType dtype = src_node->output_type(src_slot);
    dtypes[output_index] = dtype;
    inputs[output_index].Reset(src_image->name(), src_slot, dtype);
  }

  NodeDef send_def;
  NodeDefBuilder builder(strings::StrCat("outside_compilation_", subgraph_name,
                                         "_", oc_subgraph_name, "_send"),
                         kSendFromHostOp);
  builder.Device(device_);
  builder.Attr("Tinputs", dtypes);
  builder.Attr("key", strings::StrCat("host_compute_channel_", subgraph_name,
                                      "_", oc_subgraph_name));
  // The correct device_ordinal will be inserted during replication in a
  // subsequent rewrite.
  builder.Attr("device_ordinal", 0);
  builder.Attr(group_attribute, subgraph_name);
  builder.Attr(outside_compilation_attribute, oc_subgraph_name);
  builder.Input(inputs);
  builder.Input(host_compute_key_placeholder_->name(), 0, DT_STRING);
  Status s = builder.Finalize(&send_def);
  if (!s.ok()) return s;

  oc_subgraph->send_from_host = graph_out->AddNode(send_def, &s);
  if (!s.ok()) return s;
  graph_out->AddEdge(host_compute_key_placeholder_, 0,
                     oc_subgraph->send_from_host, inputs.size());

  // Add a control dependency forcing the SendFromHost to run before the
  // subgraph completes. This has no effect on execution order but prevents the
  // RecvAtHost being pruned.
  TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
  graph_out->AddControlEdge(oc_subgraph->send_from_host, sequencer_);

  return Status::OK();
}

Status Encapsulator::Subgraph::AddOutsideCompilationHostIONodes(
    const string& group_attribute, const string& subgraph_name,
    const string& outside_compilation_attribute,
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& outside_compilation_subgraph_entry :
       outside_compilation_subgraphs_) {
    const string& oc_name = outside_compilation_subgraph_entry.first;
    OutsideCompilationSubgraph& oc_subgraph =
        outside_compilation_subgraph_entry.second;

    if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty()) {
      TF_RETURN_IF_ERROR(AddRecvAtHostNode(group_attribute, subgraph_name,
                                           outside_compilation_attribute,
                                           oc_name, &oc_subgraph, graph_out));
    }

    if (!oc_subgraph.outputs_by_src.empty() ||
        !oc_subgraph.control_outputs.empty()) {
      TF_RETURN_IF_ERROR(AddSendFromHostNode(
          node_images, group_attribute, subgraph_name,
          outside_compilation_attribute, oc_name, &oc_subgraph, graph_out));
    }
  }
  return Status::OK();
}

void Encapsulator::Subgraph::GetOutsideCompilationSubgraphNames(
    std::vector<string>* names) const {
  for (auto& entry : outside_compilation_subgraphs_) {
    names->push_back(entry.first);
  }
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

Status Encapsulator::SplitIntoSubgraphs(FunctionLibraryDefinition* library) {
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
    TF_RETURN_IF_ERROR(subgraph.AddHostComputes(entry.first, node_images));
  }

  MarkGuaranteedConstants(*graph_in_, src_arg_pairs);

  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    FixupSourceAndSinkEdges(subgraph.GetGraph());
    // Verify that the graph has well-formed control flow structure.
    std::vector<ControlFlowInfo> dummy;
    TF_RETURN_IF_ERROR(BuildControlFlowInfo(subgraph.GetGraph(), &dummy));
  }

  if (VLOG_IS_ON(1)) {
    // Dump subgraphs.
    for (auto& entry : subgraphs_) {
      dump_graph::DumpGraphToFile(
          strings::StrCat("encapsulate_subgraphs_subgraph_", entry.first),
          *entry.second.GetGraph(), library);
    }
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
    Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    string outside_compilation_id;
    TF_RETURN_IF_ERROR(
        GetFunctionNameAttr(node, &func_id, &outside_compilation_id));

    // Don't copy nodes that are going to be encapsulated.
    if (IsInSubgraph(func_id, outside_compilation_id)) continue;

    Node* image = graph_out->CopyNode(node);
    (*node_images)[node] = image;
  }
  (*node_images)[graph_in_->source_node()] = graph_out->source_node();
  (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
  return Status::OK();
}

Status Encapsulator::AddFunctionCallNodes(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    TF_RETURN_IF_ERROR(
        subgraph_entry.second.AddFunctionCallNode(node_images, graph_out));
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
        group_attribute_, subgraph_name, outside_compilation_attribute_,
        node_images, graph_out));
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
      *src_image = subgraphs_.at(src_func_id).GetCallNode();
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
      *dst_image = subgraphs_.at(dst_func_id).GetCallNode();
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
    const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out,
    std::unordered_set<std::pair<OutputTensor, InputTensor>,
                       OutputInputTensorPairHasher>* edges_added) {
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
    if (edges_added
            ->emplace(OutputTensor(src_image, -1), InputTensor(dst_image, -1))
            .second) {
      graph_out->AddControlEdge(src_image, dst_image);
    }

    return Status::OK();
  }

  int src_output =
      FindOutputSlotOfEdgeSrc(src_func_id, src_outside_compilation_id,
                              dst_func_id, dst_outside_compilation_id, edge);

  int dst_input =
      FindOutputSlotOfEdgeDst(src_func_id, src_outside_compilation_id,
                              dst_func_id, dst_outside_compilation_id, edge);

  // Add the edge, if we have not already added it.
  if (edges_added
          ->emplace(OutputTensor(src_image, src_output),
                    InputTensor(dst_image, dst_input))
          .second) {
    graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
  }
  return Status::OK();
}

Status Encapsulator::AddCallNodeDependencies(Graph* graph_out) {
  for (const auto& ancestors : subgraph_ancestors_) {
    const string& subgraph = ancestors.first;
    for (const string& ancestor : ancestors.second) {
      graph_out->AddControlEdge(subgraphs_[ancestor].GetCallNode(),
                                subgraphs_[subgraph].GetCallNode());
    }
  }
  return Status::OK();
}

Status Encapsulator::AddEdgesToOutputGraph(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  // Set of edges already added to the output graph, represented as (src, dst)
  // pairs. We use the set to deduplicate edges; multiple edges in the input
  // graph may map to one edge in the output graph.
  std::unordered_set<std::pair<OutputTensor, InputTensor>,
                     OutputInputTensorPairHasher>
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
      continue;
    }

    // We have an edge that crosses a cluster boundary or is entirely within the
    // unclustered graph.
    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(
        edge, src_func_id, src_outside_compilation_id, dst_func_id,
        dst_outside_compilation_id, node_images, graph_out, &edges_added));
  }

  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    subgraph.ConnectSequencerToCallNode(graph_out);
  }
  TF_RETURN_IF_ERROR(AddCallNodeDependencies(graph_out));

  return Status::OK();
}

namespace {

// Adds a dummy Const node to graph_out. The "constant" has the type of
// data_type and the shape indicated in 'shape'. The dummy node is not a valid
// Const node because it does not have any value defined, but this doesn't
// matter because it will only be used subsequently for shape inference. (It
// would be possible to add a switch statement over data_type to create a value
// for the constant, but that would entail maintaining the logic as new types
// are added, and is not necessary.) If the node being replaced was within a
// control flow frame, adds appropriate Enter nodes so that the use of the Const
// is well-formed.
Node* AddDummyShapedNode(const Node* src_node, int src_port,
                         const std::vector<ControlFlowInfo>& control_flow_info,
                         const TensorShapeProto& shape, Graph* graph_out) {
  DataType data_type = src_node->output_type(src_port);
  TensorProto dummy_proto;
  dummy_proto.set_dtype(data_type);
  *dummy_proto.mutable_tensor_shape() = shape;
  // Don't set any value field in the proto, since it is only going to be used
  // for shape inference.

  GraphDefBuilder::Options options(graph_out, /*status=*/nullptr);
  NodeBuilder node_builder(options.GetNameForOp("KnownShape"), "Const",
                           options.op_registry());
  node_builder.Attr("dtype", data_type).Attr("value", dummy_proto);
  Node* node = options.FinalizeBuilder(&node_builder);
  // Add any Enter nodes required to bring the constant to the correct control
  // flow frame.
  while (!control_flow_info[src_node->id()].frame_name.empty()) {
    NodeBuilder enter_builder(options.GetNameForOp("Enter"), "Enter",
                              options.op_registry());
    enter_builder.Attr("frame_name",
                       control_flow_info[src_node->id()].frame_name);
    enter_builder.Attr("is_constant", true);
    enter_builder.Input(node, 0);
    Node* enter_node = options.FinalizeBuilder(&enter_builder);
    // Adopt the new Enter node as the value in the current frame.
    node = enter_node;
    // Recurse to the parent frame to see if more Enter nodes need to be added.
    src_node = control_flow_info[src_node->id()].parent_frame;
  }
  return node;
}

// Adds a copy of node_in to graph_out and adds the mapping to
// copied_node_images.
Status CopyShapeInferenceNodeToGraph(
    Node* node_in, const Node* send_node,
    const std::unordered_map<Node*, Node*>& dummy_node_images,
    FunctionLibraryDefinition* library,
    std::unordered_map<Node*, Node*>* copied_node_images, Graph* graph_out) {
  // Once all the ancestor nodes have been added to graph_out, add this node
  // and connect it to its ancestors.
  Node* node_out = graph_out->CopyNode(node_in);
  (*copied_node_images)[node_in] = node_out;
  // Don't bother to build the shape inference graph if there's a node with no
  // shape inference function, since it would just result in an error later at
  // compile time.
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(library->LookUp(node_in->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr) {
    return errors::InvalidArgument(
        "Shape inference is not possible for outside_compilation "
        "SendFromHost node ",
        send_node->name(), " because it depends on node ", node_in->name(),
        " which does not have a shape inference function registered.");
  }
  // Add all the edges to the newly copied node.
  for (const Edge* in_edge : node_in->in_edges()) {
    if (!in_edge->IsControlEdge()) {
      Node* src = in_edge->src();
      const auto iter = dummy_node_images.find(src);
      if (iter == dummy_node_images.end()) {
        // The src is a copied node so use the original output port.
        graph_out->AddEdge((*copied_node_images)[in_edge->src()],
                           in_edge->src_output(), node_out,
                           in_edge->dst_input());
      } else {
        // The src is a dummy node so use output port 0.
        graph_out->AddEdge(iter->second, 0, node_out, in_edge->dst_input());
      }
    }
  }
  // Work around the fact that Enter nodes refuse to propagate shape information
  // unless they are marked loop invariant. Since we are never going to execute
  // this graph, marking them all loop invariant is fine.
  if (node_out->type_string() == "Enter") {
    node_out->ClearAttr("is_constant");
    node_out->AddAttr("is_constant", true);
  }
  return Status::OK();
}

}  // namespace

Status Encapsulator::DoStaticShapeInferenceForOutsideCompilationSend(
    const Graph& graph_in, const BackEdgeHelper& back_edge_helper,
    const ShapeRefiner& shape_refiner,
    const std::unordered_set<string>& recv_at_host_nodes, Node* send_node,
    FunctionLibraryDefinition* library,
    std::vector<TensorShapeProto>* static_shape_out,
    std::unique_ptr<Graph>* graph_out) {
  // Get the control flow structure of the input graph so we can build
  // well-formed output graphs.
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(&graph_in, &control_flow_info));

  // Maps from nodes in graph_in to nodes in graph_out.
  //
  // When an edge has fully defined shape the source node in graph_in is
  // replaced in graph_out by a dummy constant node. The mapping from nodes
  // in graph_in to dummy nodes is stored in dummy_node_images.
  //
  // When a node in graph_in has at least one ancestor that doesn't have fully
  // defined shape, it is copied into graph_out. The mapping from nodes in
  // graph_in to copied nodes is stored in copied_node_images.
  //
  // The two types of node are treated differently because, when adding edges to
  // graph_out, an output from a dummy node always uses port 0, whereas an
  // output from a copied node uses the same port that was used in graph_in.
  std::unordered_map<Node*, Node*> dummy_node_images;
  std::unordered_map<Node*, Node*> copied_node_images;

  graph_out->reset(new Graph(graph_in.op_registry()));
  (*graph_out)->set_versions(graph_in.versions());
  // The final input to the send node is the dynamic key, which we don't include
  // in the static shapes.
  static_shape_out->resize(send_node->num_inputs() - 1);

  // We don't use the standard ReverseDFS because we want to cut off traversal
  // whenever we find an output with fully defined shape.
  struct Work {
    Node* node;
    bool leave;  // Are we entering or leaving node?
  };
  std::vector<Work> stack({{send_node, false}});
  std::vector<bool> visited(graph_in.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();
    Node* n = w.node;

    if (w.leave) {
      TF_RETURN_IF_ERROR(CopyShapeInferenceNodeToGraph(
          n, send_node, dummy_node_images, library, &copied_node_images,
          graph_out->get()));
    } else {
      if (visited[n->id()]) continue;
      visited[n->id()] = true;

      // Arrange to revisit when all done with all inputs.
      stack.push_back(Work{n, true});

      bool has_parent_with_unknown_shape = false;
      for (const Edge* in_edge : n->in_edges()) {
        if (!in_edge->IsControlEdge()) {
          Node* src_node = in_edge->src();
          int src_port = in_edge->src_output();
          shape_inference::InferenceContext* context =
              shape_refiner.GetContext(src_node);
          shape_inference::ShapeHandle shape = context->output(src_port);
          if (context->FullyDefined(shape)) {
            // This ancestor has known shape, so instead of adding it to the
            // stack, add a dummy node with that shape to graph_out and
            // continue.
            TensorShapeProto proto;
            context->ShapeHandleToProto(shape, &proto);
            VLOG(2) << "Node " << src_node->name()
                    << " has known shape: " << proto.DebugString();
            if (dummy_node_images.find(src_node) == dummy_node_images.end()) {
              dummy_node_images[src_node] =
                  AddDummyShapedNode(src_node, src_port, control_flow_info,
                                     proto, graph_out->get());
            }
            // The final input to the send node is the dynamic key, which we
            // don't include in the static shapes.
            if (n == send_node &&
                in_edge->dst_input() < static_shape_out->size()) {
              (*static_shape_out)[in_edge->dst_input()] = proto;
            }
          } else {
            has_parent_with_unknown_shape = true;
            if (!visited[src_node->id()]) {
              if (VLOG_IS_ON(2)) {
                TensorShapeProto proto;
                context->ShapeHandleToProto(shape, &proto);
                VLOG(2) << "Node " << src_node->name()
                        << " has unknown shape: " << proto.DebugString();
              }
              stack.push_back({src_node, false});
            }
          }
        }
      }
      if (!has_parent_with_unknown_shape) {
        if (n == send_node) {
          // The shapes of all the inputs to send_node are statically known. We
          // won't have to do any inference at compile time so return now: the
          // shapes were stored in static_shape_out above.
          graph_out->reset();
          return Status::OK();
        } else {
          // Any shape that is being processed is either the original send node
          // or has at least one output with statically-unknown shape. If the
          // latter and it doesn't have any inputs with statically-unknown
          // shape, then check that it is of the recv nodes that we can fill in
          // the shape of at run-time later. If it isn't one of those, then we
          // won't have any additional knowledge at compile time, so we already
          // know we won't be able to do shape inference and we can return an
          // error now.
          if (recv_at_host_nodes.find(n->name()) == recv_at_host_nodes.end()) {
            return errors::InvalidArgument(
                "Shape inference is not possible for outside_compilation "
                "SendFromHost node ",
                send_node->name(), " because shape of node ", n->name(),
                " will not be known at compilation time.");
          }
        }
      }
    }
  }

  for (const auto edge : back_edge_helper.RemovedEdges()) {
    if (copied_node_images.find(edge.dst) != copied_node_images.end()) {
      // The destination of this back edge was added to the inference graph, so
      // fix it up.
      Node* dst = copied_node_images[edge.dst];
      if (dst->type_string() != "Merge") {
        return errors::InvalidArgument(
            "outside_compilation cluster contains a back-edge to node ",
            dst->name(), " of type ", dst->type_string(),
            ". The analysis pass only supports back-edges to Merge nodes.");
      }
      const Edge* existing_input_edge;
      if (edge.dst_input != 1 || dst->num_inputs() != 2 ||
          !dst->input_edge(0, &existing_input_edge).ok()) {
        // TODO(misard) if we see graphs built with a different structure, relax
        // this constraint. Leaving it here for now to avoid writing unnecessary
        // complex code since we believe graphs generated by front ends all have
        // the back edge as the second input to the merge node.
        return errors::Internal(
            "Internal assumption failed while rewriting an outside_compilation "
            "cluster that contains a while loop. Logic assumes back-edge is to "
            "port 1 of a 2-input "
            "Merge node.");
      }
      // Connect the existing edge to both inputs of the Merge node so that the
      // graph will be well-formed.
      (*graph_out)
          ->AddEdge(existing_input_edge->src(),
                    existing_input_edge->src_output(), dst, edge.dst_input);
    }
  }

  return Status::OK();
}

namespace {

// Helper struct for building cluster dependencies and also debugging cycles in
// the dependencies. While computing dependencies we construct a mapping from
// Node* to PathDetails.
struct PathDetails {
  struct SubgraphAndCluster {
    string subgraph;
    string outside_compilation_cluster;
    bool operator==(const SubgraphAndCluster& other) const {
      return subgraph == other.subgraph &&
             outside_compilation_cluster == other.outside_compilation_cluster;
    }
  };

  struct SubgraphAndClusterHash {
    inline std::size_t operator()(const SubgraphAndCluster& v) const {
      return hash<string>()(
          strings::StrCat(v.subgraph, v.outside_compilation_cluster));
    }
  };

  typedef std::unordered_set<SubgraphAndCluster, SubgraphAndClusterHash>
      SubgraphAndClusterSet;

  // Returns the set of (subgraph, oc_cluster) pairs that should be recorded as
  // ancestors for any successor of this node. If the node is in the outer
  // graph, it returns the transitive union of the ancestors of the node's
  // inputs. If the node is in an outside_compilation cluster, it returns just
  // that cluster. If the node is compiled, it returns the empty set.
  SubgraphAndClusterSet AncestorsForSuccessor() {
    if (subgraph.empty()) {
      return ancestor_clusters;
    } else if (outside_compilation_cluster.empty()) {
      return SubgraphAndClusterSet();
    } else {
      SubgraphAndCluster entry;
      entry.subgraph = subgraph;
      entry.outside_compilation_cluster = outside_compilation_cluster;
      return SubgraphAndClusterSet({entry});
    }
  }

  // The transitive union of the ancestor's of this node's inputs. This is only
  // saved for debugging in order to print out enough information to debug a
  // discovered cycle.
  SubgraphAndClusterSet ancestor_clusters;
  // The subgraph attr on this node.
  string subgraph;
  // The outside_compilation attr on this node.
  string outside_compilation_cluster;
};

// Adds an edge from ancestor to successor to the cycle detector, and returns an
// error if that edge causes the formation of a cycle. In the error case, logs
// the contents of the node_ancestors_map to facilitate debugging.
Status CheckClusterDependencyForCycles(
    const string& ancestor, const string& successor,
    const std::unordered_map<string, std::unordered_set<string>>& ancestors,
    const std::unordered_map<Node*, PathDetails>& node_ancestors_map,
    GraphCycles* cycle_detector, std::map<string, int>* cycle_detector_map) {
  if (cycle_detector_map->find(ancestor) == cycle_detector_map->end()) {
    (*cycle_detector_map)[ancestor] = cycle_detector->NewNode();
  }
  if (cycle_detector_map->find(successor) == cycle_detector_map->end()) {
    (*cycle_detector_map)[successor] = cycle_detector->NewNode();
  }

  if (!cycle_detector->InsertEdge((*cycle_detector_map)[ancestor],
                                  (*cycle_detector_map)[successor])) {
    LOG(ERROR) << "Cycle in outside_compilation clusters";
    for (const auto& cluster : ancestors) {
      LOG(ERROR) << "Cluster " << cluster.first << " depends on:";
      for (const auto& ancestor : cluster.second) {
        LOG(ERROR) << "  " << ancestor;
      }
    }
    for (const auto& node_ancestors : node_ancestors_map) {
      LOG(ERROR) << "Node " << node_ancestors.first->name() << " ("
                 << node_ancestors.second.subgraph << ";"
                 << node_ancestors.second.outside_compilation_cluster
                 << ") has ancestor clusters:";
      for (const auto& ancestor : node_ancestors.second.ancestor_clusters) {
        LOG(ERROR) << "  " << ancestor.subgraph << ";"
                   << ancestor.outside_compilation_cluster;
      }
    }
    return errors::InvalidArgument(
        "Can't compile outside_compilation clusters because there is a "
        "dependency cycle: see error log for details.");
  }
  return Status::OK();
}

}  // namespace

Status Encapsulator::FindClusterDependencies() {
  // Map from nodes to ancestor details. A node is entered into the map if it is
  // in a compilation subgraph, and outside_compilation cluster, or appears on a
  // path in the outer graph leading from an outside_compilation subgraph.
  std::unordered_map<Node*, PathDetails> node_ancestors_map;
  // We check that clusters are acyclic using this cycle detector.
  GraphCycles cycle_detector;
  // Map from cluster name to cycle detector node id.
  std::map<string, int> cycle_detector_map;
  // Process the nodes in topologically-sorted order.
  std::vector<Node*> nodes;
  GetReversePostOrder(*graph_in_, &nodes);
  for (Node* node : nodes) {
    string subgraph_name;
    string oc_cluster;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &subgraph_name, &oc_cluster));
    // First create an entry in the ancestors map if the node is in a compiled
    // subgraph or outside_compilation cluster, or if any incoming edge is from
    // a node with an ancestor map entry; and find the union of all the
    // ancestors.
    if (!subgraph_name.empty()) {
      node_ancestors_map[node].subgraph = subgraph_name;
      node_ancestors_map[node].outside_compilation_cluster = oc_cluster;
    }
    for (Node* src : node->in_nodes()) {
      const auto iter = node_ancestors_map.find(src);
      if (iter != node_ancestors_map.end()) {
        const auto& ancestors_to_follow = iter->second.AncestorsForSuccessor();
        for (const auto& ancestor : ancestors_to_follow) {
          if (ancestor.subgraph != subgraph_name ||
              ancestor.outside_compilation_cluster != oc_cluster) {
            node_ancestors_map[node].ancestor_clusters.insert(ancestor);
          }
        }
      }
    }
    if (!subgraph_name.empty()) {
      // The node is in a compiled subgraph or an outside_compilation cluster.
      if (oc_cluster.empty()) {
        // The node is not in an outside_compilation cluster. Record the
        // subgraph's ancestor dependencies.
        for (const auto& cluster : node_ancestors_map[node].ancestor_clusters) {
          if (cluster.subgraph != subgraph_name) {
            subgraph_ancestors_[subgraph_name].insert(cluster.subgraph);
            TF_RETURN_IF_ERROR(CheckClusterDependencyForCycles(
                cluster.subgraph, subgraph_name, subgraph_ancestors_,
                node_ancestors_map, &cycle_detector, &cycle_detector_map));
          }
        }
      } else {
        Subgraph& subgraph = subgraphs_[subgraph_name];
        // The node is in an outside_compilation cluster. Record the cluster
        // and/or subgraph ancestor dependencies.
        for (const auto& cluster : node_ancestors_map[node].ancestor_clusters) {
          if (cluster.subgraph == subgraph_name) {
            // The ancestor is in the same subgraph.
            if (cluster.outside_compilation_cluster != oc_cluster) {
              // But not in the same oc_cluster, so record the dependency.
              subgraph.RecordOutsideCompilationDependency(
                  oc_cluster, cluster.outside_compilation_cluster);
              TF_RETURN_IF_ERROR(CheckClusterDependencyForCycles(
                  cluster.outside_compilation_cluster, oc_cluster,
                  subgraph.OutsideCompilationAncestorMap(), node_ancestors_map,
                  &cycle_detector, &cycle_detector_map));
            }
          } else {
            // The ancestor is in a different subgraph, so record the
            // dependency.
            subgraph_ancestors_[subgraph_name].insert(cluster.subgraph);
            TF_RETURN_IF_ERROR(CheckClusterDependencyForCycles(
                cluster.subgraph, subgraph_name, subgraph_ancestors_,
                node_ancestors_map, &cycle_detector, &cycle_detector_map));
          }
        }
      }
    }
  }
  if (VLOG_IS_ON(2)) {
    // Print debug information.
    VLOG(2) << "node_ancestors_map:";
    for (const auto& node_iter : node_ancestors_map) {
      VLOG(2) << "\t" << node_iter.first->name() << ": subgraph = '"
              << node_iter.second.subgraph
              << "', outside_compilation_cluster = '"
              << node_iter.second.outside_compilation_cluster
              << "', ancestor_clusters: "
              << (node_iter.second.ancestor_clusters.empty() ? "(empty)" : "");
      for (const auto& cluster_iter : node_iter.second.ancestor_clusters) {
        VLOG(2) << "\t\tsubgraph = '" << cluster_iter.subgraph
                << "', outside_compilation_cluster = '"
                << cluster_iter.outside_compilation_cluster << "'";
      }
    }
  }
  return Status::OK();
}

Status Encapsulator::MakePrunedGraphCopyAndInline(
    const Graph& graph, const std::vector<Node*>& sink_nodes,
    std::unique_ptr<Graph>* pruned_graph,
    std::unordered_map<const Node*, Node*>* node_images,
    FunctionLibraryDefinition* library) {
  // First copy all ancestor nodes of sink_nodes into a new graph.
  pruned_graph->reset(new Graph(library));
  (*pruned_graph)->set_versions(graph.versions());
  ReverseDFSFrom(graph, sink_nodes,
                 /*enter=*/nullptr,
                 /*leave=*/[&](Node* n) {
                   if (!n->IsSource()) {
                     Node* copied = (*pruned_graph)->CopyNode(n);
                     node_images->emplace(n, copied);
                   }
                 });

  // Add all the edges between copied nodes.
  for (auto entry : *node_images) {
    const Node* orig = entry.first;
    Node* image = entry.second;
    for (const Edge* out_edge : orig->out_edges()) {
      auto iter = node_images->find(out_edge->dst());
      if (iter != node_images->end()) {
        // The source and destination are both in the copied graph.
        (*pruned_graph)
            ->AddEdge(image, out_edge->src_output(), iter->second,
                      out_edge->dst_input());
      }
    }
  }

  // Find all the function call nodes, and inline them.
  std::vector<Node*> function_nodes;
  for (auto node : (*pruned_graph)->nodes()) {
    const OpRegistrationData* op_reg_data;
    TF_RETURN_IF_ERROR(library->LookUp(node->type_string(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      function_nodes.push_back(node);
    }
  }
  for (auto node : function_nodes) {
    VLOG(2) << "Inlining function " << node->name();
    const FunctionDef* fdef = library->Find(node->type_string());
    if (fdef == nullptr) {
      return errors::Internal("Failed to find function ", node->type_string(),
                              " in function library.");
    }
    FunctionBody* fbody = nullptr;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*fdef, node->attrs(), library,
                                [library](const string& op, const OpDef** sig) {
                                  return library->LookUpOpDef(op, sig);
                                },
                                &fbody));
    InlineFunctionBody(*library, pruned_graph->get(), node, fbody);
    delete fbody;
  }

  return Status::OK();
}

Status Encapsulator::MakeGraphForOutsideCompilationSends(
    const Graph& graph, std::unique_ptr<Graph>* pruned_graph,
    BackEdgeHelper* back_edge_helper, ShapeRefiner* shape_refiner,
    std::unordered_map<const Node*, Node*>* node_images,
    FunctionLibraryDefinition* library) {
  // Find all the send_from_host nodes in all subgraphs, to use as roots for the
  // pruning.
  std::vector<Node*> send_from_host_nodes;
  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    std::vector<string> outside_compilation_names;
    subgraph.GetOutsideCompilationSubgraphNames(&outside_compilation_names);
    for (const auto& name : outside_compilation_names) {
      Node* send_node = subgraph.GetSendFromHostNode(name);
      if (send_node != nullptr) {
        send_from_host_nodes.push_back(send_node);
      }
    }
  }

  // Make a copy of all the graph nodes needed to evaluate the send_from_host
  // nodes, inlining any functions as needed.
  TF_RETURN_IF_ERROR(MakePrunedGraphCopyAndInline(
      graph, send_from_host_nodes, pruned_graph, node_images, library));
  FixupSourceAndSinkEdges(pruned_graph->get());

  // Remove back edges from any cycles in the pruned graph to simplify shape
  // inference traversal. They will be fixed up in the per-subgraph shape
  // inference graphs stored in the function library.
  TF_RETURN_IF_ERROR(back_edge_helper->Remove(pruned_graph->get()));

  // Perform shape inference on the pruned graph.
  shape_refiner->set_require_shape_inference_fns(false);
  std::vector<Node*> post_order;
  GetReversePostOrder(*(*pruned_graph), &post_order);
  for (auto node : post_order) {
    // Ignore the status returned by the shape_refiner. At this point we want
    // the best effort shapes, even if no shape function is registered for a
    // node.
    Status status = shape_refiner->AddNode(node);
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node: " << status;
    }
  }

  return Status::OK();
}

Status Encapsulator::GetShapeInfoForOutsideCompilationSends(
    Graph* graph_out, FunctionLibraryDefinition* library) {
  BackEdgeHelper back_edge_helper;
  std::unique_ptr<Graph> pruned_graph;
  ShapeRefiner shape_refiner(graph_out->versions(), graph_out->op_registry());
  std::unordered_map<const Node*, Node*> node_images;
  TF_RETURN_IF_ERROR(MakeGraphForOutsideCompilationSends(
      *graph_out, &pruned_graph, &back_edge_helper, &shape_refiner,
      &node_images, library));

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("pruned_graph_for_shape_inference",
                                *pruned_graph, library);
  }

  for (auto& subgraph_entry : subgraphs_) {
    const string& subgraph_name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;
    // Find all the recv_at_host nodes in this subgraph.
    std::vector<string> outside_compilation_names;
    subgraph.GetOutsideCompilationSubgraphNames(&outside_compilation_names);
    std::unordered_set<string> recv_at_host_names;
    for (const auto& oc_name : outside_compilation_names) {
      Node* recv_node = subgraph.GetRecvAtHostNode(oc_name);
      if (recv_node != nullptr) {
        recv_at_host_names.insert(recv_node->name());
      }
    }
    // For each send_from_host node, do as much shape inference as possible
    // without knowing the shape of the recv_at_host nodes, and store the
    // result, along with enough information to complete the job at compile time
    // once the recv_at_host shapes are known.
    for (const auto& oc_name : outside_compilation_names) {
      Node* send_node = subgraph.GetSendFromHostNode(oc_name);
      std::vector<TensorShapeProto> static_shape;
      std::unique_ptr<Graph> graph;
      if (send_node != nullptr) {
        TF_RETURN_IF_ERROR(DoStaticShapeInferenceForOutsideCompilationSend(
            *pruned_graph, back_edge_helper, shape_refiner, recv_at_host_names,
            node_images[send_node], library, &static_shape, &graph));
        if (graph == nullptr) {
          VLOG(2) << "Send node  " << send_node->name() << " shapes";
          for (int i = 0; i < static_shape.size(); ++i) {
            VLOG(2) << static_shape[i].DebugString();
          }
        } else {
          if (VLOG_IS_ON(2)) {
            GraphDef graphdef;
            graph->ToGraphDef(&graphdef);
            VLOG(2) << "Send node " << send_node->name() << " graph\n"
                    << graphdef.DebugString();
          }
        }
      }
      TF_RETURN_IF_ERROR(subgraph.AddShapeInferenceInfo(
          subgraph_name, oc_name, static_shape, graph.get(), library));
    }
    if (!outside_compilation_names.empty()) {
      TF_RETURN_IF_ERROR(subgraph.ReplaceFunctionDef(library));
    }
  }

  return Status::OK();
}

Status Encapsulator::BuildOutputGraph(Graph* graph_out,
                                      FunctionLibraryDefinition* library) {
  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node*, Node*> node_images;

  TF_RETURN_IF_ERROR(CopyNodesToOutputGraph(graph_out, &node_images));
  TF_RETURN_IF_ERROR(AddFunctionCallNodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(AddOutsideCompilationHostIONodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(AddEdgesToOutputGraph(node_images, graph_out));

  TF_RETURN_IF_ERROR(
      GetShapeInfoForOutsideCompilationSends(graph_out, library));

  return Status::OK();
}

}  // anonymous namespace

Status EncapsulateSubgraphsInFunctions(
    string group_attribute, string outside_compilation_attribute,
    const Graph& graph_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
    bool reuse_existing_functions, std::unique_ptr<Graph>* graph_out,
    FunctionLibraryDefinition* library) {
  Status s;

  Encapsulator encapsulator(std::move(group_attribute),
                            std::move(outside_compilation_attribute),
                            &graph_in);
  TF_RETURN_IF_ERROR(encapsulator.FindClusterDependencies());
  TF_RETURN_IF_ERROR(encapsulator.SplitIntoSubgraphs(library));

  TF_RETURN_IF_ERROR(encapsulator.BuildFunctionDefs(
      rewrite_subgraph_fn, reuse_existing_functions, library));

  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  TF_RETURN_IF_ERROR(encapsulator.BuildOutputGraph(out.get(), library));

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
  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("encapsulate_subgraphs_before", **options.graph,
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

  auto rewrite_subgraph =
      [flr](const std::vector<OutputTensor>& arg_source_tensors,
            std::unique_ptr<Graph>* subgraph,
            std::vector<int>* input_permutation,
            std::vector<int>* output_permutation, NodeDef* node) {
        // Optimize the subgraph.
        OptimizeGraph(flr, subgraph);

        const int num_args = input_permutation->size();
        std::vector<bool> const_args(num_args);
        TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
            **subgraph, &const_args, /*compile_time_const_nodes=*/nullptr));

        DataTypeVector arg_types(num_args);
        TF_RETURN_IF_ERROR(GetArgTypes(**subgraph, &arg_types));

        // Compute a permutation of the arguments such that the constant
        // arguments are first.
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
        TF_RETURN_IF_ERROR(
            RenumberArguments(subgraph->get(), *input_permutation));

        // TODO(phawkins): add a forward is-constant analysis, similarly split
        // outputs into host-memory constants and device-memory non-constants.

        AddNodeAttr(kXlaCompiledKernelAttr, true, node);
        AddNodeAttr(kXlaNumConstantArgsAttr, num_consts, node);
        AddNodeAttr(kXlaNumResourceArgsAttr, num_resources, node);
        return Status::OK();
      };

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kXlaClusterAttr, kXlaOutsideCompilationAttr, **options.graph,
          rewrite_subgraph, /*reuse_existing_functions=*/false, &graph_out,
          library),
      "EncapsulateSubgraphsPass failed");

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile("encapsulate_subgraphs_after", *graph_out,
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
