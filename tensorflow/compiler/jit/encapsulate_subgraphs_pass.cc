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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_factory.h"
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
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

const char* const kXlaCompiledKernelAttr = "_XlaCompiledKernel";
const char* const kXlaNumConstantArgsAttr = "_XlaNumConstantArgs";
const char* const kXlaNumResourceArgsAttr = "_XlaNumResourceArgs";
const char* const kXlaHostTransferSequencerAttr =
    "_xla_host_transfer_sequencer";
const char* const kXlaHasReferenceVarsAttr = "_XlaHasReferenceVars";

namespace {

bool AreAllParentsGuaranteedConst(
    const Node& n,
    const absl::flat_hash_set<const Node*>& runtime_const_nodes) {
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
  absl::flat_hash_set<const Node*> guaranteed_const_nodes;
  std::vector<const Node*> srcs;
  srcs.reserve(src_arg_pairs.size());
  for (const auto& src_arg : src_arg_pairs) {
    srcs.push_back(src_arg.first);
  }
  ReverseDFSFrom(
      graph, srcs, /*enter=*/nullptr,
      /*leave=*/[&guaranteed_const_nodes](const Node* n) {
        // TODO(vinuraja): Doesn't work in the presence of loops.
        if (AreAllParentsGuaranteedConst(*n, guaranteed_const_nodes)) {
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

class Encapsulator {
 public:
  Encapsulator(string group_attribute, Graph const* graph_in)
      : group_attribute_(std::move(group_attribute)), graph_in_(graph_in) {}

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
  // value.
  //
  // In the following simple example, A, B, ..., E are nodes in the original
  // graph. The group attributes g are each shown as either 0 or empty.
  //
  //  A  -->  B  -->  C  -->  D  -->  E
  //  g:      g:0     g:0     g:0     g:
  //
  // The example is rewritten to two graphs; one on the host and one to be
  // compiled. The host graph is as follows.
  //
  //  A  -->  Call  -->  E
  //
  // The compiled cluster is as follows.
  //
  //  Arg  --> B  --> C  --> D --> Retval
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

    // Returns the Node that the inputs and outputs of the function should be
    // wired up to.
    Node* GetCallNode() const;

    // Returns the index of the arg that the dst of edge should connect to.
    int GetArgIndexForEdge(const Edge* edge) const;

    // Returns the index of the result that the src of edge should connect to.
    int GetResultIndexForEdge(const Edge* edge) const;

    // Creates an _Arg node for the src node of edge, and add its index to
    // args_by_src_, if none exists yet. Also adds its index to args_by_dst_,
    // and adds the edge within the subgraph from the _Arg node to the image of
    // the dst node.
    Status RecordArg(const Edge* edge,
                     const std::unordered_map<const Node*, Node*>& node_images,
                     std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

    // Records the src of the given edge as a control result of the graph.
    // Used during graph to function conversion to tie control results to
    // the function signature.
    Status RecordControlResult(
        const Edge* edge,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates a _Retval node for the src node of edge, and add it to results_,
    // if none exists yet. If a new _Retval node is created, also adds the edge
    // within the subgraph from the src to the _Retval node.
    Status RecordResult(
        const Edge* edge,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates the sequencer node if it doesn't exist, adding it to graph_out.
    Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);

    // If there is a sequencer node, adds a control edge from the sequencer to
    // the call node.
    void ConnectSequencerToCallNode(Graph* graph_out);

    Status ReplaceFunctionDef(FunctionLibraryDefinition* library);

   private:
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

    // Set of node names that are the source of a control output of the
    // subgraph. We store strings here so that we can tolerate nodes being
    // removed from the graph.
    absl::flat_hash_set<string> control_output_nodes_;

    // NoOp node in the output graph that is sequenced after the call node.
    Node* sequencer_ = nullptr;
  };

  // Returns the key attribute associated with a node in attr. Sets either
  // result to the empty string if the respective attribute is not found.
  Status GetFunctionNameAttr(Node const* node, string* attr) const;

  // Copies edges local to a subgraph. Adds _Arg and _Retval nodes to
  // subgraphs for data edges that cross subgraph boundaries.
  Status CopySubgraphEdges(
      const std::unordered_map<const Node*, Node*>& node_images,
      std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

  // Copies all marked nodes to a subgraph. Does nothing for unmarked nodes.
  Status CopySubgraphNodes(std::unordered_map<const Node*, Node*>* node_images);

  // Copies all nodes that aren't in a compiled subgraph to the output graph.
  Status CopyNodesToOutputGraph(
      Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images);

  // Adds function call nodes for each compiled subgraph.
  Status AddFunctionCallNodes(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Finds the image of an edge source in the output graph. If the edge crosses
  // a subgraph boundary it is the output of a call node, otherwise it is a node
  // in the output graph.
  Status FindOutputImageOfEdgeSrc(
      const string& src_func_id, const string& dst_func_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_src_node, Node** src_image);

  // Finds an edge source slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the output of a call node, otherwise it
  // is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeSrc(const string& src_func_id,
                              const string& dst_func_id,
                              const Edge* edge);

  // Finds the image of an edge destination in the output graph. If the edge
  // crosses a subgraph boundary it is the input of a call node, otherwise it is
  // a node in the output graph.
  Status FindOutputImageOfEdgeDst(
      const string& src_func_id, const string& dst_func_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_dst_node, Node** dst_image);

  // Finds an edge destination slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the input of a call node, otherwise it is
  // a slot on a node in the output graph.
  int FindOutputSlotOfEdgeDst(const string& src_func_id,
                              const string& dst_func_id,
                              const Edge* edge);

  // Copies a single edge to the output graph. The edge is either entirely
  // within the output graph, or crosses into or out of a compiled subgraph.
  Status CopyEdgeToOutputGraph(
      const Edge* edge, const string& src_func_id, const string& dst_func_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out,
      std::unordered_set<std::pair<OutputTensor, InputTensor>,
                         OutputInputTensorPairHasher>* edges_added);

  // Adds all edges to the output graph.
  Status AddEdgesToOutputGraph(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Makes a copy of graph containing only nodes that are ancestors of at least
  // one node in send_from_host_nodes and store it in pruned_graph. On exit
  // nodes_images contains a mapping from nodes in graph to nodes in
  // pruned_graph. All functions in the copied graph are inlined.
  Status MakePrunedGraphCopyAndInline(
      const Graph& graph, const std::vector<Node*>& sink_nodes,
      std::unique_ptr<Graph>* pruned_graph,
      std::unordered_map<const Node*, Node*>* node_images,
      FunctionLibraryDefinition* library);

  const string group_attribute_;
  const Graph* graph_in_;

  std::unordered_map<string, Subgraph> subgraphs_;

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

Node* Encapsulator::Subgraph::MakeNodeImage(const Graph* graph_in, Node* node) {
  if (!graph_) {
    graph_.reset(new Graph(graph_in->op_registry()));
    graph_->set_versions(graph_in->versions());
  }

  // TODO(b/116981129): Enhance how the device for the encapsulated subgraph is
  // determined. In case of hard placement, ensure all the encapsulated nodes
  // have the same requested device, which in turn will be the requested device
  // for the entire encapsulated subgraph. In case of soft placement, use a
  // deterministic approach to fill in the requested device. Handle co-location
  // constraints similarly if they exist.
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
        absl::StrCat(src_node->name(), "_", src_slot, "_arg"), kArgOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    builder.Attr("T", dtype);
    builder.Attr("index", arg_index);
    Status s = builder.Finalize(&arg_def);
    if (!s.ok()) return s;

    TF_ASSIGN_OR_RETURN(Node * arg, graph_->AddNode(arg_def));
    src_arg_pairs->push_back({src_node, arg});
    args_.push_back(arg);
  }
  Node* dst_node = edge->dst();
  Node* dst_image = node_images.at(dst_node);
  int dst_slot = edge->dst_input();
  args_by_dst_[InputTensor(dst_node, dst_slot)] = arg_index;
  graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
  return OkStatus();
}

Status Encapsulator::Subgraph::RecordControlResult(
    const Edge* edge,
    const std::unordered_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  control_output_nodes_.insert(src_image->name());
  return OkStatus();
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
        absl::StrCat(src_node->name(), "_", src_slot, "_retval"), kRetValOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = src_node->output_type(src_slot);
    builder.Attr("T", dtype);
    builder.Attr("index", ret_index);
    builder.Input(src_image->name(), src_slot, dtype);
    Status s = builder.Finalize(&ret_def);
    if (!s.ok()) return s;
    TF_ASSIGN_OR_RETURN(Node * ret, graph_->AddNode(ret_def));
    graph_->AddEdge(src_image, src_slot, ret, 0);
  }
  return OkStatus();
}

Status Encapsulator::Subgraph::MakeSequencingNode(const string& subgraph_name,
                                                  Graph* graph_out) {
  if (sequencer_ == nullptr) {
    NodeDef seq_def;
    // TODO(shikharagarwal): What source node should we use for errors?
    NodeDefBuilder builder(absl::StrCat(subgraph_name, "_sequencer"), "NoOp");
    builder.Attr(kXlaHostTransferSequencerAttr, subgraph_name);
    builder.Device(device_);
    Status s = builder.Finalize(&seq_def);
    if (!s.ok()) return s;

    TF_ASSIGN_OR_RETURN(sequencer_, graph_out->AddNode(seq_def));
  }
  return OkStatus();
}

void Encapsulator::Subgraph::ConnectSequencerToCallNode(Graph* graph_out) {
  if (sequencer_ != nullptr) {
    VLOG(2) << "ConnectSequencerToCallNode";
    graph_out->AddControlEdge(sequencer_, call_node_,
                              /* allow_duplicates= */ true);
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
  auto lookup = [this](const Node* node) -> std::optional<string> {
    if (control_output_nodes_.contains(node->name())) {
      return absl::make_optional(node->name());
    }
    return std::nullopt;
  };
  // Verify that the graph has well-formed control flow structure.
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &dummy));
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, lookup, &fdef));

  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Build function def " << name;
    DumpGraphToFile(absl::StrCat("encapsulate_fdef_graph_", name), *graph_,
                    library);
    DumpFunctionDefToFile(absl::StrCat("encapsulate_fdef_", name), fdef);
  }

  const FunctionDef* original_fdef = library->Find(name);
  if (!reuse_existing_functions || original_fdef == nullptr) {
    TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
  } else if (!FunctionDefsEqual(*original_fdef, fdef)) {
    TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  }
  return OkStatus();
}

Status Encapsulator::Subgraph::ReplaceFunctionDef(
    FunctionLibraryDefinition* library) {
  const string& name = function_def_name_;

  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));

  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Replace function def " << name;
    DumpGraphToFile(absl::StrCat("replace_encapsulate_fdef_graph_", name),
                    *graph_, library);
    DumpFunctionDefToFile(absl::StrCat("replace_encapsulate_fdef_", name),
                          fdef);
  }

  TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  return OkStatus();
}

Status Encapsulator::Subgraph::AddFunctionCallNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  TF_ASSIGN_OR_RETURN(call_node_, graph_out->AddNode(call_node_def_));

  // Copy the assigned device and the key_annotation over.
  call_node_->set_assigned_device_name(device_);

  return OkStatus();
}

Status Encapsulator::GetFunctionNameAttr(Node const* node, string* attr) const {
  AttrSlice attrs = node->attrs();
  attr->clear();
  for (const auto& node_attr : attrs) {
    if (node_attr.first == group_attribute_) {
      TF_RETURN_IF_ERROR(AttrValueHasType(node_attr.second, "string"));
      *attr = node_attr.second.s();
      break;
    }
  }
  return OkStatus();
}

bool IsInSubgraph(const string& func_id) { return !func_id.empty(); }

Status Encapsulator::CopySubgraphNodes(
    std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id));
    if (!IsInSubgraph(func_id)) continue;

    Subgraph& subgraph = subgraphs_[func_id];
    Node* image = subgraph.MakeNodeImage(graph_in_, node);
    image->ClearAttr(group_attribute_);
    (*node_images)[node] = image;
  }
  return OkStatus();
}

Status Encapsulator::CopySubgraphEdges(
    const std::unordered_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id));
    string dst_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id));
    Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
    Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);

    // Copy edges that are local to a subgraph.
    if (IsInSubgraph(src_func_id) && IsInSubgraph(dst_func_id) &&
        src_func_id == dst_func_id) {
      Graph* g = subgraphs_[src_func_id].GetGraph();
      if (edge->IsControlEdge()) {
        g->AddControlEdge(src_image, dst_image,
                          /* allow_duplicates= */ true);
      } else {
        g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
      }
      continue;
    }

    // Record 'src' as an output of its subgraph, if applicable.
    if (IsInSubgraph(src_func_id)) {
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
      if (edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(src_subgraph.RecordControlResult(edge, node_images));
      } else {
        TF_RETURN_IF_ERROR(src_subgraph.RecordResult(edge, node_images));
      }
    }

    // Record 'dst' as an input of its subgraph, if applicable.
    if (IsInSubgraph(dst_func_id)) {
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
      // Ignore control edges entering the subgraph. We will lift them onto
      // the enclosing call operators in BuildOutputGraph().
      if (!edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(
            dst_subgraph.RecordArg(edge, node_images, src_arg_pairs));
      }
    }
  }
  return OkStatus();
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
  MarkGuaranteedConstants(*graph_in_, src_arg_pairs);

  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    FixupSourceAndSinkEdges(subgraph.GetGraph());
  }

  if (VLOG_IS_ON(1)) {
    // Dump subgraphs.
    for (auto& entry : subgraphs_) {
      DumpGraphToFile(
          absl::StrCat("encapsulate_subgraphs_subgraph_", entry.first),
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
  return OkStatus();
}

Status Encapsulator::CopyNodesToOutputGraph(
    Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id));

    // Don't copy nodes that are going to be encapsulated.
    if (IsInSubgraph(func_id)) continue;

    Node* image = graph_out->CopyNode(node);
    (*node_images)[node] = image;
  }
  (*node_images)[graph_in_->source_node()] = graph_out->source_node();
  (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
  return OkStatus();
}

Status Encapsulator::AddFunctionCallNodes(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    TF_RETURN_IF_ERROR(
        subgraph_entry.second.AddFunctionCallNode(node_images, graph_out));
  }
  return OkStatus();
}

Status Encapsulator::FindOutputImageOfEdgeSrc(
    const string& src_func_id, const string& dst_func_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_src_node, Node** src_image) {
  if (IsInSubgraph(src_func_id)) {
    // The edge is from a subgraph to a regular node in the output graph so
    // use the subgraph's call node output.
    *src_image = subgraphs_.at(src_func_id).GetCallNode();
  } else {
    // The source of the edge is in the output graph so use the node image in
    // the output graph.
    *src_image = node_images.at(original_src_node);
  }
  return OkStatus();
}

int Encapsulator::FindOutputSlotOfEdgeSrc(const string& src_func_id,
                                          const string& dst_func_id,
                                          const Edge* edge) {
  if (IsInSubgraph(src_func_id)) {
    const Subgraph& src_subgraph = subgraphs_.at(src_func_id);
    // 'src' is in a subgraph and 'dst' is a regular node in the output
    // graph. Use the corresponding call output instead.
    return src_subgraph.GetResultIndexForEdge(edge);
  } else {
    // The source of the edge is in the output graph so use the regular edge
    // slot.
    return edge->src_output();
  }
}

Status Encapsulator::FindOutputImageOfEdgeDst(
    const string& src_func_id, const string& dst_func_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_dst_node, Node** dst_image) {
  if (IsInSubgraph(dst_func_id)) {
    // The edge is to a subgraph from a regular node in the output graph so
    // use the subgraph's call node input.
    *dst_image = subgraphs_.at(dst_func_id).GetCallNode();
  } else {
    // The destination of the edge is in the output graph so use the node image
    // in the output graph.
    *dst_image = node_images.at(original_dst_node);
  }
  return OkStatus();
}

int Encapsulator::FindOutputSlotOfEdgeDst(const string& src_func_id,
                                          const string& dst_func_id,
                                          const Edge* edge) {
  if (IsInSubgraph(dst_func_id)) {
    const Subgraph& dst_subgraph = subgraphs_.at(dst_func_id);
      // 'dst' is in a subgraph and 'src' is a regular node in the output
      // graph. Use the corresponding call input instead.
      return dst_subgraph.GetArgIndexForEdge(edge);
  } else {
    // The destination of the edge is in the output graph so use the regular
    // edge slot.
    return edge->dst_input();
  }
}

Status Encapsulator::CopyEdgeToOutputGraph(
    const Edge* edge, const string& src_func_id, const string& dst_func_id,
    const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out,
    std::unordered_set<std::pair<OutputTensor, InputTensor>,
                       OutputInputTensorPairHasher>* edges_added) {
  Node* src_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(
      src_func_id, dst_func_id, node_images, edge->src(), &src_image));
  Node* dst_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(
      src_func_id, dst_func_id, node_images, edge->dst(), &dst_image));

  // If this is a control edge then copy it and return. Lift control edges onto
  // the enclosing call operator.
  if (edge->IsControlEdge()) {
    // Add the control edge, if we have not already added it, using the images
    // determined above (potentially call operators or RecvAtHost/SendFromHost).
    if (edges_added
            ->emplace(OutputTensor(src_image, -1), InputTensor(dst_image, -1))
            .second) {
      graph_out->AddControlEdge(src_image, dst_image,
                                /* allow_duplicates= */ true);
    }

    return OkStatus();
  }

  int src_output = FindOutputSlotOfEdgeSrc(src_func_id, dst_func_id, edge);

  int dst_input = FindOutputSlotOfEdgeDst(src_func_id, dst_func_id, edge);

  // Add the edge, if we have not already added it.
  if (edges_added
          ->emplace(OutputTensor(src_image, src_output),
                    InputTensor(dst_image, dst_input))
          .second) {
    graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
  }
  return OkStatus();
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
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id));
    string dst_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id));

    // Ignore edges that are strictly contained within one subgraph, unless
    // we are constructing parallel check graphs.
    if (IsInSubgraph(src_func_id) && IsInSubgraph(dst_func_id) &&
        src_func_id == dst_func_id) {
      continue;
    }

    // We have an edge that crosses a cluster boundary or is entirely within the
    // unclustered graph.
    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(
        edge, src_func_id, dst_func_id, node_images, graph_out, &edges_added));
  }

  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    subgraph.ConnectSequencerToCallNode(graph_out);
  }

  return OkStatus();
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
    NodeDebugInfo debug_info(*src_node);
    NodeBuilder enter_builder(options.GetNameForOp("Enter"), "Enter",
                              options.op_registry(), &debug_info);
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

}  // namespace

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
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*fdef, node->attrs(), library, &fbody));

    InlineFunctionBodyOptions inline_opts;
    TF_RETURN_IF_ERROR(InlineFunctionBody(*library, pruned_graph->get(), node,
                                          fbody.get(), inline_opts));
  }

  return OkStatus();
}

Status Encapsulator::BuildOutputGraph(Graph* graph_out,
                                      FunctionLibraryDefinition* library) {
  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node*, Node*> node_images;

  TF_RETURN_IF_ERROR(CopyNodesToOutputGraph(graph_out, &node_images));
  TF_RETURN_IF_ERROR(AddFunctionCallNodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(AddEdgesToOutputGraph(node_images, graph_out));

  return OkStatus();
}

}  // anonymous namespace

Status EncapsulateSubgraphsInFunctions(
    string group_attribute, const Graph& graph_in,
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
    std::unique_ptr<Graph>* graph_out, FunctionLibraryDefinition* library) {
  Encapsulator encapsulator(std::move(group_attribute),
                            &graph_in);
  TF_RETURN_IF_ERROR(encapsulator.SplitIntoSubgraphs(library));

  TF_RETURN_IF_ERROR(encapsulator.BuildFunctionDefs(
      rewrite_subgraph_fn, reuse_existing_functions, library));

  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  TF_RETURN_IF_ERROR(encapsulator.BuildOutputGraph(out.get(), library));

  *graph_out = std::move(out);
  return OkStatus();
}

// Finds the types of the _Arg nodes, indexed by position.
static Status GetArgTypes(const Graph& graph, DataTypeVector* types) {
  for (Node* n : graph.op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      const int num_types = types->size();
      if (index < 0 || index >= num_types) {
        return errors::InvalidArgument("Invalid argument number");
      }
      (*types)[index] = n->output_type(0);
    }
  }
  return OkStatus();
}

// Renumber the indices of _Arg nodes in a graph, according to
// 'permutation' that maps old indices to new indices.
static Status RenumberArguments(Graph* graph,
                                const std::vector<int>& permutation) {
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      const int permutation_size = permutation.size();
      if (index < 0 || index >= permutation_size) {
        return errors::InvalidArgument("Invalid argument number");
      }
      n->AddAttr("index", permutation[index]);
    }
  }
  return OkStatus();
}

Status EncapsulateSubgraphsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "EncapsulateSubgraphsPass::Run";
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_subgraphs_before", **options.graph,
                    options.flib_def);
  }

  // TODO(b/195757077): Remove this once there is a better way to disable
  // GraphOptimizationPasses that are not needed due to MLIR bridge.
  for (Node* n : (*options.graph)->nodes()) {
    // Skip the pass if we found TPUExecute or TPUExecuteAndUpdateVariables ops
    // in the graph, which indicates the graph is produced by TPU TF-XLA bridge
    // and doesn't require auto clustering.
    if (n->type_string() == "TPUExecute" ||
        n->type_string() == "TPUExecuteAndUpdateVariables") {
      return OkStatus();
    }
  }

  std::unique_ptr<Graph> graph_out;
  FunctionLibraryDefinition* const library = options.flib_def;

  // Constant folding below might need to run part of the function to compute
  // constants. Create an FunctionLibraryRuntime with a single CPU device
  // that can run the part of the function.
  // NOTE: If this turns out to be slow, we can cache the FLRs keyed by
  // `options`.
  SessionOptions session_options;
  auto* device_count = session_options.config.mutable_device_count();
  device_count->insert({"CPU", 1});
  std::vector<std::unique_ptr<Device>> devices;

  DeviceFactory* cpu_factory = DeviceFactory::GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered. Can't run EncapsulateSubgraphsPass");
  }
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  if (devices.empty()) {
    return errors::NotFound(
        "Failed to create a CPU device for EncapsulateSubgraphsPass");
  }

  std::unique_ptr<DeviceMgr> device_mgr =
      std::make_unique<StaticDeviceMgr>(std::move(devices));
  const auto* config = &options.session_options->config;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          device_mgr.get(), options.session_options->env,
          /*config=*/config, TF_GRAPH_DEF_VERSION, library,
          config->graph_options().optimizer_options()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");
  if (flr == nullptr) {
    return errors::Internal(
        "Failed to create and retrieve function library runtime to run "
        "constant folding");
  }

  auto rewrite_subgraph =
      [flr](const std::vector<OutputTensor>& arg_source_tensors,
            std::unique_ptr<Graph>* subgraph,
            std::vector<int>* input_permutation,
            std::vector<int>* output_permutation, NodeDef* node) {
        // Optimize the subgraph.
        // Do not constant fold nodes that output DT_VARIANT type tensors.
        // XLA does not support Const nodes of Variant type since it needs
        // to know the original ops to be able to compile them to the relevant
        // XLA form.
        // TODO(srbs): This filter is a little conservative. E.g. a subgraph of
        // the form:
        //                          Const
        //                            |
        // EmptyTensorList -> TensorListPushBack -> TensorListPopBack -> Op
        //                                                  |
        //                                        (Discard popped list)
        //
        // Would have been reduced to "Const -> Op" without this filter.
        // However since we are only allowed to specify the filter at the "Node"
        // level there is no good way to allow the above behavior. So we
        // disallow any sort of constant folding on Variant nodes for now.
        bool disable_constant_folding =
            GetBuildXlaOpsPassFlags()->tf_xla_disable_constant_folding;
        auto cf_consider_fn = [disable_constant_folding](const Node* n) {
          if (disable_constant_folding) return false;
          for (const auto& output_arg : n->op_def().output_arg()) {
            if (output_arg.type() == DT_VARIANT) {
              return false;
            }
          }
          return true;
        };
        GraphOptimizer::Options graph_optimizer_options;
        graph_optimizer_options.cf_consider_fn = cf_consider_fn;
        OptimizeGraph(flr, subgraph, graph_optimizer_options);

        const int num_args = input_permutation->size();
        std::vector<bool> const_args(num_args);
        TF_RETURN_IF_ERROR(
            BackwardsConstAnalysis(**subgraph, &const_args,
                                   /*compile_time_const_nodes=*/nullptr, flr));

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
        return OkStatus();
      };

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kXlaClusterAttr, **options.graph, rewrite_subgraph,
          /*reuse_existing_functions=*/false, &graph_out, library),
      "EncapsulateSubgraphsPass failed");
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_subgraphs_after", *graph_out,
                    options.flib_def);
  }

  *options.graph = std::move(graph_out);

  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> ref_related_nodes,
                      GetNodesRelatedToRefVariables(**options.graph, flr));
  for (Node* node : (*options.graph)->nodes()) {
    bool has_ref_vars = ref_related_nodes.contains(node);
    node->AddAttr(kXlaHasReferenceVarsAttr, has_ref_vars);
    VLOG(3) << "Has ref vars = " << has_ref_vars
            << ", node: " << node->def().DebugString();
  }
  return OkStatus();
}

bool IsXlaCompiledKernel(const Node& node) {
  bool is_compiled = false;
  bool has_compilation_attr =
      TryGetNodeAttr(node.attrs(), kXlaCompiledKernelAttr, &is_compiled) &&
      is_compiled;
  return has_compilation_attr ? is_compiled : false;
}

}  // namespace tensorflow
