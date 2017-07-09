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
#include <numeric>

#include "tensorflow/compiler/jit/graph_to_functiondef.h"
#include "tensorflow/compiler/jit/legacy_flags/encapsulate_subgraphs_pass_flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
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

// A node/slot pair.
// TODO(phawkins): is there a common definition of this?
struct NodeSlot {
  NodeSlot() : node(nullptr), slot(-1) {}
  NodeSlot(const Node* node, int slot) : node(node), slot(slot) {}

  const Node* node;
  int slot;

  bool operator==(const NodeSlot& other) const {
    return node == other.node && slot == other.slot;
  }

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

class Encapsulator {
 public:
  Encapsulator(string group_attribute, Graph const* graph_in)
      : group_attribute_(std::move(group_attribute)), graph_in_(graph_in) {}

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
  // Returns the key attribute associated with a node. Returns the empty string
  // if no key attribute is found.
  string GetFunctionNameAttr(const Node* node) const;

  // A subgraph of the input, all marked with a common 'group_attribute'
  // value.
  struct Subgraph {
    // The subgraph extracted from the input graph, suitable for being turned
    // into a FunctionDef. Inputs are fed by _Arg nodes, and outputs are
    // returned by _Retval nodes.
    std::unique_ptr<Graph> graph;

    // Which device are these nodes on? Used to assign a device to the call
    // node.
    string device;

    // NodeDef for the function call node.
    NodeDef call_node_def;

    // Function call node(s) in the output graph. Not owned.
    // If parallel_checking is enabled, 'call_node_inputs' is the function call
    // node to which inputs should be fed, and 'call_node_outputs' is the
    // parallel check op from which outputs should be read. If parallel checking
    // is disabled, both point to the function call node.
    Node* call_node_inputs;
    Node* call_node_outputs;

    // Maps from source (producer node/slot) and destination
    // (consumer node/slot) tensors in the input graph to _Arg numbers in
    // the subgraph. The source map is one-to-one, whereas the dest map may be
    // many-to-one.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> args_by_src;
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> args_by_dst;

    // The _Arg nodes in the subgraph, in order by argument number.
    std::vector<Node*> args;

    // Map from source tensor in the input graph to result #.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> results;
  };

  // Builds a ParallelCheck op that compares the output of the original subgraph
  // with the encapsulated subgraph.
  Status BuildParallelCheckOp(
      const std::unordered_map<const Node*, Node*>& node_images,
      const Subgraph& subgraph, Graph* graph_out, Node** parallel_check_op);

  const string group_attribute_;
  const Graph* graph_in_;

  std::unordered_map<string, Subgraph> subgraphs_;

  TF_DISALLOW_COPY_AND_ASSIGN(Encapsulator);
};

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";

// Returns the function name attached to 'node', or the empty string if there is
// none.
string Encapsulator::GetFunctionNameAttr(Node const* node) const {
  string attr;
  if (!GetNodeAttr(node->attrs(), group_attribute_, &attr).ok()) {
    attr.clear();
  }
  return attr;
}

Status Encapsulator::SplitIntoSubgraphs() {
  Status s;

  // Map from input graph nodes to subgraph nodes.
  std::unordered_map<Node*, Node*> node_images;

  // Copy all marked nodes to a subgraph. Do nothing for unmarked nodes.
  for (Node* node : graph_in_->op_nodes()) {
    string func_id = GetFunctionNameAttr(node);
    if (func_id.empty()) continue;

    Subgraph& subgraph = subgraphs_[func_id];
    if (!subgraph.graph) {
      subgraph.graph.reset(new Graph(graph_in_->op_registry()));
      subgraph.graph->set_versions(graph_in_->versions());
    }

    Node* image = subgraph.graph->CopyNode(node);
    image->ClearAttr(group_attribute_);
    node_images[node] = image;

    if (subgraph.device.empty()) {
      subgraph.device = node->assigned_device_name().empty()
                            ? node->requested_device()
                            : node->assigned_device_name();
    }
  }

  // Copy edges local to a subgraph. Add _Arg and _Retval nodes to subgraphs for
  // data edges that cross subgraph boundaries.
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id = GetFunctionNameAttr(edge->src());
    string dst_func_id = GetFunctionNameAttr(edge->dst());
    Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
    Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);

    // Copy edges that are local to a subgraph.
    if (!src_func_id.empty() && src_func_id == dst_func_id) {
      Graph* g = subgraphs_[src_func_id].graph.get();
      if (edge->IsControlEdge()) {
        g->AddControlEdge(src_image, dst_image);
      } else {
        g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
      }
      continue;
    }

    // Ignore cross-boundary control edges for right now. We will lift them
    // onto the enclosing call operators in BuildOutputGraph().
    if (edge->IsControlEdge()) continue;

    // Add 'src' as an output of its subgraph, if applicable.
    if (!src_func_id.empty()) {
      Subgraph& src_subgraph = subgraphs_[src_func_id];
      int ret_index = src_subgraph.results.size();
      if (src_subgraph.results
              .emplace(NodeSlot(edge->src(), edge->src_output()), ret_index)
              .second) {
        // Create a new _Retval node
        DataType dtype = edge->src()->output_type(edge->src_output());

        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported: tensor ",
              edge->src()->name(), ":", edge->src_output());
        }

        NodeDef ret_def;
        ret_def.set_op(kRetValOp);
        ret_def.set_name(strings::StrCat(edge->src()->name(), "_",
                                         edge->src_output(), "_retval"));
        AddNodeAttr("T", dtype, &ret_def);
        AddNodeAttr("index", ret_index, &ret_def);
        Node* ret = src_subgraph.graph->AddNode(ret_def, &s);
        if (!s.ok()) return s;

        // Add an edge from 'src' to _Retval.
        src_subgraph.graph->AddEdge(src_image, edge->src_output(), ret, 0);
      }
    }

    // Add 'dst' as an input of its subgraph, if applicable.
    if (!dst_func_id.empty()) {
      Subgraph& dst_subgraph = subgraphs_[dst_func_id];

      // Create an _Arg node for this tensor, if none exists yet.
      std::unordered_map<NodeSlot, int, NodeSlot::Hasher>::iterator iter;
      bool inserted;
      std::tie(iter, inserted) = dst_subgraph.args_by_src.emplace(
          NodeSlot(edge->src(), edge->src_output()), dst_subgraph.args.size());
      int arg_index = iter->second;
      if (inserted) {
        // This is the first time we have seen this tensor. Create an _Arg node.
        DataType dtype = edge->dst()->input_type(edge->dst_input());

        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported: tensor ",
              edge->src()->name(), ":", edge->src_output());
        }

        NodeDef arg_def;
        NodeDefBuilder builder(strings::StrCat(edge->src()->name(), "_",
                                               edge->src_output(), "_arg"),
                               kArgOp);
        builder.Attr("T", dtype);
        builder.Attr("index", arg_index);
        s = builder.Finalize(&arg_def);
        if (!s.ok()) return s;

        Node* arg = dst_subgraph.graph->AddNode(arg_def, &s);
        if (!s.ok()) return s;

        dst_subgraph.args.push_back(arg);
      }
      // Add an edge from the _Arg node to 'dst' in the subgraph.
      dst_subgraph.args_by_dst[NodeSlot(edge->dst(), edge->dst_input())] =
          arg_index;
      dst_subgraph.graph->AddEdge(dst_subgraph.args[arg_index], 0, dst_image,
                                  edge->dst_input());
    }
  }

  for (auto& entry : subgraphs_) {
    FixupSourceAndSinkEdges(entry.second.graph.get());
  }

  return s;
}

Status Encapsulator::BuildFunctionDefs(
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
    FunctionLibraryDefinition* library) {
  // For each subgraph, build a FunctionDef.
  for (auto& subgraph_entry : subgraphs_) {
    string name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;

    subgraph.call_node_def.set_op(name);
    subgraph.call_node_def.set_name(name);
    subgraph.call_node_def.set_device(subgraph.device);

    if (rewrite_subgraph_fn) {
      // Initialize the input and output permutations to the identity.
      std::vector<int> input_permutation(subgraph.args_by_src.size());
      std::iota(input_permutation.begin(), input_permutation.end(), 0);
      std::vector<int> output_permutation(subgraph.results.size());
      std::iota(output_permutation.begin(), output_permutation.end(), 0);

      TF_RETURN_IF_ERROR(
          rewrite_subgraph_fn(&subgraph.graph, &input_permutation,
                              &output_permutation, &subgraph.call_node_def));

      // Apply the input/output permutations to the 'args_by_...' and 'results'
      // mappings in 'subgraph', so when we build edges in BuildOutputGraph() we
      // connect them to the right input/output positions.
      if (input_permutation.size() != subgraph.args_by_src.size()) {
        return errors::InvalidArgument("Input permutation has incorrect size.");
      }
      if (output_permutation.size() != subgraph.results.size()) {
        return errors::InvalidArgument(
            "Output permutation has incorrect size.");
      }
      for (auto& arg : subgraph.args_by_src) {
        arg.second = input_permutation[arg.second];
      }
      for (auto& arg : subgraph.args_by_dst) {
        arg.second = input_permutation[arg.second];
      }
      for (auto& result : subgraph.results) {
        result.second = output_permutation[result.second];
      }

      name = subgraph.call_node_def.op();
    }

    FunctionDef fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*subgraph.graph, name, &fdef));

    if (VLOG_IS_ON(1)) {
      VLOG(2) << "Build function def " << name;
      dump_graph::DumpGraphToFile(
          strings::StrCat("encapsulate_fdef_graph_", name), *subgraph.graph,
          library);
      dump_graph::DumpFunctionDefToFile(
          strings::StrCat("encapsulate_fdef_", name), fdef);
    }

    if (!reuse_existing_functions || library->Find(name) == nullptr) {
      TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
    }
  }
  return Status::OK();
}

Status Encapsulator::BuildParallelCheckOp(
    const std::unordered_map<const Node*, Node*>& node_images,
    const Encapsulator::Subgraph& subgraph, Graph* graph_out,
    Node** parallel_check_op) {
  // Build an index mapping output positions to node/slot pairs in the
  // original graph.
  std::vector<NodeSlot> results_by_num(subgraph.results.size());
  for (const auto& entry : subgraph.results) {
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
    actual_outputs[i] = NodeDefBuilder::NodeOut(subgraph.call_node_def.name(),
                                                i, result_dtypes[i]);
  }
  // Assign the parallel check op to a CPU on the same task as the cluster it is
  // checking.
  string device, dummy;
  if (!DeviceNameUtils::SplitDeviceName(
          subgraph.call_node_inputs->assigned_device_name(), &device, &dummy)) {
    return errors::InvalidArgument("Could not parse device name");
  }
  strings::StrAppend(&device, "/cpu:0");

  NodeDef check_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder(graph_out->NewName(strings::StrCat(
                         subgraph.call_node_def.name(), "_parallel_check")),
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
    graph_out->AddEdge(subgraph.call_node_inputs, i, check_op, num_results + i);
  }

  *parallel_check_op = check_op;
  return Status::OK();
}

Status Encapsulator::BuildOutputGraph(bool parallel_checking,
                                      Graph* graph_out) {
  Status s;

  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node*, Node*> node_images;

  // Copy all unmarked nodes to the output graph.
  for (Node* node : graph_in_->op_nodes()) {
    string func_id = GetFunctionNameAttr(node);

    // Don't copy nodes that going to be encapsulated, unless parallel checking
    // is enabled.
    if (!func_id.empty() && !parallel_checking) continue;

    Node* image = graph_out->CopyNode(node);
    node_images[node] = image;
  }
  node_images[graph_in_->source_node()] = graph_out->source_node();
  node_images[graph_in_->sink_node()] = graph_out->sink_node();

  // Add function call nodes for each subgraph.
  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;

    subgraph.call_node_inputs = graph_out->AddNode(subgraph.call_node_def, &s);
    if (!s.ok()) return s;

    // Copy the assigned device and the key_annotation over.
    subgraph.call_node_inputs->set_assigned_device_name(subgraph.device);
    subgraph.call_node_outputs = subgraph.call_node_inputs;

    if (parallel_checking) {
      TF_RETURN_IF_ERROR(BuildParallelCheckOp(node_images, subgraph, graph_out,
                                              &subgraph.call_node_outputs));
    }
  }

  // Set of edges already added to the output graph, represented as (src, dst)
  // pairs. We use the set to deduplicate edges; multiple edges in the input
  // graph may map to one edge in the output graph.
  std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher>
      edges_added;

  // Add edges to the graph_out graph.
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id = GetFunctionNameAttr(edge->src());
    string dst_func_id = GetFunctionNameAttr(edge->dst());

    // Ignore edges that are strictly contained within one subgraph, unless
    // we are constructing parallel check graphs.
    if (!src_func_id.empty() && src_func_id == dst_func_id) {
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

    // We have an edge that crosses a cluster boundary.
    Node* src_image = src_func_id.empty()
                          ? node_images.at(edge->src())
                          : subgraphs_.at(src_func_id).call_node_outputs;
    Node* dst_image = dst_func_id.empty()
                          ? node_images.at(edge->dst())
                          : subgraphs_.at(dst_func_id).call_node_inputs;

    // Copy control edges. Lift control edges onto the enclosing call operator.
    if (edge->IsControlEdge()) {
      // Add the control edge, if we have not already added it.
      if (edges_added.emplace(NodeSlot(src_image, -1), NodeSlot(dst_image, -1))
              .second) {
        graph_out->AddControlEdge(src_image, dst_image);
      }

      // If parallel checking is enabled, also add a control edge to the
      // corresponding parallel check op.
      if (parallel_checking) {
        graph_out->AddControlEdge(src_image, node_images.at(edge->dst()));
      }
      continue;
    }

    int src_output = edge->src_output();
    if (!src_func_id.empty()) {
      // 'src' is in a subgraph. Use the corresponding call output instead.
      const Subgraph& src_subgraph = subgraphs_.at(src_func_id);
      src_output =
          src_subgraph.results.at(NodeSlot(edge->src(), edge->src_output()));
    }

    int dst_input = edge->dst_input();

    if (!dst_func_id.empty()) {
      // 'dst' is in a subgraph. Use the corresponding call input instead.
      const Subgraph& dst_subgraph = subgraphs_.at(dst_func_id);
      dst_input =
          dst_subgraph.args_by_dst.at(NodeSlot(edge->dst(), edge->dst_input()));

      // If we are parallel checking, also feed the tensor as an input to the
      // corresponding parallel check subgraph.
      if (parallel_checking) {
        graph_out->AddEdge(src_image, src_output, node_images.at(edge->dst()),
                           edge->dst_input());
      }
    }
    // Add the edge, if we have not already added it.
    if (edges_added
            .emplace(NodeSlot(src_image, src_output),
                     NodeSlot(dst_image, dst_input))
            .second) {
      graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
    }
  }

  return s;
}

}  // anonymous namespace

Status EncapsulateSubgraphsInFunctions(
    string group_attribute, const Graph& graph_in,
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool parallel_checking,
    bool reuse_existing_functions, std::unique_ptr<Graph>* graph_out,
    FunctionLibraryDefinition* library) {
  Status s;

  Encapsulator encapsulator(std::move(group_attribute), &graph_in);
  s = encapsulator.SplitIntoSubgraphs();
  if (!s.ok()) return s;

  s = encapsulator.BuildFunctionDefs(rewrite_subgraph_fn,
                                     reuse_existing_functions, library);
  if (!s.ok()) return s;

  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  s = encapsulator.BuildOutputGraph(parallel_checking, out.get());
  if (!s.ok()) return s;

  *graph_out = std::move(out);
  return s;
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
  std::unique_ptr<FunctionLibraryRuntime> flr(
      NewFunctionLibraryRuntime(nullptr, options.session_options->env, nullptr,
                                TF_GRAPH_DEF_VERSION, library, opts));

  auto rewrite_subgraph = [&flr](
      std::unique_ptr<Graph>* subgraph, std::vector<int>* input_permutation,
      std::vector<int>* output_permutation, NodeDef* node) {
    // Optimize the subgraph.
    OptimizeGraph(flr.get(), subgraph);

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
      kXlaClusterAttr, **options.graph, rewrite_subgraph,
      flags->tf_xla_parallel_checking, /*reuse_existing_functions=*/false,
      &graph_out, library));

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
