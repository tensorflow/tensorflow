/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/encapsulate_xla_computations_pass.h"

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {

const char* const EncapsulateXlaComputationsPass::kXlaClusterAttr =
    "_xla_compile_id";

namespace {

const char* const kXlaClusterOutput = "XlaClusterOutput";

// Checks if a graph node is marked to be a guaranteed constant.
bool is_guaranteed_constant(const Node& n) {
  bool guaranteed_constant = false;
  if (!GetNodeAttr(n.attrs(), "_is_guaranteed_constant", &guaranteed_constant)
           .ok()) {
    return false;
  }
  return guaranteed_constant;
}

// Finds the `index` of an _Arg or _Retval node.
Status GetIndexAttr(const Node& n, int num_args, int* index) {
  TF_RETURN_IF_ERROR(GetNodeAttr(n.attrs(), "index", index));
  if (*index < 0 || *index >= num_args) {
    return errors::InvalidArgument("Invalid ", n.type_string(), " number ",
                                   *index);
  }
  return Status::OK();
}

// Returns the data type of the destination of an edge.
DataType EdgeType(const Edge* edge) {
  return edge->dst()->input_type(edge->dst_input());
}

// Adds the control inputs of `node` to `*deps`.
void AddControlInputs(const Node& node, absl::flat_hash_set<Node*>* deps) {
  for (const Edge* edge : node.in_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->src());
    }
  }
}

// Adds the control outputs of `node` to `*deps`.
void AddControlOutputs(const Node& node, absl::flat_hash_set<Node*>* deps) {
  for (const Edge* edge : node.out_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->dst());
    }
  }
}

// Rewrite function to be passed to EncapsulateSubgraphsInFunctions that sorts
// the arguments into the order expected by XlaLaunch computations:
// 1) arguments
// 2) resource variable arguments
// See the documentation of EncapsulateSubgraphsInFunctions for the meaning
// of the arguments.
//
// TODO(b/113166435): Ordering constraints on XlaLaunch op can be relaxed.
Status RewriteSubgraph(const std::vector<OutputTensor>& arg_source_tensors,
                       std::unique_ptr<Graph>* graph_ptr,
                       std::vector<int>* input_permutation,
                       std::vector<int>* output_permutation,
                       NodeDef* call_def) {
  Graph* graph = graph_ptr->get();
  const int num_args = input_permutation->size();
  const int num_retvals = output_permutation->size();

  std::vector<Node*> args;
  std::vector<Node*> retvals;
  args.reserve(num_args);
  retvals.reserve(num_retvals);
  for (Node* n : graph->nodes()) {
    if (n->type_string() == "_Arg") {
      // Check if this is a guaranteed constant.
      if (is_guaranteed_constant(*n)) {
        return errors::InvalidArgument(
            "Guaranteed constants are not supported (", n->name(), ")");
      }
      args.push_back(n);
    } else if (n->type_string() == "_Retval") {
      retvals.push_back(n);
    }
  }

  if (std::find(args.begin(), args.end(), nullptr) != args.end()) {
    return errors::InvalidArgument("Missing or non-consecutive arguments");
  }

  // Reorders the arguments.
  std::sort(args.begin(), args.end(), [&](Node* a, Node* b) {
    // Non-resources appear before resources
    bool a_is_resource = (a->output_type(0) == DT_RESOURCE);
    bool b_is_resource = (b->output_type(0) == DT_RESOURCE);
    // Uses the name as a tiebreaker so the output is deterministic.
    StringPiece a_name(a->name());
    StringPiece b_name(b->name());
    return std::tie(a_is_resource, a_name) < std::tie(b_is_resource, b_name);
  });

  // Sorts the retvals by name so the order is deterministic.
  std::sort(retvals.begin(), retvals.end(),
            [](Node* a, Node* b) { return a->name() < b->name(); });

  // Computes the permutation to produce the correct argument order, and update
  // the argument indices.
  int variable_start_index = num_args;
  for (int i = 0; i < num_args; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*args[i], num_args, &index));
    if (args[i]->output_type(0) == DT_RESOURCE &&
        variable_start_index == num_args) {
      variable_start_index = i;
    }
    (*input_permutation)[index] = i;
    args[i]->AddAttr("index", i);
  }
  VLOG(4) << "variable_start_index: " << variable_start_index;

  // Computes the permutation to produce the correct retval order, and update
  // the argument indices.
  for (int i = 0; i < num_retvals; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*retvals[i], num_retvals, &index));
    (*output_permutation)[index] = i;
    retvals[i]->AddAttr("index", i);
  }

  AddNodeAttr(EncapsulateXlaComputationsPass::kXlaClusterAttr, call_def->name(),
              call_def);
  AddNodeAttr("_variable_start_index", variable_start_index, call_def);

  // Uniquify the function name.
  GraphDef gdef;
  graph->ToGraphDef(&gdef);

  // Before serialization, sort each node's control inputs to achieve
  // determinism. Sorting control inputs could help (but not necessarily) create
  // a deterministic serialization and fingerprint. Other sources of
  // nondeterminism include unstable node ordering.
  SortControlInputs(&gdef);
  // Fingerprint the function.
  // Nondeterminism in serialization would not lead to incorrect results, but
  // may cause spurious cache misses. DeterministicSerialization is a
  // best-effort deterministic serialization.
  string serialized;
  TF_RET_CHECK(SerializeToStringDeterministic(gdef, &serialized));
  uint64 fingerprint = Fingerprint64(serialized);
  VLOG(1) << "Subgraph fingerprint:" << fingerprint;
  call_def->set_op(absl::StrCat(call_def->op(), "_", fingerprint));
  return Status::OK();
}

}  // namespace

/*static*/ Status EncapsulateXlaComputationsPass::Encapsulate(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def) {
  // Check for undeclared outputs before Encapsulation, so we can give a better
  // error message.
  // TODO(phawkins): merge this with the encapsulation code to avoid the extra
  // O(n) pass over the edges.
  for (const Edge* e : (*graph)->edges()) {
    if (!e->IsControlEdge() &&
        e->src()->attrs().Find(kXlaClusterAttr) != nullptr &&
        e->dst()->attrs().Find(kXlaClusterAttr) == nullptr &&
        e->dst()->type_string() != kXlaClusterOutput) {
      return errors::InvalidArgument(
          "Undeclared output of XLA computation. Some common causes of this "
          "error are: 1) variable initializers that depend on the XLA "
          "computation; 2) gradient computations that depend on the XLA "
          "computation, which can be mitigated by moving gradient computations "
          "inside XLA computation. Offending edge: ",
          e->src()->name(), ":", e->src_output(), " -> ", e->dst()->name(), ":",
          e->dst_input());
    }
  }

  auto output = absl::make_unique<Graph>((*graph)->op_registry());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kXlaClusterAttr, "", **graph, RewriteSubgraph,
          /*reuse_existing_functions=*/true, &output, flib_def),
      "EncapsulateXlaComputationsPass failed");
  graph->swap(output);
  return Status::OK();
}

/*static*/ Status EncapsulateXlaComputationsPass::BuildXlaLaunchOps(
    Graph* graph) {
  // Finds all of the XlaLaunch function calls, to avoid mutating the graph
  // while iterating.
  std::vector<Node*> launch_nodes;
  for (Node* n : graph->nodes()) {
    string name;
    if (GetNodeAttr(n->attrs(), kXlaClusterAttr, &name).ok()) {
      launch_nodes.push_back(n);
    }
  }

  // Replaces each launch function call together with its neighboring
  // XlaClusterOutput nodes with a XlaLaunch node.
  for (Node* launch : launch_nodes) {
    int variable_start_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(launch->attrs(), "_variable_start_index",
                                   &variable_start_index));

    std::vector<const Edge*> in_edges;
    TF_RETURN_IF_ERROR(launch->input_edges(&in_edges));

    const int num_inputs = in_edges.size();
    const int num_variables = num_inputs - variable_start_index;
    const int num_args = variable_start_index;

    VLOG(4) << "Launch node '" << launch->name() << "'"
            << " input edges: " << in_edges.size() << " num_args: " << num_args
            << " num_variables: " << num_variables;

    std::vector<Node*> nodes_to_remove = {launch};

    // Data and control inputs to the new XlaLaunch node.
    std::vector<std::pair<Node*, int>> data_inputs(num_inputs);
    absl::flat_hash_set<Node*> control_inputs;
    DataTypeVector arg_types(num_args);

    AddControlInputs(*launch, &control_inputs);

    for (int i = 0; i < num_args; ++i) {
      const Edge* edge = in_edges[i];
      data_inputs[i] = {edge->src(), edge->src_output()};
      arg_types[i] = EdgeType(edge);
    }

    // Appends the variable inputs.
    for (int i = 0; i < num_variables; ++i) {
      int pos = variable_start_index + i;
      const Edge* edge = in_edges[pos];
      data_inputs[pos] = {edge->src(), edge->src_output()};
    }

    // Outputs.
    const int num_outputs = launch->output_types().size();
    absl::flat_hash_set<Node*> control_outputs;
    std::vector<std::vector<std::pair<Node*, int>>> data_outputs(num_outputs);
    DataTypeVector output_types(num_outputs);

    for (const Edge* le : launch->out_edges()) {
      if (le->IsControlEdge()) {
        control_outputs.insert(le->dst());
      } else {
        TF_RET_CHECK(le->src_output() < num_outputs);
        Node* output_node = le->dst();

        TF_RET_CHECK(output_node->type_string() == kXlaClusterOutput)
            << le->DebugString();
        nodes_to_remove.push_back(output_node);

        for (const Edge* oe : output_node->out_edges()) {
          TF_RET_CHECK(!oe->IsControlEdge());
          data_outputs[le->src_output()].push_back(
              {oe->dst(), oe->dst_input()});
        }
        output_types[le->src_output()] = output_node->input_type(0);

        AddControlOutputs(*output_node, &control_outputs);
      }
    }

    NodeDef def;
    def.set_name(launch->name());
    MergeDebugInfo(NodeDebugInfo(launch->def()), &def);

    // Target the XLA CPU/GPU backends.
    VLOG(2) << "Replacing with XlaLaunch";
    VLOG(2) << "Device is " << launch->requested_device();
    def.set_op("XlaLaunch");
    def.set_device(launch->requested_device());
    AddNodeAttr("Tconstants", DataTypeVector{}, &def);
    AddNodeAttr("Targs", arg_types, &def);
    AddNodeAttr("Nresources", num_variables, &def);
    AddNodeAttr("Tresults", output_types, &def);
    NameAttrList function;
    function.set_name(launch->type_string());
    AddNodeAttr("function", function, &def);

    for (Node* node : nodes_to_remove) {
      VLOG(2) << "Deleting node " << node->DebugString();
      // Ensure that we do not attempt to add control edges to nodes that are
      // deleted.
      control_inputs.erase(node);
      control_outputs.erase(node);
      graph->RemoveNode(node);
    }

    Status status;
    Node* xla_launch = graph->AddNode(def, &status);
    if (!status.ok()) {
      return status;
    }
    for (int i = 0; i < data_inputs.size(); ++i) {
      graph->AddEdge(data_inputs[i].first, data_inputs[i].second, xla_launch,
                     i);
    }
    for (Node* n : control_inputs) {
      graph->AddControlEdge(n, xla_launch);
    }
    for (int i = 0; i < data_outputs.size(); ++i) {
      for (const auto& successor : data_outputs[i]) {
        graph->AddEdge(xla_launch, i, successor.first, successor.second);
      }
    }
    for (Node* n : control_outputs) {
      graph->AddControlEdge(xla_launch, n);
    }
  }
  return Status::OK();
}

Status EncapsulateXlaComputationsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "EncapsulateXlaComputations(): "
          << dump_graph::DumpGraphToFile("encapsulate_xla_computations_before",
                                         **options.graph, options.flib_def);

  TF_RETURN_IF_ERROR(Encapsulate(options.graph, options.flib_def));
  VLOG(1) << "EncapsulateXlaComputations() half-way: "
          << dump_graph::DumpGraphToFile("encapsulate_xla_computations_halfway",
                                         **options.graph, options.flib_def);

  TF_RETURN_IF_ERROR(BuildXlaLaunchOps(options.graph->get()));
  VLOG(1) << "EncapsulateXlaComputations() finished: "
          << dump_graph::DumpGraphToFile("encapsulate_xla_computations_after",
                                         **options.graph, options.flib_def);
  return Status::OK();
}

}  // namespace tensorflow
