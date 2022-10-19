/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/encapsulate_tpu_computations_pass.h"

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/jit/extract_outside_compilation_pass.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

const char* const kTPUReplicatedInput = "TPUReplicatedInput";
const char* const kTPUReplicatedOutput = "TPUReplicatedOutput";
const char* const kPivotForClusterAttr = "_pivot_for_cluster";
const char* const kTPUPartitionedInput = "TPUPartitionedInput";

// Finds the `index` of an _Arg or _Retval node.
Status GetIndexAttr(const Node& n, int num_args, int* index) {
  TF_RETURN_IF_ERROR(GetNodeAttr(n.attrs(), "index", index));
  if (*index < 0 || *index >= num_args) {
    return errors::InvalidArgument("Invalid ", n.type_string(), " number ",
                                   *index);
  }
  return OkStatus();
}

// Rewrite function to be passed to EncapsulateSubgraphsInFunctions that sorts
// the arguments into the order expected by TPUReplicate computations:
// 1) replicated arguments
// 2) non-replicated (broadcast) arguments
// 3) resource variable arguments
// See the documentation of EncapsulateSubgraphsInFunctions for the meaning
// of the arguments.
Status RewriteSubgraph(const std::vector<OutputTensor>& arg_source_tensors,
                       std::unique_ptr<Graph>* graph_ptr,
                       std::vector<int>* input_permutation,
                       std::vector<int>* output_permutation,
                       NodeDef* call_def) {
  // Replicated inputs have TPUReplicatedInput nodes as predecessors in the
  // input graph.
  auto is_replicated_input = [&](const Node& n, bool* is_packed = nullptr) {
    CHECK_EQ("_Arg", n.type_string());
    int index;
    TF_CHECK_OK(GetIndexAttr(n, arg_source_tensors.size(), &index));
    bool ret =
        arg_source_tensors.at(index).node->type_string() == kTPUReplicatedInput;
    if (is_packed) {
      if (!ret || !GetNodeAttr(arg_source_tensors.at(index).node->attrs(),
                               "is_packed", is_packed)
                       .ok()) {
        *is_packed = false;
      }
    }
    return ret;
  };

  auto is_guaranteed_constant = [&](const Node& n) {
    bool guaranteed_constant = false;
    if (!GetNodeAttr(n.attrs(), "_is_guaranteed_constant", &guaranteed_constant)
             .ok()) {
      return false;
    }
    // Replicated input nodes can be marked as guaranteed constants if they are
    // const.
    return guaranteed_constant && !is_replicated_input(n);
  };

  Graph* graph = graph_ptr->get();
  Node* metadata_node = nullptr;
  const int num_args = input_permutation->size();
  const int num_retvals = output_permutation->size();

  std::vector<Node*> args;
  std::vector<Node*> retvals;
  args.reserve(num_args);
  retvals.reserve(num_retvals);
  for (Node* n : graph->nodes()) {
    if (n->type_string() == "_Arg") {
      args.push_back(n);
    } else if (n->type_string() == "_Retval") {
      retvals.push_back(n);
    } else if (n->type_string() == "TPUReplicateMetadata") {
      metadata_node = n;
    } else if (!str_util::StrContains(n->requested_device(),
                                      DEVICE_TPU_REPLICATED_CORE)) {
      // If an operator isn't assigned to a TPU core device, assign it to
      // TPU_REPLICATED_CORE without a specific core ID. For some operators,
      // such as variable reads/writes, the operator may be assigned to non-TPU
      // devices due to colocation.
      n->set_assigned_device_name(
          strings::StrCat("/device:", DEVICE_TPU_REPLICATED_CORE));
    }
  }

  // Read the metadata node and remove it from the graph.
  if (metadata_node == nullptr) {
    return errors::InvalidArgument("Missing TPUReplicateMetadata node");
  }

  for (const auto& attr : metadata_node->attrs()) {
    if (attr.first == "computation_shape") {
      // Convert the deprecated computation_shape attribute into a
      // num_cores_per_replica value. If a computation_shape is present, it
      // overrides num_cores_per_replica.
      std::vector<int> shape;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(metadata_node->attrs(), "computation_shape", &shape));
      if (!shape.empty()) {
        int64_t num_cores_per_replica = 1LL;
        for (int dim : shape) {
          num_cores_per_replica *= dim;
        }
        call_def->mutable_attr()->erase("num_cores_per_replica");
        AddNodeAttr("num_cores_per_replica", num_cores_per_replica, call_def);
      }
    } else {
      call_def->mutable_attr()->insert(attr);
    }
  }
  MergeDebugInfo(NodeDebugInfo(metadata_node->def()), call_def);
  graph->RemoveNode(metadata_node);

  if (std::find(args.begin(), args.end(), nullptr) != args.end()) {
    return errors::InvalidArgument("Missing or non-consecutive arguments");
  }

  // Reorders the arguments.
  std::sort(args.begin(), args.end(), [&](Node* a, Node* b) {
    // Non-constants appear before constants
    bool a_is_guaranteed_constant = is_guaranteed_constant(*a);
    bool b_is_guaranteed_constant = is_guaranteed_constant(*b);
    // Non-packed values appear before packed values.
    bool a_is_packed;
    bool b_is_packed;
    // Replicated values appear before non-replicated values.
    bool a_not_replicated = !is_replicated_input(*a, &a_is_packed);
    bool b_not_replicated = !is_replicated_input(*b, &b_is_packed);
    // Non-resources appear before resources
    bool a_is_resource = (a->output_type(0) == DT_RESOURCE);
    bool b_is_resource = (b->output_type(0) == DT_RESOURCE);
    // Uses the name as a tiebreaker so the output is deterministic.
    StringPiece a_name(a->name());
    StringPiece b_name(b->name());
    return std::tie(a_is_guaranteed_constant, a_not_replicated, a_is_packed,
                    a_is_resource, a_name) <
           std::tie(b_is_guaranteed_constant, b_not_replicated, b_is_packed,
                    b_is_resource, b_name);
  });
  // Sorts the retvals by name so the order is deterministic.
  std::sort(retvals.begin(), retvals.end(),
            [](Node* a, Node* b) { return a->name() < b->name(); });

  // Computes the permutation to produce the correct argument order, and update
  // the argument indices.
  int variable_start_index = num_args;
  int guaranteed_const_start_index = num_args;
  for (int i = 0; i < num_args; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*args[i], num_args, &index));
    if (args[i]->output_type(0) == DT_RESOURCE &&
        !is_replicated_input(*args[i]) && variable_start_index == num_args) {
      variable_start_index = i;
    } else if (is_guaranteed_constant(*args[i]) &&
               guaranteed_const_start_index == num_args) {
      guaranteed_const_start_index = i;
    }
    (*input_permutation)[index] = i;
    args[i]->AddAttr("index", i);
  }
  VLOG(4) << "variable_start_index: " << variable_start_index
          << " guaranteed_const_start_index: " << guaranteed_const_start_index;

  // Computes the permutation to produce the correct retval order, and update
  // the argument indices.
  for (int i = 0; i < num_retvals; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*retvals[i], num_retvals, &index));
    (*output_permutation)[index] = i;
    retvals[i]->AddAttr("index", i);
  }

  AddNodeAttr(kTPUReplicateAttr, call_def->name(), call_def);
  AddNodeAttr("_variable_start_index", variable_start_index, call_def);
  AddNodeAttr("_guaranteed_const_start_index", guaranteed_const_start_index,
              call_def);

  // Uniquify the function name by fingerprinting the function.
  // Nondeterminism in serialization would not lead to incorrect results, but
  // may cause spurious cache misses. DeterministicSerialization is a
  // best-effort deterministic serialization.
  TF_ASSIGN_OR_RETURN(string serialized, SerializeGraphDeterministic(*graph));
  uint64 fingerprint =
      TpuCompileInterface::Get()->FingerprintString(serialized);
  LOG(INFO) << "Subgraph fingerprint:" << fingerprint;
  call_def->set_op(strings::StrCat(call_def->op(), "_", fingerprint));
  return OkStatus();
}

DataType EdgeType(const Edge* edge) {
  return edge->dst()->input_type(edge->dst_input());
}

// Adds the control inputs of `node` to `*deps`.
void AddControlInputs(const Node& node, gtl::FlatSet<Node*>* deps) {
  for (const Edge* edge : node.in_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->src());
    }
  }
}

// Adds the control outputs of `node` to `*deps`.
void AddControlOutputs(const Node& node, gtl::FlatSet<Node*>* deps) {
  for (const Edge* edge : node.out_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->dst());
    }
  }
}

// We add Identity nodes for _Arg/_Retval in XLA computation. Remove those
// Identity nodes to simplify furthur processing.
Status RemoveIdentityNodesForArgRetval(Graph* g) {
  // Collect Identity nodes for _Arg/_Retval.
  std::vector<Node*> identity_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "Identity" &&
        (HasNodeAttr(n->def(), "_tpu_input_identity") ||
         HasNodeAttr(n->def(), "_tpu_output_identity"))) {
      identity_nodes.push_back(n);
    }
  }

  // Remove those Identity nodes.
  for (Node* n : identity_nodes) {
    const Edge* input_edge;
    TF_RETURN_IF_ERROR(n->input_edge(0, &input_edge));

    std::vector<const Edge*> output_edges;
    for (const Edge* e : n->out_edges()) {
      output_edges.push_back(e);
    }
    for (const Edge* e : output_edges) {
      if (e->IsControlEdge()) {
        Node* dst = e->dst();
        g->RemoveEdge(e);
        g->AddControlEdge(input_edge->src(), dst);
      } else {
        Node* dst = e->dst();
        int dst_input = e->dst_input();
        g->RemoveEdge(e);
        g->AddEdge(input_edge->src(), input_edge->src_output(), dst, dst_input);
      }
    }
    g->RemoveNode(n);
  }

  return OkStatus();
}

// Updates the TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR when
// 'additional_per_replicate_inputs' are added to the inputs of `xla_node`.
Status UpdateMirroredVariableIndices(int additional_per_replica_inputs,
                                     Node* xla_node) {
  std::vector<int> mirrored_variable_indices;
  if (xla_node->attrs().Find(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR) !=
      nullptr) {
    TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(),
                                   TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                                   &mirrored_variable_indices));
  }

  if (!mirrored_variable_indices.empty()) {
    for (int i = 0; i < mirrored_variable_indices.size(); ++i)
      mirrored_variable_indices[i] += additional_per_replica_inputs;
    xla_node->ClearAttr(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR);
    xla_node->AddAttr(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                      mirrored_variable_indices);
  }
  return OkStatus();
}

// Move outside compilation nodes at the beginning of XLA computation to host.
// For XLA computation graph, we will add new _Arg nodes to replace those
// outside compilation nodes.
// For host graph, we will move those outside compilation nodes to host,
// replicate them, and use them as XLA node's input.
Status MoveHeadOutsideCompilationToHost(
    const string& outside_compilation_attr_name, const string& xla_func_name,
    const std::string& cluster_name, Graph* g, Graph* xla_graph, Node* xla_node,
    Node* pivot_node) {
  // Find outside compilation nodes that only have _Arg or other outside
  // compilation nodes as input. These nodes will be moved to host graph.
  std::vector<Node*> oc_nodes_at_head;
  const string kOnlyArgOrOcInputAttrName = "_xla_only_arg_or_oc_input";
  ReverseDFS(
      *xla_graph, /*enter=*/nullptr,
      [&](Node* n) {
        bool has_non_arg_or_oc_input = false;
        for (const Edge* e : n->in_edges()) {
          if (e->src() == xla_graph->source_node()) {
            continue;
          }
          if (!e->src()->IsArg() &&
              (!HasNodeAttr(e->src()->def(), outside_compilation_attr_name) ||
               !HasNodeAttr(e->src()->def(), kOnlyArgOrOcInputAttrName))) {
            has_non_arg_or_oc_input = true;
            break;
          }
        }
        if (HasNodeAttr(n->def(), outside_compilation_attr_name) &&
            !has_non_arg_or_oc_input &&
            !HasNodeAttr(n->def(), kXlaIsPlaceholderForArg)) {
          n->AddAttr(kOnlyArgOrOcInputAttrName, true);
          oc_nodes_at_head.push_back(n);
        }
      },
      NodeComparatorName());
  std::vector<Node*> const_nodes_to_remove;
  for (Node* n : oc_nodes_at_head) {
    // If a Const node is in "oc_nodes_at_head" but some of its successors are
    // not, copy this Const node and use the copied node for those successors.
    if (n->type_string() != "Const") {
      continue;
    }

    std::vector<const Edge*> edges_to_replace;
    for (const Edge* e : n->out_edges()) {
      if (!e->IsControlEdge() &&
          HasNodeAttr(e->dst()->def(), outside_compilation_attr_name) &&
          !HasNodeAttr(e->dst()->def(), kOnlyArgOrOcInputAttrName)) {
        edges_to_replace.push_back(e);
      }
    }
    if (edges_to_replace.empty()) {
      continue;
    }

    Node* const_copy = xla_graph->CopyNode(n);
    for (const Edge* e : edges_to_replace) {
      Node* dst = e->dst();
      int dst_input = e->dst_input();
      xla_graph->RemoveEdge(e);
      xla_graph->AddEdge(const_copy, 0, dst, dst_input);
    }
    // Make sure the copied node can be traced from source node.
    xla_graph->AddControlEdge(xla_graph->source_node(), const_copy);

    // If this Const node has no data output any more, remove it later.
    bool has_output_edge = false;
    for (const Edge* e : n->out_edges()) {
      if (!e->IsControlEdge()) {
        has_output_edge = true;
        break;
      }
    }
    if (!has_output_edge) {
      const_nodes_to_remove.push_back(n);
    }
  }
  for (Node* n : const_nodes_to_remove) {
    xla_graph->RemoveNode(n);
    oc_nodes_at_head.erase(
        std::remove(oc_nodes_at_head.begin(), oc_nodes_at_head.end(), n),
        oc_nodes_at_head.end());
  }
  if (VLOG_IS_ON(5)) {
    for (Node* n : oc_nodes_at_head) {
      VLOG(5) << "oc_nodes_at_head: " << n->DebugString();
    }
  }

  // Copy all nodes in `oc_nodes_at_head` to host graph, and also replicate
  // them.

  // Sometimes `xla_node` can have a lot of inputs, calling Node::input_edge
  // will become very expensive in this case because it is doing a linear
  // search inside. Create an input_edges vector ahead to make the lookups
  // faster.
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(xla_node->input_edges(&input_edges));

  std::vector<DataType> input_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "Tinputs", &input_types));
  int num_distributed_vars;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "num_distributed_variables",
                                 &num_distributed_vars));
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->attrs(), "num_replicas", &num_replicas));
  int old_num_per_replica_inputs =
      (input_types.size() - num_distributed_vars) / num_replicas;
  VLOG(5) << "old_num_per_replica_inputs: " << old_num_per_replica_inputs;
  std::map<Node*, std::vector<Node*>> node_images;
  for (Node* n : oc_nodes_at_head) {
    for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
      NodeDef copy_def = n->def();
      copy_def.set_name(absl::StrCat(n->name(), "_head_oc/R", replica_id));
      copy_def.clear_device();

      TF_ASSIGN_OR_RETURN(Node * copy_node, g->AddNode(copy_def));

      copy_node->AddAttr(kXlaReplicaIdAttrName, replica_id);
      copy_node->AddAttr(kTPUReplicateAttr, cluster_name);

      for (const Edge* e : n->in_edges()) {
        if (e->src() == xla_graph->source_node()) {
          continue;
        }
        // Either e->src() is _Arg node, or it's in `node_images`.
        if (e->src()->IsArg()) {
          int index;
          TF_RETURN_IF_ERROR(GetNodeAttr(e->src()->attrs(), "index", &index));
          const int new_index =
              (index < old_num_per_replica_inputs)
                  ? (old_num_per_replica_inputs * replica_id + index)
                  : (old_num_per_replica_inputs * num_replicas +
                     (index - old_num_per_replica_inputs));
          const Edge* original_edge = input_edges.at(new_index);
          g->AddEdge(original_edge->src(), original_edge->src_output(),
                     copy_node, e->dst_input());
        } else {
          g->AddEdge(node_images[e->src()][replica_id], e->src_output(),
                     copy_node, e->dst_input());
        }
      }

      // Add control edge between `copy_node` and `xla_node`, so these outside
      // compilation nodes will be executed before XLA computation happens.
      g->AddControlEdge(copy_node, xla_node);

      // Add control edge between `pivot_node` and `copy_node`, so `copy_node`
      // belongs to same while loop as `xla_node`.
      if (pivot_node) {
        g->AddControlEdge(pivot_node, copy_node);
      }

      node_images[n].push_back(copy_node);
    }
  }

  // Record output edges from `oc_nodes_at_head`. We will create an _Arg node
  // for each of these edges. An obvious optimization here is to deduplicate
  // these edges by <src, src_output>. But that optimization will complicate
  // the code, and in practice we usually do not have output edges with the
  // same <src, src_output>.
  std::vector<const Edge*> oc_output_edges;
  std::vector<DataType> new_arg_types;
  for (Node* n : oc_nodes_at_head) {
    for (const Edge* e : n->out_edges()) {
      if (!e->IsControlEdge() &&
          node_images.find(e->dst()) == node_images.end()) {
        VLOG(5) << "oc_output_edges: " << e->DebugString();
        oc_output_edges.push_back(e);
        new_arg_types.push_back(e->src()->output_type(e->src_output()));
      }
    }
  }
  int new_num_per_replica_inputs =
      old_num_per_replica_inputs + oc_output_edges.size();
  VLOG(5) << "new_num_per_replica_inputs: " << new_num_per_replica_inputs;

  // Process input edges for XLA node.
  int num_variables;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->attrs(), "NumVariables", &num_variables));
  std::vector<DataType> broadcast_input_types, guaranteed_constant_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "Tbroadcast_inputs",
                                 &broadcast_input_types));
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "Tguaranteed_constants",
                                 &guaranteed_constant_types));
  int num_other_inputs = num_distributed_vars + num_variables +
                         broadcast_input_types.size() +
                         guaranteed_constant_types.size();
  VLOG(5) << "num_other_inputs: " << num_other_inputs;

  // Update `Tinputs` attribute for `xla_node`.
  std::vector<DataType> new_input_types;
  // Order of new_input_types: old per-replica inputs -> new per-replica inputs
  // -> distributed variables
  new_input_types.reserve(num_replicas * new_num_per_replica_inputs +
                          num_distributed_vars);
  for (int replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (int i = 0; i < old_num_per_replica_inputs; ++i) {
      new_input_types.push_back(input_types[i]);
    }
    for (int i = old_num_per_replica_inputs; i < new_num_per_replica_inputs;
         ++i) {
      new_input_types.push_back(new_arg_types[i - old_num_per_replica_inputs]);
    }
  }
  const int num_new_per_replica_input_types = new_input_types.size();
  for (int i = input_types.size() - num_distributed_vars;
       i < input_types.size(); i++) {
    new_input_types.push_back(input_types[i]);
  }
  xla_node->ClearAttr("Tinputs");
  xla_node->AddAttr("Tinputs", new_input_types);

  TF_RETURN_IF_ERROR(UpdateMirroredVariableIndices(
      /*additional_per_replica_inputs=*/oc_output_edges.size(), xla_node));

  int new_variable_start_index =
      num_new_per_replica_input_types / num_replicas + num_distributed_vars +
      broadcast_input_types.size();
  if (xla_node->attrs().Find("_variable_start_index") != nullptr) {
    xla_node->ClearAttr("_variable_start_index");
    xla_node->AddAttr("_variable_start_index", new_variable_start_index);
  }
  int new_guaranteed_const_start_index =
      new_variable_start_index + num_variables;
  if (xla_node->attrs().Find("_guaranteed_const_start_index") != nullptr) {
    xla_node->ClearAttr("_guaranteed_const_start_index");
    xla_node->AddAttr("_guaranteed_const_start_index",
                      new_guaranteed_const_start_index);
  }

  // Move non per-replica input edges.
  std::vector<const Edge*> new_input_edges(
      num_replicas * new_num_per_replica_inputs + num_other_inputs);
  int end_input_index =
      num_replicas * new_num_per_replica_inputs + num_other_inputs - 1;
  int start_input_index = end_input_index + 1 - num_other_inputs;
  for (int input_index = end_input_index; input_index >= start_input_index;
       input_index--) {
    const Edge* e =
        input_edges.at(input_index - num_replicas * new_arg_types.size());
    Node* src = e->src();
    int src_output = e->src_output();
    g->RemoveEdge(e);
    const Edge* new_input_edge =
        g->AddEdge(src, src_output, xla_node, input_index);
    new_input_edges[input_index] = new_input_edge;
  }

  // Re-order old per-replica inputs edges, and add new per-replica input edges.
  std::vector<std::pair<Node*, int>> per_replica_inputs;
  std::vector<const Edge*> old_per_replica_edges;
  for (int i = 0; i < old_num_per_replica_inputs * num_replicas; i++) {
    const Edge* e = input_edges.at(i);
    per_replica_inputs.push_back(std::make_pair(e->src(), e->src_output()));
    old_per_replica_edges.push_back(e);
  }
  for (const Edge* e : old_per_replica_edges) {
    g->RemoveEdge(e);
  }
  for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
    for (int input_index = 0; input_index < old_num_per_replica_inputs;
         input_index++) {
      Node* src = per_replica_inputs[replica_id * old_num_per_replica_inputs +
                                     input_index]
                      .first;
      int src_output =
          per_replica_inputs[replica_id * old_num_per_replica_inputs +
                             input_index]
              .second;
      const Edge* new_input_edge =
          g->AddEdge(src, src_output, xla_node,
                     replica_id * new_num_per_replica_inputs + input_index);
      new_input_edges[input_index] = new_input_edge;
    }
    for (int input_index = old_num_per_replica_inputs;
         input_index < new_num_per_replica_inputs; input_index++) {
      Node* original_src =
          oc_output_edges[input_index - old_num_per_replica_inputs]->src();
      int original_src_output =
          oc_output_edges[input_index - old_num_per_replica_inputs]
              ->src_output();
      Node* src = node_images[original_src][replica_id];
      const Edge* new_input_edge =
          g->AddEdge(src, original_src_output, xla_node,
                     replica_id * new_num_per_replica_inputs + input_index);
      new_input_edges[input_index] = new_input_edge;
    }
  }

  // Adjust original _Arg nodes in `xla_graph`.
  for (Node* n : xla_graph->nodes()) {
    if (n->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (index >= old_num_per_replica_inputs) {
        index += new_arg_types.size();
        n->ClearAttr("index");
        n->AddAttr("index", index);
      }
    }
  }

  // Create new _Arg nodes in `xla_graph`.
  for (int i = old_num_per_replica_inputs; i < new_num_per_replica_inputs;
       i++) {
    NodeDefBuilder arg_builder(absl::StrCat("arg_", i),
                               FunctionLibraryDefinition::kArgOp);
    arg_builder.Attr("T", new_arg_types[i - old_num_per_replica_inputs]);
    arg_builder.Attr("index", i);
    NodeDef arg_def;
    TF_RETURN_IF_ERROR(arg_builder.Finalize(&arg_def));
    TF_ASSIGN_OR_RETURN(Node * arg_node, xla_graph->AddNode(arg_def));
    const Edge* original_edge = oc_output_edges[i - old_num_per_replica_inputs];
    Node* dst = original_edge->dst();
    int dst_input = original_edge->dst_input();
    xla_graph->RemoveEdge(original_edge);
    xla_graph->AddEdge(arg_node, 0, dst, dst_input);
  }

  // For lifted arg nodes:
  // 1. Add a Placeholder node in `xla_graph`. When we build host side graph
  //    in ExtractOutsideCompilationPass, we will use this new Placeholder node
  //    instead of lifted arg node here.
  // 2. Add an IdentityN node in `g` to indicate its inputs. We will reconnect
  //    this IdentityN node and this lifted arg node's usage nodes in
  //    DistributedTPURewritePass.
  for (Node* n : oc_nodes_at_head) {
    bool is_lifted_arg;
    string outside_compilation_attr;
    if (!TryGetNodeAttr(n->def(), kXlaIsLiftedArgAttrName, &is_lifted_arg) ||
        !TryGetNodeAttr(n->def(), kOutsideCompilationAttr,
                        &outside_compilation_attr)) {
      continue;
    }

    TF_RET_CHECK(n->IsIdentity());
    NodeDefBuilder ph_builder(absl::StrCat("placeholder_", n->name()),
                              "Placeholder");
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "T", &dtype));
    ph_builder.Attr("dtype", dtype);
    ph_builder.Attr(kXlaIsLiftedArgAttrName, true);
    ph_builder.Attr(kOutsideCompilationAttr, outside_compilation_attr);
    NodeDef ph_def;
    TF_RETURN_IF_ERROR(ph_builder.Finalize(&ph_def));
    Status s;
    xla_graph->AddNode(ph_def, &s);
    TF_RETURN_IF_ERROR(s);

    Node* input_node;
    TF_RETURN_IF_ERROR(n->input_node(0, &input_node));
    TF_RET_CHECK(input_node->type_string() == "_Arg");
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(input_node->def(), "index", &index));
    // TODO(b/74023706): for now we only support resource input (e.g. summary
    // writer), which is non-replicated input. Support replicated input as
    // well.
    TF_RET_CHECK(index >= new_num_per_replica_inputs + num_distributed_vars);
    const Edge* input_edge =
        new_input_edges.at(num_replicas * new_num_per_replica_inputs + index -
                           new_num_per_replica_inputs);
    NodeDefBuilder id_builder(absl::StrCat("lifted_arg_input_", index),
                              "IdentityN");
    DataType input_dtype =
        input_edge->src()->output_type(input_edge->src_output());
    id_builder.Attr("T", std::vector<DataType>(num_replicas, input_dtype));
    std::vector<NodeDefBuilder::NodeOut> inputs(
        num_replicas,
        NodeDefBuilder::NodeOut{input_edge->src()->name(),
                                input_edge->src_output(), input_dtype});
    id_builder.Attr(kXlaOutsideCompilationInputsAttrName,
                    outside_compilation_attr);
    id_builder.Input(inputs);
    NodeDef id_def;
    TF_RETURN_IF_ERROR(id_builder.Finalize(&id_def));
    TF_ASSIGN_OR_RETURN(Node * id_node, g->AddNode(id_def));
    for (int i = 0; i < num_replicas; i++) {
      g->AddEdge(input_edge->src(), input_edge->src_output(), id_node, i);
    }
  }

  // Remove `oc_nodes_at_head`.
  for (Node* n : oc_nodes_at_head) {
    xla_graph->RemoveNode(n);
  }

  VLOG(4) << "MoveHeadOutsideCompilationToHost host graph: "
          << DumpGraphToFile(absl::StrCat("move_head_oc_host_", xla_func_name),
                             *g);
  VLOG(4) << "MoveHeadOutsideCompilationToHost XLA graph: "
          << DumpGraphToFile(absl::StrCat("move_head_oc_xla_", xla_func_name),
                             *xla_graph);

  return OkStatus();
}

// If there are any unused _Arg nodes in `xla_graph`, remove them from
// `xla_graph` and remove corresponding input edge in host graph `g`.
Status RemoveUnusedXlaInput(const string& xla_func_name, Graph* g,
                            Graph* xla_graph, Node* xla_node) {
  // Find unused _Arg nodes, and remove them.
  std::vector<DataType> input_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(), "Tinputs", &input_types));
  std::vector<int> mirrored_variable_indices;
  if (xla_node->attrs().Find(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR) !=
      nullptr) {
    TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(),
                                   TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                                   &mirrored_variable_indices));
  }
  std::vector<DataType> broadcast_input_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(), "Tbroadcast_inputs",
                                 &broadcast_input_types));
  std::vector<DataType> guaranteed_constant_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(), "Tguaranteed_constants",
                                 &guaranteed_constant_types));
  int num_variables;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "NumVariables", &num_variables));
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "num_replicas", &num_replicas));
  int num_distributed_vars;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "num_distributed_variables",
                                 &num_distributed_vars));
  int num_per_replica_inputs =
      (input_types.size() - num_distributed_vars) / num_replicas;
  std::set<int> arg_indices_to_remove;
  std::vector<Node*> arg_nodes_to_update, nodes_to_remove;
  int num_args = 0, num_removed_per_replica_inputs = 0,
      num_removed_distributed_vars = 0;
  for (Node* n : xla_graph->nodes()) {
    if (!n->IsArg()) {
      continue;
    }

    bool has_output = false;
    for (const Edge* e : n->out_edges()) {
      if (e->dst() != xla_graph->sink_node()) {
        has_output = true;
        break;
      }
    }

    num_args++;
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    if (has_output) {
      arg_nodes_to_update.push_back(n);
      continue;
    }

    arg_indices_to_remove.insert(index);
    if (index < num_per_replica_inputs) {
      num_removed_per_replica_inputs++;
    } else if (index < num_per_replica_inputs + num_distributed_vars) {
      num_removed_distributed_vars++;
    }
    nodes_to_remove.push_back(n);
  }
  for (Node* n : nodes_to_remove) {
    xla_graph->RemoveNode(n);
  }

  // Update `index` for other _Arg nodes.
  std::map<int, int> arg_index_mapping;
  int new_arg_index = 0;
  for (int i = 0; i < num_args; i++) {
    if (arg_indices_to_remove.find(i) != arg_indices_to_remove.end()) {
      continue;
    } else {
      arg_index_mapping[i] = new_arg_index;
      new_arg_index++;
    }
  }
  for (Node* n : arg_nodes_to_update) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    n->ClearAttr("index");
    n->AddAttr("index", arg_index_mapping[index]);
  }

  // Re-order replicated index edges for `xla_node`.

  // Sometimes `xla_node` can have a lot of inputs, calling Node::input_edge
  // will become very expensive in this case because it is doing a linear search
  // inside. Create a input_edges vector ahead to make the lookups faster.
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(xla_node->input_edges(&input_edges));

  const int num_new_per_replica_inputs =
      num_per_replica_inputs - num_removed_per_replica_inputs;
  for (int i = 0; i < num_replicas; i++) {
    for (int j = 0; j < num_per_replica_inputs; j++) {
      auto iter = arg_index_mapping.find(j);
      if (iter != arg_index_mapping.end()) {
        const Edge* e = input_edges.at(i * num_per_replica_inputs + j);
        Node* src = e->src();
        int src_output = e->src_output();
        int dst_input = i * num_new_per_replica_inputs + iter->second;

        g->RemoveEdge(e);
        g->AddEdge(src, src_output, xla_node, dst_input);
      } else {
        const Edge* e = input_edges.at(i * num_per_replica_inputs + j);
        g->RemoveEdge(e);
      }
    }
  }

  // Move other data input edges.
  for (int i = num_replicas * num_per_replica_inputs;
       i < xla_node->num_inputs(); i++) {
    int arg_index =
        num_per_replica_inputs + i - num_replicas * num_per_replica_inputs;
    auto iter = arg_index_mapping.find(arg_index);
    if (iter != arg_index_mapping.end()) {
      const Edge* e = input_edges.at(i);
      Node* src = e->src();
      int src_output = e->src_output();
      int dst_input = num_replicas * num_new_per_replica_inputs + iter->second -
                      num_new_per_replica_inputs;

      g->RemoveEdge(e);
      g->AddEdge(src, src_output, xla_node, dst_input);
    } else {
      const Edge* e = input_edges.at(i);
      g->RemoveEdge(e);
    }
  }

  // Update attributes for `xla_node`.
  std::vector<DataType> new_input_types;
  for (int i = 0; i < num_replicas; i++) {
    for (int j = 0; j < num_per_replica_inputs; j++) {
      auto iter = arg_index_mapping.find(j);
      if (iter != arg_index_mapping.end()) {
        new_input_types.push_back(input_types[iter->first]);
      }
    }
  }
  for (int i = 0; i < num_distributed_vars; ++i) {
    auto iter = arg_index_mapping.find(i + num_per_replica_inputs);
    if (iter != arg_index_mapping.end()) {
      new_input_types.push_back(
          input_types[iter->first - num_per_replica_inputs +
                      num_per_replica_inputs * num_replicas]);
    }
  }
  xla_node->ClearAttr("Tinputs");
  xla_node->AddAttr("Tinputs", new_input_types);

  const int num_new_distributed_vars =
      num_distributed_vars - num_removed_distributed_vars;
  xla_node->ClearAttr("num_distributed_variables");
  xla_node->AddAttr("num_distributed_variables", num_new_distributed_vars);

  if (!mirrored_variable_indices.empty()) {
    std::vector<int> new_mirrored_variable_indices;
    absl::flat_hash_set<int> old_mirrored_variable_indices_set;
    for (int index : mirrored_variable_indices) {
      old_mirrored_variable_indices_set.insert(index);
    }
    for (int i = 0; i < num_per_replica_inputs + num_distributed_vars; i++) {
      auto iter = arg_index_mapping.find(i);
      if (iter != arg_index_mapping.end() &&
          old_mirrored_variable_indices_set.contains(iter->first)) {
        new_mirrored_variable_indices.push_back(iter->second);
      }
    }
    xla_node->ClearAttr(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR);
    xla_node->AddAttr(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                      new_mirrored_variable_indices);
  }

  int num_replicated_inputs = num_per_replica_inputs + num_distributed_vars;
  std::vector<DataType> new_broadcast_input_types;
  for (int i = 0; i < broadcast_input_types.size(); i++) {
    int arg_index = num_replicated_inputs + i;
    if (arg_index_mapping.find(arg_index) != arg_index_mapping.end()) {
      new_broadcast_input_types.push_back(broadcast_input_types[i]);
    }
  }
  xla_node->ClearAttr("Tbroadcast_inputs");
  xla_node->AddAttr("Tbroadcast_inputs", new_broadcast_input_types);
  int new_num_variables = 0;
  for (int i = 0; i < num_variables; i++) {
    int arg_index = num_replicated_inputs + broadcast_input_types.size() + i;
    if (arg_index_mapping.find(arg_index) != arg_index_mapping.end()) {
      new_num_variables++;
    }
  }
  xla_node->ClearAttr("NumVariables");
  xla_node->AddAttr("NumVariables", new_num_variables);
  std::vector<DataType> new_guaranteed_constant_types;
  for (int i = 0; i < guaranteed_constant_types.size(); i++) {
    int arg_index = num_replicated_inputs + broadcast_input_types.size() +
                    num_variables + i;
    if (arg_index_mapping.find(arg_index) != arg_index_mapping.end()) {
      new_guaranteed_constant_types.push_back(guaranteed_constant_types[i]);
    }
  }
  xla_node->ClearAttr("Tguaranteed_constants");
  xla_node->AddAttr("Tguaranteed_constants", new_guaranteed_constant_types);

  int new_variable_start_index = num_new_per_replica_inputs +
                                 num_new_distributed_vars +
                                 new_broadcast_input_types.size();
  if (xla_node->attrs().Find("_variable_start_index") != nullptr) {
    xla_node->ClearAttr("_variable_start_index");
    xla_node->AddAttr("_variable_start_index", new_variable_start_index);
  }
  int new_guaranteed_const_start_index =
      new_variable_start_index + new_num_variables;
  if (xla_node->attrs().Find("_guaranteed_const_start_index") != nullptr) {
    xla_node->ClearAttr("_guaranteed_const_start_index");
    xla_node->AddAttr("_guaranteed_const_start_index",
                      new_guaranteed_const_start_index);
  }

  VLOG(4) << "RemoveUnusedXlaInput host graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_input_host_", xla_func_name), *g);
  VLOG(4) << "RemoveUnusedXlaInput XLA graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_input_xla_", xla_func_name),
                 *xla_graph);

  return OkStatus();
}

// Move outside compilation nodes at the end of XLA computation to host.
// For XLA computation graph, we will add new _Retval nodes to replace those
// outside compilation nodes.
// For host graph, we will move those outside compilation nodes to host,
// replicate them, and use them as XLA node's output.
Status MoveTailOutsideCompilationToHost(
    const string& outside_compilation_attr_name, const string& xla_func_name,
    const std::string& cluster_name, Graph* g, Graph* xla_graph, Node* xla_node,
    Node* pivot_node) {
  // Find outside compilation nodes that only have _Retval or other outside
  // compilation nodes as output. These nodes will be moved to host graph.
  std::vector<Node*> oc_nodes_at_tail;
  const string kOnlyRetOrOcOutputAttrName = "_xla_only_ret_or_oc_output";
  DFS(
      *xla_graph, /*enter=*/nullptr,
      [&](Node* n) {
        bool has_non_ret_or_oc_output = false;
        for (const Edge* e : n->out_edges()) {
          if (e->dst() == xla_graph->sink_node()) {
            continue;
          }
          if (!e->dst()->IsRetval() &&
              (!HasNodeAttr(e->dst()->def(), outside_compilation_attr_name) ||
               !HasNodeAttr(e->dst()->def(), kOnlyRetOrOcOutputAttrName))) {
            has_non_ret_or_oc_output = true;
            break;
          }
        }
        if (HasNodeAttr(n->def(), outside_compilation_attr_name) &&
            !has_non_ret_or_oc_output) {
          n->AddAttr(kOnlyRetOrOcOutputAttrName, true);
          oc_nodes_at_tail.push_back(n);
        }
      },
      NodeComparatorName());
  if (VLOG_IS_ON(5)) {
    for (Node* n : oc_nodes_at_tail) {
      VLOG(5) << "oc_nodes_at_tail: " << n->DebugString();
    }
  }

  // Record input edges from `oc_nodes_at_tail`. We will create an _Retval node
  // for each of these edges. An obvious optimization here is to deduplicate
  // these edges by <src, src_output>. But that optimization will complicate
  // the code, and in practice we usually do not have input edges with the
  // same <src, src_output>.
  std::vector<const Edge*> oc_input_edges;
  std::vector<DataType> new_ret_types;
  for (Node* n : oc_nodes_at_tail) {
    for (const Edge* e : n->in_edges()) {
      if (!e->IsControlEdge() &&
          !HasNodeAttr(e->src()->def(), kOnlyRetOrOcOutputAttrName)) {
        VLOG(5) << "oc_input_edges: " << e->DebugString();
        oc_input_edges.push_back(e);
        new_ret_types.push_back(e->src()->output_type(e->src_output()));
      }
    }
  }
  std::vector<DataType> output_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->attrs(), "output_types", &output_types));
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->attrs(), "num_replicas", &num_replicas));
  int old_num_replicated_outputs = output_types.size() / num_replicas;
  int new_num_replicated_outputs =
      old_num_replicated_outputs + oc_input_edges.size();
  VLOG(5) << "old_num_replicated_outputs: " << old_num_replicated_outputs;
  VLOG(5) << "new_num_replicated_outputs: " << new_num_replicated_outputs;

  // Update `output_types` attribute for `xla_node`.
  std::vector<DataType> new_output_types;
  for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
    for (int i = 0; i < old_num_replicated_outputs; i++) {
      new_output_types.push_back(output_types[i]);
    }
    for (int i = old_num_replicated_outputs; i < new_num_replicated_outputs;
         i++) {
      new_output_types.push_back(new_ret_types[i - old_num_replicated_outputs]);
    }
  }
  xla_node->ClearAttr("output_types");
  xla_node->AddAttr("output_types", new_output_types);

  // Re-order old replicated output edges. Since a node could potentially
  // connect to multiple nodes, build a vector<vector<pair>> mapping of
  // output index to input nodes/index.
  // The outer vector represents the output index, the inner vector
  // represents the destination node and input index pair with the possibility
  // of multiple node/index pairs.
  std::vector<std::vector<std::pair<Node*, int>>> replicated_outputs(
      old_num_replicated_outputs * num_replicas);
  std::vector<const Edge*> old_replicated_edges;
  for (const Edge* e : xla_node->out_edges()) {
    if (e->src_output() >= 0 &&
        e->src_output() < old_num_replicated_outputs * num_replicas) {
      replicated_outputs[e->src_output()].push_back(
          std::make_pair(e->dst(), e->dst_input()));
      old_replicated_edges.push_back(e);
    }
  }
  for (const Edge* e : old_replicated_edges) {
    g->RemoveEdge(e);
  }
  for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
    for (int output_index = 0; output_index < old_num_replicated_outputs;
         output_index++) {
      for (const auto& node_input_pair :
           replicated_outputs[replica_id * old_num_replicated_outputs +
                              output_index]) {
        Node* dst = node_input_pair.first;
        int dst_input = node_input_pair.second;
        g->AddEdge(xla_node,
                   replica_id * new_num_replicated_outputs + output_index, dst,
                   dst_input);
      }
    }
  }

  // Copy all nodes in `oc_nodes_at_tail` to host graph, and also replicate
  // them.
  std::map<Node*, std::vector<Node*>> node_images;
  for (Node* n : oc_nodes_at_tail) {
    for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
      NodeDef copy_def = n->def();
      copy_def.set_name(absl::StrCat(n->name(), "_tail_oc/R", replica_id));
      copy_def.clear_device();

      TF_ASSIGN_OR_RETURN(Node * copy_node, g->AddNode(copy_def));

      copy_node->AddAttr(kXlaReplicaIdAttrName, replica_id);
      copy_node->AddAttr(kTPUReplicateAttr, cluster_name);

      for (const Edge* e : n->out_edges()) {
        if (e->dst() == xla_graph->sink_node()) {
          continue;
        }
        // Either e->dst() is _Retval, or it's in `node_images`.
        if (e->dst()->IsRetval()) {
          int index;
          TF_RETURN_IF_ERROR(GetNodeAttr(e->dst()->attrs(), "index", &index));
          for (const auto& output :
               replicated_outputs[replica_id * old_num_replicated_outputs +
                                  index]) {
            // Remove original input edge, if existent.
            const Edge* original_edge;
            Status s = output.first->input_edge(output.second, &original_edge);
            if (s.ok()) {
              g->RemoveEdge(original_edge);
            }
            g->AddEdge(copy_node, e->src_output(), output.first, output.second);
          }
        } else {
          g->AddEdge(copy_node, e->src_output(),
                     node_images[e->dst()][replica_id], e->dst_input());
        }
      }

      // Add attribute "_xla_tail_outside_compilation" to `copy_node`, and add a
      // control edge between `xla_node` and `copy_node`. As a result, in later
      // rewriting pass, a control edge will be added between `copy_node` and
      // "control_after" node for the XLA computation, so `copy_node` will be
      // executed before XLA computation's final results.
      copy_node->AddAttr("_xla_tail_outside_compilation", true);
      g->AddControlEdge(xla_node, copy_node);

      // Add control edge between `pivot_node` and `copy_node`, so `copy_node`
      // belongs to same while loop as `xla_node`.
      if (pivot_node) {
        g->AddControlEdge(pivot_node, copy_node);
      }

      node_images[n].push_back(copy_node);
    }
  }

  // Connect new output values of `xla_node` to dst nodes of `oc_input_edges`.
  for (int i = 0; i < new_ret_types.size(); i++) {
    const Edge* original_edge = oc_input_edges[i];
    for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
      int src_output = replica_id * new_num_replicated_outputs +
                       old_num_replicated_outputs + i;
      Node* dst = node_images[original_edge->dst()][replica_id];
      g->AddEdge(xla_node, src_output, dst, original_edge->dst_input());
    }
  }

  // Create new _Retval nodes in `xla_graph`.
  for (int i = old_num_replicated_outputs; i < new_num_replicated_outputs;
       i++) {
    NodeDefBuilder ret_builder(absl::StrCat("ret_", i),
                               FunctionLibraryDefinition::kRetOp);
    ret_builder.Attr("T", new_ret_types[i - old_num_replicated_outputs]);
    ret_builder.Attr("index", i);
    const Edge* original_edge = oc_input_edges[i - old_num_replicated_outputs];
    Node* src = original_edge->src();
    int src_output = original_edge->src_output();
    ret_builder.Input(src->name(), src_output, src->output_type(src_output));
    NodeDef ret_def;
    TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
    TF_ASSIGN_OR_RETURN(Node * ret_node, xla_graph->AddNode(ret_def));
    xla_graph->RemoveEdge(original_edge);
    xla_graph->AddEdge(src, src_output, ret_node, 0);
  }

  // Remove `oc_nodes_at_tail`.
  for (Node* n : oc_nodes_at_tail) {
    xla_graph->RemoveNode(n);
  }

  // We cannot leave _Retval with no input. Add a placeholder input, which will
  // be removed later with unused _Retval.
  std::vector<Node*> unused_rets;
  for (Node* n : xla_graph->nodes()) {
    if (n->IsRetval() && n->in_edges().empty()) {
      unused_rets.push_back(n);
    }
  }
  for (Node* n : unused_rets) {
    NodeDefBuilder builder(absl::StrCat("placeholder_", n->name()),
                           "Placeholder");
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "T", &dtype));
    builder.Attr("dtype", dtype);
    builder.Attr(kXlaIsPlaceholderForTailOcAttrName, true);
    NodeDef def;
    TF_RETURN_IF_ERROR(builder.Finalize(&def));
    TF_ASSIGN_OR_RETURN(Node * placeholder, xla_graph->AddNode(def));
    xla_graph->AddEdge(placeholder, 0, n, 0);
  }

  VLOG(4) << "MoveTailOutsideCompilationToHost host graph: "
          << DumpGraphToFile(absl::StrCat("move_tail_oc_host_", xla_func_name),
                             *g);
  VLOG(4) << "MoveTaildOutsideCompilationToHost XLA graph: "
          << DumpGraphToFile(absl::StrCat("move_tail_oc_xla_", xla_func_name),
                             *xla_graph);

  return OkStatus();
}

Status ReplaceArgUsedByOutsideCompilationWithPlaceholder(
    const string& outside_compilation_attr_name, const string& xla_func_name,
    Graph* g, Graph* xla_graph, Node* xla_node) {
  std::vector<DataType> input_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "Tinputs", &input_types));
  int num_distributed_vars;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "num_distributed_variables",
                                 &num_distributed_vars));
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->attrs(), "num_replicas", &num_replicas));
  int num_per_replica_inputs =
      (input_types.size() - num_distributed_vars) / num_replicas;

  for (Node* n : xla_graph->op_nodes()) {
    if (!n->IsArg()) {
      continue;
    }

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "T", &dtype));
    // TODO(b/74023706): enable moving normal data tensors.
    if (dtype != DT_RESOURCE) {
      continue;
    }

    std::vector<const Edge*> oc_out_edges;
    for (const Edge* e : n->out_edges()) {
      if (e->IsControlEdge() ||
          !HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr)) {
        continue;
      }

      oc_out_edges.push_back(e);
    }
    if (oc_out_edges.empty()) {
      continue;
    }

    // Sometimes `xla_node` can have a lot of inputs, calling Node::input_edge
    // will become very expensive in this case because it is doing a linear
    // search inside. Create an input_edges vector ahead to make the lookups
    // faster.
    std::vector<const Edge*> input_edges;
    TF_RETURN_IF_ERROR(xla_node->input_edges(&input_edges));

    // Build an IdentityN node to record inputs for this _Arg node.
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    string oc_identifier = absl::StrCat("oc_only_arg_", index);
    NodeDefBuilder id_builder(absl::StrCat(oc_identifier, "_inputs"),
                              "IdentityN");
    std::vector<DataType> dtypes(num_replicas, dtype);
    id_builder.Attr("T", dtypes);
    id_builder.Attr(kXlaOutsideCompilationInputsAttrName, oc_identifier);
    std::vector<NodeDefBuilder::NodeOut> inputs(num_replicas);
    if (index >= num_per_replica_inputs) {
      const Edge* e = input_edges.at(num_replicas * num_per_replica_inputs +
                                     (index - num_per_replica_inputs));
      for (int i = 0; i < num_replicas; i++) {
        inputs[i] =
            NodeDefBuilder::NodeOut{e->src()->name(), e->src_output(),
                                    e->src()->output_type(e->src_output())};
      }
    } else {
      for (int i = 0; i < num_replicas; i++) {
        const Edge* e = input_edges.at(i * num_per_replica_inputs + index);
        inputs[i] =
            NodeDefBuilder::NodeOut{e->src()->name(), e->src_output(),
                                    e->src()->output_type(e->src_output())};
      }
    }
    id_builder.Input(inputs);
    NodeDef id_def;
    TF_RETURN_IF_ERROR(id_builder.Finalize(&id_def));
    TF_ASSIGN_OR_RETURN(Node * id_node, g->AddNode(id_def));
    if (index >= num_per_replica_inputs) {
      const Edge* e = input_edges.at(num_replicas * num_per_replica_inputs +
                                     (index - num_per_replica_inputs));
      for (int i = 0; i < num_replicas; i++) {
        g->AddEdge(e->src(), e->src_output(), id_node, i);
      }
    } else {
      for (int i = 0; i < num_replicas; i++) {
        const Edge* e = input_edges.at(i * num_per_replica_inputs + index);
        g->AddEdge(e->src(), e->src_output(), id_node, i);
      }
    }

    for (const Edge* e : oc_out_edges) {
      // 'e' will use a new Placeholder node as input.
      NodeDefBuilder ph_builder(xla_graph->NewName("ph_for_arg_in_oc_"),
                                "Placeholder");
      ph_builder.Attr("dtype", dtype);

      string outside_compilation_attr;
      TF_RETURN_IF_ERROR(GetNodeAttr(e->dst()->def(), kOutsideCompilationAttr,
                                     &outside_compilation_attr));
      ph_builder.Attr(kOutsideCompilationAttr, outside_compilation_attr);
      ph_builder.Attr(kXlaOutsideCompilationInputsAttrName, oc_identifier);
      ph_builder.Attr(kXlaIsPlaceholderForArg, true);
      NodeDef ph_def;
      TF_RETURN_IF_ERROR(ph_builder.Finalize(&ph_def));
      TF_ASSIGN_OR_RETURN(Node * ph_node, xla_graph->AddNode(ph_def));
      Node* dst = e->dst();
      int dst_input = e->dst_input();
      xla_graph->RemoveEdge(e);
      xla_graph->AddEdge(ph_node, 0, dst, dst_input);
      xla_graph->AddControlEdge(xla_graph->source_node(), ph_node);
    }
  }
  VLOG(4) << "ReplaceOutsideCompilationOnlyArgWithPlaceholder host graph: "
          << DumpGraphToFile(
                 absl::StrCat("replace_oc_only_arg_host_", xla_func_name), *g);
  VLOG(4) << "ReplaceOutsideCompilationOnlyArgWithPlaceholder XLA graph: "
          << DumpGraphToFile(
                 absl::StrCat("replace_oc_only_arg_xla_", xla_func_name),
                 *xla_graph);
  return OkStatus();
}

// If there are any unused _Retval nodes in `xla_graph` (whose input is a
// Placeholder node), remove them from `xla_graph` and remove corresponding
// output edge in host graph `g`.
Status RemoveUnusedXlaOutput(const string& xla_func_name, Graph* g,
                             Graph* xla_graph, Node* xla_node) {
  // Find unused _Retval nodes, and remove them.
  std::vector<DataType> output_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "output_types", &output_types));
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "num_replicas", &num_replicas));
  int num_replicated_outputs = output_types.size() / num_replicas;
  std::set<int> ret_indices_to_remove;
  std::vector<Node*> ret_nodes_to_update, nodes_to_remove;
  int num_rets = 0;
  for (Node* n : xla_graph->nodes()) {
    if (!n->IsRetval()) {
      continue;
    }

    num_rets++;

    const Edge* e;
    TF_RETURN_IF_ERROR(n->input_edge(0, &e));
    if (e->src()->type_string() != "Placeholder" ||
        !HasNodeAttr(e->src()->def(), kXlaIsPlaceholderForTailOcAttrName)) {
      ret_nodes_to_update.push_back(n);
      continue;
    }

    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    ret_indices_to_remove.insert(index);
    nodes_to_remove.push_back(e->src());
    nodes_to_remove.push_back(n);
  }
  for (Node* n : nodes_to_remove) {
    xla_graph->RemoveNode(n);
  }

  // Update `index` for other _Arg nodes.
  std::map<int, int> ret_index_mapping;
  int new_ret_index = 0;
  for (int i = 0; i < num_rets; i++) {
    if (ret_indices_to_remove.find(i) != ret_indices_to_remove.end()) {
      continue;
    } else {
      ret_index_mapping[i] = new_ret_index;
      new_ret_index++;
    }
  }
  for (Node* n : ret_nodes_to_update) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "index", &index));
    n->ClearAttr("index");
    n->AddAttr("index", ret_index_mapping[index]);
  }

  // Update `output_types` attribute for `xla_node`.
  std::vector<DataType> new_output_types;
  for (int i = 0; i < num_replicas; i++) {
    for (const auto& e : ret_index_mapping) {
      new_output_types.push_back(output_types[e.first]);
    }
  }

  xla_node->ClearAttr("output_types");
  xla_node->AddAttr("output_types", new_output_types);

  // Re-order replicated output edges for `xla_node`.
  std::vector<std::vector<const Edge*>> output_edges(num_replicas *
                                                     num_replicated_outputs);
  for (const Edge* e : xla_node->out_edges()) {
    if (e->src_output() >= 0 &&
        e->src_output() < num_replicas * num_replicated_outputs) {
      output_edges[e->src_output()].push_back(e);
    }
  }
  for (int i = 0; i < num_replicas; i++) {
    for (int j = 0; j < num_replicated_outputs; j++) {
      auto iter = ret_index_mapping.find(j);
      if (iter != ret_index_mapping.end()) {
        for (const Edge* e : output_edges[i * num_replicated_outputs + j]) {
          Node* dst = e->dst();
          int dst_input = e->dst_input();
          int src_output =
              i * (num_replicated_outputs - ret_indices_to_remove.size()) +
              iter->second;
          g->RemoveEdge(e);
          g->AddEdge(xla_node, src_output, dst, dst_input);
        }
      } else {
        TF_RET_CHECK(output_edges[i * num_replicated_outputs + j].empty())
            << "Output edge not removed: "
            << output_edges[i * num_replicated_outputs + j][0]->DebugString();
      }
    }
  }

  VLOG(4) << "RemoveUnusedXlaOutput host graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_output_host_", xla_func_name), *g);
  VLOG(4) << "RemoveUnusedXlaOutput XLA graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_output_xla_", xla_func_name),
                 *xla_graph);

  return OkStatus();
}

// For data edges between _Arg and _Retval in `xla_graph`, remove them and
// change input/output edges in `g` (host graph). For now, we only consider
// replicated inputs.
Status RemoveEdgesBetweenArgAndRetval(const string& xla_func_name, Graph* g,
                                      Graph* xla_graph, Node* xla_node) {
  // Collect data edges between _Arg and _Retval.
  int num_replicas;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "num_replicas", &num_replicas));
  std::vector<DataType> input_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->def(), "Tinputs", &input_types));
  int num_distributed_vars;
  TF_RETURN_IF_ERROR(GetNodeAttr(xla_node->attrs(), "num_distributed_variables",
                                 &num_distributed_vars));
  int old_num_per_replica_inputs =
      (input_types.size() - num_distributed_vars) / num_replicas;
  std::vector<DataType> output_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(xla_node->def(), "output_types", &output_types));
  int old_num_outputs = output_types.size() / num_replicas;
  std::vector<const Edge*> edges;
  for (const Edge* e : xla_graph->edges()) {
    if (!e->IsControlEdge() && e->src()->IsArg() && e->dst()->IsRetval()) {
      edges.push_back(e);
    }
  }

  // In host graph `g`, remove output edge from `xla_node` and connect input &
  // output directly.
  std::vector<std::vector<const Edge*>> xla_node_out_edges(
      xla_node->num_outputs());
  for (const Edge* e : xla_node->out_edges()) {
    if (!e->IsControlEdge()) {
      xla_node_out_edges[e->src_output()].push_back(e);
    }
  }

  // Sometimes `xla_node` can have a lot of inputs, calling Node::input_edge
  // will become very expensive in this case because it is doing a linear
  // search inside. Create an input_edges vector ahead to make the lookups
  // faster.
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(xla_node->input_edges(&input_edges));
  for (const Edge* e : edges) {
    int arg_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(e->src()->def(), "index", &arg_index));
    int ret_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(e->dst()->def(), "index", &ret_index));

    for (int replica_id = 0; replica_id < num_replicas; replica_id++) {
      int input_index;
      if (arg_index < old_num_per_replica_inputs) {
        input_index = replica_id * old_num_per_replica_inputs + arg_index;
      } else {
        input_index = num_replicas * old_num_per_replica_inputs +
                      (arg_index - old_num_per_replica_inputs);
      }
      const Edge* input_edge = input_edges.at(input_index);

      int output_index = replica_id * old_num_outputs + ret_index;
      for (const Edge* output_edge : xla_node_out_edges[output_index]) {
        Node* dst = output_edge->dst();
        int dst_input = output_edge->dst_input();

        g->RemoveEdge(output_edge);
        g->AddEdge(input_edge->src(), input_edge->src_output(), dst, dst_input);
      }
    }
  }

  // Remove edges from `xla_graph`. Add a Placeholder node for the _Retval node,
  // which will be removed by `RemoveUnusedXlaOutput()` later.
  for (const Edge* e : edges) {
    NodeDefBuilder placeholder_builder(
        absl::StrCat("placeholder_", e->dst()->name()), "Placeholder");
    placeholder_builder.Attr("dtype", e->src()->output_type(e->src_output()));
    placeholder_builder.Attr(kXlaIsPlaceholderForTailOcAttrName, true);
    NodeDef placeholder_def;
    TF_RETURN_IF_ERROR(placeholder_builder.Finalize(&placeholder_def));
    TF_ASSIGN_OR_RETURN(Node * placeholder_node,
                        xla_graph->AddNode(placeholder_def));

    Node* dst = e->dst();
    int dst_input = e->dst_input();
    xla_graph->RemoveEdge(e);
    xla_graph->AddEdge(placeholder_node, 0, dst, dst_input);
  }

  VLOG(4) << "RemoveUnusedArgRetvalPair host graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_arg_ret_host_", xla_func_name),
                 *g);
  VLOG(4) << "RemoveUnusedArgRetvalPair XLA graph: "
          << DumpGraphToFile(
                 absl::StrCat("remove_unused_arg_ret_xla_", xla_func_name),
                 *xla_graph);

  return OkStatus();
}

// Remove any TPUReplicatedInput nodes with no output edges. Those nodes are
// usually TPUMirroredVariable handles which are not used by any computations.
void RemoveUnusedTPUReplicatedInputs(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->type_string() == kTPUReplicatedInput) {
      bool has_output = false;
      for (const Edge* e : n->out_edges()) {
        if (!e->dst()->IsSink()) {
          has_output = true;
          break;
        }
      }
      if (!has_output) {
        // Remove any TPUPartitionedInput node from the src nodes of the
        // to-be-removed TPUReplicatedInput node
        std::vector<Node*> to_be_removed_src_nodes;
        for (const auto& e_in : n->in_edges()) {
          if (!e_in->IsControlEdge() &&
              e_in->src()->type_string() == kTPUPartitionedInput)
            to_be_removed_src_nodes.push_back(e_in->src());
        }
        graph->RemoveNode(n);
        for (Node* node : to_be_removed_src_nodes) {
          graph->RemoveNode(node);
        }
      }
    }
  }
}

// We might have duplicated cluster names in the graph, e.g. when a tf.function
// containing tpu_strategy.run() is called multiple times with
// the same inputs. Find clusters with duplicated names and rename them.
Status RenameClustersWithDuplicatedNames(Graph* g) {
  // Find all TPU clusters by finding all TPUReplicateMetadata nodes.
  std::unordered_map<string, std::vector<Node*>> cluster_name_to_metadata_nodes;
  std::unordered_set<string> cluster_names;
  for (Node* n : g->nodes()) {
    if (n->type_string() != "TPUReplicateMetadata") {
      continue;
    }
    string cluster_name;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), kTPUReplicateAttr, &cluster_name));
    cluster_name_to_metadata_nodes[cluster_name].push_back(n);
    cluster_names.insert(cluster_name);
  }
  // Look for clusters with duplicated name.
  for (const auto& iter : cluster_name_to_metadata_nodes) {
    if (iter.second.size() == 1) {
      continue;
    }

    // Rename clusters.
    for (int i = 1; i < iter.second.size(); i++) {
      // Find an available cluster name.
      string new_cluster_name;
      int cluster_name_suffix = 1;
      while (true) {
        new_cluster_name = absl::StrCat(iter.first, "_", cluster_name_suffix);
        if (cluster_names.find(new_cluster_name) == cluster_names.end()) {
          break;
        }
        cluster_name_suffix++;
      }
      cluster_names.insert(new_cluster_name);

      // Change _tpu_replicate attribute for all nodes in this cluster.
      // Start with outputs of TPUReplicateMetadata and follow output edges.
      std::queue<Node*> queue;
      queue.push(iter.second.at(i));
      std::unordered_set<Node*> visited;
      while (!queue.empty()) {
        Node* n = queue.front();
        queue.pop();

        visited.insert(n);

        n->ClearAttr(kTPUReplicateAttr);
        n->AddAttr(kTPUReplicateAttr, new_cluster_name);

        string cluster_name;
        for (const Edge* e : n->out_edges()) {
          if (GetNodeAttr(e->dst()->def(), kTPUReplicateAttr, &cluster_name)
                  .ok() &&
              cluster_name == iter.first &&
              visited.find(e->dst()) == visited.end()) {
            queue.push(e->dst());
          }
        }
      }
      // Change "_tpu_compilation_status" attr for TPUCompilationResult node.
      for (const Edge* e : iter.second.at(i)->out_edges()) {
        if (e->dst()->type_string() == "TPUCompilationResult") {
          e->dst()->ClearAttr("_tpu_compilation_status");
          e->dst()->AddAttr("_tpu_compilation_status", new_cluster_name);
        }
      }
    }
  }
  return OkStatus();
}

// Instantiate a function that is associated with a functional control flow
// node. The function name is found by looking up `function_name_attr` of given
// node.
xla::StatusOr<std::unique_ptr<FunctionBody>> InstantiateAssociatedFunction(
    const Node& n, absl::string_view function_name_attr,
    FunctionLibraryDefinition* fld) {
  std::unique_ptr<FunctionBody> fbody;
  NameAttrList func_attr_list;
  TF_RETURN_IF_ERROR(GetNodeAttr(n.def(), function_name_attr, &func_attr_list));
  const FunctionDef* fdef = fld->Find(func_attr_list.name());
  if (fdef == nullptr) {
    return errors::Internal("Cannot find ", function_name_attr, " function",
                            "for node ", n.DebugString());
  }
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fdef, AttrSlice(&func_attr_list.attr()), fld, &fbody));
  return fbody;
}

// Find inputs of If node that are only used for outside compilation if used at
// all in both if/else branches
xla::StatusOr<absl::flat_hash_set<int>> FindArgsToLiftForIfNode(
    const Node& if_node, FunctionLibraryDefinition* fld) {
  absl::flat_hash_set<int> args_to_lift_indices;
  std::vector<DataType> dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(if_node.def(), "Tin", &dtypes));

  int num_args = dtypes.size();

  for (int i = 0; i < num_args; i++) {
    // TODO(b/74023706): enable non resource inputs as well.
    if (dtypes[i] == DT_RESOURCE) {
      args_to_lift_indices.insert(i);
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionBody> then_branch_fbody,
      InstantiateAssociatedFunction(if_node, "then_branch", fld));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionBody> else_branch_fbody,
      InstantiateAssociatedFunction(if_node, "else_branch", fld));

  for (int i = 0; i < num_args; ++i) {
    bool used = false;

    const Node* then_arg_node = then_branch_fbody->arg_nodes[i];
    for (const Edge* e : then_arg_node->out_edges()) {
      used = true;
      if (e->IsControlEdge() ||
          HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr))
        continue;

      args_to_lift_indices.erase(i);
      break;
    }

    const Node* else_arg_node = else_branch_fbody->arg_nodes[i];
    for (const Edge* e : else_arg_node->out_edges()) {
      used = true;
      if (e->IsControlEdge() ||
          HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr))
        continue;

      args_to_lift_indices.erase(i);
      break;
    }

    // Do not lift arguments that are not used at all. Otherwise, this unused
    // arg would be outside compiled, its output tensor will be forced to
    // transfer to host needlessly.
    if (!used) args_to_lift_indices.erase(i);
  }

  return args_to_lift_indices;
}

// Find inputs of While node that are:
// 1. not used in cond func,
// 2. only used for outside compilation in body func,
// 3. loop invariant.
// These inputs can be lifted out of the while loop.
xla::StatusOr<absl::flat_hash_set<int>> FindArgsToLiftForWhileNode(
    Node* while_node, FunctionLibraryDefinition* fld) {
  // DT_RESOURCE inputs are candidates.
  absl::flat_hash_set<int> result;
  std::vector<DataType> dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "T", &dtypes));
  for (int i = 0; i < dtypes.size(); i++) {
    // TODO(b/74023706): enable non resource inputs as well.
    if (dtypes[i] == DT_RESOURCE) {
      result.insert(i);
    }
  }

  // Remove inputs that are used in cond func.
  NameAttrList cond_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "cond", &cond_func));
  const FunctionDef* cond_fdef = fld->Find(cond_func.name());
  if (cond_fdef == nullptr) {
    return errors::Internal("Cannot find cond function ", cond_func.name(),
                            " for while node ", while_node->DebugString());
  }
  std::unique_ptr<FunctionBody> cond_fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *cond_fdef, AttrSlice(&cond_func.attr()), fld, &cond_fbody));
  for (int i = 0; i < cond_fbody->arg_nodes.size(); i++) {
    const Node* arg_node = cond_fbody->arg_nodes[i];
    for (const Edge* e : arg_node->out_edges()) {
      if (!e->IsControlEdge()) {
        result.erase(i);
      }
    }
  }

  // Remove inputs that are not loop invariant.
  NameAttrList body_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "body", &body_func));
  const FunctionDef* body_fdef = fld->Find(body_func.name());
  if (body_fdef == nullptr) {
    return errors::Internal("Cannot find body function ", body_func.name(),
                            " for while node ", while_node->DebugString());
  }
  std::unique_ptr<FunctionBody> body_fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *body_fdef, AttrSlice(&body_func.attr()), fld, &body_fbody));
  for (int i = 0; i < body_fbody->ret_nodes.size(); i++) {
    const Node* node = body_fbody->ret_nodes[i];
    do {
      TF_RETURN_IF_ERROR(node->input_node(0, &node));
    } while (node->IsIdentity());
    if (node != body_fbody->arg_nodes[i]) {
      result.erase(i);
    }
  }

  // Remove inputs that only have one output edge (loop invariant, but not used
  // in outside compilation).
  for (int i = 0; i < body_fbody->arg_nodes.size(); i++) {
    const Node* arg_node = body_fbody->arg_nodes[i];
    int data_edge_count = std::count_if(
        arg_node->out_edges().begin(), arg_node->out_edges().end(),
        [](const Edge* e) { return !e->IsControlEdge(); });
    if (data_edge_count == 1) {
      result.erase(i);
    }
  }

  // Remove inputs that have non-outside-compilation usage.
  for (int i = 0; i < body_fbody->arg_nodes.size(); i++) {
    const Node* arg_node = body_fbody->arg_nodes[i];
    for (const Edge* e : arg_node->out_edges()) {
      if (!e->dst()->IsRetval() &&
          !HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr)) {
        result.erase(i);
        break;
      }
    }
  }

  return result;
}

// Find inputs of function call node that are only used for outside compilation.
// These inputs can be lifted out of the function call node.
xla::StatusOr<absl::flat_hash_set<int>> FindArgsToLiftForCallNode(
    Node* call_node, const FunctionBody& fbody) {
  // DT_RESOURCE inputs are candidates.
  absl::flat_hash_set<int> result;
  std::vector<DataType> dtypes(call_node->input_types().begin(),
                               call_node->input_types().end());
  for (int i = 0; i < dtypes.size(); i++) {
    // TODO(b/74023706): enable for non resource inputs as well.
    if (dtypes[i] == DT_RESOURCE) {
      result.insert(i);
    }
  }

  // Remove inputs that have non-outside-compilation usage, or not used at all.
  for (int i = 0; i < fbody.arg_nodes.size(); i++) {
    const Node* arg_node = fbody.arg_nodes[i];
    if (arg_node->out_edges().empty()) {
      result.erase(i);
      continue;
    }

    for (const Edge* e : arg_node->out_edges()) {
      if (!HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr)) {
        result.erase(i);
        break;
      }
    }
  }
  return result;
}

Status LiftOutsideCompilationOnlyArgs(Graph* g, FunctionLibraryRuntime* flr,
                                      FunctionLibraryDefinition* fld,
                                      int* lifted_arg_count, bool* rewritten);

Status LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
    const FunctionBody& fbody, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, int* lifted_arg_count,
    absl::optional<string> new_func_name, bool* rewritten) {
  *rewritten = false;
  TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgs(
      fbody.graph, flr, fld, lifted_arg_count, rewritten));

  if (*rewritten) {
    FunctionDef rewritten_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(
        *(fbody.graph), fbody.fdef.signature().name(), &rewritten_fdef));
    if (new_func_name) {
      rewritten_fdef.mutable_signature()->set_name(*new_func_name);
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(rewritten_fdef));
    } else {
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(fbody.fdef.signature().name(), rewritten_fdef));
    }
  }

  return OkStatus();
}

Status MakeIdentityNodesForArgsToLift(
    const absl::flat_hash_set<int>& args_to_lift,
    const int arg_to_input_edge_offset, Graph* g, Node* n,
    absl::flat_hash_map<int, string>* lifted_arg_index_to_oc_cluster_name,
    int* lifted_arg_count) {
  int num_input = n->num_inputs();
  for (int arg_index = 0; arg_index < num_input; ++arg_index) {
    if (!args_to_lift.contains(arg_index)) continue;

    int input_edge_index = arg_index + arg_to_input_edge_offset;
    const Edge* arg_edge;
    TF_RETURN_IF_ERROR(n->input_edge(input_edge_index, &arg_edge));

    string node_name =
        g->NewName(absl::StrCat("lifted_arg", *lifted_arg_count));
    (*lifted_arg_count)++;
    (*lifted_arg_index_to_oc_cluster_name)[arg_index] = node_name;
    NodeDefBuilder id_builder(node_name, "Identity");
    id_builder.Attr("T", n->input_type(input_edge_index));
    id_builder.Attr(kOutsideCompilationAttr, id_builder.node_name());
    id_builder.Attr(kXlaIsLiftedArgAttrName, true);
    id_builder.Input(arg_edge->src()->name(), arg_edge->src_output(),
                     n->input_type(input_edge_index));
    NodeDef id_def;
    TF_RETURN_IF_ERROR(id_builder.Finalize(&id_def));
    TF_ASSIGN_OR_RETURN(Node * id_node, g->AddNode(id_def));
    g->AddEdge(arg_edge->src(), arg_edge->src_output(), id_node, 0);
    g->AddControlEdge(id_node, n);
  }

  return OkStatus();
}

// Replaces all usages of lifted args with placeholder nodes. Afterwards,
// removing these args should be safe since they no longer have users.
Status RemoveArgsToLiftFromFunctionBody(
    const absl::flat_hash_set<int>& args_to_lift,
    const std::vector<DataType>& arg_dtypes,
    const absl::flat_hash_map<int, string>& lifted_arg_index_to_oc_cluster_name,
    const absl::flat_hash_map<int, int>& index_mapping,
    const FunctionBody* fbody) {
  for (int i = 0; i < fbody->arg_nodes.size(); ++i) {
    Node* arg_node = fbody->arg_nodes[i];

    if (!args_to_lift.contains(i)) {
      int new_index = index_mapping.at(i);
      arg_node->ClearAttr("index");
      arg_node->AddAttr("index", new_index);
      arg_node->ClearAttr("T");
      arg_node->AddAttr("T", arg_dtypes[i]);
      continue;
    }

    std::vector<const Edge*> out_edges_to_oc;
    for (const Edge* e : arg_node->out_edges()) {
      if (HasNodeAttr(e->dst()->def(), kOutsideCompilationAttr)) {
        out_edges_to_oc.push_back(e);
      }
    }

    for (const Edge* e : out_edges_to_oc) {
      string outside_compilation_cluster;
      TF_RETURN_IF_ERROR(GetNodeAttr(e->dst()->def(), kOutsideCompilationAttr,
                                     &outside_compilation_cluster));
      NodeDefBuilder ph_builder(fbody->graph->NewName("lifted_arg"),
                                "Placeholder");
      ph_builder.Attr("dtype", arg_dtypes[i]);
      ph_builder.Attr(kOutsideCompilationAttr, outside_compilation_cluster);
      TF_RET_CHECK(lifted_arg_index_to_oc_cluster_name.contains(i));
      ph_builder.Attr(kXlaLiftedArgOutsideCompilationAttrName,
                      lifted_arg_index_to_oc_cluster_name.at(i));

      NodeDef ph_def;
      TF_RETURN_IF_ERROR(ph_builder.Finalize(&ph_def));

      TF_ASSIGN_OR_RETURN(Node * ph_node, fbody->graph->AddNode(ph_def));

      Node* dst = e->dst();
      int dst_input = e->dst_input();
      fbody->graph->RemoveEdge(e);
      fbody->graph->AddEdge(ph_node, 0, dst, dst_input);
    }

    fbody->graph->RemoveNode(arg_node);
  }

  return OkStatus();
}

Status CleanUpInEdges(const absl::flat_hash_map<int, int>& index_mapping,
                      const int arg_to_input_edge_offset, Graph* g, Node* n) {
  int num_inputs = n->num_inputs();
  for (int i = 0; i < num_inputs; ++i) {
    if (i < arg_to_input_edge_offset) continue;

    int arg_idx = i - arg_to_input_edge_offset;
    const Edge* e;
    TF_RETURN_IF_ERROR(n->input_edge(i, &e));

    // If an edge maps to a lifted argument, simply remove that edge from graph.
    if (!index_mapping.contains(arg_idx)) {
      g->RemoveEdge(e);
      continue;
    }

    // If an edge maps to same input port, nothing to do.
    if (index_mapping.at(arg_idx) == arg_idx) continue;

    g->AddEdge(e->src(), e->src_output(), n,
               index_mapping.at(arg_idx) + arg_to_input_edge_offset);
    g->RemoveEdge(e);
  }

  return OkStatus();
}

Status UpdateTypeAttribute(const absl::flat_hash_map<int, int>& index_mapping,
                           const string& type_attr_name,
                           const std::vector<DataType>& dtypes, Node* n) {
  std::vector<DataType> new_dtypes;
  new_dtypes.reserve(index_mapping.size());
  for (int i = 0; i < dtypes.size(); ++i) {
    if (index_mapping.contains(i)) {
      new_dtypes.emplace_back(dtypes[i]);
    }
  }

  n->ClearAttr(type_attr_name);
  n->AddAttr(type_attr_name, new_dtypes);

  return OkStatus();
}

// While V2 always creates Identity node for each While node output, which is
// not necessary for XLA computation. Remove those Identity nodes.
void RemoveOutputIdentityNodesForWhileV2(Graph* g, Node* while_node) {
  std::vector<const Edge*> edges_to_identity_node;
  for (const Edge* e : while_node->out_edges()) {
    if (!e->IsControlEdge() && e->dst()->IsIdentity()) {
      edges_to_identity_node.push_back(e);
    }
  }
  for (const Edge* e : edges_to_identity_node) {
    Node* identity = e->dst();
    std::vector<const Edge*> out_edges(identity->out_edges().begin(),
                                       identity->out_edges().end());
    for (const Edge* out_edge : out_edges) {
      if (out_edge->IsControlEdge()) {
        g->AddControlEdge(while_node, out_edge->dst());
      } else {
        Node* dst = out_edge->dst();
        int dst_input = out_edge->dst_input();
        g->RemoveEdge(out_edge);
        g->AddEdge(while_node, e->src_output(), dst, dst_input);
      }
    }
    g->RemoveNode(identity);
  }
}

// If corresponding While node output is used, change it to use While node input
// instead.
Status ReplaceOutputEdgesWithInputEdgeSourceForWhile(
    const absl::flat_hash_set<int>& args_to_lift, Graph* g, Node* while_node) {
  std::vector<const Edge*> edges_to_replace;
  for (const Edge* e : while_node->out_edges()) {
    if (args_to_lift.contains(e->src_output())) {
      edges_to_replace.push_back(e);
    }
  }
  for (const Edge* e : edges_to_replace) {
    const Edge* input_edge;
    TF_RETURN_IF_ERROR(while_node->input_edge(e->src_output(), &input_edge));
    Node* dst = e->dst();
    int dst_input = e->dst_input();
    g->RemoveEdge(e);
    g->AddEdge(input_edge->src(), input_edge->src_output(), dst, dst_input);
  }

  return OkStatus();
}

// Calculates mapping from argument index before lifting to index afterwards.
absl::flat_hash_map<int, int> ArgIndexMapping(
    const int num_args, const absl::flat_hash_set<int>& args_to_lift) {
  absl::flat_hash_map<int, int> index_mapping;
  int new_index = 0;
  for (int i = 0; i < num_args; i++) {
    if (!args_to_lift.contains(i)) {
      index_mapping[i] = new_index;
      ++new_index;
    }
  }

  return index_mapping;
}

// Remove outputs of While node body function that maps to lifted arguments.
void CleanUpRetvalsForWhileBody(
    const absl::flat_hash_map<int, int>& index_mapping,
    const std::vector<DataType>& dtypes, FunctionBody* fbody) {
  for (int i = 0; i < fbody->ret_nodes.size(); i++) {
    Node* ret_node = fbody->ret_nodes[i];
    if (index_mapping.contains(i)) {
      int new_index = index_mapping.at(i);
      ret_node->ClearAttr("index");
      ret_node->AddAttr("index", new_index);
      ret_node->ClearAttr("T");
      ret_node->AddAttr("T", dtypes[i]);
    } else {
      fbody->graph->RemoveNode(ret_node);
    }
  }
}

Status LiftOutsideCompilationOnlyArgsFromWhileNode(
    Graph* g, Node* while_node, FunctionLibraryDefinition* fld,
    int* lifted_arg_count, bool* rewritten) {
  *rewritten = false;

  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<int> args_to_lift,
                      FindArgsToLiftForWhileNode(while_node, fld));
  if (args_to_lift.empty()) return OkStatus();

  RemoveOutputIdentityNodesForWhileV2(g, while_node);

  TF_RETURN_IF_ERROR(ReplaceOutputEdgesWithInputEdgeSourceForWhile(
      args_to_lift, g, while_node));

  std::vector<DataType> dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(while_node->def(), "T", &dtypes));

  absl::flat_hash_map<int, int> index_mapping =
      ArgIndexMapping(dtypes.size(), args_to_lift);

  // For each lifted arg, add an outside compilation Identity node to send
  // it to host.
  absl::flat_hash_map<int, string> lifted_arg_index_to_oc_cluster_name;
  TF_RETURN_IF_ERROR(MakeIdentityNodesForArgsToLift(
      args_to_lift, /*arg_to_input_edge_offset=*/0, g, while_node,
      &lifted_arg_index_to_oc_cluster_name, lifted_arg_count));

  // For cond func, remove _Arg nodes.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionBody> cond_fbody,
                      InstantiateAssociatedFunction(*while_node, "cond", fld));
  TF_RETURN_IF_ERROR(RemoveArgsToLiftFromFunctionBody(
      args_to_lift, dtypes, lifted_arg_index_to_oc_cluster_name, index_mapping,
      cond_fbody.get()));

  FunctionDef rewritten_cond_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*(cond_fbody->graph),
                                        cond_fbody->fdef.signature().name(),
                                        &rewritten_cond_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(cond_fbody->fdef.signature().name(),
                                          rewritten_cond_fdef));

  // For body func, remove _Retval nodes, and replace _Arg nodes with
  // Placeholder nodes.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionBody> body_fbody,
                      InstantiateAssociatedFunction(*while_node, "body", fld));

  TF_RETURN_IF_ERROR(RemoveArgsToLiftFromFunctionBody(
      args_to_lift, dtypes, lifted_arg_index_to_oc_cluster_name, index_mapping,
      body_fbody.get()));

  CleanUpRetvalsForWhileBody(index_mapping, dtypes, body_fbody.get());

  FunctionDef rewritten_body_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*(body_fbody->graph),
                                        body_fbody->fdef.signature().name(),
                                        &rewritten_body_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(body_fbody->fdef.signature().name(),
                                          rewritten_body_fdef));

  // Remove edges from lifted args to While node, and change "T" attr of the
  // While node.
  TF_RETURN_IF_ERROR(CleanUpInEdges(
      index_mapping, /*arg_to_input_edge_offset=*/0, g, while_node));

  TF_RETURN_IF_ERROR(
      UpdateTypeAttribute(index_mapping, "T", dtypes, while_node));

  *rewritten = true;

  return OkStatus();
}

Status LiftOutsideCompilationOnlyArgsFromIfNode(Graph* g, Node* if_node,
                                                FunctionLibraryDefinition* fld,
                                                int* lifted_arg_count,
                                                bool* rewritten) {
  *rewritten = false;
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<int> args_to_lift,
                      FindArgsToLiftForIfNode(*if_node, fld));
  if (args_to_lift.empty()) return OkStatus();

  std::vector<DataType> dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(if_node->def(), "Tin", &dtypes));

  absl::flat_hash_map<int, int> index_mapping;
  int new_index = 0;
  for (int i = 0; i < dtypes.size(); i++) {
    if (!args_to_lift.contains(i)) {
      index_mapping[i] = new_index;
      ++new_index;
    }
  }

  // For each lifted arg, add an outside compilation Identity node to send
  // it to host.
  absl::flat_hash_map<int, string> lifted_arg_index_to_oc_cluster_name;
  TF_RETURN_IF_ERROR(MakeIdentityNodesForArgsToLift(
      args_to_lift, /*arg_to_input_edge_offset=*/1, g, if_node,
      &lifted_arg_index_to_oc_cluster_name, lifted_arg_count));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionBody> then_branch_fbody,
      InstantiateAssociatedFunction(*if_node, "then_branch", fld));

  TF_RETURN_IF_ERROR(RemoveArgsToLiftFromFunctionBody(
      args_to_lift, dtypes, lifted_arg_index_to_oc_cluster_name, index_mapping,
      then_branch_fbody.get()));

  FunctionDef rewritten_then_branch_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *(then_branch_fbody->graph), then_branch_fbody->fdef.signature().name(),
      &rewritten_then_branch_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(
      then_branch_fbody->fdef.signature().name(), rewritten_then_branch_fdef));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionBody> else_branch_fbody,
      InstantiateAssociatedFunction(*if_node, "else_branch", fld));

  TF_RETURN_IF_ERROR(RemoveArgsToLiftFromFunctionBody(
      args_to_lift, dtypes, lifted_arg_index_to_oc_cluster_name, index_mapping,
      else_branch_fbody.get()));

  FunctionDef rewritten_else_branch_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *(else_branch_fbody->graph), else_branch_fbody->fdef.signature().name(),
      &rewritten_else_branch_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(
      else_branch_fbody->fdef.signature().name(), rewritten_else_branch_fdef));

  // Remove edges from lifted args to If node, and change "Tin" attr of the
  // If node.
  TF_RETURN_IF_ERROR(CleanUpInEdges(
      index_mapping, /*arg_to_input_edge_offset=*/1, g, if_node));
  TF_RETURN_IF_ERROR(
      UpdateTypeAttribute(index_mapping, "Tin", dtypes, if_node));

  *rewritten = true;

  return OkStatus();
}

Status LiftOutsideCompilationOnlyArgsFromCallNode(
    Graph* g, Node* call_node, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, int* lifted_arg_count, bool* rewritten) {
  *rewritten = false;

  // Instantiate the function.
  NameAttrList func;
  if (fld->Contains(call_node->type_string())) {
    func.set_name(call_node->type_string());
    *func.mutable_attr() = call_node->def().attr();
  } else if (call_node->IsPartitionedCall()) {
    TF_RETURN_IF_ERROR(GetNodeAttr(call_node->def(), "f", &func));
  } else {
    TF_RET_CHECK(call_node->type_string() ==
                 FunctionLibraryDefinition::kGradientOp);
    func.set_name(FunctionLibraryDefinition::kGradientOp);
    *func.mutable_attr() = call_node->def().attr();
  }
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(func.name(), AttrSlice(&func.attr()), &handle));
  auto cleanup_handle = gtl::MakeCleanup(
      [&flr, &handle]() { flr->ReleaseHandle(handle).IgnoreError(); });
  const FunctionBody* fbody = flr->GetFunctionBody(handle);

  // Find _Arg nodes to lift.
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<int> args_to_lift,
                      FindArgsToLiftForCallNode(call_node, *fbody));
  if (args_to_lift.empty()) return OkStatus();

  std::vector<DataType> dtypes;
  dtypes = std::vector<DataType>(call_node->input_types().begin(),
                                 call_node->input_types().end());

  absl::flat_hash_map<int, int> index_mapping =
      ArgIndexMapping(dtypes.size(), args_to_lift);

  // For each lifted arg, add an outside compilation Identity node to send
  // it to host.
  absl::flat_hash_map<int, string> lifted_arg_index_to_oc_cluster_name;
  TF_RETURN_IF_ERROR(MakeIdentityNodesForArgsToLift(
      args_to_lift, /*arg_to_input_edge_offset=*/0, g, call_node,
      &lifted_arg_index_to_oc_cluster_name, lifted_arg_count));

  // Remove _Arg nodes.
  TF_RETURN_IF_ERROR(RemoveArgsToLiftFromFunctionBody(
      args_to_lift, dtypes, lifted_arg_index_to_oc_cluster_name, index_mapping,
      fbody));

  // Store rewritten function as a new function, because the original function
  // might be defined by user and we should not modify it.
  FunctionDef rewritten_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *(fbody->graph), fbody->fdef.signature().name(), &rewritten_fdef));
  string new_func_name =
      fld->UniqueFunctionName(fbody->fdef.signature().name());
  rewritten_fdef.mutable_signature()->set_name(new_func_name);
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(rewritten_fdef));

  // Remove edges from lifted args to call node.
  TF_RETURN_IF_ERROR(CleanUpInEdges(
      index_mapping, /*arg_to_input_edge_offset=*/0, g, call_node));

  // Rewrite the call node to use the rewritten function.
  NodeDef node_def;
  node_def.set_name(g->NewName(call_node->name()));
  node_def.set_op(new_func_name);
  if (call_node->IsPartitionedCall()) {
    NameAttrList f;
    TF_RETURN_IF_ERROR(GetNodeAttr(call_node->def(), "f", &f));
    *node_def.mutable_attr() = f.attr();
  } else if (fld->Contains(call_node->type_string())) {
    *node_def.mutable_attr() = call_node->def().attr();
  } else {
    TF_RET_CHECK(call_node->type_string() ==
                 FunctionLibraryDefinition::kGradientOp);
    *node_def.mutable_attr() = call_node->def().attr();
    node_def.mutable_attr()->erase(FunctionLibraryDefinition::kFuncAttr);
  }
  TF_ASSIGN_OR_RETURN(call_node, ReplaceNode(g, call_node, node_def));

  *rewritten = true;

  return OkStatus();
}

// Lifts outside compilation only _Arg nodes out of If/While/function nodes.
Status LiftOutsideCompilationOnlyArgs(Graph* g, FunctionLibraryRuntime* flr,
                                      FunctionLibraryDefinition* fld,
                                      int* lifted_arg_count, bool* rewritten) {
  *rewritten = false;

  // Handle deeper functional nodes first.
  std::vector<Node*> while_nodes, if_nodes, call_nodes;
  for (Node* n : g->op_nodes()) {
    if (HasNodeAttr(n->def(), kOutsideCompilationAttr)) {
      continue;
    }

    if (n->IsWhileNode()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionBody> body_fbody,
                          InstantiateAssociatedFunction(*n, "body", fld));
      bool func_rewritten = false;
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *body_fbody, flr, fld, lifted_arg_count,
          /*new_func_name=*/absl::nullopt, &func_rewritten));
      *rewritten = *rewritten || func_rewritten;

      while_nodes.push_back(n);
    } else if (n->IsIfNode()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<FunctionBody> then_branch_fbody,
          InstantiateAssociatedFunction(*n, "then_branch", fld));
      bool func_rewritten = false;
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *then_branch_fbody, flr, fld, lifted_arg_count,
          /*new_func_name=*/absl::nullopt, &func_rewritten));
      *rewritten |= func_rewritten;

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<FunctionBody> else_branch_fbody,
          InstantiateAssociatedFunction(*n, "else_branch", fld));
      func_rewritten = false;
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *else_branch_fbody, flr, fld, lifted_arg_count,
          /*new_func_name=*/absl::nullopt, &func_rewritten));
      *rewritten |= func_rewritten;

      if_nodes.push_back(n);
    } else if (IsFunctionCall(*fld, *n)) {
      // Function call nodes need to be rewritten, so handle them later.
      call_nodes.push_back(n);
    }
  }

  std::vector<Node*> rewritten_call_nodes;
  for (Node* call_node : call_nodes) {
    if (call_node->IsPartitionedCall()) {
      std::unique_ptr<FunctionBody> function_fbody;
      TF_ASSIGN_OR_RETURN(function_fbody,
                          InstantiateAssociatedFunction(*call_node, "f", fld));
      bool func_rewritten = false;
      string new_func_name =
          fld->UniqueFunctionName(function_fbody->fdef.signature().name());
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *function_fbody, flr, fld, lifted_arg_count, new_func_name,
          &func_rewritten));
      if (func_rewritten) {
        NameAttrList f;
        TF_RETURN_IF_ERROR(GetNodeAttr(call_node->def(), "f", &f));
        f.set_name(new_func_name);
        call_node->ClearAttr("f");
        call_node->AddAttr("f", f);
      }

      *rewritten |= func_rewritten;
      rewritten_call_nodes.push_back(call_node);
    } else if (fld->Contains(call_node->type_string())) {
      std::unique_ptr<FunctionBody> function_fbody;
      const FunctionDef* fdef = fld->Find(call_node->type_string());
      TF_RET_CHECK(fdef);
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, call_node->attrs(), fld,
                                                 &function_fbody));
      bool func_rewritten = false;
      string new_func_name =
          fld->UniqueFunctionName(function_fbody->fdef.signature().name());
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *function_fbody, flr, fld, lifted_arg_count, new_func_name,
          &func_rewritten));
      if (func_rewritten) {
        NodeDef node_def;
        node_def.set_name(g->NewName(call_node->name()));
        node_def.set_op(new_func_name);
        *node_def.mutable_attr() = call_node->def().attr();
        TF_ASSIGN_OR_RETURN(call_node, ReplaceNode(g, call_node, node_def));
      }

      *rewritten |= func_rewritten;
      rewritten_call_nodes.push_back(call_node);
    } else {
      TF_RET_CHECK(call_node->type_string() ==
                   FunctionLibraryDefinition::kGradientOp);
      FunctionLibraryRuntime::Handle handle;
      TF_RETURN_IF_ERROR(flr->Instantiate(call_node->type_string(),
                                          call_node->attrs(), &handle));
      auto cleanup_handle = gtl::MakeCleanup(
          [&flr, &handle]() { flr->ReleaseHandle(handle).IgnoreError(); });
      bool func_rewritten = false;
      string new_func_name = fld->UniqueFunctionName(
          absl::StrCat(call_node->name(), "_lift_args"));
      const FunctionBody* function_fbody = flr->GetFunctionBody(handle);
      TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsAndReplaceFunctionDef(
          *function_fbody, flr, fld, lifted_arg_count, new_func_name,
          &func_rewritten));
      if (func_rewritten) {
        NodeDef node_def;
        node_def.set_name(g->NewName(call_node->name()));
        node_def.set_op(new_func_name);
        *node_def.mutable_attr() = call_node->def().attr();
        node_def.mutable_attr()->erase(FunctionLibraryDefinition::kFuncAttr);
        TF_ASSIGN_OR_RETURN(call_node, ReplaceNode(g, call_node, node_def));
      }

      *rewritten |= func_rewritten;
      rewritten_call_nodes.push_back(call_node);
    }
  }

  for (Node* n : while_nodes) {
    bool node_rewritten = false;
    TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsFromWhileNode(
        g, n, fld, lifted_arg_count, &node_rewritten));
    *rewritten = *rewritten || node_rewritten;
  }

  for (Node* n : if_nodes) {
    bool node_rewritten = false;
    TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsFromIfNode(
        g, n, fld, lifted_arg_count, &node_rewritten));
    *rewritten = *rewritten || node_rewritten;
  }

  for (Node* n : rewritten_call_nodes) {
    bool node_rewritten = false;
    TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgsFromCallNode(
        g, n, flr, fld, lifted_arg_count, &node_rewritten));
    *rewritten = *rewritten || node_rewritten;
  }

  if (*rewritten) {
    VLOG(4) << DumpGraphToFile("after_lifting_args", *g, fld);
  }

  return OkStatus();
}

}  // namespace

/*static*/ Status EncapsulateTPUComputationsPass::Encapsulate(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def) {
  // Check for undeclared outputs before Encapsulation, so we can give a better
  // error message.
  // TODO(phawkins): merge this with the encapsulation code to avoid the extra
  // O(n) pass over the edges.
  for (const Edge* e : (*graph)->edges()) {
    if (!e->IsControlEdge() &&
        e->src()->attrs().Find(kTPUReplicateAttr) != nullptr &&
        e->src()->attrs().Find(kOutsideCompilationAttr) == nullptr &&
        e->dst()->attrs().Find(kTPUReplicateAttr) == nullptr &&
        e->dst()->type_string() != kTPUReplicatedOutput) {
      return errors::InvalidArgument(
          "Undeclared output of TPU computation. A common cause of this error "
          "is variable initializers that depend on the TPU computation. Edge: ",
          FormatNodeForError(*e->src()), ":", e->src_output(), " -> ",
          FormatNodeForError(*e->dst()), ":", e->dst_input());
    }
  }

  RemoveUnusedTPUReplicatedInputs(graph->get());

  TF_RETURN_IF_ERROR(RenameClustersWithDuplicatedNames(graph->get()));

  TF_RETURN_IF_ERROR(
      PerformStaticShapeInferenceBeforeEncapsulation(graph->get()));

  auto output = absl::make_unique<Graph>((*graph)->op_registry());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kTPUReplicateAttr, **graph, RewriteSubgraph,
          /*reuse_existing_functions=*/true, &output, flib_def),
      "EncapsulateTPUComputationsPass failed");
  graph->swap(output);

  return OkStatus();
}

/*static*/ Status EncapsulateTPUComputationsPass::BuildTPUReplicateOps(
    Graph* graph) {
  // Finds all of the replicate function calls, to avoid mutating the graph
  // while iterating.
  std::vector<Node*> replicate_nodes;
  std::vector<Node*> guarantee_const_nodes;
  for (Node* n : graph->nodes()) {
    string name;
    if (TryGetNodeAttr(n->attrs(), kTPUReplicateAttr, &name) &&
        !TryGetNodeAttr(n->attrs(), kOutsideCompilationAttr, &name)) {
      replicate_nodes.push_back(n);
    } else if (n->type_string() == "GuaranteeConst") {
      guarantee_const_nodes.push_back(n);
    }
  }

  // Replace any GuaranteeConst nodes with Identity nodes. These nodes have now
  // served their purpose and have no runtime effect, except increasing
  // inference latency due to executor overhead. Subsequent rewrites will remove
  // the Identity nodes.
  for (Node* n : guarantee_const_nodes) {
    std::vector<std::pair<Node*, int>> predecessors;
    for (const Edge* e : n->in_edges()) {
      predecessors.emplace_back(e->src(), e->src_output());
    }
    std::vector<std::pair<Node*, int>> successors;
    for (const Edge* e : n->out_edges()) {
      successors.emplace_back(e->dst(), e->dst_input());
    }
    NodeDef ndef;
    ndef.set_name(n->name());
    ndef.set_op("Identity");
    ndef.set_device(n->requested_device());
    MergeDebugInfo(NodeDebugInfo(n->def()), &ndef);
    AddNodeAttr("T", n->output_type(0), &ndef);

    graph->RemoveNode(n);
    TF_ASSIGN_OR_RETURN(Node * id_node, graph->AddNode(ndef));

    for (const auto& pred : predecessors) {
      if (pred.second < 0) {
        graph->AddControlEdge(pred.first, id_node);
      } else {
        graph->AddEdge(pred.first, pred.second, id_node, 0);
      }
    }
    for (const auto& succ : successors) {
      if (succ.second < 0) {
        graph->AddControlEdge(id_node, succ.first);
      } else {
        graph->AddEdge(id_node, 0, succ.first, succ.second);
      }
    }
  }

  // Replaces each replicate function call together with its neighboring
  // TPUReplicatedInput/TPUReplicatedOutput nodes with a TPUReplicate node.
  for (Node* replicate : replicate_nodes) {
    int num_replicas;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(replicate->attrs(), "num_replicas", &num_replicas));
    int variable_start_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(replicate->attrs(), "_variable_start_index",
                                   &variable_start_index));
    int guaranteed_const_start_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(replicate->attrs(),
                                   "_guaranteed_const_start_index",
                                   &guaranteed_const_start_index));

    if (HasNodeAttr(replicate->def(), "use_tpu")) {
      bool use_tpu;
      TF_RETURN_IF_ERROR(GetNodeAttr(replicate->attrs(), "use_tpu", &use_tpu));
      if (!use_tpu) {
        LOG(WARNING) << "use_tpu=false attr on a TPUReplicate node is ignored.";
      }
    }

    std::vector<const Edge*> in_edges;
    TF_RETURN_IF_ERROR(replicate->input_edges(&in_edges));

    // Counts the number of replicated, non-replicated, and variable inputs.
    int pos = 0;
    std::vector<int> mirrored_variable_indices;
    int distributed_var_start_index = 0;
    while (pos < in_edges.size() &&
           in_edges[pos]->src()->type_string() == kTPUReplicatedInput) {
      // Checks that each TPUReplicatedInput node has the correct number of
      // replicas.
      int input_num_replicas;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(in_edges[pos]->src()->attrs(), "N", &input_num_replicas));

      bool is_mirrored_variable;
      CHECK(GetNodeAttr(in_edges[pos]->src()->attrs(), "is_mirrored_variable",
                        &is_mirrored_variable)
                .ok());
      if (is_mirrored_variable) {
        mirrored_variable_indices.push_back(pos);
      }

      bool is_packed = false;
      GetNodeAttr(in_edges[pos]->src()->attrs(), "is_packed", &is_packed)
          .IgnoreError();

      bool is_distributed_variable =
          is_packed && (in_edges[pos]->src()->output_type(
                            in_edges[pos]->src_output()) == DT_RESOURCE);

      if (!is_distributed_variable && input_num_replicas != num_replicas) {
        return errors::InvalidArgument(
            "Mismatched number of replicas. Computation has ", num_replicas,
            " replicas, input '", FormatNodeForError(*in_edges[pos]->src()),
            "' has ", input_num_replicas, " replicas.");
      }

      if (!is_distributed_variable) {
        if (distributed_var_start_index < pos) {
          return errors::InvalidArgument(
              "Expect a distributed resource after index ",
              distributed_var_start_index,
              ", but got a replicated resource at index ", pos);
        } else {
          ++distributed_var_start_index;
        }
      }
      ++pos;
    }
    const int num_replicated_inputs = distributed_var_start_index;
    const int num_distributed_vars = pos - num_replicated_inputs;

    const int num_variables =
        std::max(0, guaranteed_const_start_index - variable_start_index);

    const int num_guaranteed_constants =
        in_edges.size() - guaranteed_const_start_index;
    TF_RET_CHECK(num_guaranteed_constants >= 0);

    VLOG(1) << "Replicate node '" << replicate->name() << "'"
            << " input edges: " << in_edges.size()
            << " num_replicated_inputs: " << num_replicated_inputs
            << " num_distributed_vars: " << num_distributed_vars
            << " num_variables: " << num_variables
            << " num_guaranteed_constants: " << num_guaranteed_constants
            << " num_mirrored_variables: " << mirrored_variable_indices.size();

    const int num_broadcast_inputs =
        in_edges.size() - (num_replicated_inputs + num_distributed_vars +
                           num_variables + num_guaranteed_constants);
    TF_RET_CHECK(num_broadcast_inputs >= 0);

    const int num_inputs = num_replicated_inputs * num_replicas +
                           num_distributed_vars + num_broadcast_inputs +
                           num_guaranteed_constants + num_variables;

    std::vector<Node*> nodes_to_remove = {replicate};

    // Data and control inputs to the new TPUReplicate node.
    std::vector<std::pair<Node*, int>> data_inputs(num_inputs);
    gtl::FlatSet<Node*> control_inputs;

    AddControlInputs(*replicate, &control_inputs);

    // Replicated inputs. Adds the inputs from the TPUReplicatedInput inputs,
    // in replica-major order. See the comments in
    // distributed_tpu_rewrite_pass.h for a description of the argument order.
    DataTypeVector replicated_input_types(num_replicated_inputs * num_replicas +
                                          num_distributed_vars);

    // Inputs with is_distributed_variable = false.
    for (int i = 0; i < num_replicated_inputs; ++i) {
      std::vector<const Edge*> replica_in_edges;
      TF_RETURN_IF_ERROR(in_edges[i]->src()->input_edges(&replica_in_edges));
      for (int replica = 0; replica < num_replicas; ++replica) {
        int pos = replica * num_replicated_inputs + i;
        const Edge* edge = replica_in_edges[replica];
        data_inputs[pos] = {edge->src(), edge->src_output()};
        replicated_input_types[pos] = EdgeType(edge);
      }
      AddControlInputs(*in_edges[i]->src(), &control_inputs);
      nodes_to_remove.push_back(in_edges[i]->src());
    }

    // Inputs with is_distributed_variable = true.
    for (int i = 0; i < num_distributed_vars; ++i) {
      int pos = num_replicas * num_replicated_inputs + i;
      std::vector<const Edge*> replica_in_edges;
      TF_RETURN_IF_ERROR(
          in_edges[num_replicated_inputs + i]->src()->input_edges(
              &replica_in_edges));
      TF_RET_CHECK(replica_in_edges.size() == 1);
      const Edge* edge = replica_in_edges[0];
      data_inputs[pos] = {edge->src(), edge->src_output()};
      replicated_input_types[pos] = EdgeType(edge);
      AddControlInputs(*in_edges[num_replicated_inputs + i]->src(),
                       &control_inputs);
      nodes_to_remove.push_back(in_edges[num_replicated_inputs + i]->src());
    }

    // Appends the broadcast inputs.
    DataTypeVector broadcast_input_types(num_broadcast_inputs);
    for (int i = 0; i < num_broadcast_inputs; ++i) {
      int pos = num_replicas * num_replicated_inputs + num_distributed_vars + i;
      const Edge* edge =
          in_edges[num_replicated_inputs + num_distributed_vars + i];
      data_inputs[pos] = {edge->src(), edge->src_output()};
      broadcast_input_types[i] = EdgeType(edge);
    }

    // Appends the variable inputs.
    for (int i = 0; i < num_variables; ++i) {
      int pos = num_replicas * num_replicated_inputs + num_distributed_vars +
                num_broadcast_inputs + i;
      const Edge* edge = in_edges[num_replicated_inputs + num_distributed_vars +
                                  num_broadcast_inputs + i];
      data_inputs[pos] = {edge->src(), edge->src_output()};
    }

    DataTypeVector guaranteed_constant_types(num_guaranteed_constants);
    for (int i = 0; i < num_guaranteed_constants; ++i) {
      int pos = num_replicas * num_replicated_inputs + num_distributed_vars +
                num_broadcast_inputs + num_variables + i;
      const Edge* edge = in_edges[num_replicated_inputs + num_distributed_vars +
                                  num_broadcast_inputs + num_variables + i];
      data_inputs[pos] = {edge->src(), edge->src_output()};
      guaranteed_constant_types[i] = EdgeType(edge);
    }

    // Outputs. All outputs from a replicated computation are replicated.
    const int num_outputs = replicate->output_types().size();
    gtl::FlatSet<Node*> control_outputs;
    std::vector<Node*> replicated_outputs(num_outputs);
    for (const Edge* e : replicate->out_edges()) {
      if (e->IsControlEdge()) {
        control_outputs.insert(e->dst());
      } else {
        TF_RET_CHECK(e->src_output() < num_outputs);
        TF_RET_CHECK(e->dst()->type_string() == kTPUReplicatedOutput)
            << e->DebugString();
        TF_RET_CHECK(e->dst()->output_types().size() == num_replicas);
        replicated_outputs[e->src_output()] = e->dst();
        nodes_to_remove.push_back(e->dst());

        AddControlOutputs(*e->dst(), &control_outputs);
      }
    }

    // Flattens the edges outgoing from the TPUReplicatedOutput nodes in
    // replica-major order.
    std::vector<std::vector<std::pair<Node*, int>>> data_outputs(num_replicas *
                                                                 num_outputs);
    DataTypeVector output_types(num_replicas * num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      std::vector<std::vector<const Edge*>> replica_out_edges(num_replicas);
      TF_RET_CHECK(replicated_outputs[i] != nullptr);
      for (const Edge* e : replicated_outputs[i]->out_edges()) {
        TF_RET_CHECK(!e->IsControlEdge());
        replica_out_edges[e->src_output()].push_back(e);
      }

      for (int replica = 0; replica < num_replicas; ++replica) {
        const int pos = replica * num_outputs + i;
        for (const Edge* edge : replica_out_edges[replica]) {
          data_outputs[pos].push_back({edge->dst(), edge->dst_input()});
        }
        output_types[pos] = replicated_outputs[i]->input_type(0);
      }
    }

    // TODO(b/79092708): Consolidate and cleanup to avoid TPU specialization.
    NodeDef def;
    def.set_name(replicate->name());
    def.set_op("_TPUReplicate");
    MergeDebugInfo(NodeDebugInfo(replicate->def()), &def);
    NameAttrList computation;
    computation.set_name(replicate->type_string());
    AddNodeAttr("computation", computation, &def);
    for (const auto& attr : replicate->attrs()) {
      def.mutable_attr()->insert(attr);
    }
    AddNodeAttr("Tinputs", replicated_input_types, &def);
    AddNodeAttr("Tbroadcast_inputs", broadcast_input_types, &def);
    AddNodeAttr("NumVariables", num_variables, &def);
    AddNodeAttr("Tguaranteed_constants", guaranteed_constant_types, &def);
    AddNodeAttr("output_types", output_types, &def);
    AddNodeAttr(TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR,
                mirrored_variable_indices, &def);
    AddNodeAttr("num_distributed_variables", num_distributed_vars, &def);

    for (Node* node : nodes_to_remove) {
      VLOG(2) << "Deleting node " << node->DebugString();
      // Ensure that we do not attempt to add control edges to nodes that are
      // deleted.
      control_inputs.erase(node);
      control_outputs.erase(node);
      graph->RemoveNode(node);
    }

    TF_ASSIGN_OR_RETURN(Node * tpu_replicate, graph->AddNode(def));
    for (int i = 0; i < data_inputs.size(); ++i) {
      graph->AddEdge(data_inputs[i].first, data_inputs[i].second, tpu_replicate,
                     i);
    }
    for (Node* n : control_inputs) {
      graph->AddControlEdge(n, tpu_replicate);
    }
    for (int i = 0; i < data_outputs.size(); ++i) {
      for (const auto& successor : data_outputs[i]) {
        graph->AddEdge(tpu_replicate, i, successor.first, successor.second);
      }
    }
    for (Node* n : control_outputs) {
      graph->AddControlEdge(tpu_replicate, n);
    }
  }
  return OkStatus();
}

Status EncapsulateTPUComputationsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "EncapsulateTPUComputations(): "
          << DumpGraphToFile("encapsulate_tpu_computations_before",
                             **options.graph, options.flib_def);

  TF_RETURN_IF_ERROR(Encapsulate(options.graph, options.flib_def));
  VLOG(1) << "EncapsulateTPUComputations() half-way: "
          << DumpGraphToFile("encapsulate_tpu_computations_halfway",
                             **options.graph, options.flib_def);

  TF_RETURN_IF_ERROR(BuildTPUReplicateOps(options.graph->get()));
  VLOG(1) << "EncapsulateTPUComputations() finished: "
          << DumpGraphToFile("encapsulate_tpu_computations_after",
                             **options.graph, options.flib_def);
  return OkStatus();
}

Status ExtractOutsideCompilationPass::ProcessHeadTailOutsideCompilation(
    const string& outside_compilation_attr_name, int* lifted_arg_count,
    std::unordered_map<string, XlaClusterInfo>* clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld) {
  // Gather a list of pivots by cluster so we can easily look them up.
  absl::node_hash_map<string, Node*> pivots;
  string cluster_name;
  for (Node* node : g->nodes()) {
    if (TryGetNodeAttr(node->attrs(), kPivotForClusterAttr, &cluster_name)) {
      pivots[cluster_name] = node;
    }
  }
  for (auto& iter : *clusters) {
    // Find pivot node for this XLA cluster.
    Node* pivot_node = pivots[iter.first];

    // Instantiate XLA computation function.
    string xla_func_name = iter.second.func_name_attrs.name();
    std::unique_ptr<FunctionBody> xla_fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *fld->Find(xla_func_name),
        AttrSlice(&iter.second.func_name_attrs.attr()), fld, &xla_fbody));
    Graph* xla_graph = xla_fbody->graph;

    // Make sure all nodes can be traced from sink node.
    FixupSourceAndSinkEdges(xla_graph);

    // We create Identity nodes for all _Arg/_Retval nodes in XLA computation.
    // Remove those Identity nodes to simplify furthur processing.
    TF_RETURN_IF_ERROR(RemoveIdentityNodesForArgRetval(xla_graph));

    bool rewritten;
    TF_RETURN_IF_ERROR(LiftOutsideCompilationOnlyArgs(
        xla_graph, flr, fld, lifted_arg_count, &rewritten));

    // Move head outside compilation to host.
    TF_RETURN_IF_ERROR(MoveHeadOutsideCompilationToHost(
        outside_compilation_attr_name, iter.second.func_name_attrs.name(),
        iter.second.cluster_name, g, xla_graph, iter.second.node, pivot_node));

    // Move tail outside compilation to host.
    TF_RETURN_IF_ERROR(MoveTailOutsideCompilationToHost(
        outside_compilation_attr_name, iter.second.func_name_attrs.name(),
        iter.second.cluster_name, g, xla_graph, iter.second.node, pivot_node));

    // Replace outside compilation only _Arg nodes with Placeholder nodes.
    TF_RETURN_IF_ERROR(ReplaceArgUsedByOutsideCompilationWithPlaceholder(
        outside_compilation_attr_name, xla_func_name, g, xla_graph,
        iter.second.node));

    // There might be direct data edges between _Arg node and _Retval node in
    // `xla_graph`. Remove those edges to avoid back-and-forth data transfer
    // between host and XLA.
    TF_RETURN_IF_ERROR(RemoveEdgesBetweenArgAndRetval(
        iter.second.func_name_attrs.name(), g, xla_graph, iter.second.node));

    // After `MoveHeadOutsideCompilationToHost`, there might be unused XLA
    // inputs. Remove them.
    TF_RETURN_IF_ERROR(RemoveUnusedXlaInput(iter.second.func_name_attrs.name(),
                                            g, xla_graph, iter.second.node));

    // After `MoveTailOutsideCompilationToHost`, there might be unused XLA
    // outputs. Remove them.
    TF_RETURN_IF_ERROR(RemoveUnusedXlaOutput(iter.second.func_name_attrs.name(),
                                             g, xla_graph, iter.second.node));

    // Replace original function.
    FunctionDef replace_fdef;
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*xla_graph, xla_func_name, &replace_fdef));
    TF_RETURN_IF_ERROR(fld->ReplaceFunction(xla_func_name, replace_fdef));

    FixupSourceAndSinkEdges(g);
  }

  return OkStatus();
}

Status ExtractOutsideCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  const auto* config =
      (options.session_options ? &options.session_options->config : nullptr);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          /*device_mgr=*/nullptr, options.session_options->env,
          /*config=*/config, TF_GRAPH_DEF_VERSION, options.flib_def,
          config ? config->graph_options().optimizer_options()
                 : OptimizerOptions()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  // Find XLA compile ops and their corresponding FunctionDefs.
  static std::map<string, string>* kNodeTypeToFunctionAttrMapping =
      new std::map<string, string>{
          {"_TPUReplicate", "computation"},
      };
  std::unordered_map<string, XlaClusterInfo> clusters;
  int lifted_arg_count = 0;
  for (Node* n : (*options.graph)->nodes()) {
    auto iter = kNodeTypeToFunctionAttrMapping->find(n->type_string());
    if (iter == kNodeTypeToFunctionAttrMapping->end()) {
      continue;
    }

    string xla_cluster_name = n->name();

    string func_attr = iter->second;
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), func_attr, &func));

    std::vector<string> core_list;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(n->attrs(), "host_compute_core", &core_list));
    std::map<string, int> host_compute_core;
    TF_RETURN_IF_ERROR(ParseHostComputeCoreList(core_list, &host_compute_core));

    clusters.emplace(xla_cluster_name, XlaClusterInfo{xla_cluster_name, func, n,
                                                      host_compute_core});
  }
  TF_RETURN_IF_ERROR(ProcessHeadTailOutsideCompilation(
      kOutsideCompilationAttr, &lifted_arg_count, &clusters,
      options.graph->get(), flr, options.flib_def));
  bool modified;
  TF_RETURN_IF_ERROR(ExtractOutsideCompilation(
      kTPUReplicateAttr, kOutsideCompilationAttr, clusters,
      options.graph->get(), flr, options.flib_def, &modified));
  if (modified) {
    TF_RETURN_IF_ERROR(
        PruneUnreachableFunctionsFromGraph(**options.graph, options.flib_def));
  }

  return OkStatus();
}

}  // namespace tensorflow
