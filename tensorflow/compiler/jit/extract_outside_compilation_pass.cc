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

#include "tensorflow/compiler/jit/extract_outside_compilation_pass.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

namespace {

// Add a key placeholder node to the graph. The key placeholder node will be
// used as input for XlaRecvAtHost/XlaSendFromHost nodes.
xla::StatusOr<Node*> AddHostComputeKeyPlaceholder(
    const string& xla_cluster_name, Graph* g) {
  NodeDef key_def;
  NodeDefBuilder builder(absl::StrCat(xla_cluster_name, "_key_placeholder"),
                         "Placeholder");
  builder.Attr("dtype", DT_STRING);
  builder.Attr("shape", PartialTensorShape({2}));
  builder.Attr("_host_compute_call_node", xla_cluster_name);
  Status s = builder.Finalize(&key_def);
  if (!s.ok()) return s;

  Node* n = g->AddNode(key_def, &s);
  if (!s.ok()) return s;
  return n;
}

// Returns nodes with given type.
std::vector<Node*> GatherNodesWithType(const Graph& g, const string& type) {
  std::vector<Node*> result;
  for (Node* n : g.nodes()) {
    if (n->type_string() == type) {
      result.push_back(n);
    }
  }
  return result;
}

// Gets data types from `arg_nodes` and fills them into `recv_at_host_dtypes`.
Status GetArgDataTypes(const std::vector<Node*>& arg_nodes,
                       std::vector<DataType>* recv_at_host_dtypes) {
  recv_at_host_dtypes->resize(arg_nodes.size(), DT_INVALID);
  for (auto* n : arg_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
    (*recv_at_host_dtypes)[index] = dtype;
  }
  for (int i = 0; i < recv_at_host_dtypes->size(); i++) {
    if ((*recv_at_host_dtypes)[i] == DT_INVALID) {
      return errors::Internal("Cannot get datatype for input ", i);
    }
  }
  return Status::OK();
}

// Builds XlaRecvAtHost node.
xla::StatusOr<Node*> BuildRecvAtHostNode(
    Graph* g, const string& oc_cluster_name,
    const std::vector<DataType>& recv_at_host_dtypes, Node* key_placeholder) {
  NodeDefBuilder recv_at_host_builder(
      absl::StrCat("outside_compilation_", oc_cluster_name, "_recv"),
      "_XlaRecvAtHost");
  NodeDef recv_at_host_def;
  recv_at_host_builder.Attr("Toutputs", recv_at_host_dtypes);
  // The correct device_ordinal will be inserted during replication in a
  // subsequent rewrite.
  recv_at_host_builder.Attr("device_ordinal", 0);
  recv_at_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  recv_at_host_builder.Input(key_placeholder->name(), 0, DT_STRING);
  TF_RETURN_IF_ERROR(recv_at_host_builder.Finalize(&recv_at_host_def));
  Status s;
  Node* recv_at_host_node = g->AddNode(recv_at_host_def, &s);
  TF_RETURN_IF_ERROR(s);
  return recv_at_host_node;
}

// Builds XlaRecvAtHost node, and replaces all _Arg nodes with it.
xla::StatusOr<Node*> ReplaceArgNodesWithRecvAtHostNode(
    Graph* g, const string& oc_cluster_name,
    std::vector<DataType>* recv_at_host_dtypes, Node* key_placeholder) {
  std::vector<Node*> arg_nodes = GatherNodesWithType(*g, "_Arg");
  TF_RETURN_IF_ERROR(GetArgDataTypes(arg_nodes, recv_at_host_dtypes));
  TF_ASSIGN_OR_RETURN(
      Node * recv_at_host_node,
      BuildRecvAtHostNode(g, oc_cluster_name, *recv_at_host_dtypes,
                          key_placeholder));
  for (auto* n : arg_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    // Record out edges and remove `n` before adding those edges to RecvAtHost.
    // This is to avoid multiple producers.
    std::vector<OutEdgeInfo> out_edge_info;
    for (auto edge : n->out_edges()) {
      out_edge_info.push_back(
          {edge->dst(), edge->src_output(), edge->dst_input()});
    }
    g->RemoveNode(n);
    for (const OutEdgeInfo& edge : out_edge_info) {
      if (edge.dst_input == Graph::kControlSlot) {
        g->AddControlEdge(recv_at_host_node, edge.dst);
      } else {
        g->AddEdge(recv_at_host_node, index, edge.dst, edge.dst_input);
      }
    }

    // Rewrite dst nodes because their input changed.
    for (int i = 0; i < out_edge_info.size(); i++) {
      const OutEdgeInfo edge = out_edge_info[i];
      if (edge.dst_input == Graph::kControlSlot) {
        continue;
      }

      Node* dst = edge.dst;
      NodeDef new_def = dst->def();
      *new_def.mutable_input(edge.dst_input) =
          absl::StrCat(recv_at_host_node->name(), ":", index);
      TF_ASSIGN_OR_RETURN(Node * dst_replace, ReplaceNode(g, dst, new_def));

      // Other edges might have `dst` as dst node as well. Update those edges
      // with `dst_replace`.
      for (int j = i + 1; j < out_edge_info.size(); j++) {
        if (out_edge_info[j].dst == dst) {
          out_edge_info[j].dst = dst_replace;
        }
      }
    }
  }
  g->AddEdge(key_placeholder, 0, recv_at_host_node, 0);
  return recv_at_host_node;
}

// Gets data types from `ret_nodes` and fills them into `send_from_host_dtypes`.
Status GetRetDataTypes(const std::vector<Node*>& ret_nodes,
                       std::vector<DataType>* send_from_host_dtypes) {
  send_from_host_dtypes->resize(ret_nodes.size(), DT_INVALID);
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
    (*send_from_host_dtypes)[index] = dtype;
  }
  for (int i = 0; i < send_from_host_dtypes->size(); i++) {
    if ((*send_from_host_dtypes)[i] == DT_INVALID) {
      return errors::Internal("Cannot get datatype for output ", i);
    }
  }
  return Status::OK();
}

// Builds XlaSendFromHost node.
xla::StatusOr<Node*> BuildSendFromHostNode(
    Graph* g, const string& oc_cluster_name,
    const std::vector<Node*>& ret_nodes,
    const std::vector<DataType>& send_from_host_dtypes, Node* key_placeholder) {
  NodeDefBuilder send_from_host_builder(
      absl::StrCat("outside_compilation_", oc_cluster_name, "_send"),
      "_XlaSendFromHost");
  NodeDef send_from_host_def;
  send_from_host_builder.Attr("Tinputs", send_from_host_dtypes);
  // The correct device_ordinal will be inserted during replication in a
  // subsequent rewrite.
  send_from_host_builder.Attr("device_ordinal", 0);
  send_from_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  std::vector<NodeDefBuilder::NodeOut> inputs(send_from_host_dtypes.size());
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    if (index < 0 || index >= send_from_host_dtypes.size()) {
      return errors::Internal("Invalid _Retval index: ", index);
    }
    for (auto edge : n->in_edges()) {
      inputs[index] =
          NodeDefBuilder::NodeOut{edge->src()->name(), edge->src_output(),
                                  edge->src()->output_type(edge->src_output())};
    }
  }
  send_from_host_builder.Input(inputs);
  send_from_host_builder.Input(key_placeholder->name(), 0, DT_STRING);
  TF_RETURN_IF_ERROR(send_from_host_builder.Finalize(&send_from_host_def));
  Status s;
  Node* send_from_host_node = g->AddNode(send_from_host_def, &s);
  TF_RETURN_IF_ERROR(s);
  return send_from_host_node;
}

// Builds XlaSendFromHost node, and replaces all _Retval nodes with it.
xla::StatusOr<Node*> ReplaceRetNodesWithSendFromHostNode(
    Graph* g, const string& oc_cluster_name,
    std::vector<DataType>* send_from_host_dtypes, Node* key_placeholder) {
  std::vector<Node*> ret_nodes = GatherNodesWithType(*g, "_Retval");
  TF_RETURN_IF_ERROR(GetRetDataTypes(ret_nodes, send_from_host_dtypes));
  TF_ASSIGN_OR_RETURN(
      Node * send_from_host_node,
      BuildSendFromHostNode(g, oc_cluster_name, ret_nodes,
                            *send_from_host_dtypes, key_placeholder));
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    for (auto edge : n->in_edges()) {
      if (edge->src_output() == Graph::kControlSlot) {
        g->AddControlEdge(edge->src(), send_from_host_node);
      } else {
        g->AddEdge(edge->src(), edge->src_output(), send_from_host_node, index);
      }
    }
    g->RemoveNode(n);
  }
  g->AddEdge(key_placeholder, 0, send_from_host_node,
             send_from_host_dtypes->size());
  return send_from_host_node;
}

// Returns input shapes (excluding key placeholder) for `send_from_host_node`
// if they are all fully defined; absl::nullopt otherwise.
absl::optional<std::vector<PartialTensorShape>> GetInferredInputShapes(
    int num_inputs, Node* send_from_host_node) {
  std::vector<PartialTensorShape> results(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    const Edge* e;
    if (!send_from_host_node->input_edge(i, &e).ok()) {
      return absl::nullopt;
    }

    std::vector<PartialTensorShape> shapes;
    if (!GetNodeAttr(e->src()->attrs(), kXlaInferredShapesAttrName, &shapes)
             .ok()) {
      return absl::nullopt;
    }

    const PartialTensorShape shape = shapes[e->dst_input()];
    if (!shape.IsFullyDefined()) {
      return absl::nullopt;
    }

    results[e->dst_input()] = shape;
  }
  return results;
}

}  // namespace

Status RewriteOutsideCompilationSubgraphFn::operator()(
    const std::vector<OutputTensor>& arg_source_tensors,
    std::unique_ptr<Graph>* graph, std::vector<int>* input_permutation,
    std::vector<int>* output_permutation, NodeDef* node_def) {
  string old_name = node_def->op();
  string new_name = absl::StrCat(xla_cluster_name_, "_", old_name);
  node_def->set_op(new_name);
  node_def->set_name(new_name);

  // Later we will run PruneForReverseReachability(), so make sure all original
  // nodes are reachable from sink node and won't be removed.
  FixupSourceAndSinkEdges(graph->get());

  // Step 1: create a key placeholder node.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name_, graph->get()));

  // Step 2: build RecvAtHost node, and replace all _Arg nodes with it.
  std::vector<DataType> recv_at_host_dtypes;
  TF_ASSIGN_OR_RETURN(
      Node * recv_at_host_node,
      ReplaceArgNodesWithRecvAtHostNode(graph->get(), new_name,
                                        &recv_at_host_dtypes, key_placeholder));

  // Step 3: build SendFromHost node, and replace all _Retval nodes with it.
  std::vector<DataType> send_from_host_dtypes;
  TF_ASSIGN_OR_RETURN(
      Node * send_from_host_node,
      ReplaceRetNodesWithSendFromHostNode(
          graph->get(), new_name, &send_from_host_dtypes, key_placeholder));

  // Step 4: add XLA cluster and outside compilation attr.
  for (Node* n : (*graph)->nodes()) {
    if (n->type_string() == "Placeholder" &&
        absl::EndsWith(n->name(), "_key_placeholder")) {
      continue;
    }

    n->AddAttr(xla_cluster_attr_name_, xla_cluster_name_);
    n->AddAttr(outside_compilation_attr_name_, old_name);
  }

  // Check whether we have all input shapes for XlaSendFromHost. If we do, we
  // will set `shapes` attr for the call node; otherwise we will save the
  // shape inference graph and set `shape_inference_graph` for the call node.
  absl::optional<std::vector<PartialTensorShape>> shapes =
      GetInferredInputShapes(send_from_host_dtypes.size(), send_from_host_node);

  // Step 5: add control edges for originally XLA <-> outside compilation
  // control edges.
  for (Node* n : (*graph)->nodes()) {
    if (HasNodeAttr(n->def(), kXlaConnectedToXlaComputationAttrName)) {
      (*graph)->AddControlEdge(n, send_from_host_node);
      n->ClearAttr(kXlaConnectedToXlaComputationAttrName);
    }
    if (HasNodeAttr(n->def(), kXlaConnectedFromXlaComputationAttrName)) {
      (*graph)->AddControlEdge(recv_at_host_node, n);
      n->ClearAttr(kXlaConnectedFromXlaComputationAttrName);
    }
  }

  // Step 6: RecvAtHost/SendFromHost/key_placeholder might be dead nodes. Prune
  // them if necessary.
  // - RecvAtHost should be pruned iff it has no output data/control edges. If
  //   it has any output edge, it will be reverse reachable from sink node. We
  //   don't need to do anything special.
  // - SendFromHost should be pruned iff it has no input data/control edges. If
  //   it has input edges other than key_placeholder, we connect it to sink
  //   node so it won't be pruned.
  // - key_placeholder should be pruned iff RecvAtHost/SendFromHost are pruned.
  //   We don't need to do anything special.
  if (send_from_host_node->in_edges().size() > 1) {
    (*graph)->AddControlEdge(send_from_host_node, (*graph)->sink_node());
  }
  PruneForReverseReachability(
      graph->get(), std::unordered_set<const Node*>{(*graph)->sink_node()});

  // Step 7: add necessary attributes to function call node, so we can replace
  // it with HostCompute node later.
  AddNodeAttr("_outside_compilation_subgraph", old_name, node_def);
  if (shapes) {
    AddNodeAttr("shape_inference_graph", "", node_def);
    AddNodeAttr("shapes", *shapes, node_def);
  } else {
    string shape_inference_func_name =
        absl::StrCat("_outside_compilation_shape_inference_", new_name);
    AddNodeAttr("shape_inference_graph", shape_inference_func_name, node_def);
    AddNodeAttr("shapes", std::vector<TensorShapeProto>{}, node_def);
  }
  AddNodeAttr("ancestors", std::vector<string>{}, node_def);
  AddNodeAttr("Tinputs", recv_at_host_dtypes, node_def);
  AddNodeAttr("Toutputs", send_from_host_dtypes, node_def);
  AddNodeAttr("key", absl::StrCat("host_compute_channel_", new_name), node_def);

  return Status::OK();
}

}  // namespace tensorflow
