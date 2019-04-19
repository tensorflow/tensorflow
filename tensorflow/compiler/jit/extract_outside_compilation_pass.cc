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
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/dump_graph.h"

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

// Returns if the node is a XLA computation key placeholder.
bool IsKeyPlaceholderNode(const Node& n) {
  return n.type_string() == "Placeholder" &&
         absl::EndsWith(n.name(), "_key_placeholder");
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
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");
  recv_at_host_builder.Attr("device_ordinal", device_ordinal_value);
  recv_at_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  recv_at_host_builder.Attr(kXlaHasHostTransferAttrName, true);
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
  // TODO(b/77601805): use out nodes for source node, instead of traversing all
  // nodes.
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
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");
  send_from_host_builder.Attr("device_ordinal", device_ordinal_value);
  send_from_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  send_from_host_builder.Attr(kXlaHasHostTransferAttrName, true);
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
  // TODO(b/77601805): use in nodes for sink node, instead of traversing all
  // nodes.
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

    const PartialTensorShape shape = shapes[e->src_output()];
    if (!shape.IsFullyDefined()) {
      return absl::nullopt;
    }

    results[e->dst_input()] = shape;
  }
  return results;
}

// Builds XlaHostCompute NodeDef from the outside compilation call node.
xla::StatusOr<NodeDef> BuildXlaHostComputeNodeDef(
    const Node* call_node, const std::map<string, int>& host_compute_core) {
  string original_oc_name;
  TF_RETURN_IF_ERROR(GetNodeAttr(
      call_node->attrs(), "_outside_compilation_subgraph", &original_oc_name));
  NodeDefBuilder host_compute_builder(
      absl::StrCat("outside_compilation_", original_oc_name, "_host_compute"),
      "XlaHostCompute");

  // Copy all attributes.
  for (auto attr : call_node->attrs()) {
    host_compute_builder.Attr(attr.first, attr.second);
  }

  // Populate tpu_core assignment.
  const auto iter = host_compute_core.find(original_oc_name);
  if (iter != host_compute_core.end()) {
    int core = iter->second;
    host_compute_builder.Attr("tpu_core", core);
  }

  // Set input tokens.
  host_compute_builder.Attr(kXlaTokenInputNodesAttrName,
                            std::vector<string>{kXlaTokenArgNodeName});

  // Populate inputs.
  std::vector<DataType> input_dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(call_node->attrs(), "Tinputs", &input_dtypes));
  std::vector<NodeDefBuilder::NodeOut> inputs(input_dtypes.size());
  for (auto e : call_node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    if (e->dst_input() < 0 || e->dst_input() >= input_dtypes.size()) {
      return errors::Internal("Invalid dst_input: ", e->dst_input());
    }
    inputs[e->dst_input()] = NodeDefBuilder::NodeOut{
        e->src()->name(), e->src_output(), input_dtypes[e->dst_input()]};
  }
  host_compute_builder.Input(inputs);

  NodeDef new_def;
  TF_RETURN_IF_ERROR(host_compute_builder.Finalize(&new_def));
  return new_def;
}

Status ValidateOutsideCompilationCallNode(Node* call_node) {
  // DT_INT64 as input/output for outside compilation is not supported yet:
  // b/120809951.
  for (const Edge* e : call_node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    DataType dtype = e->src()->output_type(e->src_output());
    if (dtype == DT_INT64) {
      return errors::Unimplemented(
          "int64 input for outside compilation is not supported yet: "
          "b/120809951. Please cast output of node ",
          e->src()->DebugString(),
          " to int32 before feeding it into outside compilation.");
    }
  }
  for (const Edge* e : call_node->out_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    DataType dtype = e->dst()->input_type(e->dst_input());
    if (dtype == DT_INT64) {
      return errors::Unimplemented(
          "int64 output for outside compilation is not supported yet: "
          "b/120809951. Please cast input of node ",
          e->dst()->DebugString(),
          " to int32 before returning it from outside compilation.");
    }
  }
  return Status::OK();
}

// Replace outside compilation function call node with XlaHostCompute node.
// If the function call node has no input/output edges, we will just remove it
// and not create a XlaHostCompute node.
Status ReplaceOrRemoveOutsideCompilationCallNode(
    Graph* g, Node* call_node, const std::map<string, int>& host_compute_core) {
  // If the function call node has no input/output edges, just remove it.
  bool has_edge = false;
  for (auto e : call_node->in_edges()) {
    if (!e->IsControlEdge() || e->src() != g->source_node()) {
      has_edge = true;
      break;
    }
  }
  for (auto e : call_node->out_edges()) {
    if (!e->IsControlEdge() || e->dst() != g->sink_node()) {
      has_edge = true;
      break;
    }
  }
  if (!has_edge) {
    VLOG(4) << "Did not add HostCompute node for " << call_node->DebugString();
    g->RemoveNode(call_node);
    return Status::OK();
  }

  // Build XlaHostCompute NodeDef.
  TF_ASSIGN_OR_RETURN(NodeDef node_def,
                      BuildXlaHostComputeNodeDef(call_node, host_compute_core));
  TF_ASSIGN_OR_RETURN(Node * host_compute_node,
                      ReplaceNode(g, call_node, node_def));
  VLOG(4) << "Added HostCompute node: " << host_compute_node->DebugString();

  return Status::OK();
}

// Resets "device_ordinal" attr to placeholder value for related nodes
// (XlaRecvAtHost nodes; XlaSendFromHost nodes; If/While/FuncCall nodes
// containing XlaRecvAtHost/XlaSendFromHost).
Status ResetDeviceOrdinalToPlaceholderValue(Graph* g) {
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");
  for (Node* n : g->nodes()) {
    if (!HasNodeAttr(n->def(), kXlaHasHostTransferAttrName)) {
      continue;
    }

    if (n->type_string() == "_XlaRecvAtHost" ||
        n->type_string() == "_XlaSendFromHost") {
      n->ClearAttr("device_ordinal");
      n->AddAttr("device_ordinal", device_ordinal_value);
    } else if (n->type_string() == "If") {
      for (const string& attr_name :
           std::vector<string>{"then_branch", "else_branch"}) {
        NameAttrList branch_func;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &branch_func));
        (*branch_func.mutable_attr())["device_ordinal"] = device_ordinal_value;
        n->ClearAttr(attr_name);
        n->AddAttr(attr_name, branch_func);
      }
    } else if (n->type_string() == "While") {
      for (const string& attr_name : std::vector<string>{"cond", "body"}) {
        NameAttrList branch_func;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &branch_func));
        (*branch_func.mutable_attr())["device_ordinal"] = device_ordinal_value;
        n->ClearAttr(attr_name);
        n->AddAttr(attr_name, branch_func);
      }
    } else if (HasNodeAttr(n->def(), "device_ordinal")) {
      // Function call node containing outside compilation.
      n->ClearAttr("device_ordinal");
      n->AddAttr("device_ordinal", device_ordinal_value);
    } else {
      return errors::Internal("Unknown node marked with ",
                              kXlaHasHostTransferAttrName, ": ",
                              n->DebugString());
    }
  }
  return Status::OK();
}

// For an XLA computation, builds host side graph given all outside compilation
// graphs inside it. The host side graph contains:
// 1) a "sequencer" node (we will add control edge between XlaRecvAtHost and
//    XlaSendFromHost to this sequencer node, so all outside compilation nodes
//    will be executed *before* this sequencer).
// 2) a "key placeholder" node. Later in ExpandHostGraphIntoMainGraph(), we will
//    replace this node with compilation result node.
// 3) all outside compilation graphs.
Status ConstructHostGraph(
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const std::vector<string>& outside_compilation_host_graphs,
    FunctionLibraryDefinition* fld, const string& host_graph_func_name) {
  Graph host_graph(fld);

  // Create sequencer node in host graph.
  NodeDefBuilder sequencer_builder(absl::StrCat(xla_cluster_name, "_sequencer"),
                                   "NoOp");
  sequencer_builder.Attr("_xla_host_transfer_sequencer", xla_cluster_name);
  NodeDef sequencer_def;
  TF_RETURN_IF_ERROR(sequencer_builder.Finalize(&sequencer_def));
  Status s;
  Node* sequencer = host_graph.AddNode(sequencer_def, &s);
  TF_RETURN_IF_ERROR(s);

  // Create key placeholder in host graph.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));

  // For each outside compilation graph, copy them to host graph with the
  // following changes:
  // a) Use key_placeholder in host graph instead of its own.
  // b) Add control edge from host transfer nodes (XlaRecvAtHost,
  //    XlaSendFromHost, If/While nodes containing
  //    XlaRecvAtHost/XlaSendFromHost) to sequencer node.
  // c) Clear node_def.device(), so device placer won't get confused.
  for (const string& host_func : outside_compilation_host_graphs) {
    VLOG(4) << "Expanding host graph " << host_func;
    // Temporarily use "0" as "device_ordinal". It will be reset to placeholder
    // value after we expanded all host graphs. We cannot just use placeholder
    // value here because FunctionDef instantiation does not allow placeholder
    // value for attributes.
    AttrValue device_ordinal_attr;
    device_ordinal_attr.set_i(0);
    protobuf::Map<string, AttrValue> attrs;
    attrs["device_ordinal"] = device_ordinal_attr;
    std::unique_ptr<FunctionBody> host_fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *fld->Find(host_func), AttrSlice(&attrs), fld, &host_fbody));

    // We use ReverseDFS() to copy nodes. Make sure all nodes are reverse
    // reachable from sink node so all nodes will be copied.
    // TODO(b/77601805): consolidate copy graph functions.
    FixupSourceAndSinkEdges(host_fbody->graph);

    std::map<const Node*, Node*> node_map;
    node_map[host_fbody->graph->source_node()] = host_graph.source_node();
    node_map[host_fbody->graph->sink_node()] = host_graph.sink_node();
    Status s;
    ReverseDFS(
        *host_fbody->graph, /*enter=*/nullptr,
        [&](const Node* n) {
          if (!s.ok()) {
            return;
          }

          Node* copy;
          if (node_map.find(n) != node_map.end()) {
            // Already copied this node.
            copy = node_map.at(n);
          } else if (IsKeyPlaceholderNode(*n)) {
            // Change a).
            copy = key_placeholder;
            node_map[n] = copy;
          } else {
            // Copy the node.
            NodeDef copy_def = n->def();
            // Change c).
            copy_def.clear_device();
            copy = host_graph.AddNode(copy_def, &s);
            if (!s.ok()) {
              return;
            }
            node_map[n] = copy;
          }

          // Only handle input edges. Output edges will be added later as
          // its output nodes' input edges.
          for (auto e : n->in_edges()) {
            if (node_map.find(e->src()) == node_map.end()) {
              s = errors::Internal("Cannot find node image for ",
                                   e->src()->DebugString());
              return;
            }
            host_graph.AddEdge(node_map[e->src()], e->src_output(), copy,
                               e->dst_input());
          }

          // Change b).
          if (HasNodeAttr(copy->def(), kXlaHasHostTransferAttrName)) {
            host_graph.AddControlEdge(copy, sequencer);
          }
        },
        NodeComparatorID());

    if (!s.ok()) {
      return s;
    }
  }
  // Reset "device_ordinal" to placeholder value.
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(&host_graph));

  // sequencer and key_placeholder might be dead nodes. Prune them if necessary.
  // - sequencer should be pruned iff it has no input control edges from
  //   RecvAtHost/SendFromHost. If it has input control edge, we connect it to
  //   sink node so it won't be pruned.
  // - key_placeholder should be pruned iff there's no RecvAtHost/SendFromHost.
  //   We don't need to do anything special.
  if (!sequencer->in_edges().empty()) {
    host_graph.AddControlEdge(sequencer, host_graph.sink_node());
  }
  PruneForReverseReachability(
      &host_graph, std::unordered_set<const Node*>{host_graph.sink_node()});

  // Postprocess edges between different outside compilations.
  TF_RETURN_IF_ERROR(PostprocessEdgesBetweenOutsideCompilations(
      &host_graph, outside_compilation_attr_name));

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(absl::StrCat("extract_outside_compilation_host_graph_for_",
                                 xla_cluster_name),
                    host_graph, fld);
  }

  FunctionDef host_graph_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(host_graph, host_graph_func_name, &host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(host_graph_fdef));
  }

  return Status::OK();
}

// Expand XLA computation's outside compilation host side graph into main graph.
// Add a control edge between sequencer node and the XLA computation node.
Status ExpandHostGraphIntoMainGraph(Graph* main_graph,
                                    FunctionLibraryDefinition* fld,
                                    const string& host_graph_func_name,
                                    Node* xla_computation_node,
                                    Node* pivot_node) {
  // Temporarily use "0" as "device_ordinal". It will be rewritten with the
  // correct value in a later pass. We cannot just use placeholder value here
  // because FunctionDef instantiation does not allow placeholder value for
  // attributes.
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fld->Find(host_graph_func_name),
                                             AttrSlice(&attrs), fld, &fbody));
  Graph* host_graph = fbody->graph;

  // We use ReverseDFS() to copy nodes. Make sure all nodes are reverse
  // reachable from sink node so all nodes will be copied.
  // TODO(b/77601805): consolidate copy graph functions.
  FixupSourceAndSinkEdges(host_graph);

  // Copy all nodes.
  std::map<const Node*, Node*> node_map;
  if (pivot_node) {
    node_map[host_graph->source_node()] = pivot_node;
  } else {
    node_map[host_graph->source_node()] = main_graph->source_node();
  }
  node_map[host_graph->sink_node()] = main_graph->sink_node();
  Status s = Status::OK();
  auto copy_node_fn = [&](const Node* n) {
    if (!s.ok()) {
      return;
    }

    Node* copy;
    if (node_map.find(n) != node_map.end()) {
      // Already copied this node.
      copy = node_map.at(n);
    } else {
      // Copy the node.
      NodeDef copy_def = n->def();
      copy = main_graph->AddNode(copy_def, &s);
      if (!s.ok()) {
        return;
      }
      node_map[n] = copy;
    }

    // Only handle input edges. Output edges will be added later as its output
    // nodes' input edges.
    for (auto e : n->in_edges()) {
      if (node_map.find(e->src()) == node_map.end()) {
        s = errors::Internal("Cannot find node image for ",
                             e->src()->DebugString());
        return;
      }
      main_graph->AddEdge(node_map[e->src()], e->src_output(), copy,
                          e->dst_input());
    }

    // Add control edge from sequencer to XLA computation node.
    if (copy->type_string() == "NoOp" &&
        HasNodeAttr(copy->def(), "_xla_host_transfer_sequencer")) {
      main_graph->AddControlEdge(copy, xla_computation_node);
    }
  };
  ReverseDFS(*host_graph, /*enter=*/nullptr, copy_node_fn, NodeComparatorID());
  return s;
}

// Rewrites shape inference graph for outside compilation:
// 1) If XlaSendFromHost also exists in `host_graph`, copy nodes from
//    `host_graph`. Because we might still have outside compilation to outside
//    compilation placeholder nodes in shape inference graph, which will prevent
//    us from inferring XlaSendFromHost shape. But in `host_graph`, we already
//    removed those placeholder nodes.
// 2) Remove control edges.
// 3) Prune nodes that are not useful for shape inference.
Status RewriteShapeInferenceGraph(const string& shape_inference_graph_name,
                                  Graph* host_graph, Node* pivot_node,
                                  FunctionLibraryDefinition* fld) {
  // Use "0" as "device_ordinal". It does not matter for shape inference.
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fld->Find(shape_inference_graph_name), AttrSlice(&attrs), fld, &fbody));
  Graph* g = fbody->graph;

  // Find SendFromHost node.
  Node* send_from_host = nullptr;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "_XlaSendFromHost") {
      send_from_host = n;
      break;
    }
  }
  if (!send_from_host) {
    return errors::Internal("Shape inference graph ",
                            shape_inference_graph_name,
                            " does not have _XlaSendFromHost node.");
  }

  // See if the SendFromHost node exists in `host_graph`.
  Node* send_from_host_main_graph = nullptr;
  for (Node* n : host_graph->nodes()) {
    if (n->name() == send_from_host->name()) {
      send_from_host_main_graph = n;
      break;
    }
  }
  if (send_from_host_main_graph) {
    // This is an "top-level" outside compilation. Clear the graph, and copy
    // SendFromHost and all its predecessors from `host_graph`.
    std::vector<Node*> nodes;
    for (Node* n : g->op_nodes()) {
      nodes.push_back(n);
    }
    for (Node* n : nodes) {
      g->RemoveNode(n);
    }
    Node* start_node = pivot_node ? pivot_node : host_graph->source_node();
    // Reverse DFS from send_from_host_main_graph, and stop at start_node.
    struct Visit {
      Node* n;
      bool is_exiting;
    };
    std::vector<Visit> stack{{send_from_host_main_graph, false}};
    std::map<Node*, Node*> node_map;
    node_map[host_graph->source_node()] = g->source_node();
    while (!stack.empty()) {
      Visit& curr = stack.back();
      if (curr.is_exiting) {
        if (node_map.find(curr.n) == node_map.end()) {
          Node* copy = g->CopyNode(curr.n);
          if (curr.n != start_node) {
            for (const Edge* e : curr.n->in_edges()) {
              auto node_iter = node_map.find(e->src());
              if (node_iter == node_map.end()) {
                return errors::Internal("Cannot find node image for ",
                                        e->src()->DebugString());
              }
              g->AddEdge(node_iter->second, e->src_output(), copy,
                         e->dst_input());
            }
          }
          node_map[curr.n] = copy;
        }
        stack.pop_back();
      } else {
        curr.is_exiting = true;
        if (curr.n != start_node) {
          for (const Edge* e : curr.n->in_edges()) {
            if (node_map.find(e->src()) != node_map.end()) {
              continue;
            }
            stack.push_back({e->src(), false});
          }
        }
      }
    }

    send_from_host = node_map[send_from_host_main_graph];
  } else {
    // This is an outside compilation embedded in If/While/gradient/etc.
    // It will be enough for shape inference. Leave `g` unchanged.
  }

  // Control edges are not useful for shape inference. Remove them.
  for (auto e : g->edges()) {
    if (e->IsControlEdge()) {
      g->RemoveEdge(e);
    }
  }

  // Nodes that are not reverse reachable from SendFromHost are not useful for
  // shape inference. Prune them.
  PruneForReverseReachability(g,
                              std::unordered_set<const Node*>{send_from_host});

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(shape_inference_graph_name, *g, fld);
  }

  // Replace original shape inference graph.
  FunctionDef fdef_replace;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*g, shape_inference_graph_name, &fdef_replace));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(shape_inference_graph_name, fdef_replace));

  return Status::OK();
}

// Builds XlaSendToHost node which sends cond predicate to host.
xla::StatusOr<Node*> BuildSendIfPredNode(const string& name,
                                         const string& host_transfer_key,
                                         Node* pred_node, Graph* g) {
  NodeDefBuilder send_pred_builder(name, "XlaSendToHost");
  send_pred_builder.Attr("Tinput", DT_BOOL);
  send_pred_builder.Attr("key", absl::StrCat(host_transfer_key, "_dtoh_0"));
  send_pred_builder.Attr(kXlaTokenInputNodesAttrName,
                         std::vector<string>{kXlaTokenArgNodeName});
  send_pred_builder.Input(pred_node->name(), 0, DT_BOOL);
  NodeDef send_pred_def;
  TF_RETURN_IF_ERROR(send_pred_builder.Finalize(&send_pred_def));
  Status s;
  Node* send_pred_node = g->AddNode(send_pred_def, &s);
  TF_RETURN_IF_ERROR(s);
  g->AddEdge(pred_node, 0, send_pred_node, 0);
  return send_pred_node;
}

// Replaces key placeholder node with an _Arg node.
Status ReplaceKeyPlaceholderWithArgNode(const string& xla_cluster_name,
                                        const string& func_name,
                                        FunctionLibraryDefinition* fld) {
  // Temporarily use "0" as "device_ordinal". It will be reset to placeholder
  // value after rewriting.
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fld->Find(func_name),
                                             AttrSlice(&attrs), fld, &fbody));
  Graph* g = fbody->graph;

  // Find or create the key placeholder node.
  Node* key_placeholder = nullptr;
  for (Node* n : g->nodes()) {
    if (IsKeyPlaceholderNode(*n)) {
      key_placeholder = n;
      break;
    }
  }
  if (!key_placeholder) {
    TF_ASSIGN_OR_RETURN(key_placeholder,
                        AddHostComputeKeyPlaceholder(xla_cluster_name, g));
  }

  // Build the _Arg node, and replace key placeholder node with it.
  NodeDefBuilder arg_builder("key_arg", FunctionLibraryDefinition::kArgOp);
  arg_builder.Attr("T", DT_STRING);
  arg_builder.Attr("index", 0);
  NodeDef arg_def;
  TF_RETURN_IF_ERROR(arg_builder.Finalize(&arg_def));
  TF_RETURN_IF_ERROR(ReplaceNode(g, key_placeholder, arg_def).status());

  // Reset "device_ordinal" to placeholder value.
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(g));

  FunctionDef replace_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*g, func_name, &replace_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(func_name, replace_fdef));
  return Status::OK();
}

// Builds host side graph for If node.
Status BuildHostGraphForIfNode(const string& xla_cluster_attr_name,
                               const string& outside_compilation_attr_name,
                               const string& xla_cluster_name,
                               const string& if_node_name,
                               const string& host_transfer_key,
                               const string& host_graph_func_name,
                               FunctionLibraryDefinition* fld,
                               const string& then_branch_host_func_name,
                               const string& else_branch_host_func_name) {
  Graph host_graph(fld);
  string outside_compilation_name = absl::StrCat("oc_if_", if_node_name);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");

  // Step 1: add key placeholder node.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));

  // Step 2: build XlaRecvAtHost node to recv predicate.
  NodeDefBuilder recv_pred_builder(
      absl::StrCat("recv_oc_if_pred_", if_node_name), "_XlaRecvAtHost");
  recv_pred_builder.Attr("Toutputs", std::vector<DataType>{DT_BOOL});
  recv_pred_builder.Attr("key", host_transfer_key);
  recv_pred_builder.Attr("device_ordinal", device_ordinal_value);
  recv_pred_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  recv_pred_builder.Attr(outside_compilation_attr_name,
                         outside_compilation_name);
  recv_pred_builder.Attr(kXlaHasHostTransferAttrName, true);
  recv_pred_builder.Input(key_placeholder->name(), 0, DT_STRING);
  NodeDef recv_pred_def;
  TF_RETURN_IF_ERROR(recv_pred_builder.Finalize(&recv_pred_def));
  Status s;
  Node* recv_pred_node = host_graph.AddNode(recv_pred_def, &s);
  TF_RETURN_IF_ERROR(s);
  host_graph.AddEdge(key_placeholder, 0, recv_pred_node, 0);

  // Step 3: rewrite `{then, else}_branch_host_func_name`, replace key
  // placeholder with an _Arg node.
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, then_branch_host_func_name, fld));
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, else_branch_host_func_name, fld));

  // Step 4: build If node to choose between `{then, else}_branch_host_graph`.
  NodeDefBuilder if_builder(absl::StrCat("oc_if_", if_node_name), "If");
  if_builder.Attr("Tcond", DT_BOOL);
  if_builder.Attr("Tin", std::vector<DataType>{DT_STRING});
  if_builder.Attr("Tout", std::vector<DataType>{});
  NameAttrList host_then_branch, host_else_branch;
  host_then_branch.set_name(then_branch_host_func_name);
  (*host_then_branch.mutable_attr())["device_ordinal"] = device_ordinal_value;
  host_else_branch.set_name(else_branch_host_func_name);
  (*host_else_branch.mutable_attr())["device_ordinal"] = device_ordinal_value;
  if_builder.Attr("then_branch", host_then_branch);
  if_builder.Attr("else_branch", host_else_branch);
  if_builder.Attr(kXlaHasHostTransferAttrName, true);
  if_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  if_builder.Attr(outside_compilation_attr_name, outside_compilation_name);
  if_builder.Input(recv_pred_node->name(), 0, DT_BOOL);
  std::vector<NodeDefBuilder::NodeOut> if_inputs{
      {key_placeholder->name(), 0, DT_STRING}};
  if_builder.Input(if_inputs);
  NodeDef if_def;
  TF_RETURN_IF_ERROR(if_builder.Finalize(&if_def));
  Node* if_node = host_graph.AddNode(if_def, &s);
  TF_RETURN_IF_ERROR(s);
  host_graph.AddEdge(recv_pred_node, 0, if_node, 0);
  host_graph.AddEdge(key_placeholder, 0, if_node, 1);

  // Convert `host_graph` to function, and add a "device_ordinal" attr.
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }

  return Status::OK();
}

// Rewrites loop cond to add a node which sends loop cond to host.
Status AddSendLoopPredToLoopCond(FunctionLibraryDefinition* fld,
                                 const NameAttrList& loop_cond_func,
                                 const string& while_node_name,
                                 const string& host_transfer_key) {
  // Instantiate the loop cond function.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fld->Find(loop_cond_func.name()),
                                             AttrSlice(&loop_cond_func.attr()),
                                             fld, &fbody));
  Graph* g = fbody->graph;

  // Find the _Retval node and the loop cond node.
  Node* ret_node = nullptr;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "_Retval") {
      if (ret_node) {
        return errors::Internal("Multiple return node for loop cond function ",
                                loop_cond_func.name(), ": ",
                                ret_node->DebugString(), " and ",
                                n->DebugString());
      } else {
        ret_node = n;
      }
    }
  }
  if (!ret_node) {
    return errors::Internal("No _Retval node for loop cond function ",
                            loop_cond_func.name());
  }
  Node* loop_cond;
  TF_RETURN_IF_ERROR(ret_node->input_node(0, &loop_cond));

  // Build the XlaSendToHost node.
  NodeDefBuilder send_loop_cond_builder(
      absl::StrCat("send_oc_while_cond_", while_node_name), "XlaSendToHost");
  send_loop_cond_builder.Attr("Tinput", DT_BOOL);
  send_loop_cond_builder.Attr("key",
                              absl::StrCat(host_transfer_key, "_dtoh_0"));
  send_loop_cond_builder.Attr(kXlaTokenInputNodesAttrName,
                              std::vector<string>{kXlaTokenArgNodeName});
  send_loop_cond_builder.Input(loop_cond->name(), 0, DT_BOOL);
  NodeDef send_loop_cond_def;
  TF_RETURN_IF_ERROR(send_loop_cond_builder.Finalize(&send_loop_cond_def));
  Status s;
  Node* send_loop_cond_node = g->AddNode(send_loop_cond_def, &s);
  TF_RETURN_IF_ERROR(s);
  g->AddEdge(loop_cond, 0, send_loop_cond_node, 0);

  // Replace original function.
  FunctionDef replace_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*g, loop_cond_func.name(), &replace_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(loop_cond_func.name(), replace_fdef));

  return Status::OK();
}

// Rewrites while loop cond function for host.
Status RewriteHostWhileLoopCond(
    const string& cond_host_func_name, const string& while_node_name,
    const string& host_transfer_key, const string& xla_cluster_attr_name,
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const string& outside_compilation_name, FunctionLibraryDefinition* fld) {
  // Replace key placeholder node with _Arg node.
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, cond_host_func_name, fld));

  // Instantiate cond function.
  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["device_ordinal"] = device_ordinal_temp_value;
  std::unique_ptr<FunctionBody> cond_fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fld->Find(cond_host_func_name), AttrSlice(&attrs), fld, &cond_fbody));
  Graph* cond_graph = cond_fbody->graph;
  Node* key_arg = nullptr;
  for (Node* n : cond_graph->nodes()) {
    if (n->type_string() == "_Arg") {
      key_arg = n;
    }
  }
  if (!key_arg) {
    return errors::Internal(
        "No _Arg node found for host compute key in function ",
        cond_host_func_name);
  }

  // Add an XlaRecvAtHost node to use as cond function return value.
  // We don't need to set kXlaHasHostTransferAttrName for this node, because
  // it's already added for the "While" node on the host.
  NodeDefBuilder recv_pred_builder(
      absl::StrCat("recv_oc_while_cond_", while_node_name), "_XlaRecvAtHost");
  recv_pred_builder.Attr("Toutputs", std::vector<DataType>{DT_BOOL});
  recv_pred_builder.Attr("key", host_transfer_key);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");
  recv_pred_builder.Attr("device_ordinal", device_ordinal_value);
  recv_pred_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  recv_pred_builder.Attr(outside_compilation_attr_name,
                         outside_compilation_name);
  recv_pred_builder.Input(key_arg->name(), 0, DT_STRING);
  NodeDef recv_pred_def;
  TF_RETURN_IF_ERROR(recv_pred_builder.Finalize(&recv_pred_def));
  Status s;
  Node* recv_pred_node = cond_graph->AddNode(recv_pred_def, &s);
  TF_RETURN_IF_ERROR(s);
  cond_graph->AddEdge(key_arg, 0, recv_pred_node, 0);
  NodeDefBuilder ret_builder(
      absl::StrCat("recv_oc_while_cond_ret_", while_node_name), "_Retval");
  ret_builder.Attr("T", DT_BOOL);
  ret_builder.Attr("index", 0);
  ret_builder.Input(recv_pred_node->name(), 0, DT_BOOL);
  NodeDef ret_def;
  TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
  Node* ret_node = cond_graph->AddNode(ret_def, &s);
  TF_RETURN_IF_ERROR(s);
  cond_graph->AddEdge(recv_pred_node, 0, ret_node, 0);

  // Reset device_ordinal to placeholder value.
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(cond_graph));

  // Replace original function.
  FunctionDef cond_replace_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*cond_graph, cond_host_func_name, &cond_replace_fdef));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(cond_host_func_name, cond_replace_fdef));

  return Status::OK();
}

// Rewrites while loop body function for host.
Status RewriteHostWhileLoopBody(
    const string& body_host_func_name, const string& while_node_name,
    const string& host_transfer_key, const string& xla_cluster_attr_name,
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const string& outside_compilation_name, FunctionLibraryDefinition* fld) {
  // Replace key placeholder node with _Arg node.
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, body_host_func_name, fld));

  // Instantiate body function.
  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["device_ordinal"] = device_ordinal_temp_value;
  std::unique_ptr<FunctionBody> body_fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fld->Find(body_host_func_name), AttrSlice(&attrs), fld, &body_fbody));
  Graph* body_graph = body_fbody->graph;
  Node* key_arg = nullptr;
  for (Node* n : body_graph->nodes()) {
    if (n->type_string() == "_Arg") {
      key_arg = n;
    }
  }
  if (!key_arg) {
    return errors::Internal(
        "No _Arg node found for host compute key in function ",
        body_host_func_name);
  }

  // Add a _Retval node to loop body.
  NodeDefBuilder ret_builder(
      absl::StrCat("recv_oc_while_body_ret_", while_node_name), "_Retval");
  ret_builder.Attr("T", DT_STRING);
  ret_builder.Attr("index", 0);
  ret_builder.Input(key_arg->name(), 0, DT_STRING);
  NodeDef ret_def;
  TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
  Status s;
  Node* ret_node = body_graph->AddNode(ret_def, &s);
  TF_RETURN_IF_ERROR(s);
  body_graph->AddEdge(key_arg, 0, ret_node, 0);

  // Reset device_ordinal to placeholder value.
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(body_graph));

  // Replace original function.
  FunctionDef body_replace_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*body_graph, body_host_func_name, &body_replace_fdef));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(body_host_func_name, body_replace_fdef));

  return Status::OK();
}

// Builds host side graph for while node.
Status BuildHostGraphForWhileNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const string& while_node_name, const string& host_transfer_key,
    const string& host_graph_func_name, FunctionLibraryDefinition* fld,
    const string& cond_host_func_name, const string& body_host_func_name) {
  Graph host_graph(fld);
  string outside_compilation_name = absl::StrCat("oc_while_", while_node_name);

  // Step 1: add key placeholder node.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));

  // Step 2: rewrite cond function.
  TF_RETURN_IF_ERROR(RewriteHostWhileLoopCond(
      cond_host_func_name, while_node_name, host_transfer_key,
      xla_cluster_attr_name, xla_cluster_name, outside_compilation_attr_name,
      outside_compilation_name, fld));

  // Step 3: rewrite body function.
  TF_RETURN_IF_ERROR(RewriteHostWhileLoopBody(
      body_host_func_name, while_node_name, host_transfer_key,
      xla_cluster_attr_name, xla_cluster_name, outside_compilation_attr_name,
      outside_compilation_name, fld));

  // Step 4: build While node.
  NodeDefBuilder while_builder(absl::StrCat("oc_while_", while_node_name),
                               "While");
  while_builder.Attr("T", std::vector<DataType>{DT_STRING});
  NameAttrList func;
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");
  (*func.mutable_attr())["device_ordinal"] = device_ordinal_value;
  func.set_name(cond_host_func_name);
  while_builder.Attr("cond", func);
  func.set_name(body_host_func_name);
  while_builder.Attr("body", func);
  while_builder.Attr(kXlaHasHostTransferAttrName, true);
  while_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  while_builder.Attr(outside_compilation_attr_name, outside_compilation_name);
  std::vector<NodeDefBuilder::NodeOut> while_inputs{
      {key_placeholder->name(), 0, DT_STRING}};
  while_builder.Input(while_inputs);
  NodeDef while_def;
  TF_RETURN_IF_ERROR(while_builder.Finalize(&while_def));
  Status s;
  Node* while_node = host_graph.AddNode(while_def, &s);
  TF_RETURN_IF_ERROR(s);
  host_graph.AddEdge(key_placeholder, 0, while_node, 0);

  // Convert `host_graph` to function.
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }

  return Status::OK();
}

// Builds host graph for func call nodes.
Status BuildHostGraphForFuncCallNode(const string& func_call_node_name,
                                     const string& xla_cluster_name,
                                     const string& func_call_host_func_name,
                                     const string& host_graph_func_name,
                                     FunctionLibraryDefinition* fld) {
  Graph host_graph(fld);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("device_ordinal");

  // Step 1: add key placeholder node.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));

  // Step 2: rewrite `host_func_name`, replace key placeholder with an _Arg
  // node.
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, func_call_host_func_name, fld));

  // Step 3: build a function call node with `host_func_name`, with
  // `key_placeholder` as input.
  NodeDefBuilder call_builder(absl::StrCat("oc_call_", func_call_node_name),
                              func_call_host_func_name, fld);
  call_builder.Input(key_placeholder->name(), 0, DT_STRING);
  call_builder.Attr("device_ordinal", device_ordinal_value);
  call_builder.Attr(kXlaHasHostTransferAttrName, true);
  NodeDef call_def;
  TF_RETURN_IF_ERROR(call_builder.Finalize(&call_def));
  Status s;
  Node* call_node = host_graph.AddNode(call_def, &s);
  TF_RETURN_IF_ERROR(s);
  host_graph.AddEdge(key_placeholder, 0, call_node, 0);

  // Convert `host_graph` to function, and add a "device_ordinal" attr.
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }

  return Status::OK();
}

Status ExtractOutsideCompilationForNodesWithAssociatedFunctions(
    Graph* g, const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* host_graphs,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  std::vector<Node*> if_nodes, while_nodes, func_call_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "If") {
      if_nodes.push_back(n);
    } else if (n->type_string() == "While") {
      while_nodes.push_back(n);
    } else if (fld->Contains(n->type_string())) {
      func_call_nodes.push_back(n);
    } else if (n->type_string() == FunctionLibraryDefinition::kGradientOp) {
      // Only gradient for user-defined function should be considered as
      // function call node.
      NameAttrList original_func;
      TF_RETURN_IF_ERROR(GetNodeAttr(
          n->def(), FunctionLibraryDefinition::kFuncAttr, &original_func));
      if (fld->Contains(original_func.name())) {
        func_call_nodes.push_back(n);
      }
    }
  }

  for (Node* n : func_call_nodes) {
    // Extract outside compilation for the function call.
    bool func_has_outside_compilation = false;
    NameAttrList func;
    func.set_name(n->type_string());
    typedef protobuf::Map<string, AttrValue> AttrMap;
    *func.mutable_attr() = AttrMap(n->attrs().begin(), n->attrs().end());
    string new_func_name = absl::StrCat(n->name(), "_oc");
    string host_func_name = absl::StrCat("oc_func_call_host_", n->name());
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        func, new_func_name, host_func_name, host_compute_core, flr, fld,
        shape_inference_graphs, &func_has_outside_compilation));

    // If the function call does not have outside compilation, nothing to do.
    if (!func_has_outside_compilation) {
      continue;
    }

    *has_outside_compilation = true;

    // Change `n` to call the new function directly.
    NodeDefBuilder replace_builder(n->name(), new_func_name, fld);
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        continue;
      }
      replace_builder.Input(e->src()->name(), e->src_output(),
                            e->src()->output_type(e->src_output()));
    }
    for (const auto& attr : n->attrs()) {
      replace_builder.Attr(attr.first, attr.second);
    }
    NodeDef replace_def;
    TF_RETURN_IF_ERROR(replace_builder.Finalize(&replace_def));
    TF_ASSIGN_OR_RETURN(Node * replace, ReplaceNode(g, n, replace_def));
    replace->AddAttr(kXlaTokenInputNodesAttrName,
                     std::vector<string>{kXlaTokenArgNodeName});

    // Build host side graph for the function call.
    string oc_host_graph_name =
        absl::StrCat("oc_func_host_graph_", replace->name());
    TF_RETURN_IF_ERROR(
        BuildHostGraphForFuncCallNode(replace->name(), xla_cluster_name,
                                      host_func_name, oc_host_graph_name, fld));

    // Record the host graph.
    host_graphs->push_back(oc_host_graph_name);
  }

  for (Node* n : if_nodes) {
    // Instantiate "then_branch" and "else_branch".
    NameAttrList then_branch, else_branch;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "then_branch", &then_branch));
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "else_branch", &else_branch));

    // Extract outside compilation for then_branch and else_branch.
    bool then_branch_has_outside_compilation = false;
    bool else_branch_has_outside_compilation = false;
    string then_branch_host_func_name =
               absl::StrCat("oc_then_branch_host_if_", n->name()),
           else_branch_host_func_name =
               absl::StrCat("oc_else_branch_host_if_", n->name());
    string then_branch_xla_func_name = absl::StrCat(then_branch.name(), "_oc"),
           else_branch_xla_func_name = absl::StrCat(else_branch.name(), "_oc");
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        then_branch, then_branch_xla_func_name, then_branch_host_func_name,
        host_compute_core, flr, fld, shape_inference_graphs,
        &then_branch_has_outside_compilation));
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        else_branch, else_branch_xla_func_name, else_branch_host_func_name,
        host_compute_core, flr, fld, shape_inference_graphs,
        &else_branch_has_outside_compilation));

    // If then/else branch do not have outside compilation, nothing to do.
    if (!then_branch_has_outside_compilation &&
        !else_branch_has_outside_compilation) {
      continue;
    }

    *has_outside_compilation = true;

    // Change If node to call the new functions.
    then_branch.set_name(then_branch_xla_func_name);
    n->ClearAttr("then_branch");
    n->AddAttr("then_branch", then_branch);
    else_branch.set_name(else_branch_xla_func_name);
    n->ClearAttr("else_branch");
    n->AddAttr("else_branch", else_branch);

    string host_transfer_key = absl::StrCat("oc_if_pred_", n->name());

    // XLA computation: add a SendToHost node to send cond predicate.
    Node* pred_node;
    TF_RETURN_IF_ERROR(n->input_node(0, &pred_node));
    TF_ASSIGN_OR_RETURN(
        Node * send_pred_node,
        BuildSendIfPredNode(absl::StrCat("send_oc_if_pred_", n->name()),
                            host_transfer_key, pred_node, g));
    n->AddAttr(kXlaTokenInputNodesAttrName,
               std::vector<string>{send_pred_node->name()});

    // Add a control edge from `send_pred_node` to If node, so XlaCompiler will
    // visit If node after `send_pred_node`, thus the token output for
    // `send_pred_node` has been generated.
    g->AddControlEdge(send_pred_node, n);

    // Build host side graph for the "If" node.
    string oc_host_graph_name = absl::StrCat("oc_if_host_graph_", n->name());
    TF_RETURN_IF_ERROR(BuildHostGraphForIfNode(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        n->name(), host_transfer_key, oc_host_graph_name, fld,
        then_branch_host_func_name, else_branch_host_func_name));
    host_graphs->push_back(oc_host_graph_name);
  }

  for (Node* n : while_nodes) {
    // Instantiate "cond" and "body".
    NameAttrList cond, body;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "cond", &cond));
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "body", &body));

    // Extract outside compilation for cond and body.
    bool cond_has_outside_compilation = false;
    bool body_has_outside_compilation = false;
    string cond_host_func_name = absl::StrCat("oc_cond_host_while_", n->name()),
           body_host_func_name = absl::StrCat("oc_body_host_while_", n->name());
    string cond_xla_func_name = absl::StrCat(cond.name(), "_oc"),
           body_xla_func_name = absl::StrCat(body.name(), "_oc");
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        cond, cond_xla_func_name, cond_host_func_name, host_compute_core, flr,
        fld, shape_inference_graphs, &cond_has_outside_compilation));
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        body, body_xla_func_name, body_host_func_name, host_compute_core, flr,
        fld, shape_inference_graphs, &body_has_outside_compilation));

    // If cond/body do not have outside compilation, nothing to do.
    if (!cond_has_outside_compilation && !body_has_outside_compilation) {
      continue;
    }

    *has_outside_compilation = true;

    // Change While node to call the new functions.
    cond.set_name(cond_xla_func_name);
    n->ClearAttr("cond");
    n->AddAttr("cond", cond);
    body.set_name(body_xla_func_name);
    n->ClearAttr("body");
    n->AddAttr("body", body);

    string host_transfer_key = absl::StrCat("oc_while_pred_", n->name());

    // XLA computation: rewrite cond function to add a SendToHost node to send
    // loop predicate.
    TF_RETURN_IF_ERROR(
        AddSendLoopPredToLoopCond(fld, cond, n->name(), host_transfer_key));
    n->AddAttr(kXlaTokenInputNodesAttrName,
               std::vector<string>{kXlaTokenArgNodeName});

    // Build host side graph for the "While" node.
    string oc_host_graph_name = absl::StrCat("oc_while_host_graph_", n->name());
    TF_RETURN_IF_ERROR(BuildHostGraphForWhileNode(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        n->name(), host_transfer_key, oc_host_graph_name, fld,
        cond_host_func_name, body_host_func_name));
    host_graphs->push_back(oc_host_graph_name);
  }

  return Status::OK();
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
    if (IsKeyPlaceholderNode(*n)) {
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
  for (Node* n : (*graph)->nodes()) {
    n->ClearAttr(kXlaInferredShapesAttrName);
  }

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
    NameAttrList shape_inference_graph;
    AddNodeAttr("shape_inference_graph", shape_inference_graph, node_def);
    AddNodeAttr("shapes", *shapes, node_def);
  } else {
    string shape_inference_func_name =
        absl::StrCat("_outside_compilation_shape_inference_", new_name);
    NameAttrList shape_inference_graph;
    shape_inference_graph.set_name(shape_inference_func_name);
    AddNodeAttr("shape_inference_graph", shape_inference_graph, node_def);
    AddNodeAttr("shapes", std::vector<TensorShapeProto>{}, node_def);
  }
  AddNodeAttr("ancestors", std::vector<string>{}, node_def);
  AddNodeAttr("Tinputs", recv_at_host_dtypes, node_def);
  AddNodeAttr("Toutputs", send_from_host_dtypes, node_def);
  AddNodeAttr("key", absl::StrCat("host_compute_channel_", new_name), node_def);

  return Status::OK();
}

Status ExtractOutsideCompilationForFunction(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const NameAttrList& func_name_attrs, const string& new_func_name,
    const string& host_graph_func_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  // Convert the function to graph.
  const string& func_name = func_name_attrs.name();
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(func_name, AttrSlice(&func_name_attrs.attr()), &handle));
  Status ret_status = Status::OK();
  auto cleanup_handle = gtl::MakeCleanup([&]() {
    auto s = flr->ReleaseHandle(handle);
    if (!s.ok()) {
      ret_status.Update(s);
    }
  });
  const FunctionBody* fbody = flr->GetFunctionBody(handle);

  // Check if we have outside compilation nodes.
  *has_outside_compilation = false;
  for (Node* n : fbody->graph->nodes()) {
    if (HasNodeAttr(n->def(), outside_compilation_attr_name)) {
      *has_outside_compilation = true;
      break;
    }
  }
  // We cannot early return here, because we might have outside compilation in
  // If/While function body.

  // Preprocess edges between different outside compilations. They will be
  // restored in `ConstructHostGraph()`.
  TF_RETURN_IF_ERROR(PreprocessEdgesBetweenOutsideCompilations(
      fbody->graph, outside_compilation_attr_name));
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(
        absl::StrCat("extract_outside_compilation_for_func_before_", func_name),
        *fbody->graph, fld);
  }

  // Encapsulate outside_compilation cluster into function call node.
  std::unique_ptr<Graph> graph_out;
  RewriteOutsideCompilationSubgraphFn rewrite_fn(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name);
  TF_RETURN_IF_ERROR(EncapsulateSubgraphsInFunctions(
      outside_compilation_attr_name, "", *fbody->graph, rewrite_fn,
      /*reuse_existing_functions=*/true, &graph_out, fld));

  // Replace outside_compilation function nodes with HostCompute ops.
  std::vector<Node*> outside_compilation_nodes;
  std::vector<string> outside_compilation_host_graphs;
  for (Node* n : graph_out->nodes()) {
    if (HasNodeAttr(n->def(), "_outside_compilation_subgraph")) {
      outside_compilation_nodes.push_back(n);
      outside_compilation_host_graphs.push_back(n->name());

      // If we could not infer shapes for XlaSendFromHost inputs statically, we
      // will set the "shape_inference_graph" attribute. In that case, copy
      // outside compilation subgraph as shape inference graph in `fld`.
      NameAttrList shape_inference_graph;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "shape_inference_graph",
                                     &shape_inference_graph));
      if (!shape_inference_graph.name().empty()) {
        shape_inference_graphs->push_back(shape_inference_graph.name());

        const FunctionDef* xla_fdef = fld->Find(n->name());
        if (!xla_fdef) {
          return errors::Internal("Cannot find XLA function ", n->name());
        }
        FunctionDef shape_inference_fdef = *xla_fdef;
        shape_inference_fdef.mutable_signature()->set_name(
            shape_inference_graph.name());
        if (fld->Find(shape_inference_graph.name())) {
          TF_RETURN_IF_ERROR(fld->ReplaceFunction(shape_inference_graph.name(),
                                                  shape_inference_fdef));
        } else {
          TF_RETURN_IF_ERROR(fld->AddFunctionDef(shape_inference_fdef));
        }
      }
    }
  }
  for (Node* n : outside_compilation_nodes) {
    TF_RETURN_IF_ERROR(ValidateOutsideCompilationCallNode(n));
    TF_RETURN_IF_ERROR(ReplaceOrRemoveOutsideCompilationCallNode(
        graph_out.get(), n, host_compute_core));
  }

  // Handle nodes with associated functions.
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForNodesWithAssociatedFunctions(
      graph_out.get(), xla_cluster_attr_name, outside_compilation_attr_name,
      xla_cluster_name, host_compute_core, flr, fld,
      &outside_compilation_host_graphs, shape_inference_graphs,
      has_outside_compilation));

  // Construct host graph.
  TF_RETURN_IF_ERROR(ConstructHostGraph(
      xla_cluster_name, outside_compilation_attr_name,
      outside_compilation_host_graphs, fld, host_graph_func_name));

  // Remove the outside compilation graphs from function library.
  for (const string& func : outside_compilation_host_graphs) {
    TF_RETURN_IF_ERROR(fld->RemoveFunction(func));
  }

  // Replace original function.
  FunctionDef updated_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*graph_out, new_func_name, &updated_fdef));
  const FunctionDef* original_fdef = fld->Find(func_name);
  if (original_fdef) {
    for (const auto& attr : original_fdef->attr()) {
      (*updated_fdef.mutable_attr())[attr.first] = attr.second;
    }
  }
  if (fld->Find(new_func_name)) {
    TF_RETURN_IF_ERROR(fld->ReplaceFunction(new_func_name, updated_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(updated_fdef));
  }
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(
        absl::StrCat("extract_outside_compilation_for_func_after_", func_name),
        *graph_out, fld);
  }

  return ret_status;
}

Status ExtractOutsideCompilation(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld) {
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_before", *g, fld);
  }

  auto node_name_index = g->BuildNodeNameIndex();
  for (auto& iter : clusters) {
    string xla_cluster_name = iter.first;
    Node* n = iter.second.node;
    auto const& func_name_attrs = iter.second.func_name_attrs;
    auto const& host_compute_core = iter.second.host_compute_core;

    std::vector<string> shape_inference_graphs;
    bool has_outside_compilation;
    string host_graph_func_name = absl::StrCat("oc_host_graph_", n->name());
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        func_name_attrs, func_name_attrs.name(), host_graph_func_name,
        host_compute_core, flr, fld, &shape_inference_graphs,
        &has_outside_compilation));

    string pivot_name = absl::StrCat(xla_cluster_name, "/pivot");
    Node* pivot_node = node_name_index[pivot_name];
    TF_RETURN_IF_ERROR(ExpandHostGraphIntoMainGraph(
        g, fld, host_graph_func_name, n, pivot_node));

    TF_RETURN_IF_ERROR(fld->RemoveFunction(host_graph_func_name));

    for (auto shape_inference_graph_name : shape_inference_graphs) {
      TF_RETURN_IF_ERROR(RewriteShapeInferenceGraph(shape_inference_graph_name,
                                                    g, pivot_node, fld));
    }
  }

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_after", *g, fld);
  }
  return Status::OK();
}

}  // namespace tensorflow
