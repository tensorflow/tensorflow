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
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"

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

// For an XLA computation, builds host side graph given all outside compilation
// graphs inside it. The host side graph contains:
// 1) a "sequencer" node (we will add control edge between XlaRecvAtHost and
//    XlaSendFromHost to this sequencer node, so all outside compilation nodes
//    will be executed *before* this sequencer).
// 2) a "key placeholder" node. Later in ExpandHostGraphIntoMainGraph(), we will
//    replace this node with compilation result node.
// 3) all outside compilation graphs.
Status ConstructHostGraph(
    const string& xla_cluster_name,
    const std::vector<string>& outside_compilation_host_graphs,
    FunctionLibraryDefinition* fld, std::unique_ptr<Graph>* host_graph) {
  host_graph->reset(new Graph(fld));

  // Create sequencer node in host graph.
  NodeDefBuilder sequencer_builder(absl::StrCat(xla_cluster_name, "_sequencer"),
                                   "NoOp");
  sequencer_builder.Attr("_xla_host_transfer_sequencer", xla_cluster_name);
  NodeDef sequencer_def;
  TF_RETURN_IF_ERROR(sequencer_builder.Finalize(&sequencer_def));
  Status s;
  Node* sequencer = (*host_graph)->AddNode(sequencer_def, &s);
  TF_RETURN_IF_ERROR(s);

  // Create key placeholder in host graph.
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, host_graph->get()));

  // For each outside compilation graph, copy them to host graph with the
  // following changes:
  // a) Use key_placeholder in host graph instead of its own.
  // b) Add control edge from RecvAtHost/SendFromHost to sequencer.
  // c) Clear node_def.device(), so device placer won't get confused.
  for (const string& host_func : outside_compilation_host_graphs) {
    VLOG(4) << "Expanding host graph " << host_func;
    FunctionBody* host_fbody = nullptr;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *fld->Find(host_func), AttrSlice(), fld,
        [&](const string& op, const OpDef** sig) {
          return fld->LookUpOpDef(op, sig);
        },
        &host_fbody));
    std::unique_ptr<FunctionBody> host_fbody_deleter(host_fbody);

    // We use ReverseDFS() to copy nodes. Make sure all nodes are reverse
    // reachable from sink node so all nodes will be copied.
    // TODO(b/77601805): consolidate copy graph functions.
    FixupSourceAndSinkEdges(host_fbody->graph);

    std::map<const Node*, Node*> node_map;
    node_map[host_fbody->graph->source_node()] = (*host_graph)->source_node();
    node_map[host_fbody->graph->sink_node()] = (*host_graph)->sink_node();
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
            copy = (*host_graph)->AddNode(copy_def, &s);
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
            (*host_graph)
                ->AddEdge(node_map[e->src()], e->src_output(), copy,
                          e->dst_input());
          }

          // Change b).
          if (copy->type_string() == "_XlaRecvAtHost" ||
              copy->type_string() == "_XlaSendFromHost") {
            (*host_graph)->AddControlEdge(copy, sequencer);
          }
        },
        NodeComparatorID());
    if (!s.ok()) {
      return s;
    }
  }

  // sequencer and key_placeholder might be dead nodes. Prune them if necessary.
  // - sequencer should be pruned iff it has no input control edges from
  //   RecvAtHost/SendFromHost. If it has input control edge, we connect it to
  //   sink node so it won't be pruned.
  // - key_placeholder should be pruned iff there's no RecvAtHost/SendFromHost.
  //   We don't need to do anything special.
  if (!sequencer->in_edges().empty()) {
    (*host_graph)->AddControlEdge(sequencer, (*host_graph)->sink_node());
  }
  PruneForReverseReachability(
      host_graph->get(),
      std::unordered_set<const Node*>{(*host_graph)->sink_node()});

  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("extract_outside_compilation_host_graph_for_",
                     xla_cluster_name),
        **host_graph, fld);
  }

  return Status::OK();
}

// Expand XLA computation's outside compilation host side graph into main graph.
// Add a control edge between sequencer node and the XLA computation node.
Status ExpandHostGraphIntoMainGraph(Graph* main_graph, Graph* host_graph,
                                    Node* xla_computation_node) {
  // We use ReverseDFS() to copy nodes. Make sure all nodes are reverse
  // reachable from sink node so all nodes will be copied.
  // TODO(b/77601805): consolidate copy graph functions.
  FixupSourceAndSinkEdges(host_graph);

  // Copy all nodes.
  std::map<const Node*, Node*> node_map;
  node_map[host_graph->source_node()] = main_graph->source_node();
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

// Rewrites shape inference graph for outside compilation.
// 1. If the outside compilation is a "top-level" one (not in a function of any
//    If/While/etc.), this shape inference graph might have host computation to
//    outside compilation placeholder nodes, which will cause shape inference to
//    fail. However, those nodes are not in `host_graph` any more (because we
//    have executed `PostprocessForEncapsultion`). In this case, we clear the
//    graph, and copy SendFromHost with all its predecessors from `host_graph`.
//    This case is detected by whether the SendFromHost node exists in
//    `host_graph` as well.
// 2. Remove control edges, and prune nodes that are not useful for shape
//    inference.
Status RewriteShapeInferenceGraph(const string& shape_inference_graph_name,
                                  Graph* host_graph,
                                  FunctionLibraryDefinition* fld) {
  FunctionBody* fbody = nullptr;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fld->Find(shape_inference_graph_name), AttrSlice(), fld,
      [&](const string& op, const OpDef** sig) {
        return fld->LookUpOpDef(op, sig);
      },
      &fbody));
  std::unique_ptr<FunctionBody> fbody_deleter(fbody);
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

    std::map<const Node*, Node*> node_map;
    node_map[host_graph->source_node()] = g->source_node();
    Status s;
    auto copy_node_fn = [&](const Node* n) {
      if (!s.ok()) {
        return;
      }

      if (node_map.find(n) != node_map.end()) {
        return;
      }

      NodeDef copy_def = n->def();
      Node* copy = g->AddNode(copy_def, &s);
      if (!s.ok()) {
        return;
      }
      for (auto e : n->in_edges()) {
        if (node_map.find(e->src()) == node_map.end()) {
          s = errors::Internal("Cannot find node image for ",
                               e->src()->DebugString());
          return;
        }
        g->AddEdge(node_map[e->src()], e->src_output(), copy, e->dst_input());
      }

      node_map[n] = copy;
    };
    // TODO(b/77601805): consolidate copy graph functions.
    ReverseDFSFrom(*host_graph,
                   std::vector<const Node*>{send_from_host_main_graph},
                   /*enter=*/nullptr, copy_node_fn, NodeComparatorID());
    if (!s.ok()) {
      return s;
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
    dump_graph::DumpGraphToFile(shape_inference_graph_name, *g, fld);
  }

  // Replace original shape inference graph.
  FunctionDef fdef_replace;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*g, shape_inference_graph_name, &fdef_replace));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(shape_inference_graph_name, fdef_replace));

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

Status ExtractOutsideCompilationForFunction(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const NameAttrList& func_name_attrs, const string& new_func_name,
    const std::map<string, int>& host_compute_core,
    FunctionLibraryDefinition* fld, std::unique_ptr<Graph>* host_graph,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  // Early return if function does not have any outside compilation nodes.
  const string& func_name = func_name_attrs.name();
  const FunctionDef* fdef = fld->Find(func_name);
  if (!fdef) {
    return errors::Internal("Cannot find function ", func_name);
  }
  *has_outside_compilation = false;
  for (auto& node_def : fdef->node_def()) {
    if (HasNodeAttr(node_def, outside_compilation_attr_name)) {
      *has_outside_compilation = true;
      break;
    }
  }
  if (!has_outside_compilation) {
    return Status::OK();
  }

  // Convert the function to graph.
  FunctionBody* fbody = nullptr;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *fld->Find(func_name), AttrSlice(&func_name_attrs.attr()), fld,
      [&](const string& op, const OpDef** sig) {
        return fld->LookUpOpDef(op, sig);
      },
      &fbody));
  std::unique_ptr<FunctionBody> fbody_deleter(fbody);
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
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
      string shape_inference_graph;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "shape_inference_graph",
                                     &shape_inference_graph));
      if (!shape_inference_graph.empty()) {
        shape_inference_graphs->push_back(shape_inference_graph);

        const FunctionDef* xla_fdef = fld->Find(n->name());
        if (!xla_fdef) {
          return errors::Internal("Cannot find XLA function ", n->name());
        }
        FunctionDef shape_inference_fdef = *xla_fdef;
        shape_inference_fdef.mutable_signature()->set_name(
            shape_inference_graph);
        if (fld->Find(shape_inference_graph)) {
          TF_RETURN_IF_ERROR(fld->ReplaceFunction(shape_inference_graph,
                                                  shape_inference_fdef));
        } else {
          TF_RETURN_IF_ERROR(fld->AddFunctionDef(shape_inference_fdef));
        }
      }
    }
  }
  for (Node* n : outside_compilation_nodes) {
    TF_RETURN_IF_ERROR(ReplaceOrRemoveOutsideCompilationCallNode(
        graph_out.get(), n, host_compute_core));
  }
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("extract_outside_compilation_for_func_after_", func_name),
        *graph_out, fld);
  }

  // Construct host graph.
  if (!outside_compilation_host_graphs.empty()) {
    TF_RETURN_IF_ERROR(ConstructHostGraph(
        xla_cluster_name, outside_compilation_host_graphs, fld, host_graph));
  }

  // Remove the outside compilation graphs from function library.
  for (const string& func : outside_compilation_host_graphs) {
    TF_RETURN_IF_ERROR(fld->RemoveFunction(func));
  }

  // Replace original function.
  FunctionDef updated_fdef;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*graph_out, new_func_name, &updated_fdef));
  if (fld->Find(new_func_name)) {
    TF_RETURN_IF_ERROR(fld->ReplaceFunction(new_func_name, updated_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(updated_fdef));
  }

  return Status::OK();
}

Status ExtractOutsideCompilation(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters, Graph* g,
    FunctionLibraryDefinition* fld) {
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile("extract_outside_compilation_before", *g, fld);
  }

  std::vector<string> shape_inference_graphs;
  for (auto& iter : clusters) {
    string xla_cluster_name = iter.first;
    Node* n = iter.second.node;
    auto const& func_name_attrs = iter.second.func_name_attrs;
    auto const& host_compute_core = iter.second.host_compute_core;

    bool has_outside_compilation;
    std::unique_ptr<Graph> host_graph;
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        func_name_attrs, func_name_attrs.name(), host_compute_core, fld,
        &host_graph, &shape_inference_graphs, &has_outside_compilation));
    if (host_graph) {
      TF_RETURN_IF_ERROR(ExpandHostGraphIntoMainGraph(g, host_graph.get(), n));
    }
  }

  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile("extract_outside_compilation_expanded", *g,
                                fld);
  }

  TF_RETURN_IF_ERROR(PostprocessForEncapsulation(
      g, xla_cluster_attr_name, outside_compilation_attr_name, clusters));

  for (auto shape_inference_graph_name : shape_inference_graphs) {
    TF_RETURN_IF_ERROR(
        RewriteShapeInferenceGraph(shape_inference_graph_name, g, fld));
  }

  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile("extract_outside_compilation_after", *g, fld);
  }
  return Status::OK();
}

}  // namespace tensorflow
