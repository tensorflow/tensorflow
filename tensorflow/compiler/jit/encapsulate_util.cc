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

#include "tensorflow/compiler/jit/encapsulate_util.h"
#include <algorithm>
#include <iterator>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"

namespace tensorflow {

namespace {

// Returns string attribute value for the node if the attribute is present,
// otherwise returns empty optional value.
absl::optional<string> GetStringAttr(const Node& n, const string& attr_name) {
  auto attr = n.attrs().Find(attr_name);
  if (!attr) {
    return absl::nullopt;
  } else {
    return attr->s();
  }
}

// Adds a value to the node's list attribute.
template <typename T>
Status AppendToListAttr(Node* n, const string& attr_name, const string& value) {
  std::vector<T> attr_value;
  Status s = GetNodeAttr(n->attrs(), attr_name, &attr_value);
  if (!s.ok() && s.code() != error::NOT_FOUND) {
    return s;
  }

  n->ClearAttr(attr_name);
  attr_value.push_back(value);
  n->AddAttr(attr_name, attr_value);
  return Status::OK();
}

// Replaces attribute value.
template <typename T>
void ReplaceAttr(Node* n, const string& attr_name, const T& value) {
  n->ClearAttr(attr_name);
  n->AddAttr(attr_name, value);
}

// Step 1a ~ 1d for PreprocessForEncapsulation(). See comments of
// PreprocessForEncapsulation() for details.
Status ProcessControlEdges(Graph* g, const string& xla_computation_attr_name,
                           const string& outside_compilation_attr_name) {
  // Gather edges to remove. We should not remove the edge while iterating.
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* e : g->edges()) {
    if (!e->IsControlEdge()) {
      continue;
    }

    auto src_xla_computation =
        GetStringAttr(*e->src(), xla_computation_attr_name);
    auto dst_xla_computation =
        GetStringAttr(*e->dst(), xla_computation_attr_name);
    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (!src_xla_computation && !dst_xla_computation) {
      continue;
    } else if (src_xla_computation && !dst_xla_computation) {
      if (src_outside_compilation) {
        // Case 1c: outside compilation to host computation control edge.
        edges_to_remove.push_back(e);

        TF_RETURN_IF_ERROR(AppendToListAttr<string>(
            e->dst(), kXlaControlDependenciesAttrName, e->src()->name()));
      }
    } else if (!src_xla_computation && dst_xla_computation) {
      if (dst_outside_compilation) {
        // Case 1c: host computation control to outside compilation edge.
        edges_to_remove.push_back(e);

        TF_RETURN_IF_ERROR(AppendToListAttr<string>(
            e->dst(), kXlaControlDependenciesAttrName, e->src()->name()));
      }
    } else {  // src_xla_computation && dst_xla_computation
      if (*src_xla_computation != *dst_xla_computation) {
        if (src_outside_compilation && dst_outside_compilation) {
          // Case 1b: outside compilation to outside compilation control edge.
          edges_to_remove.push_back(e);

          TF_RETURN_IF_ERROR(AppendToListAttr<string>(
              e->dst(), kXlaControlDependenciesAttrName, e->src()->name()));
        } else if (src_outside_compilation && !dst_outside_compilation) {
          // Case 1a: outside compilation to another XLA computaition control
          // edge.
          TF_RETURN_IF_ERROR(AppendToListAttr<string>(
              e->src(), kXlaConnectedToOtherXlaComputationAttrName,
              *dst_xla_computation));
        } else if (!src_outside_compilation && dst_outside_compilation) {
          // Case 1a: another XLA computaition to outside compilation control
          // edge.
          TF_RETURN_IF_ERROR(AppendToListAttr<string>(
              e->dst(), kXlaConnectedFromOtherXlaComputationAttrName,
              *src_xla_computation));
        }
      }
    }
  }

  for (auto e : edges_to_remove) {
    g->RemoveEdge(e);
  }
  return Status::OK();
}

// Step 2 for PreprocessForEncapsulation(). See comments of
// PreprocessForEncapsulation() for details.
Status ProcessXlaToXlaDataEdges(Graph* g,
                                const string& xla_computation_attr_name,
                                const string& outside_compilation_attr_name) {
  // Gather edges between XLA computations. Notice that we do not store `Edge*`
  // directly because we remove some nodes while adding Identity nodes, and
  // those Edge pointers might be invalidated.
  struct EdgeInfo {
    int dst_input, dst_node_id;
  };
  std::vector<EdgeInfo> edges;
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    auto src_xla_computation =
        GetStringAttr(*e->src(), xla_computation_attr_name);
    auto dst_xla_computation =
        GetStringAttr(*e->dst(), xla_computation_attr_name);
    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);
    if (!src_xla_computation || !dst_xla_computation) {
      continue;
    }

    if (*src_xla_computation != *dst_xla_computation) {
      if (src_outside_compilation || dst_outside_compilation) {
        edges.push_back(EdgeInfo{e->dst_input(), e->dst()->id()});
        VLOG(4) << "XLA -> XLA edge: " << e->DebugString();
      }
    }
  }

  // For each XLA -> XLA edge, add an Identity node between src and dst.
  for (int i = 0; i < edges.size(); i++) {
    Node* dst = g->FindNodeId(edges[i].dst_node_id);
    const Edge* e;
    TF_RETURN_IF_ERROR(dst->input_edge(edges[i].dst_input, &e));
    Node* src = e->src();
    int src_output = e->src_output(), dst_input = e->dst_input();
    g->RemoveEdge(e);

    // Create Identity node, and connect it between `src` and `dst`.
    string identity_node_name =
        absl::StrCat("bridge_", src->name(), "_", dst->name());
    DataType dtype = src->output_type(src_output);
    TF_ASSIGN_OR_RETURN(Node * identity_node,
                        BuildIdentityNode(g, identity_node_name, dtype, src,
                                          /*requested_device=*/absl::nullopt));
    identity_node->AddAttr(kBridgeSourceNodeAttrName, src->name());
    g->AddEdge(src, src_output, identity_node, 0);
    g->AddEdge(identity_node, 0, dst, dst_input);

    // Replace `e->dst()` because its input node changed.
    NodeDef new_def = dst->def();
    *new_def.mutable_input(dst_input) = identity_node->name();
    TF_ASSIGN_OR_RETURN(Node * dst_replace_node, ReplaceNode(g, dst, new_def));

    // Other edge in `edges` might have `e->dst()` as src or dst
    // node. Before removing `e->dst()`, replace those edges with corresponding
    // edges for `dst_replace_node`.
    for (int j = i + 1; j < edges.size(); j++) {
      if (edges[j].dst_node_id == edges[i].dst_node_id) {
        edges[j].dst_node_id = dst_replace_node->id();
      }
    }
  }
  return Status::OK();
}

// Step 3 for PreprocessForEncapsulation(). See comments of
// PreprocessForEncapsulation() for details.
Status ProcessDataEdgeBetweenOutsideCompilationAndHostComputation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name) {
  // Gather edges between outside compilation and host computation. Notice that
  // we do not store `Edge*` directly because we remove some nodes while adding
  // Identity nodes, and those Edge pointers might be invalidated.
  struct EdgeInfo {
    int dst_input, dst_node_id;
    bool is_host_to_outside_compilation;
  };
  std::vector<EdgeInfo> edges;
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    if (e->src()->attrs().Find(xla_computation_attr_name) == nullptr &&
        e->dst()->attrs().Find(xla_computation_attr_name) != nullptr &&
        e->dst()->attrs().Find(outside_compilation_attr_name) != nullptr) {
      edges.push_back(EdgeInfo{e->dst_input(), e->dst()->id(),
                               /*is_host_to_outside_compilation=*/true});
      VLOG(4) << "Host -> oc edge: " << e->DebugString();
    } else if (e->dst()->attrs().Find(xla_computation_attr_name) == nullptr &&
               e->src()->attrs().Find(xla_computation_attr_name) != nullptr &&
               e->src()->attrs().Find(outside_compilation_attr_name) !=
                   nullptr) {
      edges.push_back(EdgeInfo{e->dst_input(), e->dst()->id(),
                               /*is_host_to_outside_compilation=*/false});
      VLOG(4) << "Oc -> host edge: " << e->DebugString();
    }
  }

  // Remove the edge from host to outside compilation. Add a placeholder as
  // outside compilation node input.
  std::map<std::pair<string, int>, Node*> placeholders;
  for (int i = 0; i < edges.size(); i++) {
    Node* dst = g->FindNodeId(edges[i].dst_node_id);
    const Edge* e;
    TF_RETURN_IF_ERROR(dst->input_edge(edges[i].dst_input, &e));
    Node* src = e->src();
    int src_output = e->src_output(), dst_input = e->dst_input();
    g->RemoveEdge(e);

    // Find or create placeholder node.
    string new_name =
        edges[i].is_host_to_outside_compilation
            ? absl::StrCat(src->name(), "_host_to_oc_placeholder_", src_output)
            : absl::StrCat(src->name(), "_oc_to_host_placeholder_", src_output);
    auto placeholder_index = std::make_pair(src->name(), src_output);
    auto iter = placeholders.find(placeholder_index);
    Node* placeholder_node;
    if (iter == placeholders.end()) {
      NodeDefBuilder placeholder_builder(new_name, "Placeholder");
      placeholder_builder.Attr("dtype", src->output_type(src_output));
      if (edges[i].is_host_to_outside_compilation) {
        placeholder_builder.Attr(kHostToOutsideCompilationOriginalNodeAttrName,
                                 src->name());
        placeholder_builder.Attr(kHostToOutsideCompilationSrcOutputAttrName,
                                 src_output);
        // If this placeholder node is in outside compilation, we need to set
        // `xla_computation_attr_name` and `outside_compilation_attr_name`.
        string xla_computation_attr, outside_compilation_attr;
        TF_RETURN_IF_ERROR(GetNodeAttr(dst->attrs(), xla_computation_attr_name,
                                       &xla_computation_attr));
        TF_RETURN_IF_ERROR(GetNodeAttr(dst->attrs(),
                                       outside_compilation_attr_name,
                                       &outside_compilation_attr));
        placeholder_builder.Attr(xla_computation_attr_name,
                                 xla_computation_attr);
        placeholder_builder.Attr(outside_compilation_attr_name,
                                 outside_compilation_attr);
      } else {
        placeholder_builder.Attr(kOutsideCompilationToHostOriginalNodeAttrName,
                                 src->name());
        placeholder_builder.Attr(kOutsideCompilationToHostSrcOutputAttrName,
                                 src_output);
      }
      NodeDef placeholder_def;
      TF_RETURN_IF_ERROR(placeholder_builder.Finalize(&placeholder_def));
      Status s;
      placeholder_node = g->AddNode(placeholder_def, &s);
      TF_RETURN_IF_ERROR(s);
      placeholders[placeholder_index] = placeholder_node;
    } else {
      placeholder_node = iter->second;
    }
    g->AddEdge(placeholder_node, 0, dst, dst_input);

    // Replace `e->dst()` because its input node changed.
    NodeDef new_def = dst->def();
    *new_def.mutable_input(dst_input) = placeholder_node->name();
    TF_ASSIGN_OR_RETURN(Node * dst_replace_node, ReplaceNode(g, dst, new_def));

    // Other edge in `edges` might have `e->dst()` as src or dst
    // node. Before removing `e->dst()`, replace those edges with corresponding
    // edges for `dst_replace_node`.
    for (int j = i + 1; j < edges.size(); j++) {
      if (edges[j].dst_node_id == edges[i].dst_node_id) {
        edges[j].dst_node_id = dst_replace_node->id();
      }
    }
  }
  return Status::OK();
}

// Step 1 for `PostprocessForEncapsulation`. See comments of
// `PostprocessForEncapsulation` for details.
Status RemovePlaceholderBetweenOutsideCompilationAndHostComputation(Graph* g) {
  // Gather all outside compilation to host computation nodes.
  struct PlaceHolderNodeInfo {
    Node* n;
    bool is_host_to_oc;
  };
  std::vector<PlaceHolderNodeInfo> placeholder_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "Placeholder") {
      if (HasNodeAttr(n->def(),
                      kOutsideCompilationToHostOriginalNodeAttrName)) {
        placeholder_nodes.push_back({n, false});
      } else if (HasNodeAttr(n->def(),
                             kHostToOutsideCompilationOriginalNodeAttrName)) {
        placeholder_nodes.push_back({n, true});
      }
    }
  }

  // Remove the placeholder nodes, and reconnect original edge.
  auto node_name_index = g->BuildNodeNameIndex();
  for (auto placeholder_iter : placeholder_nodes) {
    Node* n = placeholder_iter.n;

    string node_name;
    int node_src_output;
    if (placeholder_iter.is_host_to_oc) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), kHostToOutsideCompilationOriginalNodeAttrName,
                      &node_name));
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(),
                                     kHostToOutsideCompilationSrcOutputAttrName,
                                     &node_src_output));
    } else {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), kOutsideCompilationToHostOriginalNodeAttrName,
                      &node_name));
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(),
                                     kOutsideCompilationToHostSrcOutputAttrName,
                                     &node_src_output));
    }
    auto iter = node_name_index.find(node_name);
    if (iter == node_name_index.end()) {
      return errors::Internal(
          "Cannot find original node for oc -> host placeholder node ",
          node_name);
    }

    // Change all usage node to use the original node instead.
    Node* original_node = iter->second;
    std::vector<const Edge*> control_edges;
    std::vector<OutEdgeInfo> data_edges;
    for (auto e : n->out_edges()) {
      if (e->IsControlEdge()) {
        control_edges.push_back(e);
      } else {
        data_edges.push_back({e->dst(), e->src_output(), e->dst_input()});
      }
    }
    for (const Edge* e : control_edges) {
      g->AddControlEdge(original_node, e->dst());
      g->RemoveEdge(e);
    }
    for (int i = 0; i < data_edges.size(); i++) {
      Node* dst = data_edges[i].dst;
      NodeDef new_def = dst->def();
      int dst_input = data_edges[i].dst_input;
      *new_def.mutable_input(dst_input) =
          absl::StrCat(original_node->name(), ":", node_src_output);
      TF_ASSIGN_OR_RETURN(Node * replace_node, ReplaceNode(g, dst, new_def));

      const Edge* edge_to_replace = nullptr;
      TF_RETURN_IF_ERROR(replace_node->input_edge(dst_input, &edge_to_replace));
      g->RemoveEdge(edge_to_replace);
      g->AddEdge(original_node, node_src_output, replace_node, dst_input);

      // Other edges might have `dst` as dst node. Update those edges with
      // `replace_node`.
      for (int j = i + 1; j < data_edges.size(); j++) {
        if (data_edges[j].dst == dst) {
          data_edges[j].dst = replace_node;
        }
      }

      // Other placeholder node might have `dst` as original node. Update
      // `node_name_index` with `replace_node`.
      node_name_index[replace_node->name()] = replace_node;
    }

    // Remove placeholder node.
    g->RemoveNode(n);
  }
  return Status::OK();
}

// Step 2 for `PostprocessForEncapsulation`. See comments of
// `PostprocessForEncapsulation` for details.
Status RemoveIdentityBetweenDifferentXlaComputation(Graph* g) {
  // Gather Identity nodes to remove.
  std::vector<Node*> bridge_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "Identity" &&
        HasNodeAttr(n->def(), kBridgeSourceNodeAttrName)) {
      bridge_nodes.push_back(n);
    }
  }

  // Remove the identity nodes, and reconnect the original edge.
  for (int i = 0; i < bridge_nodes.size(); i++) {
    Node* n = bridge_nodes[i];
    const Edge* src_edge = nullptr;
    TF_RETURN_IF_ERROR(n->input_edge(0, &src_edge));

    // Change all usage node to use the original node instead.
    std::vector<const Edge*> control_edges;
    std::vector<OutEdgeInfo> data_edges;
    for (auto e : n->out_edges()) {
      if (e->IsControlEdge()) {
        control_edges.push_back(e);
      } else {
        data_edges.push_back({e->dst(), e->src_output(), e->dst_input()});
      }
    }
    for (const Edge* e : control_edges) {
      g->AddControlEdge(src_edge->src(), e->dst());
      g->RemoveEdge(e);
    }
    for (int j = 0; j < data_edges.size(); j++) {
      Node* dst = data_edges[j].dst;
      NodeDef new_def = dst->def();
      int dst_input = data_edges[j].dst_input;
      *new_def.mutable_input(dst_input) =
          absl::StrCat(src_edge->src()->name(), ":", src_edge->src_output());
      TF_ASSIGN_OR_RETURN(Node * replace_node, ReplaceNode(g, dst, new_def));

      const Edge* edge_to_replace = nullptr;
      TF_RETURN_IF_ERROR(replace_node->input_edge(dst_input, &edge_to_replace));
      g->RemoveEdge(edge_to_replace);
      g->AddEdge(src_edge->src(), src_edge->src_output(), replace_node,
                 dst_input);

      // Other edges might have `dst` as dst node. Update those edges with
      // `replace_node`.
      for (int k = j + 1; k < data_edges.size(); k++) {
        if (data_edges[k].dst == dst) {
          data_edges[k].dst = replace_node;
        }
      }

      // The node we replaced might be in `bridge_nodes`. If so, update
      // `bridge_nodes` to use the replaced node.
      for (int k = i + 1; k < bridge_nodes.size(); k++) {
        if (bridge_nodes[k] == dst) {
          bridge_nodes[k] = replace_node;
        }
      }
    }

    // Remove Identity node.
    g->RemoveNode(n);
  }
  return Status::OK();
}

// Step 3 for `PostprocessForEncapsulation`. See comments of
// `PostprocessForEncapsulation` for details.
// We do not need to worry about removed nodes in step 1 and 2;
// `PreprocessForEncapsulation` will not record control dependencies for those
// remvoed nodes in the first place.
Status AddControlDependencies(
    Graph* g, const std::unordered_map<string, string>& cluster_node_names) {
  auto node_name_index = g->BuildNodeNameIndex();

  // Reconnect outside compilation to outside compilation control edge.
  for (Node* n : g->nodes()) {
    std::vector<string> control_deps;
    Status s =
        GetNodeAttr(n->attrs(), kXlaControlDependenciesAttrName, &control_deps);
    if (!s.ok()) {
      if (s.code() != error::NOT_FOUND) {
        return s;
      } else {
        continue;
      }
    } else {
      n->ClearAttr(kXlaControlDependenciesAttrName);
      for (const string& control_input : control_deps) {
        auto iter = node_name_index.find(control_input);
        if (iter == node_name_index.end()) {
          return errors::Internal("Cannot find original node for ",
                                  control_input);
        }
        g->AddControlEdge(iter->second, n);
      }
    }
  }

  // Reconnect outside compilation to XLA computation control edge.
  for (Node* n : g->nodes()) {
    std::vector<string> control_deps;
    Status s = GetNodeAttr(
        n->attrs(), kXlaConnectedToOtherXlaComputationAttrName, &control_deps);
    if (!s.ok()) {
      if (s.code() != error::NOT_FOUND) {
        return s;
      } else {
        continue;
      }
    } else {
      n->ClearAttr(kXlaConnectedToOtherXlaComputationAttrName);
      for (const string& control_input : control_deps) {
        auto iter = cluster_node_names.find(control_input);
        if (iter == cluster_node_names.end()) {
          return errors::Internal("Cannot find cluster node for ",
                                  control_input);
        }
        auto iter2 = node_name_index.find(iter->second);
        if (iter2 == node_name_index.end()) {
          return errors::Internal("Cannot find cluster node for ",
                                  iter->second);
        }
        g->AddControlEdge(n, iter2->second);
      }
    }
  }

  // Reconnect XLA computation to outside compilation control edge.
  for (Node* n : g->nodes()) {
    std::vector<string> control_deps;
    Status s =
        GetNodeAttr(n->attrs(), kXlaConnectedFromOtherXlaComputationAttrName,
                    &control_deps);
    if (!s.ok()) {
      if (s.code() != error::NOT_FOUND) {
        return s;
      } else {
        continue;
      }
    } else {
      n->ClearAttr(kXlaConnectedFromOtherXlaComputationAttrName);
      for (const string& control_input : control_deps) {
        auto iter = cluster_node_names.find(control_input);
        if (iter == cluster_node_names.end()) {
          return errors::Internal("Cannot find cluster node for ",
                                  control_input);
        }
        auto iter2 = node_name_index.find(iter->second);
        if (iter2 == node_name_index.end()) {
          return errors::Internal("Cannot find cluster node for ",
                                  iter->second);
        }
        g->AddControlEdge(iter2->second, n);
      }
    }
  }

  return Status::OK();
}

// Step 1 for `PreprocessEdgesBetweenOutsideCompilations`. See comments of
// `PreprocessEdgesBetweenOutsideCompilations` for details.
Status PreprocessControlEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  // Gather edges to remove. We should not remove the edge while iterating.
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* e : g->edges()) {
    if (!e->IsControlEdge()) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (src_outside_compilation && dst_outside_compilation) {
      if (*src_outside_compilation != *dst_outside_compilation) {
        // Case 1a: outside compilation to outside compilation control edge.
        edges_to_remove.push_back(e);

        TF_RETURN_IF_ERROR(AppendToListAttr<string>(
            e->dst(), kXlaControlDependenciesWithinXlaClusterAttrName,
            e->src()->name()));
      }
    } else if (src_outside_compilation && !dst_outside_compilation) {
      // Case 1b: outside compilation to its XLA computation control edge.
      ReplaceAttr(e->src(), kXlaConnectedToXlaComputationAttrName, true);
    } else if (!src_outside_compilation && dst_outside_compilation) {
      // Case 1b: XLA computation to outside compilation in it control edge.
      ReplaceAttr(e->dst(), kXlaConnectedFromXlaComputationAttrName, true);
    }
  }

  for (auto e : edges_to_remove) {
    g->RemoveEdge(e);
  }
  return Status::OK();
}

// Step 2 for `PreprocessEdgesBetweenOutsideCompilations`. See comments of
// `PreprocessEdgesBetweenOutsideCompilations` for details.
Status PreprocessDataEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  // Gather edges between outside compilation and host computation. Notice that
  // we do not store `Edge*` directly because we remove some nodes while adding
  // Identity nodes, and those Edge pointers might be invalidated.
  struct EdgeInfo {
    int dst_input, dst_node_id;
  };
  std::vector<EdgeInfo> edges;
  for (const Edge* e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);

    if (src_outside_compilation && dst_outside_compilation &&
        *src_outside_compilation != *dst_outside_compilation) {
      edges.push_back(EdgeInfo{e->dst_input(), e->dst()->id()});
      VLOG(4) << "Oc -> oc edge: " << e->DebugString();
    }
  }

  // Remove the edge from host to outside compilation. Add a placeholder as
  // outside compilation node input.
  std::map<std::pair<string, int>, Node*> placeholders;
  for (int i = 0; i < edges.size(); i++) {
    Node* dst = g->FindNodeId(edges[i].dst_node_id);
    const Edge* e;
    TF_RETURN_IF_ERROR(dst->input_edge(edges[i].dst_input, &e));
    Node* src = e->src();
    int src_output = e->src_output(), dst_input = e->dst_input();
    g->RemoveEdge(e);

    // Find or create placeholder node.
    string new_name =
        absl::StrCat(src->name(), "_oc_to_oc_placeholder_", src_output);
    auto placeholder_index = std::make_pair(src->name(), src_output);
    auto iter = placeholders.find(placeholder_index);
    Node* placeholder_node;
    if (iter == placeholders.end()) {
      NodeDefBuilder placeholder_builder(new_name, "Placeholder");
      placeholder_builder.Attr("dtype", src->output_type(src_output));
      string outside_compilation_attr;
      TF_RETURN_IF_ERROR(GetNodeAttr(dst->attrs(),
                                     outside_compilation_attr_name,
                                     &outside_compilation_attr));
      placeholder_builder.Attr(outside_compilation_attr_name,
                               outside_compilation_attr);
      placeholder_builder.Attr(kOutsideCompilationOriginalNodeAttrName,
                               src->name());
      placeholder_builder.Attr(kOutsideCompilationSrcOutputAttrName,
                               src_output);
      NodeDef placeholder_def;
      TF_RETURN_IF_ERROR(placeholder_builder.Finalize(&placeholder_def));
      Status s;
      placeholder_node = g->AddNode(placeholder_def, &s);
      TF_RETURN_IF_ERROR(s);
      placeholders[placeholder_index] = placeholder_node;
    } else {
      placeholder_node = iter->second;
    }
    g->AddEdge(placeholder_node, 0, dst, dst_input);

    // Replace `e->dst()` because its input node changed.
    NodeDef new_def = dst->def();
    *new_def.mutable_input(dst_input) = placeholder_node->name();
    TF_ASSIGN_OR_RETURN(Node * dst_replace_node, ReplaceNode(g, dst, new_def));

    // Other edge in `edges` might have `e->dst()` as src or dst
    // node. Before removing `e->dst()`, replace those edges with
    // corresponding edges for `dst_replace_node`.
    for (int j = i + 1; j < edges.size(); j++) {
      if (edges[j].dst_node_id == edges[i].dst_node_id) {
        edges[j].dst_node_id = dst_replace_node->id();
      }
    }
  }
  return Status::OK();
}

// Step 1 for `PostprocessEdgesBetweenOutsideCompilations`. See comments of
// `PostprocessEdgesBetweenOutsideCompilations` for details.
Status PostprocessDataEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  // Gather all outside compilation to outside compilation nodes.
  std::vector<Node*> placeholder_nodes;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "Placeholder" &&
        HasNodeAttr(n->def(), kOutsideCompilationOriginalNodeAttrName)) {
      placeholder_nodes.push_back(n);
    }
  }

  // Remove the placeholder nodes, and reconnect original edge.
  auto node_name_index = g->BuildNodeNameIndex();
  for (auto n : placeholder_nodes) {
    string node_name;
    int node_src_output;
    TF_RETURN_IF_ERROR(GetNodeAttr(
        n->attrs(), kOutsideCompilationOriginalNodeAttrName, &node_name));
    TF_RETURN_IF_ERROR(GetNodeAttr(
        n->attrs(), kOutsideCompilationSrcOutputAttrName, &node_src_output));
    auto iter = node_name_index.find(node_name);
    if (iter == node_name_index.end()) {
      return errors::Internal(
          "Cannot find original node for oc -> host placeholder node ",
          node_name);
    }

    // Change all usage node to use the original node instead.
    Node* original_node = iter->second;
    std::vector<const Edge*> control_edges;
    std::vector<OutEdgeInfo> data_edges;
    for (auto e : n->out_edges()) {
      if (e->IsControlEdge()) {
        control_edges.push_back(e);
      } else {
        data_edges.push_back({e->dst(), e->src_output(), e->dst_input()});
      }
    }
    for (const Edge* e : control_edges) {
      g->AddControlEdge(original_node, e->dst());
      g->RemoveEdge(e);
    }
    for (int i = 0; i < data_edges.size(); i++) {
      Node* dst = data_edges[i].dst;
      NodeDef new_def = dst->def();
      int dst_input = data_edges[i].dst_input;
      *new_def.mutable_input(dst_input) =
          absl::StrCat(original_node->name(), ":", node_src_output);
      TF_ASSIGN_OR_RETURN(Node * replace_node, ReplaceNode(g, dst, new_def));

      const Edge* edge_to_replace = nullptr;
      TF_RETURN_IF_ERROR(replace_node->input_edge(dst_input, &edge_to_replace));
      g->RemoveEdge(edge_to_replace);
      g->AddEdge(original_node, node_src_output, replace_node, dst_input);

      // Other edges might have `dst` as dst node. Update those edges with
      // `replace_node`.
      for (int j = i + 1; j < data_edges.size(); j++) {
        if (data_edges[j].dst == dst) {
          data_edges[j].dst = replace_node;
        }
      }

      // Other placeholder node might have `dst` as original node. Update
      // `node_name_index` with `replace_node`.
      node_name_index[replace_node->name()] = replace_node;
    }

    // Remove placeholder node.
    g->RemoveNode(n);
  }
  return Status::OK();
}

// Step 2 for `PostprocessEdgesBetweenOutsideCompilations`. See comments of
// `PostprocessEdgesBetweenOutsideCompilations` for details.
Status PostprocessControlEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  auto node_name_index = g->BuildNodeNameIndex();

  // Reconnect outside compilation to outside compilation control edge.
  for (Node* n : g->nodes()) {
    std::vector<string> control_deps;
    Status s =
        GetNodeAttr(n->attrs(), kXlaControlDependenciesWithinXlaClusterAttrName,
                    &control_deps);
    if (!s.ok()) {
      if (s.code() != error::NOT_FOUND) {
        return s;
      } else {
        continue;
      }
    } else {
      n->ClearAttr(kXlaControlDependenciesWithinXlaClusterAttrName);
      for (const string& control_input : control_deps) {
        auto iter = node_name_index.find(control_input);
        if (iter == node_name_index.end()) {
          return errors::Internal("Cannot find original node for ",
                                  control_input);
        }
        g->AddControlEdge(iter->second, n);
      }
    }
  }
  return Status::OK();
}
}  // namespace

const char kXlaInferredShapesAttrName[] = "_xla_inferred_shapes";

const char kXlaConnectedToOtherXlaComputationAttrName[] =
    "_xla_connected_to_other_xla_computation";
const char kXlaConnectedFromOtherXlaComputationAttrName[] =
    "_xla_connected_from_other_xla_computation";
const char kXlaControlDependenciesAttrName[] = "_xla_control_dependencies";
const char kBridgeSourceNodeAttrName[] = "_xla_bridge_src";
const char kOutsideCompilationToHostOriginalNodeAttrName[] =
    "_xla_oc_to_host_node_name";
const char kOutsideCompilationToHostSrcOutputAttrName[] =
    "_xla_oc_to_host_src_output";
const char kHostToOutsideCompilationOriginalNodeAttrName[] =
    "_xla_host_to_oc_node_name";
const char kHostToOutsideCompilationSrcOutputAttrName[] =
    "_xla_host_to_oc_src_output";
const char kXlaConnectedToXlaComputationAttrName[] =
    "_xla_connected_to_xla_computation";
const char kXlaConnectedFromXlaComputationAttrName[] =
    "_xla_connected_from_xla_computation";
const char kOutsideCompilationOriginalNodeAttrName[] =
    "_xla_oc_to_oc_node_name";
const char kOutsideCompilationSrcOutputAttrName[] = "_xla_oc_to_oc_src_output";
const char kXlaControlDependenciesWithinXlaClusterAttrName[] =
    "_xla_control_dependencies_within_xla_cluster";

Status PerformStaticShapeInferenceBeforeEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name) {
  // Find all outside compilation to XLA computation data edges.
  std::unordered_set<Node*> outside_compilation_send_nodes;
  for (auto e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    auto src_computation = GetStringAttr(*e->src(), xla_computation_attr_name);
    auto dst_computation = GetStringAttr(*e->dst(), xla_computation_attr_name);
    if (!src_computation || !dst_computation ||
        *src_computation != *dst_computation) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);
    if (src_outside_compilation && !dst_outside_compilation) {
      outside_compilation_send_nodes.insert(e->src());
    }
  }

  // Perform shape inference.
  std::map<int, InferredShape> arg_shapes;
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(
      InferShapes(g, arg_shapes, /*fnlib_def=*/nullptr, &shape_info));

  // Add attribute for output shapes.
  for (Node* n : outside_compilation_send_nodes) {
    auto iter = shape_info.find(n->name());
    if (iter == shape_info.end()) {
      continue;
    }

    std::vector<PartialTensorShape> output_shapes;
    std::transform(iter->second.begin(), iter->second.end(),
                   std::back_inserter(output_shapes),
                   [](const InferredShape& inferred_shape) {
                     return inferred_shape.shape;
                   });
    n->AddAttr(kXlaInferredShapesAttrName, output_shapes);
  }

  return Status::OK();
}

Status PreprocessForEncapsulation(Graph* g,
                                  const string& xla_computation_attr_name,
                                  const string& outside_compilation_attr_name) {
  TF_RETURN_IF_ERROR(ProcessControlEdges(g, xla_computation_attr_name,
                                         outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(ProcessXlaToXlaDataEdges(g, xla_computation_attr_name,
                                              outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(ProcessDataEdgeBetweenOutsideCompilationAndHostComputation(
      g, xla_computation_attr_name, outside_compilation_attr_name));
  return Status::OK();
}

Status PostprocessForEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters) {
  // The `node` pointer in `XlaClusterInfo` might be invalidated in step 1/2,
  // but the node name won't change. Record cluster node name for
  // `AddControlDependencies`.
  std::unordered_map<string, string> cluster_node_names;
  for (const auto& iter : clusters) {
    cluster_node_names[iter.first] = iter.second.node->name();
  }

  TF_RETURN_IF_ERROR(
      RemovePlaceholderBetweenOutsideCompilationAndHostComputation(g));
  TF_RETURN_IF_ERROR(RemoveIdentityBetweenDifferentXlaComputation(g));
  TF_RETURN_IF_ERROR(AddControlDependencies(g, cluster_node_names));
  return Status::OK();
}

Status PreprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  // Remove edges from source node to outside compilation nodes, and edges
  // from outside compilation nodes to sink node.
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* e : g->source_node()->out_edges()) {
    if (HasNodeAttr(e->dst()->def(), outside_compilation_attr_name)) {
      edges_to_remove.push_back(e);
    }
  }
  for (const Edge* e : g->sink_node()->in_edges()) {
    if (HasNodeAttr(e->src()->def(), outside_compilation_attr_name)) {
      edges_to_remove.push_back(e);
    }
  }
  for (auto e : edges_to_remove) {
    g->RemoveEdge(e);
  }

  TF_RETURN_IF_ERROR(PreprocessControlEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(PreprocessDataEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  return Status::OK();
}

Status PostprocessEdgesBetweenOutsideCompilations(
    Graph* g, const string& outside_compilation_attr_name) {
  TF_RETURN_IF_ERROR(PostprocessDataEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(PostprocessControlEdgesBetweenOutsideCompilations(
      g, outside_compilation_attr_name));
  return Status::OK();
}

}  // namespace tensorflow
