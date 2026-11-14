/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/JSON.h"

namespace tooling {
namespace visualization_client {

namespace {

// Converts a vector of objects to a json array.
// The objects in the vector must have a Json() method that returns a
// type convertible to llvm::json::Value.
template <typename T>
llvm::json::Array ToJsonArray(const std::vector<T>& data_list) {
  llvm::json::Array json_array;
  for (const T& data : data_list) {
    json_array.push_back(data.Json());
  }
  return json_array;
}

}  // namespace

const char Attribute::kKey[] = "key";
const char Attribute::kValue[] = "value";

llvm::json::Object Attribute::Json() const {
  llvm::json::Object json_attr;
  json_attr[kKey] = key;
  json_attr[kValue] = value;
  return json_attr;
}

const char Metadata::kId[] = "id";
const char Metadata::kAttrs[] = "attrs";

llvm::json::Object Metadata::Json() const {
  llvm::json::Object json_metadata;
  json_metadata[kId] = id;
  json_metadata[kAttrs] = ToJsonArray(attrs);
  return json_metadata;
}

const char GraphEdge::kSourceNodeId[] = "sourceNodeId";
const char GraphEdge::kSourceNodeOutputId[] = "sourceNodeOutputId";
const char GraphEdge::kTargetNodeInputId[] = "targetNodeInputId";
const char GraphEdge::kEdgeMetadata[] = "edgeMetadata";

llvm::json::Object GraphEdge::Json() const {
  llvm::json::Object json_edge;
  json_edge[kSourceNodeId] = source_node_id;
  json_edge[kSourceNodeOutputId] = source_node_output_id;
  json_edge[kTargetNodeInputId] = target_node_input_id;
  json_edge[kEdgeMetadata] = ToJsonArray(edge_metadata);
  return json_edge;
}

const char GraphNodeConfig::kPinToGroupTop[] = "pinToGroupTop";

llvm::json::Object GraphNodeConfig::Json() const {
  llvm::json::Object json_config;
  json_config[kPinToGroupTop] = pin_to_group_top;
  return json_config;
}

const char GraphNode::kNodeId[] = "id";
const char GraphNode::kNodeLabel[] = "label";
const char GraphNode::kNodeName[] = "namespace";
const char GraphNode::kSubgraphIds[] = "subgraphIds";
const char GraphNode::kNodeAttrs[] = "attrs";
const char GraphNode::kIncomingEdges[] = "incomingEdges";
const char GraphNode::kInputsMetadata[] = "inputsMetadata";
const char GraphNode::kOutputsMetadata[] = "outputsMetadata";
const char GraphNode::kConfig[] = "config";

llvm::json::Object GraphNode::Json() const {
  llvm::json::Object json_node;
  json_node[kNodeId] = node_id;
  json_node[kNodeLabel] = node_label;
  json_node[kNodeName] = node_name;
  json_node[kSubgraphIds] = subgraph_ids;
  json_node[kNodeAttrs] = ToJsonArray(node_attrs);
  json_node[kIncomingEdges] = ToJsonArray(incoming_edges);
  json_node[kInputsMetadata] = ToJsonArray(inputs_metadata);
  json_node[kOutputsMetadata] = ToJsonArray(outputs_metadata);

  if (config.has_value()) {  // Only add config if it exists
    json_node[kConfig] = config->Json();
  }

  return json_node;
}

const char Edge::kSourceNodeId[] = "sourceNodeId";
const char Edge::kTargetNodeId[] = "targetNodeId";
const char Edge::kLabel[] = "label";

llvm::json::Object Edge::Json() const {
  llvm::json::Object json_edge;
  json_edge[kSourceNodeId] = source_node_id;
  json_edge[kTargetNodeId] = target_node_id;
  if (label.has_value()) {
    json_edge[kLabel] = label.value();
  }
  return json_edge;
}

const char EdgeOverlay::kName[] = "name";
const char EdgeOverlay::kEdges[] = "edges";
const char EdgeOverlay::kEdgeColor[] = "edgeColor";
const char EdgeOverlay::kEdgeWidth[] = "edgeWidth";
const char EdgeOverlay::kEdgeLabelFontSize[] = "edgeLabelFontSize";
const char EdgeOverlay::kShowEdgesConnectedToSelectedNodeOnly[] =
    "showEdgesConnectedToSelectedNodeOnly";

llvm::json::Object EdgeOverlay::Json() const {
  llvm::json::Object json_overlay;
  json_overlay[kName] = name;
  json_overlay[kEdges] = ToJsonArray(edges);
  json_overlay[kEdgeColor] = edge_color;
  if (edge_width.has_value()) {
    json_overlay[kEdgeWidth] = edge_width.value();
  }
  if (edge_label_font_size.has_value()) {
    json_overlay[kEdgeLabelFontSize] = edge_label_font_size.value();
  }
  if (show_edges_connected_to_selected_node_only.has_value()) {
    json_overlay[kShowEdgesConnectedToSelectedNodeOnly] =
        show_edges_connected_to_selected_node_only.value();
  }
  return json_overlay;
}

const char EdgeOverlaysData::kType[] = "type";
const char EdgeOverlaysData::kName[] = "name";
const char EdgeOverlaysData::kOverlays[] = "overlays";

llvm::json::Object EdgeOverlaysData::Json() const {
  llvm::json::Object json_edge_overlays_data;
  json_edge_overlays_data[kType] = type;
  json_edge_overlays_data[kName] = name;
  json_edge_overlays_data[kOverlays] = ToJsonArray(overlays);
  return json_edge_overlays_data;
}

const char TasksData::kEdgeOverlaysDataListLeftPane[] =
    "edgeOverlaysDataListLeftPane";
const char TasksData::kEdgeOverlaysDataListRightPane[] =
    "edgeOverlaysDataListRightPane";

llvm::json::Object TasksData::Json() const {
  llvm::json::Object json_tasks_data;
  if (edge_overlays_data_list_left_pane.has_value()) {
    json_tasks_data[kEdgeOverlaysDataListLeftPane] =
        ToJsonArray(edge_overlays_data_list_left_pane.value());
  }
  if (edge_overlays_data_list_right_pane.has_value()) {
    json_tasks_data[kEdgeOverlaysDataListRightPane] =
        ToJsonArray(edge_overlays_data_list_right_pane.value());
  }
  return json_tasks_data;
}

const char GroupNodeConfig::kNamespaceRegex[] = "namespaceRegex";
const char GroupNodeConfig::kLayoutDirection[] = "layoutDirection";

llvm::json::Object GroupNodeConfig::Json() const {
  llvm::json::Object json_config;
  json_config[kNamespaceRegex] = namespace_regex;
  if (layout_direction.has_value()) {
    json_config[kLayoutDirection] = (int)layout_direction.value();
  }
  return json_config;
}

const char Subgraph::kSubgraphId[] = "id";
const char Subgraph::kNodes[] = "nodes";
const char Subgraph::kTasksData[] = "tasksData";
const char Subgraph::kGroupNodeConfigs[] = "groupNodeConfigs";
const char Subgraph::kGroupNodeAttributes[] = "groupNodeAttributes";

llvm::json::Object Subgraph::Json() const {
  llvm::json::Object json_subgraph;
  json_subgraph[kSubgraphId] = subgraph_id;
  json_subgraph[kNodes] = ToJsonArray(nodes);
  if (tasks_data.has_value()) {
    json_subgraph[kTasksData] = tasks_data->Json();
  }
  if (!group_node_configs.empty()) {
    json_subgraph[kGroupNodeConfigs] = ToJsonArray(group_node_configs);
  }
  if (!group_node_attributes.empty()) {
    llvm::json::Object json_attributes;
    for (const auto& [ns, attrs] : group_node_attributes) {  // NOLINT
      llvm::json::Object json_attrs;
      for (const auto& [key, value] : attrs) {  // NOLINT
        json_attrs[key] = value;
      }
      json_attributes[ns] = std::move(json_attrs);
    }
    json_subgraph[kGroupNodeAttributes] = std::move(json_attributes);
  }
  return json_subgraph;
}

const char Graph::kLabel[] = "label";
const char Graph::kSubgraphs[] = "subgraphs";

llvm::json::Object Graph::Json() const {
  llvm::json::Object json_graph;
  json_graph[kLabel] = label;
  json_graph[kSubgraphs] = ToJsonArray(subgraphs);
  return json_graph;
}

const char GraphCollection::kGraphs[] = "graphs";

llvm::json::Array GraphCollection::Json() const { return ToJsonArray(graphs); }

}  // namespace visualization_client
}  // namespace tooling
