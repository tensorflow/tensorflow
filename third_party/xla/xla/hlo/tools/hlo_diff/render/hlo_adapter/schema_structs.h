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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_SCHEMA_STRUCTS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_SCHEMA_STRUCTS_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/Support/JSON.h"

namespace tooling {
namespace visualization_client {

/// A key-value pair for attributes.
struct Attribute {
  Attribute(std::string key, std::string value)
      : key(std::move(key)), value(std::move(value)) {}
  /// The key of the attribute.
  std::string key;
  /// The value of the attribute.
  std::string value;

  llvm::json::Object Json() const;

 private:
  static const char kKey[];
  static const char kValue[];
};

/// Metadata for inputs and outputs, consisting of an ID and a list of
/// attributes.
struct Metadata {
  /// The unique identifier for the metadata item.
  std::string id;
  /// A list of attributes associated with this metadata.
  std::vector<Attribute> attrs;

  llvm::json::Object Json() const;

 private:
  static const char kId[];
  static const char kAttrs[];
};

/// Represents an edge in the graph, typically an incoming edge to a node.
struct GraphEdge {
  /// The ID of the node where the edge originates.
  std::string source_node_id;
  /// The specific output ID of the source node this edge comes from.
  std::string source_node_output_id;
  /// The specific input ID of the target node this edge connects to.
  std::string target_node_input_id;
  /// Metadata associated with the edge.
  std::vector<Attribute> edge_metadata;

  llvm::json::Object Json() const;

 private:
  static const char kSourceNodeId[];
  static const char kSourceNodeOutputId[];
  static const char kTargetNodeInputId[];
  static const char kEdgeMetadata[];
};

/// Configuration for a graph node.
struct GraphNodeConfig {
  /// Whether to pin the node to the top of the group it belongs to.
  bool pin_to_group_top = false;

  llvm::json::Object Json() const;

 private:
  static const char kPinToGroupTop[];
};

/// A single node in the graph.
struct GraphNode {
  /// The unique id of the node.
  std::string node_id;
  /// The label of the node, displayed on the node in the model graph.
  std::string node_label;
  /**
   * The namespace/hierarchy data of the node in the form of a "path" (e.g.
   * a/b/c). The visualizer uses this to display nodes in a nested way.
   */
  std::string node_name;
  /**
   * Ids of subgraphs that this node goes into. The visualizer allows users to
   * click this node and navigate to the selected subgraph.
   */
  std::vector<std::string> subgraph_ids;
  /// The attributes of the node.
  std::vector<Attribute> node_attrs;
  /// A list of incoming edges.
  std::vector<GraphEdge> incoming_edges;
  /// Metadata for inputs.
  std::vector<Metadata> inputs_metadata;
  /// Metadata for outputs.
  std::vector<Metadata> outputs_metadata;
  /// Custom configs for the node.
  std::optional<GraphNodeConfig> config;

  llvm::json::Object Json() const;

 private:
  static const char kNodeId[];
  static const char kNodeLabel[];
  static const char kNodeName[];
  static const char kSubgraphIds[];
  static const char kNodeAttrs[];
  static const char kIncomingEdges[];
  static const char kInputsMetadata[];
  static const char kOutputsMetadata[];
  static const char kConfig[];
};

/// An edge used for overlays, connecting a source node to a target node.
struct Edge {
  /// The ID of the source node.
  std::string source_node_id;
  /// The ID of the target node.
  std::string target_node_id;
  /// An optional label to display on the edge.
  std::optional<std::string> label;

  llvm::json::Object Json() const;

 private:
  static const char kSourceNodeId[];
  static const char kTargetNodeId[];
  static const char kLabel[];
};

/// A set of edges with a common name and styling.
struct EdgeOverlay {
  /// The name of the overlay.
  std::string name;
  /// The list of edges included in this overlay.
  std::vector<Edge> edges;
  /// The color for the edges in this overlay.
  std::string edge_color;
  /// The width for the edges in this overlay.
  std::optional<float> edge_width;
  /// The font size for the edge labels in this overlay.
  std::optional<float> edge_label_font_size;
  /// Whether to show only edges connected to the selected node.
  std::optional<bool> show_edges_connected_to_selected_node_only;

  llvm::json::Object Json() const;

 private:
  static const char kName[];
  static const char kEdges[];
  static const char kEdgeColor[];
  static const char kEdgeWidth[];
  static const char kEdgeLabelFontSize[];
  static const char kShowEdgesConnectedToSelectedNodeOnly[];
};

/// A container for a set of edge overlays.
struct EdgeOverlaysData {
  /// The type identifier, typically "edge_overlays".
  std::string type = "edge_overlays";
  /// The name for this set of overlay data.
  std::string name;
  /// A list of edge overlays.
  std::vector<EdgeOverlay> overlays;

  llvm::json::Object Json() const;

 private:
  static const char kType[];
  static const char kName[];
  static const char kOverlays[];
};

/// Data for various tasks that provide extra data to be visualized.
struct TasksData {
  /**
   * List of data for edge overlays that will be applied to the left pane
   * (2-pane view) or the only pane (1-pane view).
   */
  std::optional<std::vector<EdgeOverlaysData>>
      edge_overlays_data_list_left_pane;
  /// List of data for edge overlays that will be applied to the right pane.
  std::optional<std::vector<EdgeOverlaysData>>
      edge_overlays_data_list_right_pane;

  llvm::json::Object Json() const;

 private:
  static const char kEdgeOverlaysDataListLeftPane[];
  static const char kEdgeOverlaysDataListRightPane[];
};

/// Layout direction for group nodes.
enum class LayoutDirection {
  kTopBottom = 0,
  kLeftRight = 1,
};

/// Configuration for a group node.
struct GroupNodeConfig {
  /// The regex to match against the namespace of group nodes.
  std::string namespace_regex;
  /// Whether to expand the group node by default.
  std::optional<LayoutDirection> layout_direction;

  llvm::json::Object Json() const;

 private:
  static const char kNamespaceRegex[];
  static const char kLayoutDirection[];
};

/// A subgraph corresponds to a single renderable graph with an ID and a list of
/// nodes.
struct Subgraph {
  explicit Subgraph(std::string subgraph_id)
      : subgraph_id(std::move(subgraph_id)) {}
  /// The ID of the subgraph.
  std::string subgraph_id;
  /// A list of nodes in the subgraph.
  std::vector<GraphNode> nodes;
  /// Data for various tasks that provide extra data to be visualized.
  std::optional<TasksData> tasks_data;
  /// Custom configs for group nodes.
  std::vector<GroupNodeConfig> group_node_configs;
  /// Attributes for group nodes.
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::string>>
      group_node_attributes;

  llvm::json::Object Json() const;

 private:
  static const char kSubgraphId[];
  static const char kNodes[];
  static const char kTasksData[];
  static const char kGroupNodeConfigs[];
  static const char kGroupNodeAttributes[];
};

/// A logical grouping of subgraphs with a shared label.
struct Graph {
  /// The label of the graph collection.
  std::string label;
  /// The list of subgraphs within this graph group.
  std::vector<Subgraph> subgraphs;

  llvm::json::Object Json() const;

 private:
  static const char kLabel[];
  static const char kSubgraphs[];
};

/// A collection of graphs. This is the top-level input to the visualizer.
struct GraphCollection {
  /// The graphs inside the collection.
  std::vector<Graph> graphs;

  llvm::json::Array Json() const;

 private:
  static const char kGraphs[];
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_SCHEMA_STRUCTS_H_
