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

#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/graphnode_builder.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"

namespace tooling {
namespace visualization_client {
namespace {

void AppendAttrToMetadataImpl(const int metadata_id, absl::string_view attr_key,
                              absl::string_view attr_value,
                              std::vector<Metadata>& metadata_list) {
  Attribute attr{std::string(attr_key), std::string(attr_value)};
  std::string id_str = absl::StrCat(metadata_id);
  for (Metadata& metadata : metadata_list) {
    if (metadata.id == id_str) {
      metadata.attrs.push_back(std::move(attr));
      return;
    }
  }
  Metadata metadata;
  metadata.id = id_str;
  metadata.attrs.push_back(std::move(attr));
  metadata_list.push_back(std::move(metadata));
}

}  // namespace

void GraphNodeBuilder::SetNodeId(absl::string_view node_id_str) {
  node_.node_id = node_id_str;
}

std::string GraphNodeBuilder::GetNodeId() const { return node_.node_id; }

void GraphNodeBuilder::SetNodeLabel(absl::string_view node_label) {
  node_.node_label = node_label;
}

std::string GraphNodeBuilder::GetNodeLabel() const { return node_.node_label; }

void GraphNodeBuilder::SetNodeName(absl::string_view node_name) {
  node_.node_name = node_name;
}

std::string GraphNodeBuilder::GetNodeName() const { return node_.node_name; }

void GraphNodeBuilder::SetNodeInfo(absl::string_view node_id_str,
                                   absl::string_view node_label,
                                   absl::string_view node_name) {
  node_.node_id = node_id_str;
  node_.node_label = node_label;
  node_.node_name = node_name;
}

void GraphNodeBuilder::AppendEdgeInfo(
    absl::string_view source_node_id_str,
    absl::string_view source_node_output_id_str,
    absl::string_view target_node_input_id_str) {
  GraphEdge edge;
  edge.source_node_id = source_node_id_str;
  edge.source_node_output_id = source_node_output_id_str;
  edge.target_node_input_id = target_node_input_id_str;
  node_.incoming_edges.push_back(edge);
}

void GraphNodeBuilder::AppendSubgraphId(absl::string_view subgraph_id_str) {
  node_.subgraph_ids.push_back(std::string(subgraph_id_str));
}

void GraphNodeBuilder::AppendNodeAttribute(absl::string_view key,
                                           absl::string_view value) {
  node_.node_attrs.push_back(Attribute(std::string(key), std::string(value)));
}

absl::string_view GraphNodeBuilder::GetNodeAttribute(absl::string_view key) {
  for (const Attribute& attr : node_.node_attrs) {
    if (attr.key == key) {
      return attr.value;
    }
  }
  return "";
}

void GraphNodeBuilder::AppendAttrToMetadata(const EdgeType edge_type,
                                            const int metadata_id,
                                            absl::string_view attr_key,
                                            absl::string_view attr_value) {
  switch (edge_type) {
    case EdgeType::kInput: {
      AppendAttrToMetadataImpl(metadata_id, attr_key, attr_value,
                               node_.inputs_metadata);
      break;
    }
    case EdgeType::kOutput: {
      AppendAttrToMetadataImpl(metadata_id, attr_key, attr_value,
                               node_.outputs_metadata);
      break;
    }
  }
}

void GraphNodeBuilder::SetPinToGroupTop(bool pin_to_group_top) {
  if (node_.config.has_value()) {
    node_.config->pin_to_group_top = pin_to_group_top;
  } else {
    node_.config = GraphNodeConfig{pin_to_group_top};
  }
}

}  // namespace visualization_client
}  // namespace tooling
