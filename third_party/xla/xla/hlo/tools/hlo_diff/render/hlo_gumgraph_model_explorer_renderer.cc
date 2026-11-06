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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_model_explorer_renderer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "third_party/json/src/json.hpp"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::nlohmann::json;
using ::tooling::visualization_client::Attribute;
using ::tooling::visualization_client::GetInstructionId;
using ::tooling::visualization_client::Graph;
using ::tooling::visualization_client::GraphCollection;
using ::tooling::visualization_client::GraphNode;
using ::tooling::visualization_client::HloToGraph;
using ::tooling::visualization_client::NodeFilter;
using ::tooling::visualization_client::Subgraph;

// Diff attributes
constexpr absl::string_view kDiffType = "diff_type";
constexpr absl::string_view kMappedNodeId = "mapped_node_id";
constexpr absl::string_view kChangedDiffTypes = "changed_diff_types";

// Debug attributes
constexpr absl::string_view kMatcherType = "matcher_type";
constexpr absl::string_view kMatcherDebugInfo = "matcher_debug_info";
constexpr absl::string_view kFingerprint = "fingerprint";
constexpr absl::string_view kSubgraphFingerprint = "subgraph_fingerprint";
constexpr absl::string_view kHeight = "height";
constexpr absl::string_view kGeneration = "generation";

constexpr absl::string_view kLeftLabel = "left";
constexpr absl::string_view kRightLabel = "right";

// Sync nav keys
constexpr char kSyncNavType[] = "type";
constexpr char kSyncNavDisableMappingFallback[] = "disableMappingFallback";
constexpr char kSyncNavMapping[] = "mapping";

// Node data keys
constexpr char kNodeDataName[] = "name";
constexpr char kNodeDataShowExpandedSummaryOnGroupNode[] =
    "showExpandedSummaryOnGroupNode";
constexpr char kNodeDataShowLabelCountColumnsInChildrenStatsTable[] =
    "showLabelCountColumnsInChildrenStatsTable";
constexpr char kNodeDataResults[] = "results";
constexpr char kNodeDataValue[] = "value";
constexpr char kNodeDataBgColor[] = "bgColor";

// DiffAttr stores diff related attributes for a graph node.
struct DiffAttr {
  const HloInstruction* instruction;
  DiffType diff_type;
  DiffSide diff_side;
  std::vector<ChangedInstructionDiffType> changed_instruction_diff_types;
  std::optional<std::string> mapped_node_id;
};

struct DebugAttr {
  MatcherType matcher_type;
  uint64_t fingerprint;
  uint64_t subgraph_fingerprint;
  int64_t height;
  int64_t generation;
  std::string matcher_debug_info;
};

// Converts the diff type enum value to a string.
absl::StatusOr<std::string> DiffTypeToString(const DiffType diff_type,
                                             const DiffSide side) {
  switch (diff_type) {
    case DiffType::kUnchanged:
      return "kUnchanged";
    case DiffType::kChanged:
      return "kUpdated";
    case DiffType::kUnmatched:
      if (side == DiffSide::kLeft) {
        return "kLeftUnmatched";
      }
      return "kRightUnmatched";

    case DiffType::kMoved:
      return "kMoved";
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unexpected value for DiffType: ", static_cast<int>(diff_type)));
}

// Converts the matcher type enum value to a string.
absl::StatusOr<std::string> MatcherTypeToString(
    const MatcherType matcher_type) {
  switch (matcher_type) {
    case MatcherType::kNotSet:
      return "kNotSet";
    case MatcherType::kManual:
      return "kManual";
    case MatcherType::kComputationGraphExactFingerprintMatcher:
      return "kComputationGraphExactFingerprintMatcher";
    case MatcherType::kComputationGraphExactSignatureMatcher:
      return "kComputationGraphExactSignatureMatcher";
    case MatcherType::kGreedySubGraphExactMatcher:
      return "kGreedySubGraphExactMatcher";
    case MatcherType::kGreedyLimitedCandidatesBottomUpMatcher:
      return "kGreedyLimitedCandidatesBottomUpMatcher";
    case MatcherType::kStrictGreedyTopDownMatcher:
      return "kStrictGreedyTopDownMatcher";
    case MatcherType::kGreedyTopDownMatcher:
      return "kGreedyTopDownMatcher";
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unexpected value for MatcherType: ", static_cast<int>(matcher_type)));
}

// Returns the hex color for a diff type.
absl::StatusOr<std::string> DiffTypeToHexColor(const DiffType diff_type,
                                               const DiffSide side) {
  switch (diff_type) {
    case DiffType::kUnchanged:
      return "#F8F9FA";  // Grey 50
    case DiffType::kUnmatched:
      if (side == DiffSide::kLeft) {
        return "#FAD2CF";  // Google Red 100
      }
      return "#CEEAD6";  // Google Green 100

    case DiffType::kChanged:
      return "#FEEFC3";  // Google Yellow 100
    case DiffType::kMoved:
      return "#D2E3FC";  // Google Blue 100
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unexpected value for DiffType: ", static_cast<int>(diff_type)));
}

// Returns a json object for a node attribute.
Attribute NodeAttr(absl::string_view key, absl::string_view value) {
  return Attribute{std::string(key), std::string(value)};
}

// Renders a single HLO graph into a model explorer json and attach diff related
// attributes. Returns a pair of graph and node data JSON.
absl::StatusOr<std::pair<Graph, json>> RenderSingleMeGraph(
    const HloComputation& computation, absl::string_view label,
    const absl::flat_hash_map<std::string, DiffAttr>& diff_attrs,
    const absl::flat_hash_map<std::string, DebugAttr>& debug_attrs,
    const DiffSummary* diff_summary, const MeRenderOptions& options) {
  // Generate the graph and parse the resulting string.
  const NodeFilter node_filter = [&](const xla::HloInstruction* instruction) {
    return true;
  };
  tooling::visualization_client::HloAdapterOption hlo_adapter_options;
  hlo_adapter_options.constant_folding = false;
  ASSIGN_OR_RETURN(
      GraphCollection graph_collection,
      HloToGraph(
          computation, node_filter,
          [&diff_summary, &diff_attrs, &options](
              const xla::HloInstruction* caller_instruction,
              const xla::HloComputation* computation) {
            if (!options.collapse_unchanged_computations) {
              return true;
            }
            if (auto it = diff_attrs.find(GetInstructionId(caller_instruction));
                it != diff_attrs.end() &&
                it->second.diff_type != DiffType::kUnchanged) {
              // Expand the computation if the caller instruction is not
              // unchanged even if the called computation has no diff.
              return true;
            }
            if (diff_summary == nullptr) {
              return true;
            }
            auto it = diff_summary->computation_summary.find(computation);
            if (it == diff_summary->computation_summary.end()) {
              return true;
            }
            return !it->second.all_unchanged;
          },
          hlo_adapter_options));
  TF_RET_CHECK(graph_collection.graphs.size() == 1)
      << "The graph collection doesn't contain exactly one graph.";
  Graph& graph = graph_collection.graphs.at(0);
  json node_data = json::object();

  // Add label.
  graph.label = label;

  // Attach diff attributes to nodes.
  TF_RET_CHECK(graph.subgraphs.size() == 1)
      << "The graph doesn't contain exactly one subgraph.";
  Subgraph& subgraph = graph.subgraphs.at(0);

  // Check number of nodes in the subgraph.
  if (subgraph.nodes.size() > 100'000) {
    LOG(WARNING) << "Subgraph has more than 100k nodes: "
                 << subgraph.nodes.size()
                 << ". This might result in the Model Explorer visualization "
                    "UI to crash or be unresponsive. Consider using the "
                    "--hide_nodes_with_unchanged_ancestors flag to hide "
                    "nodes with no diff in its ancestors.";
  }

  // Add suffix to the subgraphs id to mark left and right and make the id
  // unique when the two graphs share the same id.
  std::string subgraph_id = absl::StrCat(subgraph.subgraph_id, "_", label);
  subgraph.subgraph_id = subgraph_id;

  node_data[subgraph_id] = {
      {kNodeDataName, "Diff Results"},
      {kNodeDataShowExpandedSummaryOnGroupNode, true},
      {kNodeDataShowLabelCountColumnsInChildrenStatsTable, true}};
  json& node_data_results = node_data[subgraph_id][kNodeDataResults];

  std::vector<GraphNode> nodes_to_keep;

  // Add diff attributes to nodes.
  for (GraphNode& node : subgraph.nodes) {
    std::string node_id = node.node_id;
    if (auto it = diff_attrs.find(node_id); it != diff_attrs.end()) {
      const DiffAttr& diff_attr = it->second;
      ASSIGN_OR_RETURN(
          const std::string background_color,
          DiffTypeToHexColor(diff_attr.diff_type, diff_attr.diff_side));
      ASSIGN_OR_RETURN(
          const std::string diff_type_str,
          DiffTypeToString(diff_attr.diff_type, diff_attr.diff_side));
      node.node_attrs.push_back(NodeAttr(kDiffType, diff_type_str));
      if (diff_attr.mapped_node_id.has_value()) {
        node.node_attrs.push_back(
            NodeAttr(kMappedNodeId, *diff_attr.mapped_node_id));
      }
      node_data_results[node_id] = {{kNodeDataValue, diff_type_str},
                                    {kNodeDataBgColor, background_color}};
      if (!it->second.changed_instruction_diff_types.empty()) {
        node.node_attrs.push_back(NodeAttr(
            kChangedDiffTypes,
            absl::StrJoin(it->second.changed_instruction_diff_types, ", ",
                          [](std::string* out,
                             const ChangedInstructionDiffType& diff_type) {
                            absl::StrAppend(
                                out,
                                GetChangedInstructionDiffTypeString(diff_type));
                          })));
      }
      if (diff_summary != nullptr && options.hide_unchanged_subgraphs) {
        if (auto sit =
                diff_summary->instruction_summary.find(diff_attr.instruction);
            sit != diff_summary->instruction_summary.end() &&
            !sit->second.subgraph_unchanged) {
          nodes_to_keep.push_back(node);
        }
      }
    }
    if (debug_attrs.empty()) {
      continue;
    }
    if (auto it = debug_attrs.find(node_id); it != debug_attrs.end()) {
      ASSIGN_OR_RETURN(const std::string matcher_type_str,
                       MatcherTypeToString(it->second.matcher_type));
      node.node_attrs.push_back(NodeAttr(kMatcherType, matcher_type_str));
      node.node_attrs.push_back(
          NodeAttr(kMatcherDebugInfo, it->second.matcher_debug_info));
      node.node_attrs.push_back(
          NodeAttr(kFingerprint, absl::StrCat(it->second.fingerprint)));
      node.node_attrs.push_back(NodeAttr(
          kSubgraphFingerprint, absl::StrCat(it->second.subgraph_fingerprint)));
      node.node_attrs.push_back(
          NodeAttr(kHeight, absl::StrCat(it->second.height)));
      node.node_attrs.push_back(
          NodeAttr(kGeneration, absl::StrCat(it->second.generation)));
    }
  }
  if (options.hide_unchanged_subgraphs) {
    subgraph.nodes = std::move(nodes_to_keep);
  }
  return std::make_pair(std::move(graph), std::move(node_data));
}

absl::StatusOr<json> CreateSyncNavMapping(const DiffResult& diff_result) {
  json sync_nav = json::object();
  sync_nav[kSyncNavType] = "sync_navigation";
  sync_nav[kSyncNavDisableMappingFallback] = true;
  for (const auto& [left_instr, right_instr] :
       diff_result.unchanged_instructions) {  // NOLINT
    sync_nav[kSyncNavMapping][GetInstructionId(left_instr)] =
        GetInstructionId(right_instr);
  }
  for (const auto& [left_instr, right_instr] :
       diff_result.moved_instructions) {  // NOLINT
    sync_nav[kSyncNavMapping][GetInstructionId(left_instr)] =
        GetInstructionId(right_instr);
  }
  for (const auto& [left_instr, right_instr] :
       diff_result.changed_instructions) {  // NOLINT
    sync_nav[kSyncNavMapping][GetInstructionId(left_instr)] =
        GetInstructionId(right_instr);
  }
  return sync_nav;
}

// Builds diff attributes for a pair of mapped instructions.
void BuildDiffAttr(const HloInstruction* left_instr,
                   const HloInstruction* right_instr,
                   absl::flat_hash_map<std::string, DiffAttr>& left_attrs,
                   absl::flat_hash_map<std::string, DiffAttr>& right_attrs,
                   const DiffType diff_type) {
  const std::string left_id = GetInstructionId(left_instr);
  const std::string right_id = GetInstructionId(right_instr);
  DiffAttr left_attr = {left_instr, diff_type, DiffSide::kLeft, {}, right_id};
  DiffAttr right_attr = {right_instr, diff_type, DiffSide::kRight, {}, left_id};
  if (diff_type == DiffType::kChanged) {
    std::vector<ChangedInstructionDiffType> changed_diff_types =
        GetChangedInstructionDiffTypes(*left_instr, *right_instr);
    left_attr.changed_instruction_diff_types = changed_diff_types;
    right_attr.changed_instruction_diff_types = changed_diff_types;
  }
  left_attrs[left_id] = left_attr;
  right_attrs[right_id] = right_attr;
}

struct DiffAttrs {
  absl::flat_hash_map<std::string, DiffAttr> left_attrs;
  absl::flat_hash_map<std::string, DiffAttr> right_attrs;
};

// Builds diff attributes for the given diff result.
DiffAttrs BuildDiffAttrs(const DiffResult& diff_result) {
  DiffAttrs result;
  for (const auto& left_instr :
       diff_result.left_module_unmatched_instructions) {  // NOLINT
    const std::string left_id = GetInstructionId(left_instr);
    const DiffAttr left_attr = {left_instr, DiffType::kUnmatched,
                                DiffSide::kLeft};
    result.left_attrs[left_id] = left_attr;
  }
  for (const auto& right_instr :
       diff_result.right_module_unmatched_instructions) {  // NOLINT
    const std::string right_id = GetInstructionId(right_instr);
    const DiffAttr right_attr = {right_instr, DiffType::kUnmatched,
                                 DiffSide::kRight};
    result.right_attrs[right_id] = right_attr;
  }
  for (const auto& [left_instr, right_instr] :
       diff_result.unchanged_instructions) {  // NOLINT
    BuildDiffAttr(left_instr, right_instr, result.left_attrs,
                  result.right_attrs, DiffType::kUnchanged);
  }
  for (const auto& [left_instr, right_instr] :
       diff_result.moved_instructions) {  // NOLINT
    BuildDiffAttr(left_instr, right_instr, result.left_attrs,
                  result.right_attrs, DiffType::kMoved);
  }
  for (const auto& [left_instr, right_instr] :
       diff_result.changed_instructions) {  // NOLINT
    BuildDiffAttr(left_instr, right_instr, result.left_attrs,
                  result.right_attrs, DiffType::kChanged);
  }
  return result;
}

struct DebugAttrs {
  absl::flat_hash_map<std::string, DebugAttr> left_attrs;
  absl::flat_hash_map<std::string, DebugAttr> right_attrs;
};

// Builds debug attributes for the given diff result.
DebugAttrs BuildDebugAttrs(const DiffResult& diff_result) {
  DebugAttrs result;
  for (const auto& [inst, node_props] :
       diff_result.node_props_left) {  // NOLINT
    const std::string id = GetInstructionId(inst);
    DebugAttr& debug_attr = result.left_attrs[id];
    debug_attr.fingerprint = node_props.fingerprint;
    debug_attr.subgraph_fingerprint = node_props.subgraph_fingerprint;
    debug_attr.height = node_props.height;
    debug_attr.generation = node_props.generation;
  }
  for (const auto& [inst, node_props] :
       diff_result.node_props_right) {  // NOLINT
    const std::string id = GetInstructionId(inst);
    DebugAttr& debug_attr = result.right_attrs[id];
    debug_attr.fingerprint = node_props.fingerprint;
    debug_attr.subgraph_fingerprint = node_props.subgraph_fingerprint;
    debug_attr.height = node_props.height;
    debug_attr.generation = node_props.generation;
  }
  for (const auto& [inst_pair, map_by] : diff_result.map_by) {  // NOLINT
    const std::string left_id = GetInstructionId(inst_pair.first);
    const std::string right_id = GetInstructionId(inst_pair.second);
    DebugAttr& left_debug_attr = result.left_attrs[left_id];
    DebugAttr& right_debug_attr = result.right_attrs[right_id];
    left_debug_attr.matcher_type = map_by;
    right_debug_attr.matcher_type = map_by;
  }
  for (const auto& [inst_pair, debug_info] :
       diff_result.matcher_debug_info) {  // NOLINT
    const std::string left_id = GetInstructionId(inst_pair.first);
    const std::string right_id = GetInstructionId(inst_pair.second);
    DebugAttr& left_debug_attr = result.left_attrs[left_id];
    DebugAttr& right_debug_attr = result.right_attrs[right_id];
    left_debug_attr.matcher_debug_info = debug_info;
    right_debug_attr.matcher_debug_info = debug_info;
  }
  return result;
}

}  // namespace

absl::StatusOr<MeJson> RenderMe(const HloModule& left, const HloModule& right,
                                const DiffResult& diff_result,
                                const DiffSummary* diff_summary,
                                const MeRenderOptions& options) {
  DiffAttrs diff_attrs = BuildDiffAttrs(diff_result);
  DebugAttrs debug_attrs;
  if (options.debug_mode) {
    debug_attrs = BuildDebugAttrs(diff_result);
  }

  Graph left_graph, right_graph;
  json left_node_data, right_node_data;
  absl::Status left_status, right_status;

  {
    // Concurrently render the two graphs.
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "RenderMe", 2);
    thread_pool.Schedule([&] {
      absl::StatusOr<std::pair<Graph, json>> result = RenderSingleMeGraph(
          *left.entry_computation(), kLeftLabel, diff_attrs.left_attrs,
          debug_attrs.left_attrs, diff_summary, options);
      if (result.ok()) {
        std::tie(left_graph, left_node_data) = *std::move(result);
      } else {
        left_status = result.status();
      }
    });
    thread_pool.Schedule([&] {
      absl::StatusOr<std::pair<Graph, json>> result = RenderSingleMeGraph(
          *right.entry_computation(), kRightLabel, diff_attrs.right_attrs,
          debug_attrs.right_attrs, diff_summary, options);
      if (result.ok()) {
        std::tie(right_graph, right_node_data) = *std::move(result);
      } else {
        right_status = result.status();
      }
    });
  }

  RETURN_IF_ERROR(left_status);
  RETURN_IF_ERROR(right_status);

  left_node_data.merge_patch(right_node_data);

  GraphCollection collection;
  collection.graphs.push_back(std::move(left_graph));
  collection.graphs.push_back(std::move(right_graph));

  ASSIGN_OR_RETURN(json sync_nav_mapping, CreateSyncNavMapping(diff_result));

  return MeJson(std::move(collection), std::move(sync_nav_mapping),
                std::move(left_node_data));
}

}  // namespace hlo_diff
}  // namespace xla
