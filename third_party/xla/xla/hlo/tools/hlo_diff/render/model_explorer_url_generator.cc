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

#include "xla/hlo/tools/hlo_diff/render/model_explorer_url_generator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/json/src/json.hpp"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::nlohmann::json;
using ::tooling::visualization_client::GetInstructionId;
using ::tooling::visualization_client::Graph;

// Pane state keys
constexpr char kPaneStateDeepestExpandedGroupNodeIds[] =
    "deepestExpandedGroupNodeIds";
constexpr char kPaneStateSelectedGraphId[] = "selectedGraphId";
constexpr char kPaneStateSelectedCollectionLabel[] = "selectedCollectionLabel";
constexpr char kPaneStateWidthFraction[] = "widthFraction";
constexpr char kPaneStateSelectedNodeId[] = "selectedNodeId";

// Model explorer data keys
constexpr char kUrlDataJsonUrl[] = "url";
constexpr char kUrlDataAdapterId[] = "adapterId";
constexpr char kUrlDataModels[] = "models";
constexpr char kUrlDataNodeData[] = "nodeData";
constexpr char kUrlDataUiState[] = "uiState";
constexpr char kUrlDataPaneStates[] = "paneStates";
constexpr char kUrlDataSync[] = "sync";
constexpr char kUrlDataMode[] = "mode";
constexpr char kUrlDataCnsPath[] = "cnsPath";

constexpr int kMaxJsonSize = 512 * 1024 * 1024;  // 512MB

// Generates the paneState for a model explorer graph.
absl::StatusOr<json> GeneratePaneState(absl::string_view subgraph_id,
                                       absl::string_view label,
                                       absl::string_view filename,
                                       absl::string_view selected_node_id) {
  json pane_state =
      json({{kPaneStateDeepestExpandedGroupNodeIds, json::array()},
            {kPaneStateSelectedGraphId, subgraph_id},
            {kPaneStateSelectedCollectionLabel,
             absl::StrFormat("%s (%s)", filename, label)},
            {kPaneStateWidthFraction, 0.5}});
  if (!selected_node_id.empty()) {
    pane_state[kPaneStateSelectedNodeId] = selected_node_id;
  }
  return pane_state;
}

std::string UrlEncode(absl::string_view url) {
  std::string escaped;
  escaped.reserve(url.size());
  for (char c : url) {
    if (absl::ascii_isalnum(c) || c == '-' || c == '_' || c == '.' ||
        c == '~') {
      escaped.push_back(c);
    } else {
      absl::StrAppendFormat(&escaped, "%%%02X", static_cast<unsigned char>(c));
    }
  }
  return escaped;
}

}  // namespace

absl::StatusOr<std::unique_ptr<MeUrlGenerator>> MeUrlGenerator::Create(
    const tooling::visualization_client::GraphCollection* const
        graph_collection,
    absl::string_view me_json_path, absl::string_view sync_nav_path,
    absl::Span<const std::string> node_data_paths, absl::string_view base_url) {
  if (graph_collection == nullptr) {
    return absl::InvalidArgumentError("graph_collection is null");
  }
  if (graph_collection->graphs.size() != 2) {
    return absl::InvalidArgumentError(
        "The graph collection doesn't contain exactly two graphs.");
  }
  const Graph& left_graph = graph_collection->graphs.at(0);
  const Graph& right_graph = graph_collection->graphs.at(1);
  if (left_graph.subgraphs.size() != 1) {
    return absl::InvalidArgumentError(
        "The graph collection doesn't contain exactly one subgraph in the "
        "first graph.");
  }
  if (right_graph.subgraphs.size() != 1) {
    return absl::InvalidArgumentError(
        "The graph collection doesn't contain exactly one subgraph in the "
        "second graph.");
  }
  if (me_json_path.empty()) {
    return absl::InvalidArgumentError("me_json_path is empty");
  }

  // Check file size.
  uint64_t size;
  RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSize(std::string(me_json_path), &size));
  int64_t json_size = static_cast<int64_t>(size);
  if (json_size > kMaxJsonSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The diff results are too large for Model Explorer "
                        "visualization. Please use other output options."));
  }
  return std::make_unique<MeUrlGenerator>(
      left_graph.subgraphs.at(0).subgraph_id, left_graph.label,
      right_graph.subgraphs.at(0).subgraph_id, right_graph.label, me_json_path,
      sync_nav_path, node_data_paths, base_url);
}

std::string MeUrlGenerator::GenerateWithSelectedNodes(
    absl::string_view left_selected_node_id,
    absl::string_view right_selected_node_id) {
  json data;
  data[kUrlDataModels] = json::array({{{kUrlDataJsonUrl, me_json_path_},
                                       {kUrlDataAdapterId, "builtin_json"}}});
  data[kUrlDataNodeData] = node_data_paths_;

  absl::string_view filename = tsl::io::Basename(me_json_path_);
  absl::StatusOr<json> left_pane_state = GeneratePaneState(
      left_subgraph_id_, left_label_, filename, left_selected_node_id);
  if (!left_pane_state.ok()) {
    LOG(WARNING) << "Failed to generate left pane state: "
                 << left_pane_state.status();
    return "";
  }

  absl::StatusOr<json> right_pane_state = GeneratePaneState(
      right_subgraph_id_, right_label_, filename, right_selected_node_id);
  if (!right_pane_state.ok()) {
    LOG(WARNING) << "Failed to generate right pane state: "
                 << right_pane_state.status();
    return "";
  }

  data[kUrlDataUiState][kUrlDataPaneStates] = {std::move(*left_pane_state),
                                               std::move(*right_pane_state)};
  data[kUrlDataSync] = {{kUrlDataMode, "from_cns"},
                        {kUrlDataCnsPath, sync_nav_path_}};
  return absl::StrCat(base_url_, "/?data=", UrlEncode(data.dump()));
}

std::pair<std::string, std::string> MeUrlGenerator::SelectInitialSelectedNodes(
    const DiffResult& diff_result) {
  std::string left_selected_node_id, right_selected_node_id;
  if (!diff_result.changed_instructions.empty()) {
    auto [left_instr, right_instr] = *diff_result.changed_instructions.begin();
    left_selected_node_id = GetInstructionId(left_instr);
    right_selected_node_id = GetInstructionId(right_instr);
  } else {
    if (!diff_result.left_module_unmatched_instructions.empty()) {
      left_selected_node_id = GetInstructionId(
          *diff_result.left_module_unmatched_instructions.begin());
    }
    if (!diff_result.right_module_unmatched_instructions.empty()) {
      right_selected_node_id = GetInstructionId(
          *diff_result.right_module_unmatched_instructions.begin());
    }
  }
  return {left_selected_node_id, right_selected_node_id};
}

}  // namespace hlo_diff
}  // namespace xla
