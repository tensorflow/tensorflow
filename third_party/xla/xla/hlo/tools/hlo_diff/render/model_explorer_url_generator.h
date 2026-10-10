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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_MODEL_EXPLORER_URL_GENERATOR_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_MODEL_EXPLORER_URL_GENERATOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/render/graph_url_generator.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_model_explorer_renderer.h"

namespace xla {
namespace hlo_diff {

constexpr absl::string_view kModelExplorerBaseUrl = "http://localhost:8080";

// A helper class to generate a model explorer url from the given json.
class MeUrlGenerator : public GraphUrlGenerator {
 public:
  MeUrlGenerator(absl::string_view left_subgraph_id,
                 absl::string_view left_label,
                 absl::string_view right_subgraph_id,
                 absl::string_view right_label, absl::string_view me_json_path,
                 absl::string_view sync_nav_path,
                 absl::Span<const std::string> node_data_path,
                 absl::string_view base_url)
      : base_url_(base_url),
        left_subgraph_id_(left_subgraph_id),
        left_label_(left_label),
        right_subgraph_id_(right_subgraph_id),
        right_label_(right_label),
        me_json_path_(me_json_path),
        sync_nav_path_(sync_nav_path),
        node_data_paths_(node_data_path.begin(), node_data_path.end()) {}

  // Factory function to create a MeUrlGenerator from the given json with file
  // size checks. Files larger than 512MB will be rejected.
  static absl::StatusOr<std::unique_ptr<MeUrlGenerator>> Create(
      const tooling::visualization_client::GraphCollection* graph_collection,
      absl::string_view me_json_path, absl::string_view sync_nav_path,
      absl::string_view node_data_path) {
    return Create(graph_collection, me_json_path, sync_nav_path,
                  absl::MakeConstSpan({std::string(node_data_path)}));
  }

  static absl::StatusOr<std::unique_ptr<MeUrlGenerator>> Create(
      const tooling::visualization_client::GraphCollection* graph_collection,
      absl::string_view me_json_path, absl::string_view sync_nav_path,
      absl::Span<const std::string> node_data_paths,
      absl::string_view base_url = kModelExplorerBaseUrl);

  // Generate a url for a model explorer from the given json.
  std::string GenerateWithSelectedNodes(
      absl::string_view left_selected_node_id,
      absl::string_view right_selected_node_id) override;

  // Generate a url for a model explorer from the given instruction pair.
  std::string GenerateWithSelectedNodes(
      const HloInstruction* left_inst,
      const HloInstruction* right_inst) override {
    return GenerateWithSelectedNodes(
        left_inst == nullptr ? "" : GetInstructionNodeId(left_inst),
        right_inst == nullptr ? "" : GetInstructionNodeId(right_inst));
  }

  // Generate a url for a model explorer from the given computation pair.
  std::string GenerateWithSelectedNodes(
      const HloComputation* left_comp,
      const HloComputation* right_comp) override {
    return GenerateWithSelectedNodes(
        left_comp == nullptr ? "" : GetComputationPinnedNodeId(left_comp),
        right_comp == nullptr ? "" : GetComputationPinnedNodeId(right_comp));
  }

  // Select the nodes from the diff result as initial selected nodes for the
  // model explorer url. If there are changed instructions, the changed
  // instruction pair will be selected. Otherwise, the unmatched instructions
  // will be selected.
  static std::pair<std::string, std::string> SelectInitialSelectedNodes(
      const DiffResult& diff_result);

 private:
  std::string base_url_;
  std::string left_subgraph_id_;
  std::string left_label_;
  std::string right_subgraph_id_;
  std::string right_label_;
  std::string me_json_path_;
  std::string sync_nav_path_;
  std::vector<std::string> node_data_paths_;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_MODEL_EXPLORER_URL_GENERATOR_H_
