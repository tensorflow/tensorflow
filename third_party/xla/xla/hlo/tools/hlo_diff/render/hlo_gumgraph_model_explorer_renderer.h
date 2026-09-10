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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_MODEL_EXPLORER_RENDERER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_MODEL_EXPLORER_RENDERER_H_

#include <string>

#include "absl/status/statusor.h"
#include "third_party/json/src/json.hpp"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"

namespace xla {
namespace hlo_diff {

// Model Explorer Renderer options.
struct MeRenderOptions {
  // Whether to include debug info in the model explorer JSONs.
  bool debug_mode = false;
  // Whether to collapse unchanged computations in the model explorer JSONs.
  bool collapse_unchanged_computations = true;
  // Whether to hide unchanged subgraphs in the model explorer JSONs.
  bool hide_unchanged_subgraphs = false;
};

// Model explorer JSONs.
struct MeJson {
  // Main model explorer graph collection.
  tooling::visualization_client::GraphCollection graph_collection;
  // Node mapping for sync navigation.
  nlohmann::json sync_nav_mapping;
  // Node data containing diff attributes for aggregated view.
  nlohmann::json node_data;

  std::string DumpGraphCollectionJson() const {
    std::string json_output;
    llvm::raw_string_ostream ost(json_output);
    ost << graph_collection.Json();
    return json_output;
  }
};

// Returns the pinned node id for the given computation.
inline std::string GetComputationPinnedNodeId(
    const HloComputation* computation) {
  if (computation->IsFusionComputation()) {
    return tooling::visualization_client::GetInstructionId(
        computation->FusionInstruction());
  }
  return tooling::visualization_client::GetComputationId(computation);
}

// Returns the instruction id for the given instruction.
inline std::string GetInstructionNodeId(const HloInstruction* instruction) {
  return tooling::visualization_client::GetInstructionId(instruction);
}

// Convert the graphs to a model explorer JSON annotated with diff results.
// When diff_summary is provided, the renderer will not expand computations that
// has no diff.
absl::StatusOr<MeJson> RenderMe(
    const HloModule& left, const HloModule& right,
    const DiffResult& diff_result, const DiffSummary* diff_summary = nullptr,
    const MeRenderOptions& render_options = MeRenderOptions());

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_MODEL_EXPLORER_RENDERER_H_
