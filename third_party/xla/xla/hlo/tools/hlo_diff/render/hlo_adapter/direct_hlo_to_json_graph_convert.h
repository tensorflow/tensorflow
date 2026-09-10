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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_DIRECT_HLO_TO_JSON_GRAPH_CONVERT_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_DIRECT_HLO_TO_JSON_GRAPH_CONVERT_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"

namespace tooling {
namespace visualization_client {

// Some options that control the HLO graph conversion, such as graph pruning.
struct HloAdapterOption {
  //   GTE folding: If a get-tuple-element is not the root of a computation,
  //   connects its define node and user node directly, and annotate in the user
  //   node which element it is from the define node. For instance, "ROOT of
  //   fusion.0" -> GTE(1) -> "Parameter 0 of fusion.1" will be transformed into
  //   "ROOT of fusion.0" -> "Parameter 0 of fusion.1, tuple element 1 of
  //   fusion.1".
  bool get_tuple_element_folding = true;
  // Fold constants nodes into users.
  bool constant_folding = true;
  // If a parameter node has input to async-collective-start and output to
  // async-collective-done, mark it as implementation details and hide
  // on visualization.
  bool hide_async_collective_fusion_parameter = true;
};

// Gets the instruction id.
std::string GetInstructionId(const xla::HloInstruction* instruction);

// Gets the computation id.
std::string GetComputationId(const xla::HloComputation* computation);

// NodeFilter is a lambda indicating if an HLO instruction within the scope of
// display or not.
using NodeFilter = absl::AnyInvocable<bool(const xla::HloInstruction*) const>;

// ComputationExpand is a lambda indicating if an HLO computation need to be
// expanded or not.
using ComputationExpand = absl::AnyInvocable<bool(
    const xla::HloInstruction*, const xla::HloComputation*) const>;

// Converts an HLO computation to a GraphCollection.
absl::StatusOr<GraphCollection> HloToGraph(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const ComputationExpand& computation_expand,
    const HloAdapterOption& options = HloAdapterOption());

// Converts an HLO computation to a GraphCollection. All computations will be
// expanded.
inline absl::StatusOr<GraphCollection> HloToGraph(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const HloAdapterOption& options = HloAdapterOption()) {
  return HloToGraph(
      computation, node_filter,
      [](const xla::HloInstruction* caller_instruction,
         const xla::HloComputation* computation) { return true; },
      options);
}

// Converts an HLO computation to a JSON graph.
// Creates a ME Json representation of the subgraph rooted with the given HLO
// computation.
absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloComputation& computation,
    const HloAdapterOption& options = HloAdapterOption());

// Converts an HLO instruction and its neighbors roughly within the radius to a
// JSON graph. The fusion instruction (and its fused computation) is
// treated as a single entity. The scope will not go beyond instruction's parent
// computation (i.e., traversal continues until reaching parent computation
// boundary).
absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloInstruction& instruction, int radius,
    const HloAdapterOption& options = HloAdapterOption());

}  // namespace visualization_client
}  // namespace tooling

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_ADAPTER_DIRECT_HLO_TO_JSON_GRAPH_CONVERT_H_
