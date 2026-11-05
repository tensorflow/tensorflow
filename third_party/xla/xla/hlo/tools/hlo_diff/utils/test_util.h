/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEST_UTIL_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEST_UTIL_H_

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla {
namespace hlo_diff {

// Returns the sorted names of the left instructions in `pairs`, a map from left
// to right instructions.
template <typename PairMap>
std::vector<std::string> GetSortedLeftInstructionNames(const PairMap& pairs) {
  std::vector<std::string> names;
  names.reserve(pairs.size());
  // The names are sorted below, so the iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [left, right] : pairs) {
    names.emplace_back(left->name());
  }
  absl::c_sort(names);
  return names;
}

// Returns the node with the given name.
// Returns nullptr if the node is not found.
const HloInstructionNode* GetNodeByName(const HloGumgraph& graph,
                                        absl::string_view name);

// Map Nodes, overwriting existing mappings if they are different.
void OverwriteMapInstructions(const HloInstructionNode* left,
                              const HloInstructionNode* right,
                              HloGumgraphMappings& mappings,
                              bool position_unchanged = false,
                              absl::string_view matcher_debug_info = "");

// Matches all node pairs with the same name.
void MatchAllNodesByName(const HloGumgraph& left, const HloGumgraph& right,
                         HloGumgraphMappings& mappings);

// Extracts the mapped instruction names from the HloGumgraphMappings.
absl::flat_hash_map<std::string, std::string> ExtractMappedInstructionNames(
    const HloGumgraphMappings& mappings);

// Extracts the mapped computation names from the HloGumgraphMappings.
absl::flat_hash_map<std::string, std::string> ExtractMappedComputationNames(
    const HloGumgraphMappings& mappings);

// Extracts the computation match type from the HloGumgraphMappings.
absl::flat_hash_map<std::string, ComputationMatchType>
ExtractComputationMatchType(const HloGumgraphMappings& mappings);

// Returns the instruction with the given name.
absl::StatusOr<HloInstruction*> GetInstructionByName(HloModule& module,
                                                     absl::string_view name);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEST_UTIL_H_
