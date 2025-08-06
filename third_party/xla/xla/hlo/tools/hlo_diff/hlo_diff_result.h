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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_RESULT_H_
#define XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_RESULT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"

namespace xla {
namespace hlo_diff {

// Result of diff'ng the left and right HLO modules. Contains the matched and
// unmatched instructions in the two modules.
struct DiffResult {
  // Matched instructions.
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      unchanged_instructions;
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      changed_instructions;
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      moved_instructions;

  // Unmatched instructions.
  absl::flat_hash_set<const HloInstruction*> left_module_unmatched_instructions;
  absl::flat_hash_set<const HloInstruction*>
      right_module_unmatched_instructions;

  // Debug info.
  absl::flat_hash_map<std::pair<const HloInstruction*, const HloInstruction*>,
                      MatcherType>
      map_by;
  absl::flat_hash_map<std::pair<const HloInstruction*, const HloInstruction*>,
                      std::string>
      matcher_debug_info;
  absl::flat_hash_map<const HloInstruction*, HloInstructionNodeProps>
      node_props_left;
  absl::flat_hash_map<const HloInstruction*, HloInstructionNodeProps>
      node_props_right;

  // Converts the diff result to a proto.
  DiffResultProto ToProto() const;

  // Converts the diff result from a proto.
  static DiffResult FromProto(const DiffResultProto& proto,
                              const HloModule& left_module,
                              const HloModule& right_module);
};

// Constructs the diff result from the node mappings.
std::unique_ptr<const DiffResult> ConstructDiffResult(
    const HloGumgraph& left_graph, const HloGumgraph& right_graph,
    const HloGumgraphMappings& mappings);

// Logs the diff result.
void LogDiffResult(const DiffResult& diff_result);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_HLO_DIFF_RESULT_H_
