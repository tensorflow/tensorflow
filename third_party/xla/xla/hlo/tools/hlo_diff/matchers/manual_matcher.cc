/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/matchers/manual_matcher.h"

#include "absl/log/log.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_mappings.h"

namespace xla::hlo_diff {
void ManualMatcher::Match(HloGumgraphMappings& mappings) const {
  LOG(INFO) << "Running ManualMatcher: matching input nodes";
  int current_mapping_count = mappings.left_to_right_instruction_map.size();
  for (const auto& [left_node_name, right_node_name] : nodes_to_match_) {
    HloInstructionNode* left_node = left_.GetNode(left_node_name);
    HloInstructionNode* right_node = right_.GetNode(right_node_name);
    if (left_node == nullptr || right_node == nullptr) {
      LOG(ERROR) << "Manual mapped node pair does not exist: Left: "
                 << left_node_name << " Right: " << right_node_name;
      continue;
    }
    mappings.MapInstructionsIfAbsent(left_node, right_node, type_);
  }
  LOG(INFO) << "Finished ManualMatcher. Total left to "
               "right manual mappings: "
            << mappings.left_to_right_instruction_map.size() -
                   current_mapping_count;
}
}  // namespace xla::hlo_diff
