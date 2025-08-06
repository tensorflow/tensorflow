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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_CYCLE_DETECTOR_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_CYCLE_DETECTOR_H_

#include <vector>

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla {
namespace hlo_diff {

// Detects and logs all cycles in the provided graph.
std::vector<std::vector<const HloInstructionNode*>> DetectAndLogAllCycles(
    const std::vector<HloInstructionNode*>& graph_nodes);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_CYCLE_DETECTOR_H_
