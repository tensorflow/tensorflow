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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_GRAPH_URL_GENERATOR_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_GRAPH_URL_GENERATOR_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace hlo_diff {

// A helper class to generate a url to the graph visualization.
class GraphUrlGenerator {
 public:
  virtual ~GraphUrlGenerator() = default;

  // Generates a url to the graph visualization for the given selected nodes.
  virtual std::string Generate(absl::string_view left_selected_node_id,
                               absl::string_view right_selected_node_id) = 0;

  // Generates a url to the graph visualization for the given instruction pair.
  virtual std::string Generate(const HloInstruction* left_inst,
                               const HloInstruction* right_inst) = 0;

  // Generates a url to the graph visualization for the given computation pair.
  virtual std::string Generate(const HloComputation* left_comp,
                               const HloComputation* right_comp) = 0;
};

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_GRAPH_URL_GENERATOR_H_
