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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_TEXT_RENDERER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_TEXT_RENDERER_H_

#include <sstream>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"

namespace xla {
namespace hlo_diff {

// Options for rendering the diff result to text.
struct RenderTextOptions {
  // Print the top n opcodes. If negative, print all opcodes.
  int top_n_opcodes = -1;
  // Print the top n instructions per opcode. If negative, print all
  // instructions.
  int max_instructions_per_opcode = -1;
  // If true, only print the instruction name. Otherwise, print the full details
  // of the instruction.
  bool name_only = false;
  // Opcodes to be ignored when printing summaries.
  absl::flat_hash_set<HloOpcode> opcodes_to_ignore;
  // If true, print the unchanged instructions.
  bool print_unchanged_instructions = true;
};

// Renders the diff result to a text output stream.
void RenderText(const DiffResult& diff_result, std::ostringstream& out,
                const RenderTextOptions& options = {});

// Renders the diff summary to a text output stream.
void RenderTextSummary(const DiffResult& diff_result, std::ostringstream& out);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_TEXT_RENDERER_H_
