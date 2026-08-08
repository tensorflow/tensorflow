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

#include "xla/backends/autotuner/hlo_extractor.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tsl/util/sorted_range.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

bool CompareFingerprintPairs(
    const std::pair<tsl::Fprint128, EquivalentInstructions>& a,
    const std::pair<tsl::Fprint128, EquivalentInstructions>& b) {
  if (a.first.high64 != b.first.high64) {
    return a.first.high64 < b.first.high64;
  }
  return a.first.low64 < b.first.low64;
}

}  // namespace

std::vector<EquivalentInstructions> ExtractEquivalentInstructions(
    const HloModule& module, const InstructionFilterFn& filter_fn) {
  absl::flat_hash_map<tsl::Fprint128, EquivalentInstructions,
                      tsl::Fprint128Hasher>
      grouped_instructions;
  for (HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (filter_fn(*instr)) {
        grouped_instructions[GetHloFingerprint(*instr)].push_back(instr);
      }
    }
  }
  std::vector<EquivalentInstructions> result;
  result.reserve(grouped_instructions.size());
  // Ensure deterministic iteration using sorted range.
  for (auto& [fingerprint, nodes] :
       tsl::SortedRange(grouped_instructions, CompareFingerprintPairs)) {
    result.push_back(std::move(nodes));
  }
  return result;
}

}  // namespace xla
