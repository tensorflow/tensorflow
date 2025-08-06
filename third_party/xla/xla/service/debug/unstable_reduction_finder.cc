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

#include "xla/service/debug/unstable_reduction_finder.h"

#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

bool HasMaxReducer(const HloReduceInstruction* reduction) {
  return Match(reduction->to_apply()->root_instruction(),
               m::MaximumAnyOrder(m::Parameter(), m::Parameter()));
}

bool HasMinReducer(const HloReduceInstruction* reduction) {
  return Match(reduction->to_apply()->root_instruction(),
               m::MinimumAnyOrder(m::Parameter(), m::Parameter()));
}

// Returns true if the given reduction instruction is known to be stable.
//
// We err on the side of overreporting instability, by whitelisting specific
// reducers.
bool IsKnownStableReduction(const HloReduceInstruction* reduction) {
  PrimitiveType element_type = reduction->shape().element_type();
  if (!primitive_util::IsFloatingPointType(element_type)) {
    return true;
  }

  if (primitive_util::BitWidth(element_type) >= 32) {
    return true;
  }

  // At this point, we know that the reduction accumulates into a floating point
  // type smaller than f32. It's still possible that it is stable, e.g. if its
  // reducer is a min or max.
  if (HasMaxReducer(reduction) || HasMinReducer(reduction)) {
    return true;
  }

  return false;
}

}  // namespace

std::vector<const HloInstruction*> FindUnstableReductionInstructions(
    const HloModule* module) {
  std::vector<const HloInstruction*> results;

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kReduce) {
        if (!IsKnownStableReduction(Cast<HloReduceInstruction>(instruction))) {
          results.push_back(instruction);
        }
      }
    }
  }

  return results;
}

}  // namespace xla
