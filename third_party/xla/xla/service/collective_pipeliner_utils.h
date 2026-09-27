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

#ifndef XLA_SERVICE_COLLECTIVE_PIPELINER_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_PIPELINER_UTILS_H_

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace collective_pipeliner_utils {

enum PipeliningDirection {
  kBackward,
  kForward,
  kForwardSink,
};

// Returns true if the operation is considered a default acceptable
// formatting operation that is safe to hoist/sink.
inline bool IsBaseAcceptableFormattingOp(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCustomCall) {
    return !Cast<HloCustomCallInstruction>(inst)->custom_call_has_side_effect();
  }

  return HloPredicateIsOp<
      HloOpcode::kSlice, HloOpcode::kDynamicSlice, HloOpcode::kPad,
      HloOpcode::kCollectivePermute, HloOpcode::kConvert, HloOpcode::kReshape,
      HloOpcode::kAllReduce, HloOpcode::kTranspose, HloOpcode::kBroadcast,
      HloOpcode::kAllGather, HloOpcode::kGetTupleElement, HloOpcode::kReduce,
      HloOpcode::kConcatenate, HloOpcode::kReduceScatter, HloOpcode::kBitcast>(
      inst);
}

}  // namespace collective_pipeliner_utils
}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_PIPELINER_UTILS_H_
