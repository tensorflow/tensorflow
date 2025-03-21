/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_query.h"

#include <algorithm>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_query {

bool IsCollectiveCommunicationOp(HloOpcode op) {
  switch (op) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kCollectivePermuteStart:
      return true;
    default:
      return false;
  }
}

bool IsAsyncCollectiveStartOp(const HloInstruction* instruction,
                              bool include_send_recv) {
  HloOpcode op = instruction->opcode();
  switch (op) {
    case HloOpcode::kAsyncStart:
      return IsCollectiveCommunicationOp(instruction->async_wrapped_opcode());
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kCollectivePermuteStart:
      return true;
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return include_send_recv;
    default:
      return false;
  }
}

bool IsAsyncCollectiveDoneOp(const HloInstruction* instruction,
                             bool include_send_recv) {
  HloOpcode op = instruction->opcode();
  switch (op) {
    case HloOpcode::kAsyncDone:
      return IsCollectiveCommunicationOp(instruction->async_wrapped_opcode());
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectivePermuteDone:
      return true;
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
      return include_send_recv;
    default:
      return false;
  }
}

bool IsConstantR0F32(HloInstruction* instruction, float* out) {
  if (instruction->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsScalarWithElementType(instruction->shape(), F32)) {
    *out = instruction->literal().Get<float>({});
    return true;
  }

  return false;
}

bool AllOperandsAreParametersOrConstants(const HloInstruction& instruction) {
  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreParametersOrConstantsWithSingleUser(
    const HloInstruction& instruction) {
  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
    if (operand->user_count() > 1) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreParameters(const HloInstruction& instruction) {
  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreConstants(const HloInstruction& instruction) {
  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

HloInstruction* GetMatchingOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction) {
  for (HloInstruction* op : instruction->operands()) {
    if (matcher(op)) {
      return op;
    }
  }
  return nullptr;
}

bool MatchBinaryInstructionOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction,
                                   HloInstruction** matching_operand,
                                   HloInstruction** other_operand) {
  CHECK_EQ(instruction->operand_count(), 2);
  if (matcher(instruction->operand(0))) {
    *matching_operand = instruction->mutable_operand(0);
    *other_operand = instruction->mutable_operand(1);
    return true;
  }
  if (matcher(instruction->operand(1))) {
    *matching_operand = instruction->mutable_operand(1);
    *other_operand = instruction->mutable_operand(0);
    return true;
  }
  return false;
}

bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand) {
  return MatchBinaryInstructionOperand(
      [opcode](const HloInstruction* instruction) {
        return instruction->opcode() == opcode;
      },
      instruction, matching_operand, other_operand);
}

bool IsScalarConstant(const HloInstruction* instruction) {
  return instruction->IsConstant() && ShapeUtil::IsScalar(instruction->shape());
}

bool IsBroadcastedConstantOrScalar(const HloInstruction& instr) {
  return instr.IsConstant() || ShapeUtil::IsScalar(instr.shape()) ||
         (HloOpcode::kBroadcast == instr.opcode() &&
          (instr.operand(0)->IsConstant() ||
           ShapeUtil::IsScalar(instr.operand(0)->shape())));
}

bool IsBroadcastOfScalarConstant(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kBroadcast &&
         IsScalarConstant(instr.operand(0));
}

bool IsBroadcastOfParameter(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kBroadcast &&
         instr.operand(0)->opcode() == HloOpcode::kParameter;
}

bool IsEffectiveParameter(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kParameter ||
         ((instr.opcode() == HloOpcode::kBitcast ||
           instr.opcode() == HloOpcode::kGetTupleElement) &&
          IsEffectiveParameter(*instr.operand(0)));
}

HloInstruction* GetFirstInstructionWithOpcode(const HloComputation& computation,
                                              const HloOpcode opcode) {
  auto instructions = computation.instructions();
  auto it = absl::c_find_if(instructions, [&](HloInstruction* instr) {
    return instr->opcode() == opcode;
  });
  return it == instructions.end() ? nullptr : *it;
}

bool ContainsInstrWithOpcode(const HloComputation* comp,
                             const absl::flat_hash_set<HloOpcode>& opcodes) {
  for (const auto* instr : comp->instructions()) {
    if (opcodes.count(instr->opcode())) {
      return true;
    }
    for (const HloComputation* subcomp : instr->called_computations()) {
      if (ContainsInstrWithOpcode(subcomp, opcodes)) {
        return true;
      }
    }
  }
  return false;
}

bool ContainsLayoutConstrainedCollective(const HloModule& module,
                                         HloOpcode op) {
  CHECK(IsCollectiveCommunicationOp(op));

  for (auto computation : module.computations()) {
    for (auto hlo : computation->instructions()) {
      if (hlo->opcode() == op &&
          DynCast<HloCollectiveInstruction>(hlo)->constrain_layout()) {
        return true;
      }
    }
  }
  return false;
}

int64_t NextChannelId(const HloModule& module) {
  int64_t next_channel_id = 1;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* hlo : comp->instructions()) {
      const HloChannelInstruction* channel_instr =
          DynCast<HloChannelInstruction>(hlo);
      if (channel_instr && channel_instr->channel_id()) {
        next_channel_id =
            std::max(next_channel_id, *channel_instr->channel_id() + 1);
      }
    }
  }
  return next_channel_id;
}

bool HasX64TransformedHostTransfer(const HloModule& module) {
  for (auto computation : module.computations()) {
    for (auto hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kSend) {
        auto send = DynCast<HloSendInstruction>(hlo);
        if (send->is_host_transfer() && send->operand(0)->shape().IsTuple()) {
          return true;
        }
      } else if (hlo->opcode() == HloOpcode::kRecv) {
        auto recv = DynCast<HloRecvInstruction>(hlo);
        if (recv->is_host_transfer() &&
            recv->shape().tuple_shapes(0).IsTuple()) {
          return true;
        }
      }
    }
  }
  return false;
}

HloInstruction* GetUniqueGteInstruction(const HloInstruction* operand,
                                        int64_t index) {
  HloInstruction* gte = nullptr;
  for (HloInstruction* instr : operand->parent()->MakeInstructionPostOrder()) {
    if (!Match(instr, match::GetTupleElement().WithTupleIndex(index))) {
      continue;
    }
    if (instr->operand(0) != operand) {
      continue;
    }
    // If gte is not unique, return nullptr.
    if (gte != nullptr) {
      return nullptr;
    }
    gte = instr;
  }
  return gte;
}

HloComputation* FindComputation(HloModule* module, absl::string_view name) {
  auto computations = module->computations();
  auto it = absl::c_find_if(
      computations, [&](HloComputation* c) { return c->name() == name; });
  if (it == computations.end()) {
    return nullptr;
  }
  return *it;
}

HloInstruction* FindInstruction(const HloComputation* computation,
                                absl::string_view name) {
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->name() == name) return instruction;
  }
  return nullptr;
}

HloInstruction* FindInstruction(const HloComputation* computation,
                                HloOpcode opcode) {
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == opcode) return instruction;
  }
  return nullptr;
}

}  // namespace hlo_query
}  // namespace xla
