/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/fusion_wrapper.h"

#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> FusionWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto instructions = module->entry_computation()->MakeInstructionPostOrder();
  bool changed = false;

  std::function<absl::Status(HloInstruction*)> handle_instruction;
  handle_instruction = [&](HloInstruction* instruction) -> absl::Status {
    switch (instruction->opcode()) {
      case HloOpcode::kConditional:
      case HloOpcode::kWhile:
        for (auto* computation : instruction->called_computations()) {
          for (auto* inner_instruction :
               computation->MakeInstructionPostOrder()) {
            TF_RETURN_IF_ERROR(handle_instruction(inner_instruction));
          }
        }
        break;
      case HloOpcode::kAbs:
      case HloOpcode::kAdd:
      case HloOpcode::kAnd:
      case HloOpcode::kAtan2:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kBroadcast:
      case HloOpcode::kCeil:
      case HloOpcode::kCbrt:
      case HloOpcode::kClamp:
      case HloOpcode::kClz:
      case HloOpcode::kCompare:
      case HloOpcode::kComplex:
      case HloOpcode::kConcatenate:
      case HloOpcode::kConvert:
      case HloOpcode::kCopy:
      case HloOpcode::kCos:
      case HloOpcode::kDivide:
      case HloOpcode::kDot:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kErf:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kGather:
      case HloOpcode::kImag:
      case HloOpcode::kIota:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kMap:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kMultiply:
      case HloOpcode::kNegate:
      case HloOpcode::kNot:
      case HloOpcode::kOr:
      case HloOpcode::kPad:
      case HloOpcode::kPopulationCount:
      case HloOpcode::kPower:
      case HloOpcode::kReal:
      case HloOpcode::kReshape:
      case HloOpcode::kReduce:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kRemainder:
      case HloOpcode::kReverse:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kRsqrt:
      case HloOpcode::kScatter:
      case HloOpcode::kSelect:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightLogical:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kSign:
      case HloOpcode::kSin:
      case HloOpcode::kSlice:
      case HloOpcode::kSqrt:
      case HloOpcode::kSubtract:
      case HloOpcode::kStochasticConvert:
      case HloOpcode::kTan:
      case HloOpcode::kTanh:
      case HloOpcode::kTranspose:
      case HloOpcode::kXor: {
        auto* computation = instruction->parent();
        auto* fusion_instruction =
            computation->AddInstruction(HloInstruction::CreateFusion(
                instruction->shape(),
                ChooseFusionKind(*instruction, *instruction), instruction));
        const absl::string_view wrapped_opcode =
            HloOpcodeString(instruction->opcode());
        module->SetAndUniquifyInstrName(
            fusion_instruction, absl::StrCat("wrapped_", wrapped_opcode));
        module->SetAndUniquifyComputationName(
            fusion_instruction->fused_instructions_computation(),
            absl::StrCat("wrapped_", wrapped_opcode, "_computation"));
        if (module->has_schedule()) {
          module->schedule().replace_instruction(computation, instruction,
                                                 fusion_instruction);
        }
        TF_RETURN_IF_ERROR(
            fusion_instruction->CopyAllControlDepsFrom(instruction));
        TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
        TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(fusion_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
        break;
      }
      default:
        break;
    }
    return absl::OkStatus();
  };

  for (auto* instruction : instructions) {
    TF_RETURN_IF_ERROR(handle_instruction(instruction));
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
