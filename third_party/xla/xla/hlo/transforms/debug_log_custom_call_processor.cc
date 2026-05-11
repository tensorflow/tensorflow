/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/debug_log_custom_call_processor.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"

namespace xla {
namespace {

bool IsDebugLogCustomCall(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         instruction->custom_call_target() == "xla.debug.Log";
}

bool ShouldRetainCustomCall(const HloInstruction* instruction) {
  CHECK(IsDebugLogCustomCall(instruction));
  if (!instruction->has_backend_config()) {
    return false;
  }
  auto bc = instruction->backend_config<DebugLogBackendConfigProto>();
  if (!bc.ok() || !bc->has_debug_attributes_config()) {
    return false;
  }
  return bc->debug_attributes_config().log_mode() ==
         DebugAttributesProto::GUARANTEED;
}

bool ShouldInsertOptimizationBarrier(const HloInstruction* instruction) {
  CHECK(IsDebugLogCustomCall(instruction));
  if (!instruction->has_backend_config()) {
    return false;
  }
  auto bc = instruction->backend_config<DebugLogBackendConfigProto>();
  if (!bc.ok() || !bc->has_debug_attributes_config()) {
    return false;
  }
  return bc->debug_attributes_config().log_mode() ==
         DebugAttributesProto::NO_FUSION;
}

std::optional<HloModule::DebugAttributes> GetDebugAttributes(
    const HloInstruction* instruction) {
  CHECK(IsDebugLogCustomCall(instruction));
  if (!instruction->has_backend_config()) {
    return std::nullopt;
  }
  auto bc = instruction->backend_config<DebugLogBackendConfigProto>();
  if (!bc.ok() || !bc->has_debug_attributes_config()) {
    return std::nullopt;
  }
  const auto& config = bc->debug_attributes_config();

  HloModule::DebugAttributes::DebugLogMode log_mode =
      HloModule::DebugAttributes::DebugLogMode::kDefault;
  if (config.log_mode() == DebugAttributesProto::FUSION_DEBUGGER) {
    log_mode = HloModule::DebugAttributes::DebugLogMode::kFusionDebugger;
  }
  return HloModule::DebugAttributes{
      log_mode,
      config.callback_id(),
      config.partitioned(),
  };
}

}  // namespace

absl::StatusOr<bool> DebugLogCustomCallProcessor::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> debug_log_calls;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsDebugLogCustomCall(instruction)) {
        debug_log_calls.push_back(instruction);
      }
    }
  }
  if (debug_log_calls.empty()) {
    return false;
  }

  absl::flat_hash_set<int64_t> callback_ids;
  for (HloInstruction* instruction : debug_log_calls) {
    std::optional<HloModule::DebugAttributes> debug_attributes =
        GetDebugAttributes(instruction);
    if (!debug_attributes.has_value()) {
      continue;
    }
    if (!callback_ids.insert(debug_attributes->callback_id).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate callback_id found in debug log custom call: ",
                       debug_attributes->callback_id));
    }
  }

  for (HloInstruction* instruction : debug_log_calls) {
    HloComputation* computation = instruction->parent();
    std::optional<HloModule::DebugAttributes> debug_attributes =
        GetDebugAttributes(instruction);
    if (!debug_attributes.has_value()) {
      RETURN_IF_ERROR(
          computation->RemoveInstructionAndUnusedOperands(instruction));
      continue;
    }

    if (ShouldRetainCustomCall(instruction)) {
      // For guaranteed logging, we retain the custom call as-is to act as an
      // optimization barrier and process it with a custom emitter during TPU
      // lowering.
      continue;
    }

    for (int i = 0; i < instruction->operand_count(); ++i) {
      auto operand = instruction->mutable_operand(i);

      if (ShouldInsertOptimizationBarrier(instruction)) {
        HloInstruction* barrier =
            computation->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kOptimizationBarrier, operand));
        barrier->CopyOriginalValue(operand);
        RETURN_IF_ERROR(operand->ReplaceAllUsesWith(
            barrier, "DebugLogCustomCallProcessor"));
        operand = barrier;
      }
      auto orig_val = operand->original_value();
      if (orig_val != nullptr) {
        auto original_array = orig_val->original_array({});
        if (original_array.has_value()) {
          HloModule::DebugAttributes attrs = debug_attributes.value();
          attrs.op_id = i;
          module->AddDebugAttributes(original_array.value(), attrs);
        }
      }
    }

    RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(instruction));
  }

  return true;
}

}  // namespace xla
