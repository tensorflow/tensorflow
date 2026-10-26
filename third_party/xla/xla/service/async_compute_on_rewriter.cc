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

#include "xla/service/async_compute_on_rewriter.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/async_custom_call_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

bool IsComputeOnCustomCall(const HloInstruction* instr,
                           absl::string_view suffix) {
  if (instr->opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  absl::string_view target = instr->custom_call_target();
  if (!absl::ConsumeSuffix(&target, suffix)) {
    return false;
  }
  return target == "compute-on";
}

// Validates that `called_comp` is a compute-on computation that this pass knows
// how to rewrite, i.e. it consists only of parameter instructions and exactly
// one all-gather, which must have a single operand.
// Returns that all-gather on success.
absl::StatusOr<HloAllGatherInstruction*> FindComputeOnAllGather(
    HloComputation* called_comp) {
  HloAllGatherInstruction* all_gather = nullptr;
  for (HloInstruction* inst : called_comp->instructions()) {
    if (inst->opcode() == HloOpcode::kAllGather) {
      if (all_gather != nullptr) {
        return absl::UnimplementedError(absl::StrCat(
            "Rewriting compute-on is only supported when the called "
            "computation contains a single all-gather, but ",
            called_comp->name(), " contains at least two: ", all_gather->name(),
            " and ", inst->name()));
      }
      all_gather = DynCast<HloAllGatherInstruction>(inst);
      continue;
    }
    if (inst->opcode() != HloOpcode::kParameter) {
      return absl::UnimplementedError(absl::StrCat(
          "Rewriting compute-on is only supported when the called computation "
          "consists of all-gather and parameter instructions, but ",
          called_comp->name(), " contains a ", HloOpcodeString(inst->opcode()),
          " instruction: ", inst->name()));
    }
  }
  if (all_gather == nullptr) {
    return absl::UnimplementedError(absl::StrCat(
        "Rewriting compute-on is only supported when the called computation "
        "contains an all-gather, but none was found in ",
        called_comp->name()));
  }
  if (all_gather->operand_count() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Rewriting compute-on is only supported when the all-gather in the "
        "called computation has a single operand, but ",
        all_gather->name(), " in ", called_comp->name(), " has ",
        all_gather->operand_count(), " operands"));
  }
  return all_gather;
}

absl::StatusOr<bool> SimplifyEmptyComputeOn(
    HloComputation* computation, HloComputation* called_comp,
    HloInstruction* start_call, HloInstruction* done_call,
    absl::Span<const hlo_instruction_utils::async::AsyncTraceStep>
        forward_path) {
  if (absl::c_any_of(called_comp->instructions(),
                     [](const HloInstruction* inst) {
                       return inst->opcode() != HloOpcode::kParameter;
                     })) {
    return false;
  }
  HloInstruction* call_inst =
      computation->AddInstruction(HloInstruction::CreateCall(
          done_call->shape(), done_call->operands(), called_comp));
  // We do not propagate annotations from the compute-on-start back upwards to
  // its operand (start_call->mutable_operand(0)).
  ABSL_RETURN_IF_ERROR(FinishRewrite(computation, start_call, done_call,
                                /*async_start=*/start_call->mutable_operand(0),
                                /*async_done=*/call_inst, forward_path,
                                /*final_result=*/call_inst,
                                /*propagate_annotations=*/false));
  return true;
}

}  // namespace

absl::StatusOr<bool> AsyncComputeOnRewriter::ProcessComputeOn(
    HloComputation* computation, HloInstruction* start_call,
    HloInstruction* done_call,
    absl::Span<const hlo_instruction_utils::async::AsyncTraceStep> forward_path,
    bool use_legacy_collectives) {
  CHECK_EQ(start_call->called_computations().size(), 1);
  absl::string_view called_comp_name =
      start_call->called_computations().front()->name();
  HloComputation* called_comp =
      start_call->GetModule()->GetComputationWithName(called_comp_name);

  ABSL_ASSIGN_OR_RETURN(bool simplified,
                   SimplifyEmptyComputeOn(computation, called_comp, start_call,
                                          done_call, forward_path));
  if (simplified) {
    return true;
  }
  ABSL_ASSIGN_OR_RETURN(HloAllGatherInstruction * all_gather,
                   FindComputeOnAllGather(called_comp));
  HloInstruction* async_start;
  HloInstruction* async_done;
  if (use_legacy_collectives) {
    Shape shape = start_call->shape();
    Shape start_shape =
        ShapeUtil::MakeTupleShape({start_call->operand(0)->shape(), shape});
    async_start =
        computation->AddInstruction(HloInstruction::CreateAllGatherStart(
            start_shape, start_call->operands(),
            all_gather->all_gather_dimension(), all_gather->device_list(),
            /*constrain_layout=*/false, all_gather->channel_id(),
            all_gather->use_global_device_ids()));
    async_done = computation->AddInstruction(HloInstruction::CreateUnary(
        shape, HloOpcode::kAllGatherDone, async_start));
    async_start->set_frontend_attributes(start_call->frontend_attributes());
  } else {
    HloInstruction* call_inst =
        computation->AddInstruction(HloInstruction::CreateCall(
            done_call->shape(), start_call->operands(), called_comp));
    ABSL_ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         call_inst, /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false,
                         /*override_names=*/false));
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(call_inst));
    async_start = async_done->mutable_operand(0);
  }

  ABSL_RETURN_IF_ERROR(FinishRewrite(computation, start_call, done_call, async_start,
                                async_done, forward_path,
                                /*final_result=*/async_done));

  if (compute_on_helper_ != nullptr) {
    ABSL_RETURN_IF_ERROR(
        compute_on_helper_->AddBackendSpecializations(start_call, async_start));
  }
  return true;
}

absl::StatusOr<bool> AsyncComputeOnRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> done_calls;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsComputeOnCustomCall(instr, "-done")) {
        VLOG(1) << "Candidate done: " << instr->name();
        done_calls.push_back(instr);
      }
    }

    for (HloInstruction* done_call : done_calls) {
      auto expected_start =
          MatchingStartTarget(done_call->custom_call_target());
      if (!expected_start.has_value()) {
        VLOG(1) << "No matching start for " << done_call->name();
        continue;
      }
      auto trace = hlo_instruction_utils::async::TraceDataflowPath(
          done_call, [&](const HloInstruction* instr) {
            return instr->parent() == computation &&
                   instr->IsCustomCall(*expected_start);
          });
      if (!trace.has_value()) {
        VLOG(1) << "No trace found for " << done_call->name();
        continue;
      }
      auto [start_call, forward_path] = *std::move(trace);
      ABSL_ASSIGN_OR_RETURN(bool pair_changed,
                       ProcessComputeOn(computation, start_call, done_call,
                                        forward_path, use_legacy_collectives_));
      changed |= pair_changed;
    }
  }
  return changed;
}

}  // namespace xla
