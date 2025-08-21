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

#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/core/host_offloading/annotate_host_compute_offload.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/offloaded_instruction_wrapper.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/host_offload_utils.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/tuple_util.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

namespace {
bool IsPassThroughShardingOp(const HloInstruction& instr) {
  return instr.IsCustomCall("Sharding") ||
         instr.IsCustomCall("SPMDShardToFullShape") ||
         instr.IsCustomCall("SPMDFullToShardShape");
}

void RemoveComputeTypeFrontendAttribute(HloInstruction& instr) {
  FrontendAttributes copy_of_frontend_attributes = instr.frontend_attributes();
  copy_of_frontend_attributes.mutable_map()->erase(kXlaComputeTypeAttr);
  instr.set_frontend_attributes(copy_of_frontend_attributes);
}

void RemoveComputeTypeFrontendAttribute(HloComputation& computation) {
  for (HloInstruction* instr : computation.instructions()) {
    RemoveComputeTypeFrontendAttribute(*instr);
  }
}

// Offloads instructions marked as host compute that reside within
// `computation`.
absl::StatusOr<bool> OffloadHostInstructions(
    HloComputation& computation,
    const HloHostDeviceTypeCallWrapper::Options& options) {
  auto should_offload_to_host_compute =
      [&](const HloInstruction* instr) -> bool {
    if (host_offload_utils::ComputeTypeIsHost(instr)) {
      return true;
    }
    while (IsPassThroughShardingOp(*instr)) {
      instr = instr->operand(0);
    }
    return host_offload_utils::ComputeTypeIsHost(instr);
  };

  TF_ASSIGN_OR_RETURN(
      auto offloaded_instructions_and_calls,
      offloader_util::FindAndWrapOffloadedComputations(
          computation,
          /*should_offload=*/should_offload_to_host_compute,
          /*should_fuse*/
          [](const HloInstruction&, const HloInstruction& hlo) {
            // If the computation has a schedule, we cannot fuse.
            // Otherwise incremental HloSchedule::Update() will
            // fail. Before:
            //   a = ...
            //   copy_a = copy(a) // host-to-host copy
            //   c = ds(copy_a)
            //   b = ...
            //   copy_b = copy(b) // host-to-host copy
            //   d = ds(copy_b)
            //
            // After offloading and fusing:
            //   a = ...
            //   b = ...
            //   host_call = host-call(a, b)
            //   copy_a = gte(host_call, 0)
            //   copy_b = gte(host_call, 1)
            //   c = ds(copy_a)
            //   d = ds(copy_b)
            //
            // Now b has to be scheduled before c, which an
            // incremental HloSchedule::Update() cannot do since it
            // only schedules new instructions and doesn't change
            // the original sequence.
            return !hlo.GetModule()->has_schedule();
          },
          options.clear_backend_config_device_type, "host-call"));

  bool modified = false;
  for (const auto& [_, instr_call] : offloaded_instructions_and_calls) {
    CHECK_EQ(instr_call->opcode(), HloOpcode::kCall) << absl::StreamFormat(
        "Host instruction must be a call. %s", instr_call->ToString());
    CHECK_EQ(instr_call->called_computations().size(), 1);
    HloComputation* host_computation =
        instr_call->called_computations().front();

    TF_RET_CHECK(instr_call->operands().size() ==
                 host_computation->num_parameters())
        << "Expected the number of operands to match the number of parameters "
           "of the host called computation.";
    TF_RET_CHECK(ShapeUtil::Equal(host_computation->root_instruction()->shape(),
                                  instr_call->shape()))
        << "Shape mismatch between the host computation and the corresponding "
           "host call.";

    TF_RETURN_IF_ERROR(options.set_backend_config_fn(instr_call));

    for (HloComputation* called_computation :
         instr_call->called_computations()) {
      TF_RETURN_IF_ERROR(
          offloader_util::RecursivelyClearComputeTypeFrontendAttribute(
              called_computation));
    }

    modified = true;
  }

  return modified;
}
}  // namespace

/*static*/ absl::StatusOr<HloCallInstruction*>
HloHostDeviceTypeCallWrapper::RemoveTupleParameters(HloCallInstruction* call) {
  std::vector<HloInstruction*> new_call_operands;
  std::vector<int32_t> tuple_operand_indices;

  // Add all non-tuple operands to the new call operands vector.
  for (int64_t call_operand_no = 0; call_operand_no < call->operand_count();
       ++call_operand_no) {
    HloInstruction* operand_instr = call->mutable_operand(call_operand_no);
    if (operand_instr->shape().IsTuple()) {
      tuple_operand_indices.push_back(call_operand_no);
      continue;
    }
    new_call_operands.push_back(operand_instr);
  }

  if (tuple_operand_indices.empty()) {
    // No tuple operands.
    return call;
  }

  HloComputation* called_computation = call->called_computation();

  for (int32_t tuple_operand_index : tuple_operand_indices) {
    HloInstruction* tuple_operand_instr =
        call->mutable_operand(tuple_operand_index);

    ShapeTree<HloInstruction*> operand_tuple_shape_tree =
        TupleUtil::DisassembleTupleInstruction(tuple_operand_instr);

    HloInstruction* called_comp_parameter =
        called_computation->parameter_instruction(tuple_operand_index);

    TF_RETURN_IF_ERROR(operand_tuple_shape_tree.ForEachElementWithStatus(
        [&](const ShapeIndex& operand_tuple_index,
            HloInstruction* operand_tuple_element) -> absl::Status {
          HloInstruction* called_comp_leaf_instr =
              TupleUtil::GetTupleInstructionAtIndex(*called_comp_parameter,
                                                    operand_tuple_index);

          if (!operand_tuple_shape_tree.IsLeaf(operand_tuple_index)) {
            TF_RET_CHECK(absl::c_all_of(called_comp_leaf_instr->users(),
                                        [&](HloInstruction* user) -> bool {
                                          return user->opcode() ==
                                                 HloOpcode::kGetTupleElement;
                                        }))
                << "Unsupported: Expected host tuple parameter to be "
                   "decomposed into GTEs"
                << called_comp_leaf_instr->ToString();
            return absl::OkStatus();
          }

          if (called_comp_leaf_instr == nullptr) {
            // Unused tuple element.
            return absl::OkStatus();
          }

          HloInstruction* leaf_param =
              called_computation->AddParameter(HloInstruction::CreateParameter(
                  called_computation->num_parameters(),
                  operand_tuple_element->shape(), "param"));

          TF_RETURN_IF_ERROR(
              called_comp_leaf_instr->ReplaceAllUsesWith(leaf_param));

          new_call_operands.push_back(operand_tuple_element);

          return absl::OkStatus();
        }));
  }

  // Remove any unused tuple parameter gte instructions.
  TF_RETURN_IF_ERROR(
      HloDCE().RunOnComputation(called_computation, false).status());
  // Remove the tuple parameters that are no longer used.
  TF_RETURN_IF_ERROR(
      called_computation->RemoveUnusedParametersFromAnyComputation());

  HloInstruction* new_call = call->parent()->AddInstruction(
      call->CloneWithNewOperands(call->shape(), new_call_operands));

  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWith(new_call));
  TF_RETURN_IF_ERROR(call->parent()->RemoveInstruction(call));
  return tsl::down_cast<HloCallInstruction*>(new_call);
}

/*static*/ absl::StatusOr<HloCallInstruction*>
HloHostDeviceTypeCallWrapper::MaterializeConstantsOnHostComputation(
    HloCallInstruction* call) {
  std::vector<HloInstruction*> non_constant_operands;
  HloComputation* called_computation = call->called_computation();

  absl::flat_hash_set<int64_t> dead_param_indices;
  for (HloInstruction* param : called_computation->parameter_instructions()) {
    if (param->IsDead()) {
      dead_param_indices.insert(param->parameter_number());
    }
  }
  // Remove any existing dead parameters.
  TF_RETURN_IF_ERROR(
      called_computation->RemoveUnusedParametersFromAnyComputation());

  int skipped_params = 0;
  for (int64_t operand_no = 0; operand_no < call->operand_count();
       ++operand_no) {
    if (dead_param_indices.contains(operand_no)) {
      ++skipped_params;
      continue;
    }
    HloInstruction* operand = call->mutable_operand(operand_no);
    TF_RET_CHECK(!operand->shape().IsTuple())
        << "Tuple inputs for async host computations should be flattened.";
    HloInstruction* constant_operand = nullptr;
    if (operand->IsConstant()) {
      constant_operand = operand;
    } else if (operand->IsCustomCall(
                   memory_annotations::kMoveToHostCustomCallTarget) &&
               operand->operand(0)->IsConstant()) {
      constant_operand = operand->mutable_operand(0);
    }

    if (constant_operand != nullptr) {
      HloInstruction* cloned_constant =
          called_computation->AddInstruction(constant_operand->Clone());
      HloInstruction* called_computation_parameter =
          called_computation->parameter_instruction(operand_no -
                                                    skipped_params);
      TF_RETURN_IF_ERROR(
          called_computation_parameter->ReplaceAllUsesWith(cloned_constant));
    } else {
      non_constant_operands.push_back(operand);
    }
  }

  if (non_constant_operands.size() == call->operand_count()) {
    return call;
  }

  // Remove any parameter that were constants and are now unused.
  TF_RETURN_IF_ERROR(
      called_computation->RemoveUnusedParametersFromAnyComputation());

  HloInstruction* new_call = call->parent()->AddInstruction(
      call->CloneWithNewOperands(call->shape(), non_constant_operands));

  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWith(new_call));
  TF_RETURN_IF_ERROR(call->parent()->RemoveInstruction(call));
  return tsl::down_cast<HloCallInstruction*>(new_call);
}

absl::StatusOr<bool> HloHostDeviceTypeCallWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_host_compute_instr = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (absl::c_any_of(computation->instructions(), [&](HloInstruction* instr) {
          return host_offload_utils::ComputeTypeIsHost(instr);
        })) {
      has_host_compute_instr = true;
      break;
    };
  }

  if (!has_host_compute_instr) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      AnnotateHostComputeOffload().Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(CallInliner().Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(TupleSimplifier().Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* computation :
       module->MakeNonfusionComputations({execution_threads})) {
    std::vector<HloInstruction*> callers =
        call_graph->GetComputationCallers(computation);
    bool caller_is_single_host_instr =
        callers.size() == 1 &&
        host_offload_utils::ComputeTypeIsHost(callers.front());
    if (caller_is_single_host_instr) {
      // Skip offloading instructions inside of computations that are already
      // part of an offloaded program (while, conditional, host utility
      // function).
      RemoveComputeTypeFrontendAttribute(*computation);
      continue;
    }
    TF_RETURN_IF_ERROR(
        OffloadHostInstructions(*computation, options_).status());
  }

  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());

  if (module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }

  return true;
}

}  // namespace xla
