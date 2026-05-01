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
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
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

std::string GetDevicePlacement(const HloInstruction* instr) {
  CHECK(instr->IsCustomCall(memory_annotations::kDevicePlacement))
      << "Input " << instr->name() << " must be a device placement annotation";
  CHECK(instr->has_frontend_attributes())
      << "Input " << instr->name() << " must have frontend attributes";
  const auto& frontend_attribute_map = instr->frontend_attributes().map();
  auto buffer_placement_it =
      frontend_attribute_map.find(kXlaBufferPlacementAttr);
  CHECK(buffer_placement_it != frontend_attribute_map.end())
      << "Input " << instr->name()
      << " must have a buffer placement frontend attribute";
  return buffer_placement_it->second;
}

absl::flat_hash_set<HloInstruction*> CollectAllowedDevicePlacementAnnotations(
    const HloComputation* computation) {
  // Collect a list of allowed annotations. We only expect annotations in one of
  // two locations in host computations currently:
  //  1. The ROOT instruction, if the computation returns a single value.
  //  2. The items feeding into the ROOT tuple instruction, if the computation
  //  returns a tuple.
  absl::flat_hash_set<HloInstruction*> allowed_device_placement_annotations;
  HloInstruction* root_instr = computation->root_instruction();
  if (root_instr->opcode() == HloOpcode::kTuple) {
    // Is a tuple
    for (int64_t i = 0; i < root_instr->operand_count(); ++i) {
      HloInstruction* operand = root_instr->mutable_operand(i);
      if (operand->IsCustomCall(memory_annotations::kDevicePlacement)) {
        allowed_device_placement_annotations.insert(operand);
      }
    }
  } else {
    // Is not a tuple
    if (root_instr->IsCustomCall(memory_annotations::kDevicePlacement)) {
      allowed_device_placement_annotations.insert(root_instr);
    }
  }
  return allowed_device_placement_annotations;
}

absl::StatusOr<std::vector<HloInstruction*>>
CheckRemainingDevicePlacementAnnotations(
    const HloComputation* computation,
    const absl::flat_hash_set<HloInstruction*>&
        allowed_device_placement_annotations) {
  // Look for annotations which are not in the allowed set. If any annotation is
  // redundant, return it in a list so that the caller of this function can
  // remove it. Any other annotation is an error.
  std::vector<HloInstruction*> redundant_annotations;
  for (HloInstruction* instr : computation->instructions()) {
    if (instr->IsCustomCall(memory_annotations::kDevicePlacement)) {
      if (allowed_device_placement_annotations.contains(instr)) {
        continue;
      }
      const std::string device_placement = GetDevicePlacement(instr);
      if (device_placement == memory_annotations::kMemoryTargetPinnedHost ||
          device_placement == memory_annotations::kMemoryTargetUnpinnedHost) {
        // An annotation in host computation annotating the buffer to be on the
        // host is redundant.
        redundant_annotations.push_back(instr);
      } else {
        // An annotation in host computation annotating the buffer to be
        // somewhere other than the host is not allowed.
        return absl::InvalidArgumentError(
            absl::StrFormat("Host computation %s contains a device placement "
                            "annotation %s that is not allowed.",
                            computation->name(), instr->ToString()));
      }
    }
  }
  return redundant_annotations;
}

// Returns true if any redundant annotations were removed.
absl::StatusOr<bool> CleanUpHostComputationDevicePlacementAnnotations(
    const HloComputation* computation) {
  const absl::flat_hash_set<HloInstruction*>
      allowed_device_placement_annotations =
          CollectAllowedDevicePlacementAnnotations(computation);
  TF_ASSIGN_OR_RETURN(
      const std::vector<HloInstruction*> redundant_device_placement_annotations,
      CheckRemainingDevicePlacementAnnotations(
          computation, allowed_device_placement_annotations));

  // Remove redundant annotations
  for (HloInstruction* redundant_annotation :
       redundant_device_placement_annotations) {
    VLOG(1) << "Removing redundant annotation: "
            << redundant_annotation->ToString();
    CHECK_EQ(redundant_annotation->operand_count(), 1)
        << "A device placement annotation must have exactly one operand.";
    for (HloInstruction* user : redundant_annotation->users()) {
      for (int64_t operand_index :
           user->operand_indices(redundant_annotation)) {
        TF_RETURN_IF_ERROR(user->ReplaceOperandWith(
            operand_index, redundant_annotation->mutable_operand(0)));
      }
    }
    TF_RETURN_IF_ERROR(redundant_annotation->parent()->RemoveInstruction(
        redundant_annotation));
  }

  return !redundant_device_placement_annotations.empty();
}

bool DevicePlacementMemorySpaceIsSame(const HloInstruction* a,
                                      const HloInstruction* b) {
  CHECK(a->IsCustomCall(memory_annotations::kDevicePlacement))
      << "Input a: " << a->name() << " must be a device placement annotation";
  CHECK(b->IsCustomCall(memory_annotations::kDevicePlacement))
      << "Input b: " << b->name() << " must be a device placement annotation";
  return GetDevicePlacement(a) == GetDevicePlacement(b);
}

absl::Status CloneAnnotationToDestination(
    HloComputation* destination_computation,
    HloInstruction* destination_computation_caller_instruction,
    const HloInstruction* original_annotation,
    HloInstruction* destination_instruction) {
  HloInstruction* moved_annotation = destination_computation->AddInstruction(
      original_annotation->CloneWithNewOperands(original_annotation->shape(),
                                                {destination_instruction},
                                                "move_to_caller"));

  bool used_new_annotation = false;
  for (HloInstruction* destination_user : destination_instruction->users()) {
    if (destination_user == moved_annotation) {
      // Do not replace the annotation with itself.
      continue;
    }
    if (destination_user->IsCustomCall(memory_annotations::kDevicePlacement)) {
      // The destination already has an annotation.
      if (!DevicePlacementMemorySpaceIsSame(original_annotation,
                                            destination_user)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Found conflicting host computation output memory "
                            "space. Call %s wants output memory space %s but "
                            "call %s wants output memory space %s",
                            original_annotation->operand(0)->name(),
                            GetDevicePlacement(original_annotation),
                            destination_computation_caller_instruction->name(),
                            GetDevicePlacement(destination_user)));
      }
      // Annotation already exists, nothing to do.
      continue;
    }
    for (int64_t operand_index :
         destination_user->operand_indices(destination_instruction)) {
      TF_RETURN_IF_ERROR(destination_user->ReplaceOperandWith(
          operand_index, moved_annotation));
    }
    used_new_annotation = true;
  }

  // All the places where this annotation would be placed already have this
  // exact annotation.
  if (!used_new_annotation) {
    TF_RETURN_IF_ERROR(
        destination_computation->RemoveInstruction(moved_annotation));
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> MoveAnnotationsToCallerTuple(
    HloComputation* host_computation) {
  bool changed = false;
  for (int64_t operand_index = 0;
       operand_index < host_computation->root_instruction()->operand_count();
       ++operand_index) {
    HloInstruction* root_operand =
        host_computation->root_instruction()->mutable_operand(operand_index);
    if (!root_operand->IsCustomCall(memory_annotations::kDevicePlacement)) {
      // Instruction is not a device placement annotation; nothing to do.
      continue;
    }
    // Root is a device placement annotation.
    CHECK_EQ(root_operand->operand_count(), 1)
        << "A device placement annotation must have exactly one operand.";

    // Clone the annotation to each of the callers.
    for (HloInstruction* caller_instruction :
         host_computation->caller_instructions()) {
      HloComputation* caller_computation = caller_instruction->parent();
      for (HloInstruction* caller_user_gte : caller_instruction->users()) {
        if (caller_user_gte->opcode() != HloOpcode::kGetTupleElement) {
          return absl::UnimplementedError(
              "When moving device placement annotations out of a host "
              "computation, the tuple is used by something other than a "
              "get-tuple-element. This is currently not supported.");
        }
        if (caller_user_gte->tuple_index() != operand_index) {
          // This get-tuple-element is getting a different index than the one we
          // are currently looking at.
          continue;
        }
        TF_RETURN_IF_ERROR(
            CloneAnnotationToDestination(caller_computation, caller_instruction,
                                         root_operand, caller_user_gte));
        changed = true;
      }
    }

    TF_RETURN_IF_ERROR(host_computation->root_instruction()->ReplaceOperandWith(
        operand_index, root_operand->mutable_operand(0)));
    TF_RETURN_IF_ERROR(host_computation->RemoveInstruction(root_operand));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> MoveAnnotationToCallerNonTuple(
    HloComputation* host_computation) {
  HloInstruction* root_instr = host_computation->root_instruction();
  if (!root_instr->IsCustomCall(memory_annotations::kDevicePlacement)) {
    // Root is not a device placement annotation; nothing to do.
    return false;
  }
  // Root is a device placement annotation.
  CHECK_EQ(root_instr->operand_count(), 1)
      << "A device placement annotation must have exactly one operand.";

  // Clone the annotation to each of the callers.
  for (HloInstruction* caller_instruction :
       host_computation->caller_instructions()) {
    HloComputation* caller_computation = caller_instruction->parent();
    TF_RETURN_IF_ERROR(
        CloneAnnotationToDestination(caller_computation, caller_instruction,
                                     root_instr, caller_instruction));
  }

  // Remove the annotation from inside this computation.
  host_computation->set_root_instruction(root_instr->mutable_operand(0));
  TF_RETURN_IF_ERROR(host_computation->RemoveInstruction(root_instr));
  return true;
}

// Move host device placement annotations out of this computation to the calling
// computation.
absl::StatusOr<bool> MoveAnnotationsToCaller(HloComputation* computation) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(
      bool cleaned_up,
      CleanUpHostComputationDevicePlacementAnnotations(computation));
  changed = changed || cleaned_up;
  // All annotations at this point are valid.
  if (computation->root_instruction()->opcode() == HloOpcode::kTuple) {
    // When the computation returns a tuple, the annotation is on the operands
    // of the root tuple.
    TF_ASSIGN_OR_RETURN(bool moved, MoveAnnotationsToCallerTuple(computation));
    changed = changed || moved;
  } else {
    // When the computation returns a single value, the annotation is the root
    // instruction.
    TF_ASSIGN_OR_RETURN(bool moved,
                        MoveAnnotationToCallerNonTuple(computation));
    changed = changed || moved;
  }
  return changed;
}

absl::StatusOr<bool> RemoveDevicePlacementAnnotationsFromHostComputations(
    HloModule* module) {
  // The only time we currently find device placement annotations in host
  // computations are when the host computation calls another host computation
  // and that called host computation has an output memory space annotated. That
  // output memory space annotation is usually on the users of the host call (or
  // users of the get-tuple-elements if the call returns a tuple).
  //
  // Visit host computations in post-order. We will push annotations out of host
  // computations into their callers.
  std::vector<HloComputation*> host_computations;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // Check if this computation is a host computation.
    for (const HloInstruction* caller_instruction :
         computation->caller_instructions()) {
      if (caller_instruction->has_frontend_attributes()) {
        FrontendAttributes frontend_attributes =
            caller_instruction->frontend_attributes();
        if (frontend_attributes.map().contains(kXlaComputeTypeAttr) &&
            frontend_attributes.map().at(kXlaComputeTypeAttr) ==
                kXlaComputeTypeHost) {
          // The computation is a host computation.
          host_computations.push_back(computation);
          break;
        }
      }
    }
  }

  bool changed = false;
  for (HloComputation* computation : host_computations) {
    TF_ASSIGN_OR_RETURN(bool moved, MoveAnnotationsToCaller(computation));
    changed = changed || moved;
  }
  return changed;
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
  return absl::down_cast<HloCallInstruction*>(new_call);
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
  return absl::down_cast<HloCallInstruction*>(new_call);
}

absl::StatusOr<bool> HloHostDeviceTypeCallWrapper::RunImpl(
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

  // Before any other passes run, move device placement annotations out of host
  // computations.
  TF_ASSIGN_OR_RETURN(
      bool modified,
      RemoveDevicePlacementAnnotationsFromHostComputations(module));
  // At this point, this pass will always modify the module. The return value of
  // this function, which indicates whether the module was modified, is not
  // useful.
  (void)modified;

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
