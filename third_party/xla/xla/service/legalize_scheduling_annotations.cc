/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/legalize_scheduling_annotations.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Given a group of annotated instructions (sources), find all reachable
// instructions from them in the same computation.
absl::flat_hash_set<HloInstruction*> PropagateAnnotationFromSources(
    const std::vector<HloInstruction*>& sources,
    const HloComputation* computation) {
  absl::flat_hash_set<HloInstruction*> to_annotate;
  auto reachability = HloReachabilityMap::Build(computation);
  // worklist contains instructions that can reach any source instruction.
  std::queue<HloInstruction*> work_queue;
  absl::flat_hash_set<HloInstruction*> visited;
  absl::flat_hash_set<HloInstruction*> sources_set(sources.begin(),
                                                   sources.end());
  for (HloInstruction* instr : sources) {
    for (HloInstruction* another_instr : sources) {
      if (instr == another_instr) {
        continue;
      }
      if (reachability->IsReachable(instr, another_instr)) {
        work_queue.push(instr);
        visited.insert(instr);
        break;
      }
    }
  }

  while (!work_queue.empty()) {
    auto* instr = work_queue.front();
    work_queue.pop();
    if (!sources_set.contains(instr)) {
      to_annotate.insert(instr);
    }
    for (const PtrVec<HloInstruction*>& users :
         {instr->users(), instr->control_successors()}) {
      for (HloInstruction* user : users) {
        if (visited.contains(user)) {
          continue;
        }
        // Add user to work queue if it reaches any source instruction.
        for (HloInstruction* source : sources) {
          if (reachability->IsReachable(user, source)) {
            work_queue.push(user);
            visited.insert(user);
            break;
          }
        }
      }
    }
  }
  return to_annotate;
}

// Attach the annotation ID to the given instructions. Returns error if any of
// the instructions already has an annotation.
absl::Status AttachAnnotation(
    Annotation annotation,
    const absl::flat_hash_set<HloInstruction*>& instructions) {
  for (HloInstruction* instr : instructions) {
    TF_ASSIGN_OR_RETURN(std::optional<Annotation> instr_annotation,
                        GetSchedulingAnnotation(instr));
    if (instr_annotation) {
      return absl::InternalError("Trying to propagate scheduling annotation " +
                                 annotation.ToString() + " to " +
                                 std::string(instr->name()) +
                                 " but it has an existing annotation: " +
                                 instr_annotation->ToString());
    }
    LOG(INFO) << "Propagating annotation " << annotation.ToString() << " to "
              << instr->name();
    TF_RETURN_IF_ERROR(SetSchedulingAnnotation(instr, annotation));
  }
  return absl::OkStatus();
}

bool IsSupportedAsyncOp(HloInstruction* instr) {
  return HloPredicateIsOp<
      HloOpcode::kAllGatherDone, HloOpcode::kAllGatherStart,
      HloOpcode::kAllReduceDone, HloOpcode::kAllReduceStart,
      HloOpcode::kCollectivePermuteDone, HloOpcode::kCollectivePermuteStart,
      HloOpcode::kAsyncDone, HloOpcode::kAsyncStart, HloOpcode::kSendDone,
      HloOpcode::kSend, HloOpcode::kRecvDone, HloOpcode::kRecv>(instr);
}

absl::Status CheckStartDoneAnnotationConsistency(
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation) {
  for (const auto& [annotation, comp_inst_vector] : annotation_to_instruction) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      for (HloInstruction* instr : annotated_instructions) {
        CHECK(instruction_to_annotation.contains(instr));
        CHECK(instruction_to_annotation.at(instr) == annotation);
        if (HloPredicateIsOp<
                HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                HloOpcode::kCollectivePermuteDone, HloOpcode::kAsyncDone>(
                instr) &&
            (!instruction_to_annotation.contains(instr->operand(0)) ||
             instruction_to_annotation.at(instr->mutable_operand(0)) !=
                 annotation)) {
          return absl::InternalError(absl::StrCat(
              "Done instruction's operand is not annotated with the same id: ",
              instr->operand(0)->name(),
              ", annotation: ", annotation.ToString()));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CheckGapBetweenAnnotatedInstructions(
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation) {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> parent;
  for (const auto& [annotation, comp_inst_vector] : annotation_to_instruction) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      // First find the frontier nodes that are not annotated with id but use an
      // annotated instruction with id.
      std::vector<HloInstruction*> stack;
      absl::flat_hash_set<HloInstruction*> visited;
      for (HloInstruction* instr : annotated_instructions) {
        CHECK(instruction_to_annotation.contains(instr));
        CHECK(instruction_to_annotation.at(instr) == annotation);
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            if (!visited.contains(user) &&
                (!instruction_to_annotation.contains(user) ||
                 instruction_to_annotation.at(user) != annotation)) {
              stack.push_back(user);
              parent[user] = instr;
              visited.insert(user);
              VLOG(2) << "Annotation : " << annotation.ToString()
                      << ", frontier using a root: " << user->name();
            }
          }
        }
      }
      VLOG(2) << "Annotation : " << annotation.ToString() << ", frontier has "
              << stack.size() << " instructions";
      // Traverse the HLO graph starting from the frontier instructions and move
      // to the users. If there are gaps in the annotation, the traversal will
      // hit an instruction that is annotated with the same id.
      while (!stack.empty()) {
        HloInstruction* instr = stack.back();
        stack.pop_back();
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            const auto log_inst = [&](HloInstruction* inst) {
              LOG(INFO) << "PATH: " << inst->name() << ", annotation: "
                        << GetSchedulingAnnotation(inst)
                               ->value_or(Annotation{})
                               .ToString();
            };

            if (instruction_to_annotation.contains(user) &&
                instruction_to_annotation.at(user) == annotation) {
              log_inst(user);
              HloInstruction* current = instr;
              log_inst(current);
              while (parent.contains(current)) {
                current = parent[current];
                log_inst(current);
              }
              return absl::UnimplementedError(absl::StrCat(
                  "Support for annotation groups with gaps doesn't "
                  "exist yet, annotation: ",
                  annotation.ToString(), ", instr: ", user->name(),
                  " has the same annotation in its operand tree but "
                  "has gaps on the way from that operand to itself."));
            }
            if (visited.contains(user)) {
              continue;
            }
            stack.push_back(user);
            parent[user] = instr;
            visited.insert(user);
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> HaulAnnotationToFusionInstruction(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation,
    std::function<bool(HloInstruction*)> keep_sync_annotation) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation() ||
        !keep_sync_annotation(computation->FusionInstruction()) ||
        instruction_to_annotation.contains(computation->FusionInstruction())) {
      continue;
    }
    changed = true;
    std::optional<Annotation> seen_annotation;
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(instr));
      if (!annotation) {
        continue;
      }
      if (!seen_annotation) {
        seen_annotation = annotation;
        continue;
      }
      if (seen_annotation != annotation) {
        return absl::InternalError(absl::StrCat(
            "Found a fusion with multiple annotations in the fused "
            "computation. fusion: ",
            computation->FusionInstruction()->name(), ", annotations: ",
            seen_annotation->ToString(), " and ", annotation->ToString()));
      }
    }
    // No fused instructions are annotated, nothing to do.
    if (!seen_annotation) {
      continue;
    }
    TF_RETURN_IF_ERROR(SetSchedulingAnnotation(computation->FusionInstruction(),
                                               seen_annotation->ToString()));
  }
  return changed;
}

absl::StatusOr<bool> RemoveLoopIterationAnnotation(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(bool removed,
                          RemoveSchedulingAnnotationIterationId(instr));
      changed |= removed;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> LegalizeSchedulingAnnotations::PropagateAnnotations(
    const HloComputation* computation,
    const absl::btree_map<Annotation, std::vector<HloInstruction*>>&
        annotation_to_instruction) {
  bool changed = false;
  for (auto& [annotation, sources] : annotation_to_instruction) {
    absl::flat_hash_set<HloInstruction*> to_annotate =
        PropagateAnnotationFromSources(sources, computation);
    changed |= (!to_annotate.empty());
    auto status = AttachAnnotation(annotation, to_annotate);
    if (!status.ok()) {
      return status;
    }
  }
  return changed;
}

bool LegalizeSchedulingAnnotations::KeepSchedulingAnnotation(
    HloInstruction* instr) {
  const auto& attrs = instr->frontend_attributes().map();
  if (attrs.contains(kXlaSchedulingGroupIdAttr) &&
      attrs.at(kXlaSchedulingGroupIdAttr) == kXlaNoOpSchedulingGroup) {
    return false;
  }

  return IsSupportedAsyncOp(instr) || config_.keep_sync_annotation(instr);
}

bool LegalizeSchedulingAnnotations::RemoveTrivialGroups(
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction) {
  absl::flat_hash_map<
      AnnotationGroupId,
      absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>
      group_id_to_instruction;
  for (const auto& [annotation, comp_inst_vector] : annotation_to_instruction) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      for (const auto& annotated_instruction : annotated_instructions) {
        if (annotation.group_id.has_value()) {
          group_id_to_instruction[annotation.group_id.value()][comp].push_back(
              annotated_instruction);
        }
      }
    }
  }

  bool changed = false;
  for (const auto& [group_id, annotated_instructions] :
       group_id_to_instruction) {
    for (const auto& [comp, annotated_instructions] : annotated_instructions) {
      if (annotated_instructions.size() == 1 &&
          !config_.keep_trivial_sync_annotation(annotated_instructions[0])) {
        // Remove annotations from synchronous operations (control flow, TC
        // custom calls) since they won't do anything and will just get in the
        // way of scheduling.
        VLOG(2) << "Removing trivial group: " << group_id
                << " from instruction: " << annotated_instructions[0]->name()
                << " in computation: " << comp->name();
        changed |= RemoveSchedulingAnnotation(annotated_instructions[0]);
      } else {
        VLOG(3) << "Retaining nontrivial group: " << group_id;
      }
    }
  }

  return changed;
}

absl::Status LegalizeSchedulingAnnotations::Verify(HloModule* module) {
  VLOG(1) << "Verifying scheduling annotations for module: " << module->name();
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    auto reachability_map = HloReachabilityMap::Build(computation);
    absl::flat_hash_map<Annotation, std::vector<HloInstruction*>> grp_map;
    for (HloInstruction* instr : computation->instructions()) {
      if (HasSchedulingAnnotation(instr)) {
        auto scheduling_annotation_or = GetSchedulingAnnotation(instr);
        if (!scheduling_annotation_or.ok()) {
          continue;
        }
        std::optional<Annotation> scheduling_annotation =
            scheduling_annotation_or.value();
        if (scheduling_annotation.has_value()) {
          grp_map[scheduling_annotation.value()].push_back(instr);
        }
      }
    }
    // Check reachability for each group.
    for (auto& [id_a, instrs_a] : grp_map) {
      for (auto& [id_b, instrs_b] : grp_map) {
        if (id_a == id_b) {
          continue;
        }
        bool b_is_reachable_from_a = false;
        bool a_is_reachable_from_b = false;
        std::pair<HloInstruction*, HloInstruction*> a_b_pair;
        std::pair<HloInstruction*, HloInstruction*> b_a_pair;
        for (int64_t i = 0; i < instrs_a.size(); ++i) {
          for (int64_t j = 0; j < instrs_b.size(); ++j) {
            auto* a = instrs_a[i];
            auto* b = instrs_b[j];
            if (reachability_map->IsReachable(a, b)) {
              b_is_reachable_from_a = true;
              a_b_pair = std::make_pair(a, b);
            }
            if (reachability_map->IsReachable(b, a)) {
              a_is_reachable_from_b = true;
              b_a_pair = std::make_pair(b, a);
            }
          }
          if (a_is_reachable_from_b && b_is_reachable_from_a) {
            VLOG(1) << "a_b_pair: " << a_b_pair.first->name() << " "
                    << a_b_pair.second->name();
            VLOG(1) << "b_a_pair: " << b_a_pair.first->name() << " "
                    << b_a_pair.second->name();
            return absl::InternalError(
                absl::StrCat("ERROR: Detected scheduling group annotation "
                             "cycle between scheduling_group_id: ",
                             id_a.ToString(),
                             " and scheduling_group_id: ", id_b.ToString()));
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> LegalizeSchedulingAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Run verification if requested.
  if (config_.run_verification) {
    TF_RETURN_IF_ERROR(Verify(module));
  }

  bool changed = false;
  // Remove loop iteration annotation if requested.
  if (config_.remove_loop_iteration_annotation_only) {
    TF_ASSIGN_OR_RETURN(bool removed, RemoveLoopIterationAnnotation(module));
    changed |= removed;
    return changed;
  }

  absl::flat_hash_map<HloInstruction*, Annotation> instruction_to_annotation;
  absl::flat_hash_map<
      Annotation,
      absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>
      annotation_to_instruction;
  // Filter the annotated ops (using config) to keep the annotations only in the
  // desired sync ops. Annotations in all async ops are kept.
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (HasSchedulingAnnotation(instr) && !KeepSchedulingAnnotation(instr)) {
        changed |= RemoveSchedulingAnnotation(instr);
      }
    }
  }

  // Find the annotated instructions and save relevant information.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(instr));
      if (!annotation) {
        continue;
      }
      instruction_to_annotation[instr] = *annotation;
      annotation_to_instruction[*annotation][computation].push_back(instr);
    }
  }

  // Move the annotation from inside fusion computation to the caller
  // instruction if the caller doesn't have an annotation. Return an error if
  // there are some fused instructions with different annotations.
  TF_ASSIGN_OR_RETURN(
      bool haul_annotation_to_top_level,
      HaulAnnotationToFusionInstruction(
          module, execution_threads, annotation_to_instruction,
          instruction_to_annotation, config_.keep_sync_annotation));
  changed |= haul_annotation_to_top_level;

  if (annotation_to_instruction.empty()) {
    return changed;
  }

  if (config_.check_start_done_annotation_consistency) {
    auto status = CheckStartDoneAnnotationConsistency(
        annotation_to_instruction, instruction_to_annotation);
    if (!status.ok()) {
      return status;
    }
  }

  changed |= RemoveTrivialGroups(annotation_to_instruction);

  // Either propagate the annotation to fill the gaps between instructions with
  // the same annotation ID or check (and return error) if there are gaps.
  if (config_.propagate_annotation) {
    // Propagate the annotation to fill the gaps between instructions with the
    // same annotation ID.
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      absl::btree_map<Annotation, std::vector<HloInstruction*>>
          per_computation_annotation_to_instruction;
      for (const auto& [annotation, comp_inst_vector] :
           annotation_to_instruction) {
        if (comp_inst_vector.contains(computation)) {
          per_computation_annotation_to_instruction[annotation] =
              comp_inst_vector.at(computation);
        }
      }
      if (per_computation_annotation_to_instruction.empty()) {
        continue;
      }
      auto result = PropagateAnnotations(
          computation, per_computation_annotation_to_instruction);
      if (!result.ok()) {
        return result.status();
      }
      changed |= result.value();
    }
  } else {
    auto result = CheckGapBetweenAnnotatedInstructions(
        annotation_to_instruction, instruction_to_annotation);
    if (!result.ok()) {
      return result;
    }
  }

  return changed;
}
}  // namespace xla
