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
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<int64_t> ExtractAnnotation(
    const ::google::protobuf::Map<std::string, std::string>& attrs,
    absl::string_view instr_name) {
  int64_t annotation_id;
  if (!absl::SimpleAtoi(attrs.at(kXlaSchedulingGroupIdAttr), &annotation_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction has a non-integer scheduling annotation, inst: ",
        instr_name, ", annotation: ", attrs.at(kXlaSchedulingGroupIdAttr)));
  }
  if (annotation_id < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction has a negative scheduling annotation, inst: ", instr_name,
        ", annotation: ", attrs.at(kXlaSchedulingGroupIdAttr)));
  }
  return annotation_id;
}

void DropSchedulingAnnotation(HloInstruction* instr) {
  VLOG(2) << "Dropping annotation from " << instr->name();
  FrontendAttributes frontend_attributes = instr->frontend_attributes();
  frontend_attributes.mutable_map()->erase("_scheduling_group_id");
  instr->set_frontend_attributes(frontend_attributes);
}

bool IsSupportedAsyncOp(HloInstruction* instr) {
  return HloPredicateIsOp<
      HloOpcode::kAllGatherDone, HloOpcode::kAllGatherStart,
      HloOpcode::kAllReduceDone, HloOpcode::kAllReduceStart,
      HloOpcode::kCollectivePermuteDone, HloOpcode::kCollectivePermuteStart,
      HloOpcode::kAsyncDone, HloOpcode::kAsyncStart, HloOpcode::kSendDone,
      HloOpcode::kSend, HloOpcode::kRecvDone, HloOpcode::kRecv>(instr);
}

bool LegalizeSchedulingAnnotations::KeepSchedulingAnnotation(
    HloInstruction* instr) {
  return IsSupportedAsyncOp(instr) || config_.keep_sync_annotation(instr);
}

absl::StatusOr<bool> LegalizeSchedulingAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<HloInstruction*, int64_t> annotation;
  absl::flat_hash_map<int64_t, HloComputation*> annotation_to_computation;
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>>
      annotation_to_instructions;
  // Filter the annotated ops (using config) to keep the annotations only in the
  // desired sync ops. Annotations in all async ops are kept.
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (!instr->frontend_attributes().map().contains(
              "_scheduling_group_id") ||
          KeepSchedulingAnnotation(instr)) {
        continue;
      }
      DropSchedulingAnnotation(instr);
    }
  }
  // Find the annotated instructions and save relevant information.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      const auto& attrs = instr->frontend_attributes().map();
      if (!attrs.contains(kXlaSchedulingGroupIdAttr)) {
        continue;
      }
      VLOG(1) << "Annotated instruction: " << instr->name() << " "
              << attrs.at(kXlaSchedulingGroupIdAttr);
      TF_ASSIGN_OR_RETURN(int64_t annotation_id,
                          ExtractAnnotation(attrs, instr->name()));
      if (annotation_to_computation.contains(annotation_id) &&
          annotation_to_computation[annotation_id] != computation) {
        return absl::UnimplementedError(absl::StrCat(
            "Support for cross computation annotations doesn't exist yet. Saw "
            "a scheduling annotation that is already used in another "
            "computation, inst: ",
            instr->name(), ", annotation: ", annotation_id,
            ", computation: ", annotation_to_computation[annotation_id]->name(),
            ", computation: ", computation->name()));
      }
      annotation[instr] = annotation_id;
      annotation_to_instructions[annotation_id].push_back(instr);
      annotation_to_computation[annotation_id] = computation;
    }
  }
  // Move the annotation from inside fusion computation to the caller
  // instruction if the caller doesn't have an annotation. Return an error if
  // there are some fused instructions with different annotations.
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation() ||
        !config_.keep_sync_annotation(computation->FusionInstruction()) ||
        annotation.contains(computation->FusionInstruction())) {
      continue;
    }
    int64_t seen_annotation = -1;
    for (HloInstruction* instr : computation->instructions()) {
      const auto& attrs = instr->frontend_attributes().map();
      if (!attrs.contains(kXlaSchedulingGroupIdAttr)) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(int64_t annotation_id,
                          ExtractAnnotation(attrs, instr->name()));
      if (seen_annotation == -1) {
        seen_annotation = annotation_id;
        continue;
      }
      if (seen_annotation != annotation_id) {
        return absl::InternalError(absl::StrCat(
            "Found a fusion with multiple annotations in the fused "
            "computation. fusion: ",
            computation->FusionInstruction()->name(),
            ", annotations: ", seen_annotation, " and ", annotation_id));
      }
    }
    // No fused instructions are annotated, nothing to do.
    if (seen_annotation == -1) {
      continue;
    }
    FrontendAttributes frontend_attributes =
        computation->FusionInstruction()->frontend_attributes();
    frontend_attributes.mutable_map()->insert(
        {kXlaSchedulingGroupIdAttr, std::to_string(seen_annotation)});
    computation->FusionInstruction()->set_frontend_attributes(
        frontend_attributes);
  }
  if (annotation_to_computation.empty()) {
    return false;
  }
  absl::flat_hash_map<HloInstruction*, HloInstruction*> parent;
  for (const auto& [id, annotated_instructions] : annotation_to_instructions) {
    // First find the frontier nodes that are not annotated with id but use an
    // annotated instruction with id.
    std::vector<HloInstruction*> stack;
    absl::flat_hash_set<HloInstruction*> visited;
    for (HloInstruction* instr : annotated_instructions) {
      CHECK(annotation.contains(instr));
      CHECK_EQ(annotation[instr], id);
      if (HloPredicateIsOp<HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                           HloOpcode::kCollectivePermuteDone,
                           HloOpcode::kAsyncDone>(instr) &&
          (!annotation.contains(instr->operand(0)) ||
           annotation[instr->mutable_operand(0)] != id)) {
        return absl::InternalError(absl::StrCat(
            "Done instruction's operand is not annotated with the same id: ",
            instr->operand(0)->name(), ", annotation: ", id));
      }
      for (HloInstruction* user : instr->users()) {
        if (!visited.contains(user) &&
            (!annotation.contains(user) || annotation[user] != id)) {
          stack.push_back(user);
          parent[user] = instr;
          visited.insert(user);
          VLOG(2) << "Annotation group: " << id
                  << ", frontier using a root: " << user->name();
        }
      }
    }
    VLOG(2) << "Annotation group: " << id << ", frontier has " << stack.size()
            << " instructions";
    // Traverse the HLO graph starting from the frontier instructions and move
    // to the users. If there are gaps in the annotation, the traversal will hit
    // an instruction that is annotated with the same id.
    while (!stack.empty()) {
      HloInstruction* instr = stack.back();
      stack.pop_back();
      for (HloInstruction* user : instr->users()) {
        if (annotation.contains(user) && annotation[user] == id) {
          LOG(INFO) << "PATH: " << user->name();
          HloInstruction* current = instr;
          LOG(INFO) << "PATH: " << current->name();
          while (parent.contains(current)) {
            current = parent[current];
            LOG(INFO) << "PATH: " << current->name();
          }
          return absl::UnimplementedError(
              absl::StrCat("Support for annotation groups with gaps doesn't "
                           "exist yet, annotation: ",
                           id, ", instr: ", user->name(),
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
  return true;
}
}  // namespace xla
