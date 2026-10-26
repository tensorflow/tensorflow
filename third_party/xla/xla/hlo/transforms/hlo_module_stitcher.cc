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

#include "xla/hlo/transforms/hlo_module_stitcher.h"

#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/stack_frames.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

// Recursively interns `src_id` (and its ancestor chain) from `src_frames` into
// `dst_frames`, returning the corresponding ID in `dst_frames`.
StackFrameId InternStackFrame(
    const StackFrames& src_frames, StackFrameId src_id, StackFrames& dst_frames,
    absl::flat_hash_map<StackFrameId, StackFrameId>& interned) {
  if (auto it = interned.find(src_id); it != interned.end()) {
    return it->second;
  }
  HloStackFrame frame = src_frames.GetStackFrame(src_id);
  if (frame.empty()) {
    return StackFrameId{};
  }
  frame.parent_frame_id =
      InternStackFrame(src_frames, frame.parent_frame_id, dst_frames, interned);
  StackFrameId dst_id = dst_frames.AddStackFrame(frame);
  interned[src_id] = dst_id;
  return dst_id;
}

}  // namespace

absl::StatusOr<bool> HloModuleStitcher::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Reset visited set on top-level entry to support pass reuse.
  if (visiting_modules_.empty()) {  // Top-level call.
    visited_modules_.clear();
  }

  if (visiting_modules_.contains(module)) {
    return InternalStrCat(
        "Circular dependency detected in submodule stitching: ",
        module->name());
  }

  if (visited_modules_.contains(module)) {
    return false;
  }

  visiting_modules_.insert(module);
  auto cleanup =
      absl::MakeCleanup([this, module]() { visiting_modules_.erase(module); });

  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (HloComputation* comp : computations) {
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : instructions) {
      if (inst->opcode() == HloOpcode::kCustomCall &&
          inst->custom_call_target() == kMultiModuleCustomCallTarget) {
        std::string sub_module_name = inst->raw_backend_config_string();
        auto it = optimized_modules_.find(sub_module_name);
        if (it == optimized_modules_.end()) {
          return NotFoundStrCat("Sub-module ", sub_module_name, " not found");
        }

        HloModule* sub_module = it->second;
        if (sub_module == nullptr) {
          return absl::InternalError("sub_module is null");
        }
        // Resolve all nested custom calls in the submodule first recursively.
        ABSL_RETURN_IF_ERROR(Run(sub_module).status());
        HloComputation* sub_entry = sub_module->entry_computation();

        if (inst->operand_count() != sub_entry->num_parameters()) {
          return InvalidArgumentStrCat(
              "Operand count mismatch: custom call has ", inst->operand_count(),
              " operands but sub-module expects ", sub_entry->num_parameters());
        }

        HloCloneContext context(module);
        HloComputation* cloned_sub_entry = context.FindComputation(sub_entry);
        if (cloned_sub_entry == nullptr) {
          cloned_sub_entry = module->DeepCloneComputation(sub_entry, &context);
        }
        // Intern the submodule's stack frames into the parent module and remap
        // cloned instructions' stack_frame_ids.
        const StackFrames& src_frames = sub_module->stack_frames();
        StackFrames& dst_frames = module->mutable_stack_frames();
        absl::flat_hash_map<StackFrameId, StackFrameId> frame_id_map;
        for (const HloComputation* sub_comp : sub_module->computations()) {
          HloComputation* cloned_comp = context.FindComputation(sub_comp);
          if (cloned_comp == nullptr) {
            continue;
          }
          for (HloInstruction* new_inst : cloned_comp->instructions()) {
            int src_id = new_inst->metadata().stack_frame_id();
            if (src_id != 0) {
              new_inst->mutable_metadata().set_stack_frame_id(
                  InternStackFrame(src_frames, StackFrameId{src_id}, dst_frames,
                                   frame_id_map)
                      .value);
            }
          }
        }

        std::vector<HloInstruction*> operands;
        operands.reserve(inst->operand_count());
        for (int i = 0; i < inst->operand_count(); ++i) {
          HloInstruction* operand = inst->mutable_operand(i);
          const Shape& expected_shape =
              cloned_sub_entry->parameter_instruction(i)->shape();
          if (!ShapeUtil::Equal(operand->shape(), expected_shape)) {
            if (!ShapeUtil::Compatible(operand->shape(), expected_shape)) {
              return InvalidArgumentStrCat(
                  "Incompatible operand shape at index ", i, ": expected ",
                  ShapeUtil::HumanString(expected_shape), ", got ",
                  ShapeUtil::HumanString(operand->shape()));
            }
            operand = comp->AddInstruction(HloInstruction::CreateUnary(
                expected_shape, HloOpcode::kCopy, operand));
          }
          operands.push_back(operand);
        }

        const Shape& result_shape =
            cloned_sub_entry->root_instruction()->shape();
        HloInstruction* call = comp->AddInstruction(HloInstruction::CreateCall(
            result_shape, operands, cloned_sub_entry));
        call->set_frontend_attributes(inst->frontend_attributes());
        call->set_metadata(inst->metadata());

        HloInstruction* replacement = call;
        if (!ShapeUtil::Equal(result_shape, inst->shape())) {
          if (!ShapeUtil::Compatible(result_shape, inst->shape())) {
            return InvalidArgumentStrCat("Incompatible result shape: expected ",
                                         ShapeUtil::HumanString(inst->shape()),
                                         ", got ",
                                         ShapeUtil::HumanString(result_shape));
          }
          replacement = comp->AddInstruction(HloInstruction::CreateUnary(
              inst->shape(), HloOpcode::kCopy, call));
        }

        ABSL_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(replacement));
        ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(inst));
        changed = true;
      }
    }
  }

  visited_modules_.insert(module);
  return changed;
}

}  // namespace xla
