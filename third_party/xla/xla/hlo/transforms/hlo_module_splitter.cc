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

#include "xla/hlo/transforms/hlo_module_splitter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_module_stitcher.h"
#include "xla/service/name_uniquer.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

absl::StatusOr<HloInstruction*> CreateBoundaryCopy(HloComputation* comp,
                                                   HloInstruction* inst) {
  if (inst->shape().IsToken()) {
    return inst;
  }
  if (!inst->shape().IsTuple()) {
    return comp->AddInstruction(
        HloInstruction::CreateUnary(inst->shape(), HloOpcode::kCopy, inst));
  }
  ShapeTree<bool> indices_to_copy(inst->shape(), true);
  ShapeUtil::ForEachSubshape(
      inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          *indices_to_copy.mutable_element(index) = false;
        }
      });
  return comp->DeepCopyInstruction(inst, &indices_to_copy);
}

}  // namespace

bool HloModuleSplitter::ShouldSplitCall(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kCall) {
    return false;
  }

  if (inst->GetModule() != nullptr &&
      inst->GetModule()->config().use_shardy_partitioner()) {
    const HloComputation* callee = inst->to_apply();
    if (callee != nullptr) {
      absl::string_view callee_name = callee->name();
      if (absl::StrContains(callee_name, "shmap_body") ||
          absl::StrContains(callee_name,
                            sdy::kManualComputationFuncName.data())) {
        return false;
      }
    }
  }

  const auto& map = inst->frontend_attributes().map();
  auto inlineable_it = map.find("inlineable");
  if (inlineable_it != map.end() && inlineable_it->second == "xla_late") {
    return true;
  }
  return map.contains("compilation_unit");
}

absl::StatusOr<bool> HloModuleSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  submodules_.clear();
  bool changed = false;
  absl::flat_hash_map<HloComputation*, HloModule*> extracted_modules;
  NameUniquer name_uniquer(".");

  // We use post-order to process callees before callers.
  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (HloComputation* comp : computations) {
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : instructions) {
      if (ShouldSplitCall(inst)) {
        HloComputation* callee = inst->to_apply();

        auto [map_it, inserted] =
            extracted_modules.try_emplace(callee, nullptr);
        if (inserted) {
          auto it = inst->frontend_attributes().map().find("compilation_unit");
          std::string base_name =
              (it != inst->frontend_attributes().map().end() &&
               !it->second.empty())
                  ? it->second
                  : std::string(callee->name());
          std::string name = name_uniquer.GetUniqueName(base_name);

          auto sub_module = std::make_unique<HloModule>(name, module->config());
          HloCloneContext context(sub_module.get());
          HloComputation* cloned_callee =
              sub_module->DeepCloneComputation(callee, &context);
          sub_module->ReplaceEntryComputation(cloned_callee);
          map_it->second = sub_module.get();
          submodules_.push_back(std::move(sub_module));
        }

        // Replace call with custom-call.
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : inst->operands()) {
          ASSIGN_OR_RETURN(HloInstruction * copy,
                           CreateBoundaryCopy(comp, operand));
          operands.push_back(copy);
        }

        auto* custom_call = Cast<HloCustomCallInstruction>(
            comp->AddInstruction(HloInstruction::CreateCustomCall(
                inst->shape(), operands, kMultiModuleCustomCallTarget,
                /*opaque=*/"",
                CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)));
        custom_call->set_raw_backend_config_string(map_it->second->name());

        if (callee->HasSideEffect()) {
          custom_call->set_custom_call_has_side_effect(true);
        }
        custom_call->set_frontend_attributes(inst->frontend_attributes());

        RETURN_IF_ERROR(comp->ReplaceInstruction(inst, custom_call));
        changed = true;
      }
      }
    }

  return changed;
}

}  // namespace xla
