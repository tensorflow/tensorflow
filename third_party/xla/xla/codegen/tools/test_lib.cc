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

#include "xla/codegen/tools/test_lib.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_module_loader.h"

namespace xla {

static void FlattenIntoHelper(std::vector<HloInstruction*>& elements,
                              HloComputation::Builder& builder,
                              HloInstruction* value) {
  const auto& result_shape = value->shape();
  if (result_shape.IsTuple()) {
    for (int64_t i = 0; i < result_shape.tuple_shapes().size(); ++i) {
      FlattenIntoHelper(
          elements, builder,
          builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              result_shape.tuple_shapes()[i], value, i)));
    }
  } else {
    elements.push_back(value);
  }
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadTestModule(
    absl::string_view filename) {
  auto module = *xla::LoadModuleFromFile(std::string(filename));
  int num_fusions = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instruction) {
        return instruction->opcode() == xla::HloOpcode::kFusion;
      });
  TF_RET_CHECK(num_fusions <= 1) << "HLO must contain at most one fusion";

  if (num_fusions == 0) {
    // Generate a fusion from the entry computation.
    HloComputation::Builder builder("generated_main");
    std::vector<HloInstruction*> params;
    for (const auto* param :
         module->entry_computation()->parameter_instructions()) {
      params.push_back(*builder.AddParameter(param->Clone(/*suffix=*/"")));
    }
    auto* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
        module->entry_computation()->root_instruction()->shape(),
        HloInstruction::FusionKind::kLoop /* irrelevant */, params,
        module->entry_computation()));
    auto result_shape = fusion->shape();
    if (result_shape.IsTuple()) {
      bool needs_flattening = false;
      for (auto& leaf_shape : result_shape.tuple_shapes()) {
        if (leaf_shape.IsTuple()) {
          needs_flattening = true;
        }
      }
      if (needs_flattening) {
        std::vector<HloInstruction*> elements;
        FlattenIntoHelper(elements, builder, fusion);
        builder.AddInstruction(HloInstruction::CreateTuple(elements));
      }
    }

    auto* new_entry = module->AddComputationAndUnifyNamesAndIds(
        builder.Build(), /*is_entry=*/false);
    module->ReplaceEntryComputation(new_entry);
    *module->mutable_entry_computation_layout() =
        module->compute_computation_layout();
  }

  return module;
}

}  // namespace xla
