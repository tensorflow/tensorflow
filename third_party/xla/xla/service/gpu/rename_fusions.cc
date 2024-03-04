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

#include "xla/service/gpu/rename_fusions.h"

#include <memory>
#include <string>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view FusionKindToString(
    HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kCustom:
      return "custom";
    case HloInstruction::FusionKind::kLoop:
      return "loop";
    case HloInstruction::FusionKind::kInput:
      return "input";
    case HloInstruction::FusionKind::kOutput:
      return "output";
  }
}

std::string MakeFusionHeroNames(const HloInstruction* instruction) {
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(instruction);
  absl::btree_set<absl::string_view> heroes;

  for (auto root : fusion_adaptor->GetRoots()) {
    heroes.insert(HloOpcodeString(
        FindNonTrivialHero(root.instruction(), *fusion_adaptor).opcode()));
  }
  return absl::StrReplaceAll(absl::StrJoin(heroes, "_"), {{"-", "_"}});
}

void RenameFusion(HloModule* module, HloInstruction* instruction) {
  std::string hero_names = MakeFusionHeroNames(instruction);
  module->SetAndUniquifyInstrName(
      instruction, absl::StrCat(FusionKindToString(instruction->fusion_kind()),
                                "_", hero_names, "_fusion"));
  module->SetAndUniquifyComputationName(
      instruction->fused_instructions_computation(),
      absl::StrCat("fused_", hero_names));
}

}  // namespace

absl::StatusOr<bool> RenameFusions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion ||
          instruction->fusion_kind() == HloInstruction::FusionKind::kCustom) {
        continue;
      }
      RenameFusion(module, instruction);
    }
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
