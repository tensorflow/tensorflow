/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/custom_fusion_rewriter.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/kernels/custom_fusion_pattern.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

CustomFusionRewriter::CustomFusionRewriter(
    const CustomFusionPatternRegistry* patterns)
    : patterns_(patterns) {}

// Creates custom fusion computation and moves all matched instructions into it.
static StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, const CustomFusionPattern::Match& match) {
  HloComputation::Builder builder(match.config.name());

  // We do not currently support matching custom fusions with more than one
  // instruction.
  HloInstruction* root = match.instructions[0];

  // Fusion computation parameters inferred from a matched instruction.
  absl::InlinedVector<HloInstruction*, 4> parameters;
  for (HloInstruction* operand : root->operands()) {
    parameters.push_back(builder.AddInstruction(
        HloInstruction::CreateParameter(parameters.size(), operand->shape(),
                                        absl::StrCat("p", parameters.size()))));
  }

  builder.AddInstruction(root->CloneWithNewOperands(root->shape(), parameters));

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

static StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, const CustomFusionPattern::Match& match,
    HloComputation* body) {
  // We'll be replacing the root operation of a custom fusion with a fusion
  // instruction calling fusion computation.
  HloInstruction* fusion_root = match.instructions[0];
  HloComputation* fusion_parent = fusion_root->parent();

  HloInstruction* fusion =
      fusion_parent->AddInstruction(HloInstruction::CreateFusion(
          fusion_root->shape(), HloInstruction::FusionKind::kCustom,
          fusion_root->operands(), body));

  // Assign unique name to a new fusion instruction.
  module->SetAndUniquifyInstrName(fusion, match.config.name());

  // Set backends config to a matched custom fusion config.
  FusionBackendConfig backend_config;
  backend_config.set_kind("__custom_fusion");
  *backend_config.mutable_custom_fusion_config() = match.config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(backend_config)));

  // Replace fusion root with a fusion instruction.
  TF_RETURN_IF_ERROR(fusion_parent->ReplaceInstruction(fusion_root, fusion));

  return fusion;
}

StatusOr<bool> CustomFusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<CustomFusionPattern::Match> matches;

  // Collect all potential custom fusion matches in the module.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      auto matched = patterns_->Match(instr);
      matches.insert(matches.end(), matched.begin(), matched.end());
    }
  }

  if (matches.empty()) return false;

  for (const CustomFusionPattern::Match& match : matches) {
    if (match.instructions.size() != 1)
      return absl::InternalError(
          "Custom fusions with multiple instruction are not yet supported");

    TF_ASSIGN_OR_RETURN(HloComputation * fusion_body,
                        CreateFusionBody(module, match));

    TF_ASSIGN_OR_RETURN(HloInstruction * fusion,
                        CreateFusionInstruction(module, match, fusion_body));

    VLOG(5) << "Added a fusion instruction: " << fusion->name()
            << " for custom fusion " << match.config.name()
            << " (instruction count = " << match.instructions.size() << ")";
  }

  return true;
}

}  // namespace xla::gpu
