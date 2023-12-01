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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

// Returns instructions that have to become custom fusion parameters. Returns an
// error if matched pattern can't be outlined as a fusion.
static StatusOr<absl::InlinedVector<HloInstruction*, 4>> GetPatternCaptures(
    const CustomFusionPattern::Match& match) {
  HloInstruction* root = match.instructions.back();
  absl::InlinedVector<HloInstruction*, 4> captures;

  // Instruction that will go into the fusion body.
  absl::flat_hash_set<HloInstruction*> instructions_set(
      match.instructions.begin(), match.instructions.end());

  // Check that intermediate instructions do not have users outside of the
  // matched pattern. Only root instruction can have external users.
  for (HloInstruction* instr : match.instructions) {
    for (HloInstruction* user : instr->users()) {
      if (instr != root && !instructions_set.contains(user)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Custom fusion intermediate result ", instr->name(),
            " has users outside of a matched pattern: ", user->name()));
      }
    }
  }

  // Collect instructions captured by a matched pattern.
  for (HloInstruction* instr : match.instructions) {
    for (HloInstruction* operand : instr->operands()) {
      if (!instructions_set.contains(operand) &&
          absl::c_find(captures, operand) == captures.end()) {
        captures.push_back(operand);
      }
    }
  }

  return captures;
}

// Creates custom fusion computation and moves all matched instructions into it.
static StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, const CustomFusionPattern::Match& match,
    absl::Span<HloInstruction* const> captures) {
  HloComputation::Builder builder(match.config.name());

  // A mapping from original instructions to instructions in the fusion body.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;

  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(instr_mapping.at(operand));
    }
    return operands;
  };

  // For every parameter create a parameter instruction in the computation body
  // and set up instruction mapping.
  for (const HloInstruction* capture : captures) {
    int64_t index = instr_mapping.size();
    instr_mapping[capture] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, capture->shape(), absl::StrCat("p", index)));
  }

  // TODO(ezhulenev): Instructions in the pattern must be topologically sorted,
  // otherwise we'll get a crash! Figure out how to do it!
  for (HloInstruction* instr : match.instructions) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

static StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, const CustomFusionPattern::Match& match,
    absl::Span<HloInstruction* const> captures, HloComputation* body) {
  // We'll be replacing the root operation of a custom fusion with a fusion
  // instruction calling fusion computation.
  HloInstruction* root = match.instructions.back();
  HloComputation* parent = root->parent();

  // Add a fusion operation calling outlined fusion computation.
  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      root->shape(), HloInstruction::FusionKind::kCustom, captures, body));
  module->SetAndUniquifyInstrName(fusion, match.config.name());

  // Set backends config to a matched custom fusion config.
  FusionBackendConfig backend_config;
  backend_config.set_kind("__custom_fusion");
  *backend_config.mutable_custom_fusion_config() = match.config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(backend_config)));

  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(root, fusion));
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
    // Check if pattern can be outlined as a fusion and collect captured
    // parameters (instructions defined outside of a fusion).
    auto captures = GetPatternCaptures(match);
    if (!captures.ok()) {
      VLOG(2) << "Skip custom fusion " << match.config.name() << ": "
              << captures.status();
      continue;
    }

    TF_ASSIGN_OR_RETURN(HloComputation * fusion_body,
                        CreateFusionBody(module, match, *captures));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * fusion,
        CreateFusionInstruction(module, match, *captures, fusion_body));

    VLOG(2) << "Added a fusion instruction: " << fusion->name()
            << " for custom fusion " << match.config.name()
            << " (instruction count = " << match.instructions.size() << ")";
  }

  return true;
}

}  // namespace xla::gpu
