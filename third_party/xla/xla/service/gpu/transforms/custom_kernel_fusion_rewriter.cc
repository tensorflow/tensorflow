/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

CustomKernelFusionRewriter::CustomKernelFusionRewriter(
    const se::DeviceDescription* device, int kernel_index,
    const CustomKernelFusionPatternRegistry* patterns)
    : device_(device), kernel_index_(kernel_index), patterns_(patterns) {}

// Returns a set of instruction that have users outside of a matched pattern
// and have a replacement that must be applied after building a new custom
// fusion instruction. Only root instruction can have external users and does
// not require a replacement, as the fusion itself is a replacement. If
// instruction has external users and does not have a replacement returns empty
// optional.
static std::optional<absl::flat_hash_set<HloInstruction*>>
GetPatternReplacements(const CustomKernelFusionPattern::Match& match) {
  absl::flat_hash_set<HloInstruction*> requires_replacement;
  absl::flat_hash_set<HloInstruction*> instructions_set(
      match.instructions().begin(), match.instructions().end());

  for (HloInstruction* instr : match.instructions()) {
    for (HloInstruction* user : instr->users()) {
      if (instr == match.root() || instructions_set.contains(user)) continue;

      if (match.HasReplacement(instr)) {
        requires_replacement.insert(instr);
        continue;
      }

      VLOG(3) << "Custom kernel fusion intermediate result " << instr->name()
              << " has users outside of a matched pattern: " << user->name();
      return std::nullopt;
    }
  }

  return requires_replacement;
}

// Returns instructions that have to become custom kernel fusion parameters.
// Returns an error if matched pattern can't be outlined as a fusion.
static absl::InlinedVector<HloInstruction*, 4> GetPatternCaptures(
    const CustomKernelFusionPattern::Match& match) {
  absl::InlinedVector<HloInstruction*, 4> captures;

  absl::flat_hash_set<HloInstruction*> instructions_set(
      match.instructions().begin(), match.instructions().end());

  for (HloInstruction* instr : match.instructions()) {
    for (HloInstruction* operand : instr->operands()) {
      if (!instructions_set.contains(operand) &&
          absl::c_find(captures, operand) == captures.end()) {
        captures.push_back(operand);
      }
    }
  }

  return captures;
}

// Creates custom kernel fusion computation and moves all matched instructions
// into it.
static absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, const CustomKernelFusionPattern::Match& match,
    absl::Span<HloInstruction* const> captures) {
  HloComputation::Builder builder(match.config().name());

  // A mapping from original instructions to instructions in the fusion body.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;

  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(instr_mapping.at(operand));
    }
    return operands;
  };

  // For every captured value create a parameter instruction in the computation
  // body and set up instruction mapping.
  for (const HloInstruction* capture : captures) {
    int64_t index = instr_mapping.size();
    instr_mapping[capture] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, capture->shape(), absl::StrCat("p", index)));
  }

  // TODO(ezhulenev): Instructions in the pattern must be topologically sorted,
  // otherwise we'll get a crash! Figure out how to do it!
  for (HloInstruction* instr : match.instructions()) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
  }

  HloInstruction* root = builder.last_added_instruction();

  // If custom kernel fusion requires a workspace we add a custom call that
  // allocates workspace and return a tuple of "real" result and a workspace.
  if (match.workspace_size_bytes() > 0) {
    auto workspace_shape =
        ShapeUtil::MakeShape(PrimitiveType::U8, {match.workspace_size_bytes()});
    HloInstruction* workspace =
        builder.AddInstruction(HloInstruction::CreateCustomCall(
            workspace_shape, {}, CustomKernelFusionPattern::kWorkspace, "",
            CustomCallApiVersion::API_VERSION_TYPED_FFI));
    builder.AddInstruction(HloInstruction::CreateTuple({root, workspace}));
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

namespace {
absl::StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, const CustomKernelFusionPattern::Match& match,
    absl::Span<HloInstruction* const> captures, HloComputation* body,
    int kernel_index) {
  // We'll be replacing the root operation of a custom kernel fusion with a
  // fusion instruction calling fusion computation.
  HloInstruction* root = match.root();
  HloComputation* parent = root->parent();

  // Add a fusion operation calling outlined fusion computation.
  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      body->root_instruction()->shape(), HloInstruction::FusionKind::kCustom,
      captures, body));
  module->SetAndUniquifyInstrName(fusion, match.config().name());

  // Set backends config to a matched custom fusion config.
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind("__custom_fusion");
  *backend_config.mutable_custom_fusion_config() = match.config();
  backend_config.mutable_custom_fusion_config()->set_kernel_index(kernel_index);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

  // If we don't have workspace we can return constructed fusion instruction.
  if (match.workspace_size_bytes() == 0) return fusion;

  // Otherwise have to get result corresponding to the original value;
  return parent->AddInstruction(
      HloInstruction::CreateGetTupleElement(fusion, 0));
}
}  // namespace

absl::StatusOr<bool> CustomKernelFusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<CustomKernelFusionPattern::Match> matches;

  // Collect all potential custom fusion matches in the module.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      auto matched = patterns_->Match(*device_, instr);
      matches.insert(matches.end(), matched.begin(), matched.end());
    }
  }

  if (matches.empty()) return false;

  for (const CustomKernelFusionPattern::Match& match : matches) {
    VLOG(2) << "Matched custom kernel fusion " << match.config().name()
            << "; root instruction: " << match.instructions().back()->name();

    auto replacememts = GetPatternReplacements(match);
    if (!replacememts.has_value()) continue;

    auto captures = GetPatternCaptures(match);

    TF_ASSIGN_OR_RETURN(HloComputation * fusion_body,
                        CreateFusionBody(module, match, captures));
    TF_ASSIGN_OR_RETURN(HloInstruction * fusion,
                        CreateFusionInstruction(module, match, captures,
                                                fusion_body, kernel_index_));

    VLOG(2) << "Added a fusion instruction: " << fusion->name()
            << " for custom kernel fusion " << match.config().name()
            << " (instruction count = " << match.instructions().size() << ")";

    for (HloInstruction* instr : *replacememts) {
      VLOG(2) << "Replace matched instruction: " << instr->name()
              << " with a pattern replacement";

      TF_ASSIGN_OR_RETURN(
          HloInstruction * replacement,
          match.BuildReplacement(instr, Cast<HloFusionInstruction>(fusion)));

      TF_RETURN_IF_ERROR(
          instr->ReplaceAllUsesWith(replacement, match.config().name()));

      VLOG(2) << "Replaced instruction: " << instr->name()
              << " with: " << replacement->name();
    }

    VLOG(2) << "Replace custom kernel fusion root instruction "
            << match.root()->name() << "with " << fusion->name();
    HloComputation* parent = match.root()->parent();
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(match.root(), fusion));
  }

  return true;
}

}  // namespace xla::gpu
