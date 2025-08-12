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

#include "xla/backends/cpu/transforms/dot_library_rewriter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/transforms/onednn_matcher.h"
#include "xla/backends/cpu/transforms/xnn_matcher.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

bool IsCustomFusionWithKind(const HloInstruction* instr,
                            absl::string_view fusion_kind) {
  return instr->IsCustomFusion() &&
         instr->backend_config<BackendConfig>()->fusion_config().kind() ==
             fusion_kind;
}

// Fuses a consumer `to_fuse` into `fusion` as the new fusion root.
// - We cannot call `HloFusionInstruction::FuseInstruction()` because it only
//   supports fusing a producer into the fusion instruction.
// - Putting `to_fuse` in a new fusion instruction and calling
//   `to_fuse->MergeFusionInstruction(fusion)` is also not ideal because we will
//   have to copy many instructions in `fusion` into the new fusion node.
absl::Status FuseConsumerInstruction(HloFusionInstruction* fusion,
                                     HloInstruction* to_fuse) {
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  HloInstruction* old_root = fusion->fused_expression_root();
  std::vector<HloInstruction*> new_operands;
  for (auto operand : to_fuse->operands()) {
    if (operand == fusion) {
      new_operands.push_back(old_root);
      continue;
    }

    // Check if the operand is already a fusion operand.
    int fusion_param_idx = 0;
    for (auto fusion_operand : fusion->operands()) {
      if (fusion_operand == operand) {
        break;
      }
      fusion_param_idx++;
    }
    if (fusion_param_idx < fusion->operand_count()) {
      // Reuse the existing fusion parameter.
      new_operands.push_back(
          fused_computation->parameter_instruction(fusion_param_idx));
    } else {
      // Add a new fusion operand.
      HloInstruction* new_operand = fusion->AddCallOperand(operand);
      new_operands.push_back(new_operand);
    }
  }

  HloInstruction* new_root = fused_computation->AddInstruction(
      to_fuse->CloneWithNewOperands(to_fuse->shape(), new_operands));
  fused_computation->set_root_instruction(
      new_root,
      /*accept_different_shape=*/old_root->shape() != new_root->shape());

  TF_RETURN_IF_ERROR(fusion->parent()->ReplaceInstruction(to_fuse, fusion));
  return absl::OkStatus();
}

}  // namespace

class DotLibraryRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  explicit DotLibraryRewriteVisitor(
      const TargetMachineFeatures* target_machine_features,
      const DotLibraryRewriterOptions& options)
      : target_machine_features_(target_machine_features), options_(options) {
    if (options.use_onednn) {
      libs_.push_back(
          std::make_unique<OneDnnMatcher>(target_machine_features_));
    }
    if (options.use_xnnpack) {
      libs_.push_back(std::make_unique<XnnMatcher>(target_machine_features_));
    }
    for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
      supported_ops_.merge(lib->SupportedOps());
    }
  }

  // Groups possible `Dot` with elementwise instructions into a `Fusion` op
  // with kind `kCustom`.
  absl::Status HandleDot(HloInstruction* instr) override {
    // Replace the dot with a fusion node if any library supports it.
    for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
      TF_ASSIGN_OR_RETURN(bool op_supported, lib->IsOpSupported(instr));
      if (op_supported) {
        Shape fusion_shape = instr->shape();
        PrimitiveType out_dtype = fusion_shape.element_type();
        PrimitiveType lib_out_dtype = lib->LibraryOpOutputType(instr);
        std::unique_ptr<HloInstruction> convert;
        HloInstruction* fusion_root = instr;

        // Add a convert node if the library op does not support the original
        // output type.
        bool convert_output = lib_out_dtype != out_dtype;
        if (convert_output) {
          instr->mutable_shape()->set_element_type(lib_out_dtype);
          fusion_root = instr->parent()->AddInstruction(
              HloInstruction::CreateConvert(fusion_shape, instr));
        }

        // Create a fusion with `dot` as root if we don't need output
        // conversion. Otherwise, the fusion will have `convert` as root.
        auto fusion = HloInstruction::CreateFusion(
            fusion_shape, HloInstruction::FusionKind::kCustom, fusion_root,
            absl::StrCat(lib->fusion_prefix(), "dot_"));

        if (convert_output) {
          // Convert is the root. Fuse the dot into the fusion too.
          fusion->FuseInstruction(instr);

          // `ReplaceWithNewInstruction` checks that the instruction to replace
          // has a matching shape with the original instruction, so we set the
          // dtype back to original.
          instr->mutable_shape()->set_element_type(out_dtype);

          // Remove the first `convert` we created in the parent computation
          // of the fusion op.
          TF_RETURN_IF_ERROR(instr->parent()->RemoveInstruction(fusion_root));
        }

        BackendConfig backend_config;
        FusionBackendConfig* fusion_config =
            backend_config.mutable_fusion_config();
        fusion_config->set_kind(std::string(lib->fusion_kind()));
        TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

        return ReplaceWithNewInstruction(instr, std::move(fusion));
      }
    }

    // Do nothing if no libraries support this Dot.
    return absl::OkStatus();
  }

  absl::Status DefaultAction(HloInstruction* instr) override {
    // Skip this op if no library supports it.
    if (!supported_ops_.contains(instr->opcode())) {
      return absl::OkStatus();
    }

    // If this op follows a library fusion node and is fusible, fuse it.
    for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
      TF_ASSIGN_OR_RETURN(bool op_supported, lib->IsOpSupported(instr));
      if (!op_supported) {
        continue;
      }

      // One of the operands must be a fusion of the same library kind.
      // If there are more than one, we fuse with the first one.
      HloFusionInstruction* fusion = nullptr;
      for (auto operand : instr->operands()) {
        if (IsCustomFusionWithKind(operand, lib->fusion_kind())) {
          fusion = Cast<HloFusionInstruction>(
              const_cast<HloInstruction*>(operand));  // NOLINT
          break;
        }
      }

      // If the custom fusion has multiple users, fusing `instr` into it will
      // require multi-output support. So we do not fuse it for now.
      // TODO(penporn): Support multi-output fusion.
      if (fusion == nullptr || fusion->user_count() > 1) {
        continue;
      }

      TF_RETURN_IF_ERROR(FuseConsumerInstruction(fusion, instr));
      return absl::OkStatus();
    }
    return absl::OkStatus();
  }

 private:
  std::vector<std::unique_ptr<LibraryMatcher>> libs_;
  absl::flat_hash_set<HloOpcode> supported_ops_;
  const TargetMachineFeatures* target_machine_features_;
  const DotLibraryRewriterOptions options_;
};

absl::StatusOr<bool> DotLibraryRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  DotLibraryRewriteVisitor visitor(target_machine_features_, options_);
  TF_ASSIGN_OR_RETURN(auto result,
                      visitor.RunOnModule(module, execution_threads));
  return result;
}

}  // namespace xla::cpu
