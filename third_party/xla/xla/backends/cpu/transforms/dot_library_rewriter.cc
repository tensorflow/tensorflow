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

  // Add new operands as fusion parameters.
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

  bool shape_changed = old_root->shape() != to_fuse->shape();
  HloInstruction* new_root = fused_computation->AddInstruction(
      to_fuse->CloneWithNewOperands(to_fuse->shape(), new_operands));
  fused_computation->set_root_instruction(
      new_root,
      /*accept_different_shape=*/shape_changed);
  if (shape_changed) {
    *fusion->mutable_shape() = new_root->shape();
  }

  TF_RETURN_IF_ERROR(
      fusion->parent()->ReplaceInstructionWithDifferentShape(to_fuse, fusion));
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
    if (options.use_xnnpack && options_.xnn_fusion_types != nullptr &&
        !options_.xnn_fusion_types->empty()) {
      libs_.push_back(std::make_unique<XnnMatcher>(target_machine_features_,
                                                   options_.xnn_fusion_types));
    }
    for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
      supported_ops_.merge(lib->SupportedOps());
    }
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

      // Find if an operand is a fusion of the same library kind.
      // If there are more than one, we fuse with the first one.
      HloFusionInstruction* fusion = nullptr;
      for (auto operand : instr->operands()) {
        if (IsCustomFusionWithKind(operand, lib->fusion_kind())) {
          fusion = Cast<HloFusionInstruction>(
              const_cast<HloInstruction*>(operand));  // NOLINT
          break;
        }
      }

      // If no fusion is found, we check if we should start a new fusion.
      if (fusion == nullptr && !lib->ShouldCreateFusion(instr)) {
        continue;
      }
      // If the custom fusion has multiple users, fusing `instr` into it will
      // require multi-output support. So we do not fuse it for now.
      // TODO(penporn): Support multi-output fusion.
      if (fusion != nullptr && fusion->user_count() > 1) {
        continue;
      }

      PrimitiveType out_dtype = instr->shape().element_type();
      PrimitiveType lib_out_dtype = lib->LibraryOpOutputType(instr);
      if (fusion == nullptr) {
        // Start a new fusion.
        HloComputation* computation = instr->parent();
        fusion = Cast<HloFusionInstruction>(
            computation->AddInstruction(HloInstruction::CreateFusion(
                instr->shape(), HloInstruction::FusionKind::kCustom, instr,
                lib->fusion_prefix())));

        // Set the fusion kind.
        BackendConfig backend_config;
        FusionBackendConfig* fusion_config =
            backend_config.mutable_fusion_config();
        fusion_config->set_kind(lib->fusion_kind());
        TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

        // Replace the instruction.
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstructionWithDifferentShape(instr, fusion));

      } else {
        // One of the operands is a fusion. Fuse with it.
        TF_RETURN_IF_ERROR(FuseConsumerInstruction(fusion, instr));
      }

      // If the library can't output the exact type, we set the type of the op
      // to what the library supports, and add a convert node to change to the
      // desired type.
      if (out_dtype != lib_out_dtype) {
        HloComputation* fused_computation =
            fusion->fused_instructions_computation();
        HloInstruction* root = fusion->fused_expression_root();
        root->mutable_shape()->set_element_type(lib_out_dtype);
        HloInstruction* convert = fused_computation->AddInstruction(
            HloInstruction::CreateConvert(fusion->shape(), root));
        fused_computation->set_root_instruction(convert);
      }
      MarkAsChanged();
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
