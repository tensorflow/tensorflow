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

#include "xla/backends/gpu/transforms/sort_iota_fusion.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

class SortIotaFusionGroupVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleSort(HloInstruction* sort) override {
    VLOG(4) << "Input: " << sort->ToString();
    std::vector<HloInstruction*> iota_operands;
    absl::flat_hash_set<HloInstruction*> different_iotas;
    for (HloInstruction* operand : sort->mutable_operands()) {
      if (HloPredicateIsOp<HloOpcode::kIota>(operand)) {
        if (different_iotas.insert(operand).second) {
          iota_operands.push_back(operand);
        }
      }
    }
    if (iota_operands.empty()) {
      return absl::OkStatus();
    }
    HloInstruction* fusion =
        sort->parent()->AddInstruction(HloInstruction::CreateFusion(
            sort->shape(), HloInstruction::FusionKind::kCustom, sort));
    for (HloInstruction* iota : iota_operands) {
      fusion->FuseInstruction(iota);
    }
    VLOG(5) << "Generated fusion: " << fusion->ToString();
    return ReplaceInstruction(sort, fusion);
  }
};
}  // namespace

absl::StatusOr<bool> SortIotaFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ASSIGN_OR_RETURN(bool changed, SortIotaFusionGroupVisitor().RunOnModule(
                                     module, execution_threads));
  return changed;
}
}  // namespace xla::gpu
