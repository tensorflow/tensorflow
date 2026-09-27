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

#include "xla/backends/gpu/transforms/constant_fill_copy_rewriter.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/decision.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

// Splitting small fills can cost more in launches than it saves in memory.
constexpr int64_t kMinFillBytes = 1024 * 1024;

// Returns the instruction that ultimately consumes `user`'s value, looking
// through the same single-user bitcast and copy that the rewrite looks through.
const HloInstruction* Consumer(const HloInstruction* user) {
  if (user->opcode() == HloOpcode::kBitcast && user->user_count() == 1) {
    user = user->users()[0];
  }
  if (user->opcode() == HloOpcode::kCopy && user->user_count() == 1) {
    user = user->users()[0];
  }
  return user;
}

Decision CanReplaceCopyWithFill(const HloInstruction& copy) {
  const HloInstruction* source = copy.operand(0);
  if (!copy.shape().IsArray() || copy.shape() != source->shape() ||
      copy.HasControlDependencies() || copy.has_sharding() || copy.IsDead()) {
    return Decision::Forbid(
        "Copy changes layout, has ordering or sharding, or is dead");
  }
  const HloInstruction* fill = source;
  if (source->opcode() == HloOpcode::kBitcast) {
    // CopyFusion looks through such bitcasts as well, so leaving the copy in
    // place would fuse it back into the fill.
    if (source->user_count() != 1 || source->HasControlDependencies() ||
        source->has_sharding() ||
        !hlo_instruction_utils::KeepsBitwidth(*source)) {
      return Decision::Forbid("Bitcast is shared or changes bitwidth");
    }
    fill = source->operand(0);
  }
  if (fill->opcode() != HloOpcode::kFusion ||
      fill->fusion_kind() != HloInstruction::FusionKind::kLoop ||
      fill->operand_count() != 0 || fill->HasControlDependencies() ||
      fill->has_sharding()) {
    return Decision::Forbid("Not an independent loop fill");
  }
  if (fill->fused_instructions_computation()->instruction_count() != 2 ||
      !hlo_query::IsBroadcastOfScalarConstant(*fill->fused_expression_root())) {
    return Decision::Forbid("Not a broadcast of a scalar literal");
  }
  if (ShapeUtil::ByteSizeOfElements(copy.shape()) < kMinFillBytes) {
    return Decision::Forbid("Fill is too small to split");
  }
  // When every use of the fill reaches the copy's consumer, all of those
  // buffers are live there anyway. A separate fill would only add a launch and
  // free the scheduler to pull it away from the consumer, so leave the copy for
  // CopyFusion to fold into the fill.
  if (copy.user_count() == 1 &&
      absl::c_all_of(fill->users(), [&](const HloInstruction* user) {
        return Consumer(user) == copy.users()[0];
      })) {
    return Decision::Forbid("All uses of the fill feed the same consumer");
  }
  return Decision::Allow();
}

}  // namespace

absl::StatusOr<bool> ConstantFillCopyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return false;
  }
  bool changed = false;
  // Each decision sees the replacements made so far: once the copies feeding
  // other consumers have their own fills, the remaining copies of a fill that
  // all feed one consumer stay with it.
  for (HloInstruction* copy : entry->MakeInstructionPostOrder()) {
    if (copy->opcode() != HloOpcode::kCopy) {
      continue;
    }
    if (Decision decision = CanReplaceCopyWithFill(*copy);
        decision.IsForbidden()) {
      VLOG(4) << "Not rematerializing " << copy->name() << ": "
              << decision.Explain();
      continue;
    }
    HloInstruction* source = copy->mutable_operand(0);
    HloInstruction* fill = source->opcode() == HloOpcode::kBitcast
                               ? source->mutable_operand(0)
                               : source;
    HloInstruction* replacement = source;
    if (fill->user_count() > 1) {
      // Other uses keep the original fill alive, so the copy gets a fill of its
      // own: still a separate writable value, but one that no longer pins the
      // original's buffer until this consumer runs.
      replacement = entry->AddInstruction(fill->Clone("rematerialized"));
      if (source != fill) {
        replacement = entry->AddInstruction(
            HloInstruction::CreateBitcast(copy->shape(), replacement));
      }
    }
    // Otherwise the copy is the fill's only use and the fill stands in for it
    // directly. Either way the original fill keeps a user, so its fusion body
    // is never orphaned.
    ABSL_RETURN_IF_ERROR(entry->ReplaceInstruction(copy, replacement));
    changed = true;
  }
  return changed;
}

}  // namespace xla::gpu
