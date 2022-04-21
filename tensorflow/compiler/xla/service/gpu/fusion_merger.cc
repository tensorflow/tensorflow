/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

namespace {

// Traverses users of tuple shape, adding leaf instructions to 'instructions'.
void MaybeResolveTupleElements(HloInstruction* instruction,
                               std::vector<HloInstruction*>* instructions) {
  if (instruction->shape().IsTuple()) {
    for (auto tuple_user : instruction->users()) {
      MaybeResolveTupleElements(tuple_user, instructions);
    }
  } else {
    instructions->push_back(instruction);
  }
}

// Returns the bytes read by fusion parameter 'param', by returning the byte
// size of 'param' shape (or the cumulative byte sizes of all leaf tuple
// elements if 'param' is tuple-shaped).
//
// In the special case where all users of 'param' (or all users of a leaf
// tuple element if 'param' is tuple-shaped) are Slice instructions, the size
// of each slice instruction is accumulated instead, to give a more accurate
// value for bytes read.
double CalculateBytesReadByFusionParameter(HloInstruction* param) {
  CHECK_EQ(HloOpcode::kParameter, param->opcode());

  // Adds all leaf tuple elements to 'instructions' if 'param' is tuple-shaped.
  // Adds 'param' to 'instructions' otherwise.
  std::vector<HloInstruction*> instructions;
  MaybeResolveTupleElements(param, &instructions);

  // Iterate through 'instructions' accumulating byte sizes of each instruction
  // shape. For each 'instruction' in 'instructions', if all users of
  // 'instruction' are Slice instructions, accumulates the byte sizes of each
  // Slice for a more accurate estimate of bytes read.
  double bytes = 0.0;
  for (auto& instruction : instructions) {
    if (absl::c_all_of(
            instruction->users(), [](const HloInstruction* instruction) {
              return instruction->opcode() == HloOpcode::kSlice ||
                     instruction->opcode() == HloOpcode::kDynamicSlice;
            })) {
      // All users are slice: accumulate bytes of all user slice instructions.
      for (auto& user : instruction->users()) {
        bytes += ShapeUtil::ByteSizeOf(user->shape());
      }
    } else {
      // Some users are not slice: accumulate full size of 'instruction'.
      bytes += ShapeUtil::ByteSizeOf(instruction->shape());
    }
  }
  return bytes;
}

// Returns the bytes read by all fusion parameters of instruction 'fusion'.
double CalculateBytesReadByFusionInstruction(HloInstruction* fusion) {
  double bytes = 0.0;
  for (auto* fused_instruction : fusion->fused_instructions()) {
    if (fused_instruction->opcode() != HloOpcode::kParameter) {
      continue;
    }
    bytes += CalculateBytesReadByFusionParameter(fused_instruction);
  }
  return bytes;
}

// Returns bytes transferred by instruction 'fusion', including the bytes
// that would be read by all users.
double GetCurrentBytesTransferred(HloInstruction* fusion) {
  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  const double bytes_read = CalculateBytesReadByFusionInstruction(fusion);
  double bytes_written = 0;
  if (fusion->IsMultiOutputFusion()) {
    for (auto& operand : fusion->fused_expression_root()->operands()) {
      bytes_written += ShapeUtil::ByteSizeOf(operand->shape());
    }
  } else {
    bytes_written =
        ShapeUtil::ByteSizeOf(fusion->fused_expression_root()->shape());
  }
  // Current bytes transferred (ignoring non 'fusion' user operands) is bytes
  // read and written by 'fusion', plus reads of size 'bytes_written' for each
  // user.
  return bytes_read + bytes_written * (fusion->user_count() + 1);
}

// Returns bytes transferred if 'fusion' were to be merged into its users.
double GetMergedBytesTransferred(HloInstruction* fusion) {
  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  return CalculateBytesReadByFusionInstruction(fusion) * fusion->user_count();
}

}  // anonymous namespace

// For each fusion F, attempts to fuse F into *all* of F's users (does not fuse
// if can't fuse into at least one).
class FusionInstructionMerger {
 public:
  explicit FusionInstructionMerger(HloComputation* computation)
      : computation_(computation),
        dump_fusion_visualization_(computation->parent()
                                       ->config()
                                       .debug_options()
                                       .xla_dump_fusion_visualization()) {}

  Status Run();

  bool changed() const { return changed_; }

 private:
  FusionDecision HandleFusion(HloInstruction* fusion);
  Status FuseIntoAllUsers(HloInstruction* instruction);

  HloComputation* computation_;

  bool changed_ = false;
  bool dump_fusion_visualization_ = false;

  // Fusion instruction merge stats.
  int total_visited_ = 0;
  int total_merged_ = 0;
  int num_fail_no_users_ = 0;
  int num_fail_not_loop_fusion_ = 0;
  int num_fail_merge_all_users_ = 0;
  int num_fail_expensive_fused_instruction_ = 0;
  int num_fail_net_bytes_transferred_ratio_ = 0;
  int num_fail_inefficient_fusion_emitter_ = 0;
  int num_fail_fusion_too_large_ = 0;

  FusionInstructionMerger(const FusionInstructionMerger&) = delete;
  FusionInstructionMerger& operator=(const FusionInstructionMerger&) = delete;
};

Status FusionInstructionMerger::FuseIntoAllUsers(HloInstruction* instruction) {
  // Merge fused instructions from 'fusion' into each user.
  std::vector<HloInstruction*> users = instruction->users();
  for (HloInstruction* user : users) {
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("About to fuse |", instruction->name(), "| into |",
                       user->name(), "| inside FusionMerger"),
          /*consumer=*/*user,
          /*producer=*/instruction);
    }

    // Wrap consumers which are not fusions first.
    HloInstruction* consumer = user;
    if (consumer->opcode() != HloOpcode::kFusion) {
      consumer = computation_->AddInstruction(HloInstruction::CreateFusion(
          user->shape(), ChooseFusionKind(*instruction, *user), user));
      TF_CHECK_OK(computation_->ReplaceInstruction(user, consumer));
    }
    consumer->MergeFusionInstruction(instruction);
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("Fused |", instruction->name(), "| into |", user->name(),
                       "| inside FusionMerger"),
          *consumer);
    }
    changed_ = true;
  }

  CHECK_EQ(0, instruction->user_count()) << instruction->ToString();
  TF_RETURN_IF_ERROR(computation_->RemoveInstruction(instruction));
  VLOG(2) << "Merged fusion instruction: " << instruction->name()
          << " into users { "
          << absl::StrJoin(users, ", ",
                           [](std::string* out, HloInstruction* user) {
                             absl::StrAppend(out, user->name());
                           })
          << " }";
  return Status::OK();
}

Status FusionInstructionMerger::Run() {
  for (HloInstruction* instruction : computation_->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kFusion) {
      FusionDecision was_fused = HandleFusion(instruction);
      if (!was_fused) {
        VLOG(2) << "Not fusing fusion |" << instruction->name()
                << "| with all of it's users due to: " << was_fused.Explain();
        if (dump_fusion_visualization_ && !instruction->users().empty()) {
          RegisterFusionState(
              *computation_,
              absl::StrCat(
                  "Not fusing fusion |", instruction->name(),
                  "| into all of its users due to: ", was_fused.Explain()),
              // Just pick any consumer, since we are trying to merge into all.
              /*consumer=*/*instruction->users()[0],
              /*producer=*/instruction);
        }
      } else {
        TF_RETURN_IF_ERROR(FuseIntoAllUsers(instruction));
      }
    }
  }

  VLOG(1) << "FusionInstructionMerger EXIT"
          << " computation: " << computation_->name()
          << " total_visited: " << total_visited_
          << " total_merged: " << total_merged_ << " merge failures { "
          << " no_users: " << num_fail_no_users_
          << " not_loop_fusion: " << num_fail_not_loop_fusion_
          << " merge_all_users: " << num_fail_merge_all_users_
          << " expensive_instruction: " << num_fail_expensive_fused_instruction_
          << " net_bytes_transferred: " << num_fail_net_bytes_transferred_ratio_
          << " inefficient_fusion_emitter: "
          << num_fail_inefficient_fusion_emitter_
          << " fusion_too_large: " << num_fail_fusion_too_large_ << " }";
  return Status::OK();
}

FusionDecision FusionInstructionMerger::HandleFusion(HloInstruction* fusion) {
  ++total_visited_;

  // Skip 'fusion' instruction if there are no users into which we can merge.
  if (fusion->users().empty()) {
    ++num_fail_no_users_;
    return "fusion has no users";
  }

  // Skip 'fusion' instruction if it is not a loop fusion. Library fusion
  // instructions match specific patterns, so they shouldn't be further fused.
  // Input fusion instructions need to be rooted at a particular HLO (e.g.
  // kReduce), so they shouldn't be further fused either.
  if (!fusion->IsLoopFusion()) {
    ++num_fail_not_loop_fusion_;
    return "not a loop fusion";
  }

  for (const HloInstruction* user : fusion->users()) {
    FusionDecision fusible = IsProducerConsumerFusible(*fusion, *user)
                                 .And({user->opcode() != HloOpcode::kBitcast,
                                       "not fusing bitcast ops"});
    if (!fusible) {
      ++num_fail_merge_all_users_;
      return fusible;
    }
  }

  // Skip 'fusion' instruction if merging it into all users would result in a
  // net increase in bytes transferred (currently allowing the net bytes
  // transferred to be exceeded up to ~10% in exchange for eliminating the
  // overhead from a GPU kernel launch).
  const double current_bytes_transferred = GetCurrentBytesTransferred(fusion);
  const double merged_bytes_transferred = GetMergedBytesTransferred(fusion);
  const double merged_to_current_bytes_ratio =
      merged_bytes_transferred / std::max(1.0, current_bytes_transferred);
  if (merged_to_current_bytes_ratio > 1.10) {
    ++num_fail_net_bytes_transferred_ratio_;
    return FusionDecision{} << "merged-to-current-bytes-ratio of "
                            << merged_to_current_bytes_ratio
                            << " is not favorable";
  }

  // Skip 'fusion' instruction if any of its fused instructions are expensive.
  // This is done to avoid the duplication of expensive instructions, which
  // would occur if 'fusion' were merged into multiple users.
  //
  // Also, we don't want to fuse expensive instructions with instructions which
  // reuse its operand values (e.g. Broadcast instructions).
  //
  // However, if we are going to save a "lot" in memory bandwidth then we
  // ignore how expensive the fusion instructions are.  The heuristic used to
  // determine "a lot" is the following: merging must reduce memory traffic by a
  // factor of 0.3, and the amount of memory accessed must not be entirely
  // trivial (above 1K).  This likely has room for improvement in the future.

  bool allow_expensive_ops =
      (fusion->user_count() == 1 || (merged_to_current_bytes_ratio < 0.3 &&
                                     current_bytes_transferred > 1024)) &&
      !absl::c_any_of(fusion->users(), [fusion](const HloInstruction* user) {
        int64_t operand_index = user->operand_index(fusion);
        return user->ReusesOperandElements(operand_index);
      });
  if (!allow_expensive_ops) {
    for (const HloInstruction* instruction : fusion->fused_instructions()) {
      if (instruction->opcode() != HloOpcode::kParameter &&
          GpuInstructionFusion::IsExpensive(*instruction)) {
        ++num_fail_expensive_fused_instruction_;
        return FusionDecision{} << "fusion contains an expensive instruction |"
                                << instruction->name() << "|";
      }
    }
  }

  // Skip 'fusion' instruction if merging it into at least one of the users
  // would cause too much code duplication because of inefficiencies in the
  // fusion emitter.
  // TODO(b/119692968): Remove this once the fusion emitter can handle arbitrary
  // fusion nodes.
  for (const HloInstruction* user : fusion->users()) {
    if (FusedIrEmitter::IsFusedIrEmitterInefficient(/*consumer=*/*user,
                                                    /*producer=*/*fusion)) {
      ++num_fail_inefficient_fusion_emitter_;
      return FusionDecision{}
             << "fusion contains user |" << user->ToShortString()
             << "| which would cause inefficiency in fusion emitter";
    }

    // Skip 'fusion' instruction if merging it into at least one of the users
    // would make the fusion too big.
    FusionDecision fits = FusionFitsInBudget(*fusion, *user);
    if (!fits) {
      ++num_fail_fusion_too_large_;
      return fits;
    }
  }

  ++total_merged_;
  return {};
}

StatusOr<bool> FusionMerger::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "FusionMerger for module: " << module->name();
  for (auto* computation : module->MakeNonfusionComputations()) {
    VLOG(1) << "Before running FusionInstructionMerger for computation: "
            << computation->name();
    XLA_VLOG_LINES(3, computation->ToString());

    FusionInstructionMerger fusion_merger(computation);
    TF_RETURN_IF_ERROR(fusion_merger.Run());
    changed |= fusion_merger.changed();

    VLOG(1) << "After running FusionInstructionMerger for computation: "
            << computation->name() << " changed: " << changed;
    XLA_VLOG_LINES(3, computation->ToString());
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
