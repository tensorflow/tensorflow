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

#include "xla/backends/cpu/transforms/library_rewriter.h"

#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

bool IsCustomFusionWithKind(const HloInstruction* instr,
                            absl::string_view fusion_kind) {
  return instr->IsCustomFusion() &&
         instr->backend_config<BackendConfig>()->fusion_config().kind() ==
             fusion_kind;
}

absl::string_view FusionDirectionToString(FusionDirection dir) {
  switch (dir) {
    case FusionDirection::kUp:
      return "Up";
    case FusionDirection::kDown:
      return "Down";
    case FusionDirection::kBoth:
      return "Both";
  }
}

// Creates a new custom library fusion instruction containing a single
// instruction `instr`, with the given name prefix and kind. Returns the pointer
// to the newly created fusion instruction.
absl::StatusOr<HloFusionInstruction*> CreateLibraryFusion(
    HloInstruction* instr, absl::string_view fusion_name_prefix,
    absl::string_view fusion_kind) {
  // Start a new fusion.
  HloComputation* computation = instr->parent();
  HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      computation->AddInstruction(HloInstruction::CreateFusion(
          instr->shape(), HloInstruction::FusionKind::kCustom, instr,
          fusion_name_prefix)));

  // Set the fusion kind.
  BackendConfig backend_config;
  FusionBackendConfig* fusion_config = backend_config.mutable_fusion_config();
  fusion_config->set_kind(fusion_kind);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

  // Replace the instruction.
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstructionWithDifferentShape(instr, fusion));

  return fusion;
}

// Fuses a consumer `to_fuse` into `fusion` as the new fusion root.
// - We cannot call `HloFusionInstruction::FuseInstruction()` because it only
//   supports fusing a producer into the fusion instruction.
// - Putting `to_fuse` in a new fusion instruction and calling
//   `to_fuse->MergeFusionInstruction(fusion)` is also not ideal because we will
//   have to copy many instructions in `fusion` into the new fusion node.
absl::StatusOr<HloInstruction*> FuseConsumerInstruction(
    HloFusionInstruction* fusion, HloInstruction* to_fuse) {
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
      ++fusion_param_idx;
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
  return new_root;
}

// For `instr` with `dtype`, set its output type to `new_instr_out_dtype` and
// add a convert node to change the type back to `dtype` instead.
// This is for when the library can't output the exact type. -- We set the type
// of the op to what the library supports, and add a convert node to change to
// the desired type.
inline absl::Status InsertConvertIfNecessary(
    HloInstruction* instr, PrimitiveType new_instr_out_dtype) {
  if (instr->shape().element_type() == new_instr_out_dtype) {
    return absl::OkStatus();
  }
  HloComputation* computation = instr->parent();
  HloInstruction* convert = computation->AddInstruction(
      HloInstruction::CreateConvert(instr->shape(), instr));
  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(convert));
  instr->mutable_shape()->set_element_type(new_instr_out_dtype);

  if (instr == computation->root_instruction()) {
    computation->set_root_instruction(convert);
  }
  return absl::OkStatus();
}

inline bool IsElementwiseAndNotConstant(const HloInstruction* instr) {
  return instr->IsElementwise() && !instr->IsConstant();
}

}  // namespace

absl::StatusOr<LibraryMatcher*> LibraryRewriter::ChooseLibrary(
    HloInstruction* instr) {
  for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
    TF_ASSIGN_OR_RETURN(bool op_supported, lib->IsOpSupported(instr));
    if (op_supported && lib->ShouldCreateFusion(instr)) {
      return lib.get();
    }
  }
  return nullptr;
}

void LibraryRewriter::AddFusionCandidates(
    HloInstruction* fusion, HloInstruction* instr, FusionDirection dir,
    std::queue<std::pair<HloInstruction*, FusionDirection>>& queue) {
  // Don't add anything that has already been fused or require multi-output
  // fusion support. (We don't support that yet.)
  if (dir == FusionDirection::kUp || dir == FusionDirection::kBoth) {
    for (HloInstruction* operand : instr->operands()) {
      if (!fused_.contains(operand) &&
          absl::c_all_of(operand->users(), [&](HloInstruction* user) {
            return user == fusion || user == instr;
          })) {
        queue.push(std::make_pair(operand, FusionDirection::kUp));
      }
    }
  }
  // When fusing down, we can only have one user without multi-output support.
  if (instr->user_count() != 1) {
    return;
  }
  HloInstruction* user = instr->users().front();
  if ((dir == FusionDirection::kDown || dir == FusionDirection::kBoth) &&
      !fused_.contains(user)) {
    queue.push(std::make_pair(user, FusionDirection::kDown));
  }
}

absl::StatusOr<HloFusionInstruction*> LibraryRewriter::MergeFusionInstructions(
    HloFusionInstruction* main, HloFusionInstruction* neighbor,
    FusionDirection dir) {
  VLOG(3) << "  " << FusionDirectionToString(dir)
          << ": Fusing with: " << neighbor->ToString();
  if (dir == FusionDirection::kUp) {
    main->MergeFusionInstruction(neighbor);
    TF_RETURN_IF_ERROR(main->parent()->RemoveInstruction(neighbor));
    return main;
  }
  if (dir == FusionDirection::kDown) {
    neighbor->MergeFusionInstruction(main);
    TF_RETURN_IF_ERROR(neighbor->parent()->RemoveInstruction(main));
    return neighbor;
  }
  return InvalidArgument("Invalid fusion direction: %s",
                         FusionDirectionToString(dir));
}

absl::StatusOr<HloInstruction*> LibraryRewriter::GrowFusion(
    HloFusionInstruction* fusion, HloInstruction* to_fuse,
    FusionDirection dir) {
  HloInstruction* new_instr = nullptr;
  VLOG(3) << "  " << FusionDirectionToString(dir)
          << ": Fusing with: " << to_fuse->ToString();
  if (dir == FusionDirection::kUp) {
    new_instr = fusion->FuseInstruction(to_fuse);
    if (to_fuse->user_count() == 0) {
      TF_RETURN_IF_ERROR(to_fuse->parent()->RemoveInstruction(to_fuse));
    }
  } else if (dir == FusionDirection::kDown) {
    TF_ASSIGN_OR_RETURN(new_instr, FuseConsumerInstruction(fusion, to_fuse));
  }
  return new_instr;
}

absl::Status LibraryRewriter::FuseNeighbors(HloFusionInstruction* fusion,
                                            LibraryMatcher* lib) {
  // A queue storing potential candidates for fusion: Each item is a pair of
  //   - Pointer to immediate neighbors of the current fusion node.
  //   - Travel direction: up (parents) and down (children).
  // This queue only tracks original HLO instructions in the parent computation,
  // not any new instructions created during the fusion process.
  std::queue<std::pair<HloInstruction*, FusionDirection>> frontier;
  AddFusionCandidates(fusion, fusion, FusionDirection::kBoth, frontier);

  // BFS and fuse as many neighbors as possible.
  while (!frontier.empty()) {
    auto [instr, dir] = frontier.front();
    frontier.pop();
    if (dir != FusionDirection::kUp && dir != FusionDirection::kDown) {
      return InvalidArgument("Invalid travel direction: %s",
                             FusionDirectionToString(dir));
    }

    // If `instr` is another fusion of the same library type, fuse it.
    // We don't need to add its neighbors to the frontier because anything that
    // can be fused would have already been fused into `instr`.
    if (IsCustomFusionWithKind(instr, lib->fusion_kind())) {
      TF_ASSIGN_OR_RETURN(fusion,
                          MergeFusionInstructions(
                              fusion, Cast<HloFusionInstruction>(instr), dir));
      continue;
    }

    // Skip this instruction if it can't be fused.
    TF_ASSIGN_OR_RETURN(bool op_supported, lib->IsOpSupported(instr));
    if (!op_supported) {
      VLOG(4) << "  Skipping unsupported instruction: " << instr->ToString();
      continue;
    }

    // Add neighbors to the frontier.
    fused_.insert(instr);
    AddFusionCandidates(fusion, instr, dir, frontier);

    // Fuse `instr` into `fusion` according to the travel direction.
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        GrowFusion(fusion, instr, dir));
    TF_RETURN_IF_ERROR(
        InsertConvertIfNecessary(new_instr, lib->LibraryOpOutputType(instr)));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> LibraryRewriter::ProcessComputation(
    HloComputation* computation) {
  // Construct a list of instructions that can start a library fusion, starting
  // from the root up to the top. Prioritize dot and reduce ops over
  // element-wise ops.
  // TODO(penporn): Use priority queue when we have a cost model.
  std::vector<HloInstruction*> fusion_starters;
  std::vector<HloInstruction*> eltwise_ops;
  auto instructions = computation->MakeInstructionPostOrder();
  for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
    if (fuse_dot_ && (*it)->opcode() == HloOpcode::kDot) {
      fusion_starters.push_back(*it);
    } else if (fuse_reduce_ && ((*it)->opcode() == HloOpcode::kReduce ||
                                (*it)->opcode() == HloOpcode::kReduceWindow)) {
      fusion_starters.push_back(*it);
    } else if (fuse_eltwise_ && IsElementwiseAndNotConstant(*it)) {
      eltwise_ops.push_back(*it);
    }
  }
  fusion_starters.insert(fusion_starters.end(), eltwise_ops.begin(),
                         eltwise_ops.end());

  // Grow the fusion around each to-fuse ops.
  fused_.clear();
  for (HloInstruction* centroid : fusion_starters) {
    // Skip this instruction if it has already been fused into some fusion.
    if (fused_.contains(centroid)) {
      continue;
    }

    // Find the best library to use for the current instruction.
    TF_ASSIGN_OR_RETURN(LibraryMatcher * lib, ChooseLibrary(centroid));
    if (lib == nullptr) {
      continue;
    }

    // Start a fusion node.
    fused_.insert(centroid);
    VLOG(3) << "Starting a fusion with: " << centroid->ToString();
    TF_ASSIGN_OR_RETURN(HloFusionInstruction * fusion,
                        CreateLibraryFusion(centroid, lib->fusion_prefix(),
                                            lib->fusion_kind()));
    TF_RETURN_IF_ERROR(InsertConvertIfNecessary(
        fusion->fused_expression_root(), lib->LibraryOpOutputType(centroid)));

    // Fuse as many neighbors as as we can.
    TF_RETURN_IF_ERROR(FuseNeighbors(fusion, lib));
  }
  return !fused_.empty();
}

absl::StatusOr<bool> LibraryRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool module_changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Skip computations of reductions.
    if (absl::c_any_of(computation->caller_instructions(),
                       [](const HloInstruction* caller) {
                         return caller->has_to_apply() &&
                                caller->opcode() != HloOpcode::kCall;
                       })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool comp_changed, ProcessComputation(computation));
    module_changed |= comp_changed;
  }
  return module_changed;
}

}  // namespace xla::cpu
