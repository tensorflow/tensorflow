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

#include "xla/backends/gpu/transforms/hoist_fused_bitcasts.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "xla/backends/gpu/transforms/bitcast_utils.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// This is meant to only be run on fusions meant for Triton. Skip cuDNN fusions.
bool ShouldRunOnFusion(const HloInstruction& fusion) {
  auto backend_config = fusion.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return true;
  }
  const FusionBackendConfig& fusion_backend_config =
      backend_config.value().fusion_backend_config();
  if (fusion_backend_config.kind() == "__cudnn$fusion") {
    VLOG(2) << "Skipping cuDNN fusion";
    return false;
  }
  return true;
}

using HloInstructionSetVector =
    llvm::SetVector<HloInstruction*, std::vector<HloInstruction*>,
                    HloInstructionSet>;

// Returns the set of instructions that are reachable from 'instruction' using
// the given accessor.
template <typename T>
HloInstructionSetVector GetTransitiveInstructionSet(
    const HloInstruction* instruction, T (HloInstruction::*get)() const) {
  std::deque<HloInstruction*> worklist;
  auto append = [&](const auto& instructions) {
    worklist.insert(worklist.end(), instructions.begin(), instructions.end());
  };
  append((instruction->*get)());
  HloInstructionSetVector result;
  while (!worklist.empty()) {
    HloInstruction* front = worklist.front();
    worklist.pop_front();
    if (result.insert(front)) {
      append((front->*get)());
    }
  }
  return result;
}

// Returns the set of producers reachable from 'instruction' in use-before-def
// order.
HloInstructionSetVector GetProducerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::operands);
}
// Returns the set of consumers reachable from 'instruction' in def-before-use
// order.
HloInstructionSetVector GetConsumerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::users);
}

// Verifies that the set of instructions is closed under the given accessor,
// i.e. that the set of instructions reachable through the given accessor are
// either in the set itself or the root.
template <typename T>
absl::Status VerifyIsClosedInstructionSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root,
    T (HloInstruction::*get)() const) {
  for (HloInstruction* instruction : instructions) {
    for (HloInstruction* reachable : (instruction->*get)()) {
      if (reachable != root && instructions.count(reachable) == 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Instruction ", reachable->ToString(),
                         " is reachable from ", instruction->ToString(),
                         ", which is not in the recursive set of, or ",
                         root->ToString(), " itself."));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyIsClosedProducerSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::users);
}
}  // namespace

namespace {

// Simulates a rewrite of all producers of a given bitcast/reshape, moving the
// instruction outside of the computation. Returns the new shapes of affected
// instructions in order of traversal from consumers to producers.
absl::StatusOr<std::vector<std::pair<HloInstruction*, Shape>>>
PlanHoistBitcastUpwardsToCallers(const HloInstruction* bitcast) {
  // Check that all producers only affect the bitcast. If there are any
  // other consumers: refuse the hoisting.
  // It is possible to support more cases by sinking the bitcast from such
  // producers downward.
  HloInstructionSetVector producers = GetProducerSet(bitcast);
  RETURN_IF_ERROR(VerifyIsClosedProducerSet(producers, bitcast));
  if (bitcast->shape().element_type() !=
      bitcast->operand(0)->shape().element_type()) {
    return absl::UnimplementedError(
        absl::StrCat("Hoisting bitcast with type conversion is not supported: ",
                     bitcast->ToString()));
  }

  HloInstructionMap<Shape> result_shapes;
  auto set_result_shape =
      [&](const absl::Span<HloInstruction* const> instructions,
          const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      // Only update the dimensions keeping the type intact.
      Shape new_shape(shape);
      CopyElementType(instruction->shape(), &new_shape);
      CHECK_EQ(ShapeUtil::ArrayDataSize(new_shape),
               ShapeUtil::ArrayDataSize(instruction->shape()))
          << " instruction " << instruction->ToString()
          << " updating result shape from "
          << ShapeUtil::HumanStringWithLayout(instruction->shape()) << " to "
          << ShapeUtil::HumanStringWithLayout(new_shape)
          << " with different data size";
      auto it = result_shapes.find(instruction);
      if (it == result_shapes.end()) {
        VLOG(2) << "updating the result shape of " << instruction->ToString()
                << " to " << ShapeUtil::HumanStringWithLayout(new_shape);
        result_shapes.emplace(instruction, new_shape);
      } else if (it->second != new_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(set_result_shape(bitcast->operands(), bitcast->shape()));

  std::vector<std::pair<HloInstruction*, Shape>> result;
  // We want to visit instructions in order from consumers to producers: we
  // hoist the bitcast upwards and having a valid HLO at every rewrite step
  // helps a lot. A simple DFS or BFS over operands will not work in non-tree
  // situations when there are multiple consumers of the same producer. Instead
  // of writing a custom traversal we can simply walk the post-order (producers
  // before consumers) list backward and only update the instructions affected.
  // TODO(b/393299275): use MakeInstructionPostOrderFrom(bitcast) - that should
  // be slightly more efficient.
  auto def_before_use = bitcast->parent()->MakeInstructionPostOrder();
  for (HloInstruction* instruction :
       llvm::make_range(def_before_use.rbegin(), def_before_use.rend())) {
    auto it = result_shapes.find(instruction);
    if (it == result_shapes.end()) {
      continue;  // Not affected.
    }
    Shape& result_shape = it->second;
    if (instruction->shape() == result_shape) {
      continue;  // No change.
    }
    result.emplace_back(instruction, result_shape);
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
        // No operands.
        break;
      case HloOpcode::kReshape:  // Reshape is a bitcast.
      case HloOpcode::kBitcast:
        // Other bitcast will be hoisted separately so we don't need to
        // update its operand.
        break;
      case HloOpcode::kBroadcast: {
        ASSIGN_OR_RETURN(
            BitcastParams params,
            CalculateBitcastOfBroadcast(
                Cast<HloBroadcastInstruction>(instruction), result_shape));
        RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), params.new_shape));
        break;
      }
      case HloOpcode::kTranspose: {
        ASSIGN_OR_RETURN(
            BitcastParams params,
            CalculateBitcastOfTranspose(
                Cast<HloTransposeInstruction>(instruction), result_shape));
        RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), params.new_shape));
        break;
      }
      default:
        if (!instruction->IsElementwise()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast past ", instruction->ToString()));
        }
        RETURN_IF_ERROR(
            set_result_shape(instruction->operands(), result_shape));
        break;
    }
  }
  return result;
}

// Returns the shape of the root instruction after hoisting bitcasts away from
// the dot instruction. If traversal encounters an instruction we cannot hoist
// bitcasts past we try to sink the bitcast starting from that instruction.
//
// For example, given:
//
// dot = dot_shape dot
// bitcast = bitcast(dot)
// ROOT root = transpose(bitcast)
//
// Returns root_shape for:
//
// dot = dot_shape dot
// ROOT root = roots_shape transpose(dot)
//
absl::StatusOr<Shape> ComputeRootShapeAfterHoistingBitcasts(
    const HloInstruction* dot) {
  if (dot->IsRoot()) {
    return dot->shape();
  }

  HloInstructionMap<Shape> operand_shapes;
  auto set_operand_shape =
      [&](const absl::Span<HloInstruction* const> instructions,
          const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      // Only update the dimensions keeping the type intact.
      Shape new_shape(shape);
      const HloInstruction* operand = instruction->operand(0);
      CopyElementType(operand->shape(), &new_shape);
      CHECK_EQ(ShapeUtil::ArrayDataSize(new_shape),
               ShapeUtil::ArrayDataSize(operand->shape()))
          << " instruction " << instruction->ToString()
          << " updating operand shape from "
          << ShapeUtil::HumanStringWithLayout(operand->shape()) << " to "
          << ShapeUtil::HumanStringWithLayout(new_shape)
          << " with different data size";
      auto it = operand_shapes.find(instruction);
      if (it == operand_shapes.end()) {
        VLOG(2) << "updating the operand shape of "
                << instruction->ToString(
                       HloPrintOptions().set_print_operand_shape(true))
                << " to " << ShapeUtil::HumanStringWithLayout(new_shape);
        operand_shapes.emplace(instruction, new_shape);
      } else if (it->second != new_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(set_operand_shape(dot->users(), dot->shape()));

  for (HloInstruction* instruction : GetConsumerSet(dot)) {
    auto it = operand_shapes.find(instruction);
    if (it == operand_shapes.end()) {
      continue;  // Not affected.
    }
    Shape& operand_shape = it->second;
    ASSIGN_OR_RETURN(Shape result_shape, [&]() -> absl::StatusOr<Shape> {
      switch (instruction->opcode()) {
        case HloOpcode::kBroadcast: {
          auto paramsOr = CalculateBroadcastOfBitcast(
              Cast<HloBroadcastInstruction>(instruction), operand_shape);
          if (paramsOr.ok()) {
            return paramsOr->new_shape;
          }
          VLOG(2) << "Failed to calculate broadcast of bitcast: "
                  << paramsOr.status();
          return instruction->shape();
        }
        case HloOpcode::kTranspose: {
          auto paramsOr = CalculateTransposeOfBitcast(
              Cast<HloTransposeInstruction>(instruction), operand_shape);
          if (paramsOr.ok()) {
            return paramsOr->new_shape;
          }
          VLOG(2) << "Failed to calculate transpose of bitcast: "
                  << paramsOr.status();
          return instruction->shape();
        }
        case HloOpcode::kReshape:
        case HloOpcode::kBitcast:
          return operand_shape;
        default:
          if (instruction->IsElementwise()) {
            return operand_shape;
          }
          // TODO(b/467421789): we can probably allow sinking from this op down.
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast past ", instruction->ToString()));
      }
    }());
    if (instruction->IsRoot()) {
      CopyElementType(instruction->shape(), &result_shape);
      return result_shape;
    }
    RETURN_IF_ERROR(set_operand_shape(instruction->users(), result_shape));
  }
  return absl::InternalError("No root found");
}

// Hoists the given 'bitcast' upwards out of its computation, to the parent of
// each caller.
absl::Status HoistBitcastUpwardsToCallers(HloInstruction* bitcast,
                                          absl::Span<HloInstruction*> callers) {
  ASSIGN_OR_RETURN(auto rewrite_plan,
                   PlanHoistBitcastUpwardsToCallers(bitcast));
  for (auto [instruction, result_shape] : rewrite_plan) {
    VLOG(2) << absl::StrCat("rewriting result shape of ",
                            instruction->ToString(), " to ",
                            ShapeUtil::HumanStringWithLayout(result_shape));
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        // Create a new bitcast in callers.
        int64_t number = instruction->parameter_number();
        for (HloInstruction* caller : callers) {
          // Create a more generic `bitcast` even if the caller has a
          // `reshape`.
          HloInstruction* new_bitcast =
              caller->AddInstruction(HloInstruction::CreateBitcast(
                  result_shape, caller->mutable_operand(number)));
          RETURN_IF_ERROR(
              caller->ReplaceOperandWithDifferentShape(number, new_bitcast));
        }
        break;
      }
      case HloOpcode::kBroadcast: {
        auto* broadcast = Cast<HloBroadcastInstruction>(instruction);
        auto params = CalculateBitcastOfBroadcast(broadcast, result_shape);
        // Must be OK, already succeeded in PlanHoistBitcasUpwardsToCallers.
        QCHECK_OK(params);
        broadcast->mutable_dimensions()->assign(params->new_dims.begin(),
                                                params->new_dims.end());
        break;
      }
      case HloOpcode::kTranspose: {
        auto* transpose = Cast<HloTransposeInstruction>(instruction);
        auto params = CalculateBitcastOfTranspose(transpose, result_shape);
        // Must be OK, already succeeded in PlanHoistBitcastUpwardsToCallers.
        QCHECK_OK(params);
        transpose->mutable_dimensions()->assign(params->new_dims.begin(),
                                                params->new_dims.end());
        break;
      }
      default:
        break;
    }
    *instruction->mutable_shape() = result_shape;
    // Sharding is not necessary here and changing the shape can cause an
    // HloVerifier error.
    instruction->clear_sharding();
  }
  RETURN_IF_ERROR(bitcast->ReplaceAllUsesWith(bitcast->mutable_operand(0)));
  RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Inserts a bitcast at the root if the root shape is different from the dot
// shape. The bitcast is chosen so that it cancels out bitcasts and reshapes
// along the way up to the dot. Updates the callers of the dot to expect the new
// root shape.
absl::StatusOr<bool> MaybeInsertRootBitcast(
    HloInstruction* dot, absl::Span<HloInstruction*> callers) {
  ASSIGN_OR_RETURN(Shape root_shape,
                   ComputeRootShapeAfterHoistingBitcasts(dot));

  HloComputation* computation = dot->parent();
  HloInstruction* root = computation->root_instruction();
  if (root->shape() == root_shape) {
    return false;
  }

  // Insert a new bitcast at the root.
  computation->set_root_instruction(
      root->AddInstruction(HloInstruction::CreateBitcast(root_shape, root)));

  // Insert new bitcast for each caller's result.
  for (HloInstruction* caller : callers) {
    HloInstruction* new_bitcast = caller->AddInstruction(
        HloInstruction::CreateBitcast(caller->shape(), caller));
    RETURN_IF_ERROR(caller->ReplaceAllUsesWith(new_bitcast));
    *caller->mutable_shape() = root_shape;
  }

  return true;
}

// Try hoisting bitcasts and reshapes in the computation away from 'dot' to the
// callers of the computation. Some bitcasts or reshapes may remain in the
// computation, because they cannot be hoisted across all ops, e.g. across some
// transposes and broadcasts. This is not reported as an error.
absl::StatusOr<bool> TryHoistBitcastsInComputationToCallers(
    HloInstruction* dot, CallGraph* call_graph) {
  bool changed = false;
  // Instead of implementing a logic to hoist bitcast upwards and downwards
  // we insert a bitcast at the root that and always hoist bitcasts upwards.
  // That significantly simplifies the implementation.
  VLOG(2) << "Before hoisting bitcasts: " << dot->parent()->ToString();

  auto callers = call_graph->GetComputationCallers(dot->parent());
  absl::StatusOr<bool> inserted =
      MaybeInsertRootBitcast(dot, absl::MakeSpan(callers));
  if (!inserted.ok()) {
    VLOG(2) << "Failed to insert root bitcast: " << inserted.status();
  } else {
    changed |= *inserted;
  }
  VLOG(2) << "After inserting root bitcast: " << dot->parent()->ToString();

  auto def_before_use = dot->parent()->MakeInstructionPostOrder();
  for (HloInstruction* instruction :
       llvm::make_range(def_before_use.rbegin(), def_before_use.rend())) {
    if (!HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape>(
            instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast upwards " << instruction->ToString();
    auto status =
        HoistBitcastUpwardsToCallers(instruction, absl::MakeSpan(callers));
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist " << instruction->ToString()
              << " upwards: " << status;
    } else {
      changed = true;
    }
  }

  VLOG(2) << "After hoisting bitcasts: " << dot->parent()->ToString();
  return changed;
}

class HoistFusedBitcastsVisitor : public DfsHloRewriteVisitor {
 public:
  explicit HoistFusedBitcastsVisitor(CallGraph* call_graph)
      : call_graph_(call_graph) {}

 private:
  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             CallGraph* call_graph) {
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        return absl::InternalError(absl::StrCat("Computation of fusion ",
                                                fusion->ToString(),
                                                " has no dot instruction"));
      }
    }

    ASSIGN_OR_RETURN(bool changed,
                     TryHoistBitcastsInComputationToCallers(instr, call_graph));
    if (changed) {
      MarkAsChanged();
    }
    return absl::OkStatus();
  }

  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);

    // Check if we target this fusion.
    if (!ShouldRunOnFusion(*fusion)) {
      return absl::OkStatus();
    }
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        VLOG(2) << "Skipping fusion as it has no dot instruction";
        return absl::OkStatus();
      }
    }
    return RewriteFusion(fusion, call_graph_);
  }

 private:
  CallGraph* call_graph_;
};

}  // namespace

absl::StatusOr<bool> HoistFusedBitcasts::RunOnModule(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HoistFusedBitcastsVisitor visitor(call_graph.get());
    RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

absl::StatusOr<bool> HoistFusedBitcasts::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().debug_options().xla_gpu_experimental_gemm_fusion_v2()) {
    VLOG(2) << "Skipping HoistFusedBitcasts because "
               "xla_gpu_experimental_gemm_fusion_v2 is set.";
    return false;
  }
  return RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu
