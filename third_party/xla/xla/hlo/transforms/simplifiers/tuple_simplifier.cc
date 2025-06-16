/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

TupleSimplifier::TupleSimplifier(bool exclude_entry_computation)
    : exclude_entry_computation_(exclude_entry_computation) {}

absl::StatusOr<bool> TupleSimplifier::RemoveWholeTuple(HloInstruction* tuple) {
  HloInstruction* top_tuple = nullptr;
  for (int64_t operand_number = 0; operand_number < tuple->operand_count();
       ++operand_number) {
    HloInstruction* operand = tuple->mutable_operand(operand_number);
    if (operand->opcode() != HloOpcode::kGetTupleElement ||
        operand->tuple_index() != operand_number) {
      return false;
    }
    if (top_tuple == nullptr) {
      top_tuple = operand->mutable_operand(0);
      if (!ShapeUtil::Compatible(top_tuple->shape(), tuple->shape())) {
        return false;
      }
    } else if (top_tuple != operand->operand(0)) {
      return false;
    }
  }
  if (top_tuple == nullptr) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool changed,
                      tuple->parent()->ReplaceInstruction(
                          tuple, top_tuple, /*preserve_sharding=*/true));
  return changed;
}

namespace {
HloInstruction* ToFlatTuple(HloInstruction* instr) {
  if (!ShapeUtil::IsNestedTuple(instr->shape())) {
    return instr;
  }
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(ShapeUtil::GetLeafCount(instr->shape()));
  auto element_tree = TupleUtil::DisassembleTupleInstruction(instr);
  for (const auto& leaf : element_tree.leaves()) {
    tuple_elements.push_back(leaf.second);
  }
  return instr->parent()->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}

absl::Status FlattenAndUpdateUses(HloInstruction* instr) {
  if (!ShapeUtil::IsNestedTuple(instr->shape())) {
    return absl::OkStatus();
  }
  // Take a snapshot of all users before inserting get-tuple-element's.
  std::vector<HloInstruction*> actual_users(instr->users().begin(),
                                            instr->users().end());
  int32_t leaf_index = 0;
  ShapeTree<HloInstruction*> tuple_elements(instr->shape());
  for (auto& leaf : tuple_elements.leaves()) {
    leaf.second =
        instr->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(instr->shape(), leaf.first), instr,
            leaf_index++));
  }
  TF_RETURN_IF_ERROR(instr->ReplaceUsesWith(
      actual_users,
      TupleUtil::AssembleTupleInstruction(instr->parent(), tuple_elements)));
  *instr->mutable_shape() = ShapeUtil::ToFlatTupleShape(instr->shape());
  return absl::OkStatus();
}

bool IsAsyncWrappedOrCalledComputation(HloComputation* computation) {
  bool is_async_called_computation =
      computation->caller_instructions().size() == 1 &&
      computation->caller_instructions()
          .front()
          ->parent()
          ->IsAsyncComputation();
  return computation->IsAsyncComputation() || is_async_called_computation;
}
}  // namespace

absl::StatusOr<bool> TupleSimplifier::FlattenRootTuple(
    HloComputation* computation) {
  if (!ShapeUtil::IsNestedTuple(computation->root_instruction()->shape())) {
    VLOG(4) << "Skipping because root instruction is not a nested tuple: "
            << computation->root_instruction()->ToString();
    return false;
  }
  if (!IsAsyncWrappedOrCalledComputation(computation)) {
    VLOG(4) << "Skipping because computation is not async wrapped or called: "
            << computation->name();
    return false;
  }
  // Replace the root instruction with the flattened tuple.
  computation->set_root_instruction(
      ToFlatTuple(computation->root_instruction()),
      /*accept_different_shape=*/true);
  return true;
}

absl::Status TupleSimplifier::FlattenCaller(HloInstruction* caller) {
  switch (caller->opcode()) {
    case HloOpcode::kAsyncStart: {
      // async-start returns a tuple of (arguments, result, sflag).
      constexpr int64_t kResultIndex = 1;
      Shape flattened_shape = ShapeUtil::ToFlatTupleShape(
          caller->shape().tuple_shapes(kResultIndex));
      // Update all async ops on the same chain.
      for (auto* async : Cast<HloAsyncInstruction>(caller)->GetAsyncChain()) {
        if (async->opcode() == HloOpcode::kAsyncDone) {
          TF_RETURN_IF_ERROR(FlattenAndUpdateUses(async));
          continue;
        }
        *async->mutable_shape()->mutable_tuple_shapes(kResultIndex) =
            flattened_shape;
      }
      return absl::OkStatus();
    }
    case HloOpcode::kCall: {
      TF_RETURN_IF_ERROR(FlattenAndUpdateUses(caller));
      return absl::OkStatus();
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Did not expect ", HloOpcodeString(caller->opcode()),
          " async-wrapped computation to produce a nested tuple."));
  }
}

absl::StatusOr<bool> TupleSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  XLA_VLOG_LINES(5, "Before tuple simplification:\n" + module->ToString());
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (exclude_entry_computation_ &&
        computation == module->entry_computation()) {
      continue;
    }
    // If the computation returns a nested tuple, flatten it and update the
    // computation's callers.
    TF_ASSIGN_OR_RETURN(bool flattened, FlattenRootTuple(computation));
    if (flattened) {
      for (auto* caller : computation->caller_instructions()) {
        TF_RETURN_IF_ERROR(FlattenCaller(caller));
      }
      changed = true;
    }
    // Initially add all GTE and Tuple instructions to the worklist.
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTuple) {
        TF_ASSIGN_OR_RETURN(bool c, RemoveWholeTuple(instruction));
        changed |= c;
      } else {
        auto [ancestor, index] = instruction->LatestNonGteAncestorAndIndex();
        if (ancestor == instruction) {
          continue;
        }
        // If possible replace a chain of GTE with the operation which produces
        // the element. For example, replace uses of GTE with below with just
        // 'Op' (assuming 'Op' is at the index of the GTE instruction):
        //
        //     ...  Op ...
        //       \  |   /
        //        Tuple
        //          |
        //         GTE
        //         ...
        //          |
        //         GTE
        //          |
        //         GTE
        //
        // Note that this deletes the Tuple instruction altogether. In addition,
        // if only a subset of tuple's elements are used, this transform
        // optimizes them one at a time, and after the last use is optimized,
        // the Tuple will also be deleted.
        HloInstruction* replacement = ancestor;
        for (int i = 0; i < index.size(); ++i) {
          if (replacement->opcode() != HloOpcode::kTuple) {
            replacement = nullptr;
            break;
          }
          replacement = replacement->mutable_operand(index[i]);
        }

        if (replacement) {
          TF_ASSIGN_OR_RETURN(bool replaced,
                              computation->ReplaceInstruction(
                                  instruction, replacement,
                                  /*preserve_sharding=*/true,
                                  /*relay_control_dependency=*/true));
          changed |= replaced;
        }
      }
    }
  }

  if (module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }
  XLA_VLOG_LINES(5, "After tuple simplification:\n" + module->ToString());

  return changed;
}

}  // namespace xla
