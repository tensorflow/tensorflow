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

#include "xla/service/gpu/dynamic_slicing_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"

namespace xla::gpu {

namespace {
// A dataflow path flowing from a definition to a user.
using DefUseDataflowPath = absl::InlinedVector<HloInstruction*, 2>;

// All dataflow paths flowing from a definition to all users. Each user will
// have a separate entry in the vector.
using DefUseDataflowPaths = absl::InlinedVector<DefUseDataflowPath, 4>;

// A dataflow path flowing from a user to a definition.
using UseDefDataflowPath = absl::InlinedVector<HloInstruction*, 4>;

// All dataflow paths flowing from a user to all definitions of its operands.
using UseDefDataflowPaths = absl::InlinedVector<HloInstruction*, 8>;

using DataflowPathView = absl::Span<HloInstruction* const>;
using DataflowPathsView = absl::Span<DataflowPathView>;

using InstructionSet = absl::flat_hash_set<HloInstruction*>;

bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
}

// Returns true if the slice is 128-byte-aligned. The slice starting
// address is determined by the product of all non-sliced dimensions and an
// offset defined by `slice_starts` of the slice op.
//
// For dynamic cases, we don't have info about the start indices, so we have to
// be conservative by only accepting sliced shapes that have the product of all
// non-sliced dimensions being a multiple of `kXlaAllocatedBufferAlignBytes`.
bool IsAlignedSlice(const HloInstruction* slice) {
  DCHECK((HloPredicateIsOp<HloOpcode::kSlice, HloOpcode::kDynamicSlice,
                           HloOpcode::kDynamicUpdateSlice>(slice)))
      << "Unknown slice operation: " << slice->ToString();

  if (!IsContiguousSlice(*slice)) return false;

  auto [full_shape, slice_shape] = [&] {
    if (auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(slice)) {
      return std::make_pair(dus->shape(), dus->update()->shape());
    }
    return std::make_pair(slice->operand(0)->shape(), slice->shape());
  }();

  auto strides = ShapeUtil::ByteStrides(slice_shape);
  if (!strides.has_value()) return false;

  for (auto dim : slice_shape.layout().minor_to_major()) {
    if ((strides.value()[dim] % kXlaAllocatedBufferAlignBytes) == 0) {
      return true;
    }
    if (slice_shape.dimensions(dim) < full_shape.dimensions(dim)) {
      return (HloPredicateIsOp<HloOpcode::kSlice>(slice) &&
              (((*strides)[dim] * slice->slice_starts(dim)) %
                   kXlaAllocatedBufferAlignBytes ==
               0));
    }
  }
  return true;
}

// Returns true if the `consumer` only depends on the `producer` and no other
// instructions. This is a recursive function checking all paths from the
// `consumer` to the parameters of the computation and if there is any path
// without `producer`, then it returns false.
bool IsOnlyDependentOn(const HloInstruction* consumer,
                       HloInstruction* producer) {
  if (consumer == producer ||
      HloPredicateIsOp<HloOpcode::kConstant>(consumer)) {
    return true;
  }
  if (consumer->operand_count() == 0) {
    return false;
  }
  return absl::c_all_of(consumer->operands(),
                        [producer](const HloInstruction* operand) {
                          return IsOnlyDependentOn(operand, producer);
                        });
};

// Returns true if the value is a function of the induction variable within a
// while loop.
bool IsValueFunctionOfLoopInductionVariable(const HloInstruction& value,
                                            const CallGraph* call_graph) {
  std::vector<HloInstruction*> callers =
      call_graph->GetComputationCallers(value.parent());
  if (callers.size() != 1) {
    VLOG(2) << "Computation has multiple callers: "
            << absl::StrJoin(callers, ",",
                             [](std::string* out, const HloInstruction* instr) {
                               out->append(instr->name());
                             });
    return false;
  }
  HloInstruction* while_op = callers[0];
  if (HloPredicateIsNotOp<HloOpcode::kWhile>(while_op)) {
    VLOG(2) << "Computation caller is not while, it is "
            << while_op->ToString();
    return false;
  }
  HloComputation* while_body = while_op->while_body();
  std::optional<int64_t> loop_induction_variable_tuple_idx =
      GetLoopInductionVarTupleIdx(while_op);
  if (!loop_induction_variable_tuple_idx.has_value()) {
    VLOG(2) << "Induction variable tuple index is nullopt";
    return false;
  }
  // The verifier makes sure that there is exactly one parameter. So, it is okay
  // to directly access the parameter here. The function
  // `GetLoopInductionVarTupleIdx` above makes sure that the parameter is a
  // tuple.
  HloInstruction* indvar = hlo_query::GetUniqueGteInstruction(
      while_body->parameter_instruction(0), *loop_induction_variable_tuple_idx);
  if (!indvar) {
    VLOG(2) << "Unable to find unique GTE for while induction variable idx: "
            << *loop_induction_variable_tuple_idx
            << ", while op: " << while_op->ToString();
    return false;
  }
  const HloInstruction* update = while_body->root_instruction()->operand(
      *loop_induction_variable_tuple_idx);
  const HloInstruction* indvar_init =
      while_op->operand(0)->operand(*loop_induction_variable_tuple_idx);

  // The `update` instruction and `value` should only depend on the induction
  // variable.
  return IsOnlyDependentOn(/*consumer=*/update, /*producer=*/indvar) &&
         IsOnlyDependentOn(/*consumer=*/&value, /*producer=*/indvar) &&
         IsOnlyDependentOn(indvar_init, nullptr);
}

// This returns true for the constants that are handled in the dynamic slice
// fusion runtime. These constants do not force a D2H copy and hence preserve
// the cuda graph.
bool IsHandledConstantForDynamicSliceFusion(const HloInstruction& offset) {
  if (auto* cst = DynCast<HloConstantInstruction>(&offset)) {
    switch (cst->shape().element_type()) {
      case PrimitiveType::S32:
      case PrimitiveType::S64:
      case PrimitiveType::U32:
      case PrimitiveType::U64:
        return true;
      default:
        return false;
    };
  }
  return false;
}

// This checks whether a dynamic index operation has all offsets that are either
// constant or loop iteration offsets.
bool HasConstantOrLoopIterationOffsets(const HloDynamicIndexInstruction& instr,
                                       const CallGraph* call_graph) {
  return absl::c_all_of(
      instr.index_operands(), [call_graph](const HloInstruction* offset) {
        return IsValueFunctionOfLoopInductionVariable(*offset, call_graph) ||
               IsHandledConstantForDynamicSliceFusion(*offset);
      });
}

}  // namespace

UseDefDataflowPaths GetSlicedOperandPaths(const HloInstruction& instr,
                                          const CallGraph& call_graph) {
  UseDefDataflowPaths sliced_operand_paths;

  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
      aliasing_pairs;
  if (HloPredicateIsOp<HloOpcode::kCustomCall>(&instr)) {
    aliasing_pairs =
        Cast<HloCustomCallInstruction>(&instr)->output_to_operand_aliasing();
  }
  absl::flat_hash_set<int64_t> aliased_operands;
  for (const auto& pair : aliasing_pairs) {
    aliased_operands.insert(pair.second.first);
  }

  for (const auto* operand : instr.operands()) {
    // output_to_operand_aliasing means the operand is to be materialized, which
    // is against the whole idea of address computation fusion. Skip this
    // operand.
    if (aliased_operands.contains(instr.operand_index(operand))) {
      continue;
    }
    UseDefDataflowPath maybe_sliced_operand_path;
    bool slice_found = false;
    // TODO: currently HloFindIf exits upon encountering the first node that
    // matches. This works well if each operand only has 1 data flow (i.e. only
    // flows through unary op). We might want to keep finding until the queue is
    // empty: if the operand is a tuple, it might have different data flows
    // (i.e. 1 for each element).
    auto maybe_slice_instr =
        HloBfsFindIf({operand}, [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the traversal.
          if (processed_instrs.contains(cur)) {
            return true;
          }

          maybe_sliced_operand_path.push_back(const_cast<HloInstruction*>(cur));

          if (IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(cur)) {
            if (IsAlignedSlice(cur)) {
              slice_found = true;
              return slice_found;
            }
          }

          return !IsNoOp(cur);
        });

    if (maybe_slice_instr == std::nullopt) {
      continue;
    }
    auto dynamic_index_operation =
        DynCast<HloDynamicIndexInstruction>(maybe_slice_instr.value());
    bool valid_slice_found =
        slice_found && ((dynamic_index_operation &&
                         HasConstantOrLoopIterationOffsets(
                             *dynamic_index_operation, &call_graph)) ||
                        (*maybe_slice_instr)->opcode() == HloOpcode::kSlice);
    if (valid_slice_found ||
        processed_instrs.contains(maybe_slice_instr.value())) {
      // Even in the case of stopping at a match that has been processed, we
      // still need to add instructions encountered in the sliced operand path
      // during the latest traversal.
      sliced_operand_paths.insert(sliced_operand_paths.end(),
                                  maybe_sliced_operand_path.rbegin(),
                                  maybe_sliced_operand_path.rend());
      processed_instrs.insert(maybe_sliced_operand_path.begin(),
                              maybe_sliced_operand_path.end());
    }
  }

  sliced_operand_paths.push_back(const_cast<HloInstruction*>(&instr));
  return sliced_operand_paths;
}

DefUseDataflowPaths GetSlicedUserPaths(const HloInstruction& instr,
                                       const CallGraph& call_graph) {
  DefUseDataflowPaths sliced_user_paths;
  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  auto traverse_hlo_and_collect = [&](HloInstruction* start) {
    DefUseDataflowPath maybe_sliced_user_path;
    bool dus_found = false;
    auto maybe_dus_instr = HloBfsFindIf(
        {start},
        [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the
          // traversal.
          if (processed_instrs.contains(cur)) {
            return true;
          }
          maybe_sliced_user_path.push_back(const_cast<HloInstruction*>(cur));
          if (const auto slice_instr =
                  DynCast<HloDynamicUpdateSliceInstruction>(cur)) {
            if (IsAlignedSlice(slice_instr)) {
              dus_found = true;
              return true;
            }
          }
          return cur->user_count() > 1 || !IsNoOp(cur);
        },
        /*visit_operands=*/false);
    if (maybe_dus_instr == std::nullopt) {
      return;
    }
    auto dynamic_index_operation =
        DynCast<HloDynamicIndexInstruction>(maybe_dus_instr.value());
    bool valid_dus_found = dus_found && dynamic_index_operation &&
                           HasConstantOrLoopIterationOffsets(
                               *dynamic_index_operation, &call_graph);
    if (valid_dus_found || processed_instrs.contains(maybe_dus_instr.value())) {
      // Even in the case of stopping at a match that has been processed, we
      // still need to add instructions encountered in the sliced user path
      // during the latest traversal.
      processed_instrs.insert(maybe_sliced_user_path.begin(),
                              maybe_sliced_user_path.end());
      sliced_user_paths.push_back(std::move(maybe_sliced_user_path));
    }
  };

  if (instr.shape().IsTuple()) {
    for (auto* user : instr.users()) {
      if (DynCast<HloGetTupleElementInstruction>(user)) {
        traverse_hlo_and_collect(user);
      }
    }
  } else {
    if (instr.user_count() == 1) {
      traverse_hlo_and_collect(instr.users().front());
    }
  }

  return sliced_user_paths;
}
}  // namespace xla::gpu
