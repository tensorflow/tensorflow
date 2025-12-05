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

#include "xla/hlo/transforms/simplifiers/dead_dynamic_update_slice_elimination.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

std::optional<int64_t> GetConstantAsInt64(const HloInstruction* inst) {
  if (!inst->IsConstant() || !ShapeUtil::IsScalar(inst->shape())) {
    return std::nullopt;
  }
  return primitive_util::PrimitiveTypeSwitch<std::optional<int64_t>>(
      [&](auto primitive_type_constant) -> std::optional<int64_t> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return static_cast<int64_t>(
              inst->literal().GetFirstElement<NativeT>());
        }
        return std::nullopt;
      },
      inst->shape().element_type());
}

std::optional<std::vector<int64_t>> GetStartIndices(const HloInstruction* dus) {
  absl::Span<HloInstruction* const> start_indices_operands =
      absl::MakeSpan(dus->operands())
          .subspan(xla::Cast<HloDynamicUpdateSliceInstruction>(dus)
                       ->first_index_operand_number());
  std::vector<int64_t> start_indices;
  for (HloInstruction* operand : start_indices_operands) {
    std::optional<int64_t> start_index = GetConstantAsInt64(operand);
    if (!start_index.has_value()) {
      return std::nullopt;
    }
    start_indices.push_back(*start_index);
  }
  return start_indices;
}

// Checks if the ranges [start1, end1) and [start2, end2) overlap.
//
// Example:
// RangesOverlap(0, 10, 5, 15) -> true
// RangesOverlap(0, 10, 10, 20) -> false
// RangesOverlap(0, 10, 15, 20) -> false
bool RangesOverlap(int64_t start1, int64_t end1, int64_t start2, int64_t end2) {
  return start1 < end2 && start2 < end1;
}

// If true, the updated elements of the dynamic-update-slice is not accessed
// by the slice user.
bool IsDusUpdateUnused(const std::vector<int64_t>& dus_starts,
                       const Shape& update_shape,
                       const HloInstruction* slice_user) {
  if (slice_user->opcode() != HloOpcode::kSlice) {
    return false;
  }
  // Get Slice ranges
  const std::vector<int64_t>& slice_starts = slice_user->slice_starts();
  const std::vector<int64_t>& slice_limits = slice_user->slice_limits();

  // The slice accesses the updated part IFF there is an overlap in *ALL*
  // dimensions. If there is no overlap in any dimension, the slice is safe,
  // i.e., it doesn't access the updated elements.
  for (int dim = 0; dim < update_shape.dimensions().size(); ++dim) {
    int64_t dus_start = dus_starts[dim];
    int64_t dus_limit = dus_start + update_shape.dimensions(dim);
    int64_t slice_start = slice_starts[dim];
    int64_t slice_limit = slice_limits[dim];
    if (RangesOverlap(dus_start, dus_limit, slice_start, slice_limit)) {
      continue;
    }
    // Disjoint in this dimension, so slice does not overlap with update.
    return true;
  }
  // Overlap in all dimensions, so slice reads updated values.
  return false;
}

// Helper function to process a single DynamicUpdateSlice instruction.
// Returns true if the module was changed.
absl::StatusOr<bool> ProcessDynamicUpdateSlice(HloInstruction* dus,
                                               HloComputation* comp) {
  const std::optional<std::vector<int64_t>> dus_starts = GetStartIndices(dus);
  if (!dus_starts.has_value()) {
    // Not a constant start index, cannot simplify.
    return false;
  }
  const std::vector<int64_t>& dus_starts_vec = *dus_starts;
  HloInstruction* update_operand = dus->mutable_operand(1);
  if (dus_starts_vec.size() != update_operand->shape().dimensions().size()) {
    // DUS start indices size does not match update operand shape dimensions
    // size.
    VLOG(1) << "DUS start indices size does not match update operand shape "
               "dimensions size: "
            << dus->ToString();
    return false;
  }

  bool is_dus_update_unused =
      dus->user_count() > 0 &&
      absl::c_all_of(dus->users(), [&](HloInstruction* user) {
        return IsDusUpdateUnused(dus_starts_vec, update_operand->shape(), user);
      });
  VLOG(2) << "  is_dus_update_unused: " << is_dus_update_unused;
  if (is_dus_update_unused) {
    TF_RETURN_IF_ERROR(dus->ReplaceAllUsesWith(dus->mutable_operand(0)));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(dus));
    return true;  // Changed
  }
  return false;  // Not changed
}

}  // namespace

absl::StatusOr<bool> DeadDynamicUpdateSliceElimination::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto computations_range = module->computations(execution_threads);
  std::vector<HloComputation*> computations(computations_range.begin(),
                                            computations_range.end());
  for (HloComputation* computation : computations) {
    std::vector<HloInstruction*> post_order_instructions =
        computation->MakeInstructionPostOrder();
    for (auto it = post_order_instructions.rbegin();
         it != post_order_instructions.rend(); ++it) {
      HloInstruction* instruction = *it;
      if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
        continue;
      }
      VLOG(2) << "Processing DUS: " << instruction->ToString();
      TF_ASSIGN_OR_RETURN(bool dus_changed,
                          ProcessDynamicUpdateSlice(instruction, computation));
      if (dus_changed) {
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
