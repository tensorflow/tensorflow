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

#include "xla/service/scatter_determinism_expander.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/scatter_utils.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Canonicalizes the scatter_updates in order to keep them uniform while
// performing the scatter operation.
static absl::StatusOr<std::vector<HloInstruction*>> CanonicalizeScatterUpdates(
    const std::vector<HloInstruction*>& scatter_updates,
    HloInstruction* scatter_indices, const ScatterDimensionNumbers& dim_numbers,
    int64_t scatter_loop_trip_count) {
  std::vector<HloInstruction*> adjusted_updates;
  adjusted_updates.reserve(scatter_updates.size());
  for (HloInstruction* update : scatter_updates) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * canonical_update,
        PermuteScatterAndWindowDims(update, dim_numbers.update_window_dims()));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * adjusted_update,
        AdjustScatterDims(scatter_indices->shape(), canonical_update,
                          dim_numbers.index_vector_dim()));
    CHECK_EQ(scatter_loop_trip_count, adjusted_update->shape().dimensions(0));
    adjusted_updates.push_back(adjusted_update);
  }
  return adjusted_updates;
}

// Create the out-of-bound tensor for the scatter operation.
HloInstruction* CreateOutOfBoundTensor(HloComputation* parent,
                                       HloInstruction* scatter_indices,
                                       const Shape& scatter_shape) {
  if (scatter_indices->shape().rank() == 1) {
    CHECK_EQ(scatter_shape.dimensions_size(), 1);
    Array<int32_t> out_of_bound_array({scatter_indices->shape().dimensions(0)},
                                      scatter_shape.dimensions(0));
    return parent->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromArray(out_of_bound_array)));
  }
  // More than one dimension in scatter_indices
  Array2D<int32_t> out_of_bound_array(scatter_indices->shape().dimensions(0),
                                      scatter_indices->shape().dimensions(1));
  for (int i = 0; i < scatter_indices->shape().dimensions(0); ++i) {
    for (int j = 0; j < scatter_indices->shape().dimensions(1); ++j) {
      out_of_bound_array(i, j) = scatter_shape.dimensions(j);
    }
  }
  return parent->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<int>(out_of_bound_array)));
}

// Computation for sorting the scalar scatter indices and updates together
HloComputation* ScalarSortingComparison(HloModule* module,
                                        const Shape key_shape,
                                        const Shape update_shape,
                                        int64_t num_updates) {
  HloComputation::Builder builder("sorting_computation");
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, key_shape, "lhs_key"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, key_shape, "rhs_key"));
  const int kExistingParams = 2;
  for (int i = 0; i < num_updates; ++i) {
    builder.AddInstruction(
        HloInstruction::CreateParameter(kExistingParams + i, update_shape,
                                        absl::StrFormat("lhs_update_%d", i)));
    builder.AddInstruction(
        HloInstruction::CreateParameter(kExistingParams + 1 + i, update_shape,
                                        absl::StrFormat("rhs_update_%d", i)));
  }
  builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                    param1, ComparisonDirection::kLt));
  return module->AddEmbeddedComputation(builder.Build());
}

static std::vector<HloInstruction*> SortIndicesAndUpdates(
    HloInstruction* scatter_indices,
    const std::vector<HloInstruction*>& scatter_updates, int64_t num_indices,
    HloScatterInstruction* scatter, HloComputation* parent) {
  const Shape& indices_shape = scatter_indices->shape();
  const Shape& updates_shape = scatter_updates[0]->shape();
  auto updates_dims = updates_shape.dimensions();
  // Since we canonicalized the scatter updates, the first dim will always be
  // the number of updates and the rest will be the shape of each update

  HloInstruction* scalar_indices = scatter_indices;

  std::vector<int64_t> single_update_dimensions(updates_dims.begin() + 1,
                                                updates_dims.end());

  const Shape update_shape = ShapeUtil::MakeShape(updates_shape.element_type(),
                                                  single_update_dimensions);

  const Shape& scalar_index_shape =
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices});

  auto* comparison = ScalarSortingComparison(
      scatter->GetModule(),
      ShapeUtil::MakeShape(indices_shape.element_type(), {}),
      ShapeUtil::MakeShape(updates_shape.element_type(), {}),
      scatter_updates.size());

  std::vector<HloInstruction*> sort_operands = {scalar_indices};
  std::vector<Shape> sort_shapes = {scalar_index_shape};
  for (auto update : scatter_updates) {
    sort_operands.push_back(update);
    sort_shapes.push_back(update->shape());
  }

  auto* sorting = parent->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape(sort_shapes), 0, sort_operands, comparison,
      /*is_stable=*/false));
  auto* sorted_scalar_indices =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          scalar_indices->shape(), sorting, 0));

  std::vector<HloInstruction*> sorted_updates(scatter_updates.size());
  for (int i = 0; i < scatter_updates.size(); i++) {
    sorted_updates[i] =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            scatter_updates[i]->shape(), sorting, i + 1));
  }
  std::vector<HloInstruction*> sorted_tensors = {sorted_scalar_indices};
  sorted_tensors.insert(sorted_tensors.end(), sorted_updates.begin(),
                        sorted_updates.end());
  return sorted_tensors;
}

// CreateScanWithIndices performs a prefix scan operation (akin to parallel
// prefix sum) on the updates and indices, to compute the accumulated updates in
// log(n) time.
//
// High-level algorithm:
//
// Iteration through log2(num_updates):
//   - For each iteration, the `updates` tensor will be sliced and padded to
//   perform shifting by `offset`.
//   - Similarly, the `indices` tensor is also sliced and padded.
//   - A mask is created that compares each element of shifted `indices` and
//   original `indices` are equal (used to avoid combining updates from
//   different indices).
//   - The `to_apply` function is used to combine the original and shifted
//   updates to generate a combined update tensor.
//   - Based on the mask, the new update tensor will choose from either the
//   combined update or the original update.
//   - The result becomes the `new_updates`, which is then used as the
//   input for the next iteration.
static absl::StatusOr<HloInstruction*> CreateScanWithIndices(
    HloComputation* parent, HloInstruction* updates, HloInstruction* indices,
    HloComputation* to_apply) {
  const Shape& updates_shape = updates->shape();
  const Shape& indices_shape = indices->shape();
  // Get the length of the input array
  int64_t num_updates = updates_shape.dimensions(0);

  // Calculate the number of iterations needed (log_2(n))
  int64_t log_n = Log2Ceiling(static_cast<uint64_t>(num_updates));

  HloInstruction* current_updates = updates;

  std::vector<int64_t> start_indices = {0};
  std::vector<int64_t> strides = {1};

  for (int64_t iteration = 0; iteration < log_n; ++iteration) {
    int64_t offset = static_cast<int64_t>(1) << iteration;
    std::vector<int64_t> end_indices = {num_updates - offset};

    auto shifted_updates_shape = ShapeUtil::MakeShape(
        updates_shape.element_type(), {num_updates - offset});
    auto padding_updates_shape =
        ShapeUtil::MakeShape(updates_shape.element_type(), {offset});

    auto shifted_indices_shape = ShapeUtil::MakeShape(
        indices_shape.element_type(), {num_updates - offset});
    auto padding_indices_shape =
        ShapeUtil::MakeShape(indices_shape.element_type(), {offset});

    auto* shifted_updates = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_updates_shape, current_updates,
                                    start_indices, end_indices, strides));
    auto* padding_updates =
        parent->AddInstruction(HloInstruction::CreateBroadcast(
            padding_updates_shape,
            parent->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0(updates_shape.element_type(), 0))),
            {}));

    auto* shifted_indices = parent->AddInstruction(HloInstruction::CreateSlice(
        shifted_indices_shape, indices, start_indices, end_indices, strides));
    auto* padding_indices =
        parent->AddInstruction(HloInstruction::CreateBroadcast(
            padding_indices_shape,
            parent->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0(indices_shape.element_type(), 0))),
            {}));

    auto* concatenated_updates =
        parent->AddInstruction(HloInstruction::CreateConcatenate(
            updates_shape, {padding_updates, shifted_updates}, 0));
    auto* concatenated_indices =
        parent->AddInstruction(HloInstruction::CreateConcatenate(
            indices_shape, {padding_indices, shifted_indices}, 0));

    auto* indices_mask = parent->AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {num_updates}), indices,
        concatenated_indices, ComparisonDirection::kEq));
    std::vector<HloInstruction*> map_operands = {current_updates,
                                                 concatenated_updates};
    TF_ASSIGN_OR_RETURN(HloInstruction * reduced_updates,
                        MakeMapHlo(map_operands, to_apply));
    current_updates = parent->AddInstruction(HloInstruction::CreateTernary(
        updates_shape, HloOpcode::kSelect, indices_mask, reduced_updates,
        current_updates));
  }
  return current_updates;
}

absl::StatusOr<std::vector<HloInstruction*>> ComputePrefixScan(
    const std::vector<HloInstruction*>& sorted_updates,
    HloInstruction* sorted_scalar_indices, HloScatterInstruction* scatter,
    HloComputation* parent) {
  std::vector<HloInstruction*> prefix_scans(sorted_updates.size());
  HloInstruction* prefix_scan_update = nullptr;
  for (int i = 0; i < sorted_updates.size(); i++) {
    // TODO(chenhao) change to use the extracted computation
    TF_ASSIGN_OR_RETURN(
        HloComputation * to_apply,
        CallComputationAndGetIthOutputWithBinaryParams(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(prefix_scan_update,
                        CreateScanWithIndices(parent, sorted_updates[i],
                                              sorted_scalar_indices, to_apply));
    CHECK(prefix_scan_update != nullptr) << i << "th update is nullptr";
    prefix_scans[i] = prefix_scan_update;
  }
  return prefix_scans;
}

static HloInstruction* FindLastOccurrenceIndices(
    HloInstruction* scatter_indices, HloInstruction* sorted_scalar_indices,
    HloInstruction* scatter, HloComputation* parent, int64_t num_indices) {
  int64_t indices_len = sorted_scalar_indices->shape().dimensions(0);
  HloInstruction* sorted_indices = sorted_scalar_indices;
  auto* sorted_indices_preceding_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                               {indices_len - 1}),
          sorted_scalar_indices, {0}, {indices_len - 1}, {1}));
  auto* sorted_indices_following_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                               {indices_len - 1}),
          sorted_scalar_indices, {1}, {indices_len}, {1}));
  auto* indices_mask_without_padding =
      parent->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {indices_len - 1}),
          sorted_indices_preceding_part, sorted_indices_following_part,
          ComparisonDirection::kNe));
  // Pad the comparison with a true value at the end
  auto* true_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto* padding = parent->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, {1}), true_constant, {}));
  std::vector<HloInstruction*> padding_operands = {indices_mask_without_padding,
                                                   padding};
  auto* indices_mask = parent->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(PRED, {indices_len}), padding_operands, 0));

  // Mask the indices
  indices_mask = parent->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, scatter_indices->shape().dimensions()),
      indices_mask, {0}));

  auto* out_of_bound_tensor =
      CreateOutOfBoundTensor(parent, scatter_indices, scatter->shape());

  auto* masked_indices = parent->AddInstruction(HloInstruction::CreateTernary(
      sorted_indices->shape(), HloOpcode::kSelect, indices_mask, sorted_indices,
      out_of_bound_tensor));
  return masked_indices;
}

absl::StatusOr<HloInstruction*> ScatterDeterminismExpander::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  std::vector<HloInstruction*> scatter_updates(
      scatter->scatter_updates().begin(), scatter->scatter_updates().end());
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();

  // If the updates tensors are empty, there is no need to update the operands.
  // The operands can be forwarded.
  if (ShapeUtil::IsZeroElementArray(scatter_updates[0]->shape())) {
    if (scatter_operands.size() == 1) {
      return scatter_operands[0];
    }
    return scatter->parent()->AddInstruction(
        HloInstruction::CreateTuple(scatter_operands));
  }

  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  int64_t scatter_indices_count = ScatterIndicesCount(scatter);
  if (!IsInt32(scatter_indices_count)) {
    // 2147483647 is the maximum value for a 32-bit signed integer (INT32_MAX).
    return Unimplemented(
        "Scatter operations with more than 2147483647 scatter indices are not "
        "supported. This error occurred for %s.",
        scatter->ToString());
  }

  // Canonicalize the scatter_indices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(scatter_indices,
                      CanonicalizeScatterIndices(
                          scatter_indices, dim_numbers.index_vector_dim()));
  CHECK_EQ(scatter_indices_count, scatter_indices->shape().dimensions(0));

  // Canonicalize the updates, after which the size of their most-major
  // dimensions must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(scatter_updates, CanonicalizeScatterUpdates(
                                           scatter_updates, scatter_indices,
                                           dim_numbers, scatter_indices_count));

  HloComputation* parent = scatter->parent();

  // Sort the scatter indices and updates together based on the scatter indices.
  int64_t num_indices = ShapeUtil::ElementsIn(scatter_updates[0]->shape());
  std::vector<HloInstruction*> sorted_tensors = SortIndicesAndUpdates(
      scatter_indices, scatter_updates, num_indices, scatter, parent);
  HloInstruction* sorted_scalar_indices = sorted_tensors[0];
  std::vector<HloInstruction*> sorted_updates(sorted_tensors.begin() + 1,
                                              sorted_tensors.end());

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> prefix_scan_updates,
                      ComputePrefixScan(sorted_updates, sorted_scalar_indices,
                                        scatter, parent));

  HloInstruction* last_occurrence_indices = FindLastOccurrenceIndices(
      scatter_indices, sorted_scalar_indices, scatter, parent, num_indices);

  CHECK(last_occurrence_indices != nullptr)
      << "Last occurrence indices should not be nullptr";

  // Finally, recreate the scatter instruction with unique indices
  return parent->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, last_occurrence_indices,
      prefix_scan_updates, scatter->to_apply(), dim_numbers,
      /*indices_are_sorted=*/true, /*unique_indices=*/true));
}

namespace {
void RecursivelyGetInputParamNumbers(
    const HloInstruction* instruction, std::vector<int64_t>& param_numbers,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.emplace(instruction).second) {
    return;
  }

  if (instruction->opcode() == HloOpcode::kParameter) {
    param_numbers.push_back(instruction->parameter_number());
    return;
  }
  for (HloInstruction* operand : instruction->operands()) {
    RecursivelyGetInputParamNumbers(operand, param_numbers, visited);
  }
}

// Check if every output of the scatter computation only depends on the
// corresponding operand and updates
bool CheckOutputDependency(HloComputation* to_apply, int operand_size) {
  HloInstruction* root = to_apply->root_instruction();
  if (!root->shape().IsTuple()) {
    return true;
  }
  CHECK_EQ(operand_size, root->operand_count());

  // traverse the tuple output of the computation
  for (int i = 0; i < operand_size; ++i) {
    const HloInstruction* output = root->operand(i);
    std::vector<int64_t> param_numbers;
    absl::flat_hash_set<const HloInstruction*> visited;
    RecursivelyGetInputParamNumbers(output, param_numbers, visited);
    // The input dependencies can be at most 2
    if (param_numbers.size() > 2) {
      return false;
    }
    for (int64_t param_number : param_numbers) {
      if (param_number != i && param_number != operand_size + i) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool ScatterDeterminismExpander::InstructionMatchesPattern(
    HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  // Need to check if updates and indices are scalar, as the current pass does
  // not expand scatter with multi-dimensional updates or indices. This is
  // temporary and will be removed in a future PR soon.
  if (scatter == nullptr) {
    return false;
  }

  const Shape& indices_shape = scatter->scatter_indices()->shape();
  const Shape& updates_shape = scatter->scatter_updates()[0]->shape();

  // Check if indices and updates are effectively 1D.
  bool indices_are_1d =
      (indices_shape.rank() == 1 ||
       (indices_shape.rank() == 2 && indices_shape.dimensions(1) == 1));
  bool updates_are_1d =
      (updates_shape.rank() == 1 ||
       (updates_shape.rank() == 2 && updates_shape.dimensions(1) == 1));

  return indices_are_1d && updates_are_1d && !IsScatterDeterministic(scatter) &&
         CheckOutputDependency(scatter->to_apply(),
                               scatter->scatter_operands().size());
}

}  // namespace xla
