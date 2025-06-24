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

#include "xla/service/batched_gather_scatter_normalizer.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {
bool IsBatchGather(const HloGatherInstruction* gather) {
  const auto& dims = gather->gather_dimension_numbers();
  return !dims.operand_batching_dims().empty();
}

bool IsBatchScatter(const HloScatterInstruction* scatter) {
  const auto& dims = scatter->scatter_dimension_numbers();
  return !dims.input_batching_dims().empty();
}

// If `type` is an integral type in which `size` doesn't fit, promote it to S32
// or S64 (depending on `size`).
PrimitiveType PromoteTypeForSize(PrimitiveType type, int64_t size) {
  // Gather/Scatter should have an integral type, but we check just in case.
  if (!primitive_util::IsIntegralType(type) ||
      primitive_util::FitsInIntegralType(size, type)) {
    return type;
  }
  if (primitive_util::FitsInIntegralType(size, PrimitiveType::S32)) {
    return PrimitiveType::S32;
  }
  return PrimitiveType::S64;
}

// If `indices_batching_dims` and `updated_index_map` are both sorted, then the
// `indices_are_sorted` property is preserved.
//
// This is because each concatenated iota is monotonically increasing, sorted
// indices batching dims mean their order corresponds to the order of batching
// dims in the operand, and a sorted updated start index map means the order of
// the index vector dim corresponds to the order of operand dims.
bool GetUpdatedIndicesAreSorted(bool indices_are_sorted,
                                absl::Span<const int64_t> indices_batching_dims,
                                absl::Span<const int64_t> updated_index_map) {
  return indices_are_sorted && absl::c_is_sorted(indices_batching_dims) &&
         absl::c_is_sorted(updated_index_map);
}

// Update gather/scater indices by adding fake batching iota dimensions.
HloInstruction* CreateConcatIndices(
    HloInstruction* inst, HloInstruction* indices, int64_t index_vector_dim,
    absl::Span<const int64_t> indices_batching_dims,
    BatchedGatherScatterNormalizer* normalizer) {
  // The batching dim sizes might not fit in the existing element type,
  // in which case we need to promote it.
  PrimitiveType element_type = indices->shape().element_type();
  for (int64_t indices_batching_dim : indices_batching_dims) {
    element_type = PromoteTypeForSize(
        element_type, indices->shape().dimensions(indices_batching_dim));
  }
  if (element_type != indices->shape().element_type()) {
    Shape indices_shape = indices->shape();
    indices_shape.set_element_type(element_type);
    indices = inst->parent()->AddInstruction(
        HloInstruction::CreateConvert(indices_shape, indices));
  }

  Shape iota_shape = indices->shape();
  const bool index_vector_dim_on_last_dim =
      index_vector_dim == iota_shape.dimensions().size();
  if (index_vector_dim_on_last_dim) {
    std::vector<int64_t> dimensions(iota_shape.dimensions().begin(),
                                    iota_shape.dimensions().end());
    dimensions.push_back(1);
    iota_shape = ShapeUtil::MakeShape(element_type, dimensions);
    indices = inst->AddInstruction(
        HloInstruction::CreateReshape(iota_shape, indices));
  }
  iota_shape.set_dimensions(index_vector_dim, 1);
  normalizer->UpdateLayout(&iota_shape);

  std::vector<HloInstruction*> indices_to_concat;
  indices_to_concat.reserve(indices_batching_dims.size() + 1);
  for (int64_t indices_batching_dim : indices_batching_dims) {
    indices_to_concat.push_back(inst->parent()->AddInstruction(
        HloInstruction::CreateIota(iota_shape, indices_batching_dim)));
  }
  indices_to_concat.push_back(indices);

  Shape concat_shape = iota_shape;
  concat_shape.set_dimensions(
      index_vector_dim,
      indices_batching_dims.size() +
          (index_vector_dim_on_last_dim
               ? 1
               : indices->shape().dimensions(index_vector_dim)));
  normalizer->UpdateLayout(&concat_shape);
  return inst->AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, indices_to_concat, index_vector_dim));
}

absl::StatusOr<HloInstruction*> NormalizeBatchGather(
    HloGatherInstruction* gather, BatchedGatherScatterNormalizer* normalizer) {
  HloInstruction* gather_operand = gather->mutable_operand(0);
  HloInstruction* gather_indices = gather->mutable_operand(1);
  const auto& dims = gather->gather_dimension_numbers();
  CHECK_EQ(dims.operand_batching_dims_size(),
           dims.start_indices_batching_dims_size());
  // Update start_index_map.
  std::vector<int64_t> start_index_map(dims.operand_batching_dims().begin(),
                                       dims.operand_batching_dims().end());
  absl::c_copy(dims.start_index_map(), std::back_inserter(start_index_map));
  gather_indices =
      CreateConcatIndices(gather, gather_indices, dims.index_vector_dim(),
                          dims.start_indices_batching_dims(), normalizer);
  // Update collapsed_slice_dims.
  std::vector<int64_t> collapsed_slice_dims(dims.collapsed_slice_dims().begin(),
                                            dims.collapsed_slice_dims().end());
  absl::c_copy(dims.operand_batching_dims(),
               std::back_inserter(collapsed_slice_dims));
  absl::c_sort(collapsed_slice_dims);

  GatherDimensionNumbers updated_dims =
      HloGatherInstruction::MakeGatherDimNumbers(
          dims.offset_dims(), collapsed_slice_dims, start_index_map,
          dims.index_vector_dim());
  return gather->AddInstruction(HloInstruction::CreateGather(
      gather->shape(), gather_operand, gather_indices, updated_dims,
      gather->gather_slice_sizes(),
      GetUpdatedIndicesAreSorted(gather->indices_are_sorted(),
                                 dims.start_indices_batching_dims(),
                                 start_index_map)));
}

absl::StatusOr<HloInstruction*> NormalizeBatchScatter(
    HloScatterInstruction* scatter,
    BatchedGatherScatterNormalizer* normalizer) {
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  auto scatter_updates = scatter->scatter_updates();
  const auto& dims = scatter->scatter_dimension_numbers();
  CHECK_EQ(dims.input_batching_dims_size(),
           dims.scatter_indices_batching_dims_size());
  // Update scatter_dims_to_operand_dims.
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dims.input_batching_dims().begin(), dims.input_batching_dims().end());
  absl::c_copy(dims.scatter_dims_to_operand_dims(),
               std::back_inserter(scatter_dims_to_operand_dims));
  scatter_indices =
      CreateConcatIndices(scatter, scatter_indices, dims.index_vector_dim(),
                          dims.scatter_indices_batching_dims(), normalizer);
  // Update inserted_window_dims.
  std::vector<int64_t> inserted_window_dims(dims.inserted_window_dims().begin(),
                                            dims.inserted_window_dims().end());
  absl::c_copy(dims.input_batching_dims(),
               std::back_inserter(inserted_window_dims));
  absl::c_sort(inserted_window_dims);

  ScatterDimensionNumbers updated_dims =
      HloScatterInstruction::MakeScatterDimNumbers(
          dims.update_window_dims(), inserted_window_dims,
          scatter_dims_to_operand_dims, dims.index_vector_dim());
  return scatter->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, scatter_indices, scatter_updates,
      scatter->to_apply(), updated_dims,
      GetUpdatedIndicesAreSorted(scatter->indices_are_sorted(),
                                 dims.scatter_indices_batching_dims(),
                                 scatter_dims_to_operand_dims),
      scatter->unique_indices()));
}

}  // namespace

absl::StatusOr<HloInstruction*>
BatchedGatherScatterNormalizer::ExpandInstruction(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    auto* gather = DynCast<HloGatherInstruction>(inst);
    return NormalizeBatchGather(gather, this);
  }
  if (inst->opcode() == HloOpcode::kScatter) {
    auto* scatter = DynCast<HloScatterInstruction>(inst);
    return NormalizeBatchScatter(scatter, this);
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Instruction: %s is not a batch gather or scatter.", inst->ToString()));
}

bool BatchedGatherScatterNormalizer::InstructionMatchesPattern(
    HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    auto* gather = DynCast<HloGatherInstruction>(inst);
    return IsBatchGather(gather);
  }
  if (inst->opcode() == HloOpcode::kScatter) {
    auto* scatter = DynCast<HloScatterInstruction>(inst);
    return IsBatchScatter(scatter);
  }
  return false;
}

}  // namespace xla
