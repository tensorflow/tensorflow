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

#include "xla/backends/gpu/transforms/one_hot_gather_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

PrimitiveType PromoteTypeForMaxIndex(PrimitiveType type, int64_t max_index) {
  if (!primitive_util::IsIntegralType(type) ||
      primitive_util::FitsInIntegralType(max_index, type)) {
    return type;
  }
  if (primitive_util::FitsInIntegralType(max_index, PrimitiveType::S32)) {
    return PrimitiveType::S32;
  }
  return PrimitiveType::S64;
}

HloInstruction* BroadcastScalarToShape(HloComputation* computation,
                                       HloInstruction* scalar,
                                       const Shape& target_shape,
                                       const OpMetadata* metadata) {
  if (target_shape.dimensions().empty()) {
    return scalar;
  }
  return computation->AddInstruction(
      HloInstruction::CreateBroadcast(target_shape, scalar, {}), metadata);
}

absl::StatusOr<HloInstruction*> CreateInBoundsMask(HloInstruction* indices,
                                                   int64_t num_classes,
                                                   const OpMetadata* metadata) {
  HloComputation* computation = indices->parent();
  const PrimitiveType indices_type = indices->shape().element_type();
  Shape pred_shape = ShapeUtil::ChangeElementType(indices->shape(), PRED);
  ABSL_ASSIGN_OR_RETURN(
      Literal max_class_lit,
      LiteralUtil::CreateR0<int64_t>(num_classes - 1).Convert(indices_type));
  HloInstruction* max_class_const = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(max_class_lit)));
  HloInstruction* max_class_val = BroadcastScalarToShape(
      computation, max_class_const, indices->shape(), metadata);
  HloInstruction* in_bounds = computation->AddInstruction(
      HloInstruction::CreateCompare(pred_shape, indices, max_class_val,
                                    ComparisonDirection::kLe),
      metadata);
  if (ShapeUtil::ElementIsSigned(indices->shape())) {
    HloInstruction* zero_idx = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(indices_type)));
    HloInstruction* zero_idx_val = BroadcastScalarToShape(
        computation, zero_idx, indices->shape(), metadata);
    HloInstruction* ge_zero = computation->AddInstruction(
        HloInstruction::CreateCompare(pred_shape, indices, zero_idx_val,
                                      ComparisonDirection::kGe),
        metadata);
    in_bounds = computation->AddInstruction(
        HloInstruction::CreateBinary(pred_shape, HloOpcode::kAnd, ge_zero,
                                     in_bounds),
        metadata);
  }
  return in_bounds;
}

struct IndexedSourceDim {
  int64_t source_dim;
  bool is_contract_dim;
  int64_t indices_dim;
};

}  // namespace

bool IsValidOneHotIndexType(PrimitiveType indices_type, int64_t num_classes) {
  // `IsIntegralType` is false for `PRED` but true for sub-byte integer types
  // (e.g. `S4`, `U4`), which must not be used as gather `start_indices`.
  return num_classes > 1 && primitive_util::IsIntegralType(indices_type) &&
         primitive_util::BitWidth(indices_type) >= 8 &&
         primitive_util::FitsInIntegralType(num_classes - 1, indices_type);
}

absl::StatusOr<HloInstruction*> BuildOneHotGatherAndSelect(
    HloComputation* computation, const OneHotGatherSpec& spec,
    const OpMetadata* metadata) {
  absl::Span<const int64_t> indices_dims = spec.indices->shape().dimensions();
  const int64_t k = indices_dims.size();
  absl::Span<const int64_t> source_dims =
      spec.gather_source->shape().dimensions();
  const int64_t num_classes = source_dims[spec.source_contract_dim];

  std::vector<IndexedSourceDim> indexed_dims;
  indexed_dims.reserve(1 + spec.source_batch_to_indices_dim.size());
  indexed_dims.push_back({spec.source_contract_dim,
                          /*is_contract_dim=*/true,
                          /*indices_dim=*/-1});
  for (const BatchDimMapping& mapping : spec.source_batch_to_indices_dim) {
    indexed_dims.push_back({mapping.source_dim,
                            /*is_contract_dim=*/false, mapping.indices_dim});
  }
  absl::c_sort(indexed_dims,
               [](const IndexedSourceDim& a, const IndexedSourceDim& b) {
                 return a.source_dim < b.source_dim;
               });
  const int64_t num_indexed_dims = indexed_dims.size();

  const PrimitiveType indices_type = spec.indices->shape().element_type();
  PrimitiveType gather_indices_type = indices_type;
  for (const IndexedSourceDim& entry : indexed_dims) {
    if (!entry.is_contract_dim && source_dims[entry.source_dim] > 1) {
      gather_indices_type = PromoteTypeForMaxIndex(
          gather_indices_type, source_dims[entry.source_dim] - 1);
    }
  }
  HloInstruction* gather_indices = spec.indices;
  if (gather_indices_type != indices_type) {
    gather_indices = computation->AddInstruction(
        HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(spec.indices->shape(),
                                         gather_indices_type),
            spec.indices),
        metadata);
  }

  std::vector<int64_t> slice_coord_dims(indices_dims.begin(),
                                        indices_dims.end());
  slice_coord_dims.push_back(1);
  Shape slice_coord_shape =
      ShapeUtil::MakeShape(gather_indices_type, slice_coord_dims);

  HloInstruction* start_indices = nullptr;
  if (num_indexed_dims == 1) {
    start_indices = computation->AddInstruction(
        HloInstruction::CreateReshape(slice_coord_shape, gather_indices),
        metadata);
  } else {
    std::vector<HloInstruction*> coord_slices;
    coord_slices.reserve(num_indexed_dims);
    for (const IndexedSourceDim& entry : indexed_dims) {
      if (entry.is_contract_dim) {
        coord_slices.push_back(computation->AddInstruction(
            HloInstruction::CreateReshape(slice_coord_shape, gather_indices),
            metadata));
      } else {
        coord_slices.push_back(computation->AddInstruction(
            HloInstruction::CreateIota(slice_coord_shape, entry.indices_dim),
            metadata));
      }
    }
    Shape concat_shape = slice_coord_shape;
    concat_shape.set_dimensions(k, num_indexed_dims);
    start_indices = computation->AddInstruction(
        HloInstruction::CreateConcatenate(concat_shape, coord_slices, k),
        metadata);
  }

  std::vector<int64_t> collapsed_slice_dims;
  collapsed_slice_dims.reserve(num_indexed_dims);
  for (const IndexedSourceDim& entry : indexed_dims) {
    collapsed_slice_dims.push_back(entry.source_dim);
  }

  const int64_t m = source_dims.size() - num_indexed_dims;
  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(source_dims.size());
  std::vector<int64_t> source_slice_dims;
  source_slice_dims.reserve(m);
  for (int64_t d = 0; d < source_dims.size(); ++d) {
    if (absl::c_binary_search(collapsed_slice_dims, d)) {
      slice_sizes.push_back(1);
    } else {
      slice_sizes.push_back(source_dims[d]);
      source_slice_dims.push_back(source_dims[d]);
    }
  }

  const int64_t offset_insert_pos =
      spec.lhs_is_one_hot ? k : spec.num_batch_dims;
  std::vector<int64_t> offset_dims(m);
  absl::c_iota(offset_dims, offset_insert_pos);

  std::vector<int64_t> gather_dims;
  gather_dims.reserve(k + m);
  gather_dims.insert(gather_dims.end(), indices_dims.begin(),
                     indices_dims.begin() + offset_insert_pos);
  gather_dims.insert(gather_dims.end(), source_slice_dims.begin(),
                     source_slice_dims.end());
  gather_dims.insert(gather_dims.end(),
                     indices_dims.begin() + offset_insert_pos,
                     indices_dims.end());

  Shape gather_shape = ShapeUtil::MakeShape(spec.out_type, gather_dims);
  Shape raw_gather_shape = ShapeUtil::ChangeElementType(
      gather_shape, spec.gather_source->shape().element_type());

  GatherDimensionNumbers gather_dnums =
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/collapsed_slice_dims,
          /*start_index_map=*/collapsed_slice_dims,
          /*index_vector_dim=*/k);
  // GPU gather emitters clamp out-of-bounds indices to [0, num_classes - 1] in
  // IR, and the `select` below masks out-of-bounds entries to zero.
  HloInstruction* gather = computation->AddInstruction(
      HloInstruction::CreateGather(raw_gather_shape, spec.gather_source,
                                   start_indices, gather_dnums, slice_sizes,
                                   /*indices_are_sorted=*/false),
      metadata);
  if (spec.weights_round_mantissa_bits.has_value()) {
    gather = computation->AddInstruction(
        HloInstruction::CreateReducePrecision(
            gather->shape(), gather, /*exponent_bits=*/8,
            /*mantissa_bits=*/*spec.weights_round_mantissa_bits),
        metadata);
  }
  if (gather->shape().element_type() != spec.out_type) {
    gather = computation->AddInstruction(
        HloInstruction::CreateConvert(gather_shape, gather), metadata);
  }

  ABSL_ASSIGN_OR_RETURN(HloInstruction * in_bounds,
                   CreateInBoundsMask(spec.indices, num_classes, metadata));

  HloInstruction* in_bounds_bcast = in_bounds;
  if (m > 0) {
    std::vector<int64_t> bcast_dims;
    bcast_dims.reserve(k);
    for (int64_t i = 0; i < offset_insert_pos; ++i) {
      bcast_dims.push_back(i);
    }
    for (int64_t i = offset_insert_pos; i < k; ++i) {
      bcast_dims.push_back(i + m);
    }
    in_bounds_bcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(
            ShapeUtil::ChangeElementType(gather_shape, PRED), in_bounds,
            bcast_dims),
        metadata);
  }

  HloInstruction* zero_scalar = spec.zero_scalar;
  if (zero_scalar == nullptr) {
    zero_scalar = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(spec.out_type)));
  }
  HloInstruction* zero_gather =
      BroadcastScalarToShape(computation, zero_scalar, gather_shape, metadata);
  return computation->AddInstruction(
      HloInstruction::CreateTernary(gather_shape, HloOpcode::kSelect,
                                    in_bounds_bcast, gather, zero_gather),
      metadata);
}

}  // namespace xla::gpu
