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

#include "xla/hlo/transforms/simplifiers/hlo_rewrite_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// All not fusion, constant and copy hlos are trivial to transform to different
// shapes.
bool IsTrivialElementwise(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kFusion || hlo.opcode() == HloOpcode::kRng ||
      hlo.opcode() == HloOpcode::kCopy ||
      hlo.opcode() == HloOpcode::kConstant ||
      hlo.opcode() == HloOpcode::kIota) {
    return false;
  }
  return hlo.IsElementwise();
}

// Is this an iota that cannot be transformed by algebraic simplifier.
bool IsNonTrivialIota(const Shape& target_shape, const HloInstruction* hlo) {
  if (const auto* iota = DynCast<HloIotaInstruction>(hlo)) {
    auto opt_dims = ShapeUtil::ReshapeLeavesDimensionsUnmodified(
        iota->shape(), target_shape, {iota->iota_dimension()});
    if (!opt_dims.has_value()) {
      return true;
    }
  }
  return false;
}

HloInstruction* TransformConcat(const Shape& shape, HloInstruction* concat,
                                bool transform_trivial) {
  Shape new_concat_shape = shape;
  new_concat_shape.set_element_type(concat->shape().element_type());
  if (auto maybe_dim = ShapeUtil::ReshapeLeavesDimensionsUnmodified(
          concat->shape(), new_concat_shape,
          {concat->concatenate_dimension()})) {
    VLOG(3) << "***** Transform concat " << concat->ToString() << "->"
            << new_concat_shape.ToString();
    int64_t new_dim = maybe_dim->at(0);
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(concat->operand_count());
    for (HloInstruction* operand : concat->operands()) {
      Shape operand_shape = new_concat_shape;
      operand_shape.set_dimensions(
          new_dim,
          operand->shape().dimensions(concat->concatenate_dimension()));
      new_operands.push_back(MakeReshapeHlo(operand_shape, operand).value());
      concat->SetupDerivedInstruction(new_operands.back());
    }
    HloInstruction* new_concat = MakeConcatHlo(new_operands, new_dim).value();
    concat->SetupDerivedInstruction(new_concat);
    return new_concat;
  }

  bool do_leading_dimension_transform = true;
  if (concat->shape().dimensions().size() - 1 ==
          concat->concatenate_dimension() &&
      !(shape.dimensions().size() == 2 &&
        shape.dimensions(0) == concat->operand_count())) {
    for (int64_t i = 0; i < concat->shape().dimensions().size() - 1; ++i) {
      if (concat->shape().dimensions(i) != shape.dimensions(i)) {
        VLOG(3) << "INVALID LEADING SHAPE " << shape.ToString() << " "
                << concat->ToString();
        do_leading_dimension_transform = false;
        break;
      }
    }
  } else {
    do_leading_dimension_transform = false;
  }

  if (do_leading_dimension_transform &&
      new_concat_shape.dimensions(concat->concatenate_dimension()) %
              concat->operand_count() ==
          0 &&
      absl::c_all_of(concat->operands(), [&](HloInstruction* hlo) {
        return concat->operand(0)->shape().dimensions(
                   concat->concatenate_dimension()) ==
               hlo->shape().dimensions(concat->concatenate_dimension());
      })) {
    VLOG(3) << "***** Transform R" << concat->shape().dimensions().size()
            << " concat " << concat->ToString() << "->"
            << new_concat_shape.ToString();
    Shape new_operand_shape = new_concat_shape;
    new_operand_shape.set_dimensions(
        concat->concatenate_dimension(),
        new_concat_shape.dimensions(concat->concatenate_dimension()) /
            concat->operand_count());
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(concat->operand_count());
    for (HloInstruction* operand : concat->operands()) {
      new_operands.push_back(
          MakeReshapeHlo(new_operand_shape, operand).value());
      concat->SetupDerivedInstruction(new_operands.back());
    }
    HloInstruction* new_concat =
        MakeConcatHlo(new_operands, concat->concatenate_dimension()).value();
    concat->SetupDerivedInstruction(new_concat);
    return new_concat;
  }

  if (transform_trivial) {
    HloInstruction* reshape = MakeReshapeHlo(new_concat_shape, concat).value();
    concat->SetupDerivedInstruction(reshape);
    return reshape;
  }
  VLOG(3) << "INVALID CONCAT " << concat->ToString();
  return nullptr;
}

HloInstruction* TransformSlice(const Shape& shape, HloInstruction* slice,
                               bool transform_trivial) {
  Shape new_slice_shape = shape;
  new_slice_shape.set_element_type(slice->shape().element_type());
  auto early_return = [&]() -> HloInstruction* {
    VLOG(3) << "XXXX NOT transform slice " << slice->ToString() << "->"
            << new_slice_shape.ToString();
    if (transform_trivial) {
      HloInstruction* reshape = MakeReshapeHlo(new_slice_shape, slice).value();
      slice->SetupDerivedInstruction(reshape);
      return reshape;
    }
    return nullptr;
  };
  if (Product(slice->slice_strides()) != 1 ||
      slice->shape().dimensions().size() != 1 ||
      shape.dimensions().size() < 2) {
    VLOG(3) << "INVALID SLICE ";
    return early_return();
  }
  int64_t slice_leading_dim = 0;
  int64_t leading_dim = 0;

  while (slice_leading_dim < slice->shape().dimensions().size() &&
         leading_dim < shape.dimensions().size()) {
    int64_t slice_size = slice->shape().dimensions(slice_leading_dim);
    int64_t slice_operand_size =
        slice->operand(0)->shape().dimensions(slice_leading_dim);
    if (slice_operand_size > slice_size) {
      break;
    }
    int64_t size = shape.dimensions(leading_dim);
    if (slice_size == size) {
      ++slice_leading_dim;
      ++leading_dim;
      continue;
    }
    if (slice_size == 1) {
      ++slice_leading_dim;
      continue;
    }
    if (size == 1) {
      ++leading_dim;
      continue;
    }
    break;
  }

  int64_t slice_start = slice->slice_starts(slice_leading_dim);
  int64_t slice_limit = slice->slice_limits(slice_leading_dim);
  int64_t sliced_elements = slice->shape().dimensions(slice_leading_dim);
  int64_t operand_elements =
      slice->operand(0)->shape().dimensions(slice_leading_dim);
  int64_t i = shape.dimensions().size() - 1;
  int64_t minor_size = 1;
  for (; i > leading_dim; --i) {
    const int64_t dim_size = shape.dimensions(i);
    if (operand_elements % dim_size != 0 || sliced_elements % dim_size != 0 ||
        slice_start % dim_size != 0 || slice_limit % dim_size != 0) {
      break;
    }
    slice_start /= dim_size;
    slice_limit /= dim_size;
    sliced_elements /= dim_size;
    operand_elements /= dim_size;
    minor_size *= dim_size;
  }
  // The most major non one-sized dimension is the only transformation currently
  // supported.
  if (i > leading_dim && (sliced_elements > shape.dimensions(i) ||
                          operand_elements > shape.dimensions(i))) {
    return early_return();
  }
  VLOG(3) << "***** Transform slice " << slice->ToString() << "->"
          << new_slice_shape.ToString();
  Shape new_slice_operand_shape = new_slice_shape;
  new_slice_operand_shape.set_dimensions(i, operand_elements);
  HloInstruction* reshaped_operand =
      MakeReshapeHlo(new_slice_operand_shape, slice->mutable_operand(0))
          .value();
  slice->SetupDerivedInstruction(reshaped_operand);
  DimensionVector new_slice_starts(new_slice_shape.dimensions().size(), 0);
  CHECK_EQ(slice->slice_starts(slice_leading_dim) % minor_size, 0)
      << slice->ToString() << " " << new_slice_shape.ToString();
  new_slice_starts[i] = slice->slice_starts(slice_leading_dim) / minor_size;
  DimensionVector new_slice_strides(new_slice_shape.dimensions().size(), 1);
  DimensionVector new_slice_limits(new_slice_shape.dimensions().begin(),
                                   new_slice_shape.dimensions().end());
  CHECK_EQ(slice->slice_limits(0) % minor_size, 0)
      << slice->ToString() << " " << new_slice_shape.ToString();
  new_slice_limits[i] = slice->slice_limits(slice_leading_dim) / minor_size;

  HloInstruction* new_slice = MakeSliceHlo(reshaped_operand, new_slice_starts,
                                           new_slice_limits, new_slice_strides)
                                  .value();
  slice->SetupDerivedInstruction(new_slice);
  return new_slice;
}

std::optional<Shape const*>
FindElementwiseSubgraphSurroundedByReshapesAndBroadcastsWithLimit(
    HloInstruction* root, std::optional<Shape const*> target_shape,
    absl::flat_hash_set<HloInstruction*>* finds, HloOpcode opc, int d) {
  // Avoid stack overflows on borg.
  if (d > 128) {
    return nullptr;
  }
  if (target_shape && *target_shape == nullptr) {
    return nullptr;
  }
  if (!finds->insert(root).second) {
    return target_shape;
  }

  auto indent = [d] { return std::string(d, ' '); };
  std::vector<HloInstruction*> to_recurse;
  for (HloInstruction* operand : root->operands()) {
    if (operand->opcode() == HloOpcode::kBroadcast ||
        operand->opcode() == HloOpcode::kGather ||
        operand->opcode() == HloOpcode::kReduce ||
        operand->opcode() == HloOpcode::kReduceWindow ||
        operand->opcode() == HloOpcode::kSelectAndScatter ||
        operand->opcode() == HloOpcode::kConstant ||
        (operand->opcode() == HloOpcode::kConcatenate &&
         opc == HloOpcode::kReshape) ||
        operand->opcode() == HloOpcode::kIota ||
        (operand->opcode() == HloOpcode::kSlice &&
         operand->shape().dimensions().size() == 1 &&
         operand->slice_strides(0) == 1 && target_shape && *target_shape &&
         ShapeUtil::TrueNumDimensions(**target_shape) != 1) ||
        operand->opcode() == HloOpcode::kRng) {
      VLOG(3) << indent() << " " << operand->ToString();
      continue;
    }

    if (opc == HloOpcode::kTranspose &&
        (operand->opcode() == HloOpcode::kDot ||
         operand->opcode() == HloOpcode::kConvolution)) {
      VLOG(3) << indent() << " " << operand->ToString();
      continue;
    }

    if (operand->opcode() == HloOpcode::kGetTupleElement &&
        operand->operand(0)->opcode() == HloOpcode::kParameter &&
        operand->parent() != operand->GetModule()->entry_computation() &&
        absl::c_any_of(operand->users(), [&](HloInstruction* use) {
          return use->opcode() == HloOpcode::kTuple &&
                 operand->tuple_index() < use->operand_count() &&
                 operand == use->operand(operand->tuple_index());
        })) {
      VLOG(3) << indent() << " " << operand->ToString();
      continue;
    }

    if (operand->opcode() == opc) {
      if (!target_shape) {
        VLOG(3) << indent() << " " << "EUREKA "
                << operand->operand(0)->ToString();
        target_shape = operand->mutable_operand(0)->mutable_shape();
      }
      VLOG(3) << indent() << " " << operand->ToString();
      continue;
    }

    if (IsTrivialElementwise(*operand)) {
      if (!target_shape) {
        VLOG(3) << indent() << " " << "RECURSE NO SHAPE "
                << operand->ToString();
      }
      to_recurse.push_back(operand);
      continue;
    }
    VLOG(3) << indent() << "FAIL " << operand->ToString();
    return nullptr;
  }

  for (HloInstruction* operand : to_recurse) {
    if (std::optional<Shape const*> new_target_shape =
            FindElementwiseSubgraphSurroundedByReshapesAndBroadcastsWithLimit(
                operand, target_shape, finds, opc, d + 1)) {
      if (*new_target_shape == nullptr) {
        VLOG(3) << indent() << "FAIL " << operand->ToString();
        return nullptr;
      }
      VLOG(3) << indent() << " " << operand->ToString();
      if (!target_shape) {
        target_shape = new_target_shape;
        continue;
      }
      if (!ShapeUtil::SameDimensions(**target_shape, **new_target_shape)) {
        VLOG(3) << indent() << "DIFF " << (*target_shape)->ToString() << " -- "
                << operand->ToString();
      }
    }
  }
  if (target_shape && *target_shape) {
    VLOG(3) << indent() << "ROOT" << root->ToString();
  } else if (!target_shape) {
    VLOG(3) << indent() << "UNKNOWN ROOT " << root->ToString();
  } else {
    VLOG(3) << indent() << "BROKEN ROOT " << root->ToString();
  }
  return target_shape;
}

std::optional<Shape> FindElementwiseSubgraphSurroundedByReshapesAndBroadcasts(
    HloInstruction* root, std::optional<Shape> target_shape,
    absl::flat_hash_set<HloInstruction*>* finds, HloOpcode opc) {
  // Don't bother recursing down scalars.
  if (ShapeUtil::IsEffectiveScalar(root->shape()) ||
      (target_shape && ShapeUtil::IsEffectiveScalar(*target_shape))) {
    return std::nullopt;
  }
  std::optional<Shape const*> result;
  if (target_shape) {
    result = FindElementwiseSubgraphSurroundedByReshapesAndBroadcastsWithLimit(
        root, &(*target_shape), finds, opc);
  } else {
    result = FindElementwiseSubgraphSurroundedByReshapesAndBroadcastsWithLimit(
        root, std::nullopt, finds, opc);
  }
  if (result) {
    if (*result) {
      return **result;
    }
    VLOG(3) << "Invalid Traversal";
  }
  VLOG(3) << "No reshape found";
  return std::nullopt;
}

// Convert a broadcast of one shape to a broadcast of another shape. For example
// to make a [B, H, W, C] out of a broadcast of [B, 2] that produces
// [B, H, W, C/2, 2] need to broadcast [B, 2] to [B, C/2, 2] reshape that to
// [B,C] and finally broadcast that to [B, H, W, C].
HloInstruction* TransformBroadcast(const Shape& shape,
                                   HloInstruction* broadcast,
                                   HloComputation* computation,
                                   bool transform_trivial) {
  // An iota with a dimension modified by a reshape can be split into a single
  // iota dimension and a rank 1 broadcast that can then be transformed by this
  // function.
  if (const auto* iota = DynCast<HloIotaInstruction>(broadcast)) {
    HloInstruction* new_iota =
        broadcast->AddInstruction(HloInstruction::CreateIota(
            ShapeUtil::FilterDimensions(
                [&](int64_t dim) { return dim == iota->iota_dimension(); },
                iota->shape()),
            0));
    HloInstruction* new_broadcast =
        broadcast->AddInstruction(HloInstruction::CreateBroadcast(
            iota->shape(), new_iota, {iota->iota_dimension()}));
    HloInstruction* replacement = TransformBroadcast(
        shape, new_broadcast, computation, transform_trivial);
    if (replacement != nullptr) {
      return replacement;
    }
    CHECK(!transform_trivial);
    CHECK_OK(computation->RemoveInstructionAndUnusedOperands(new_broadcast));
    return nullptr;
  }

  Shape final_shape =
      ShapeUtil::ChangeElementType(shape, broadcast->shape().element_type());
  HloInstruction* operand = broadcast->mutable_operand(0);
  if (ShapeUtil::ElementsIn(final_shape) ==
      ShapeUtil::ElementsIn(operand->shape())) {
    auto trivial_reshape = MakeReshapeHlo(final_shape, operand).value();
    broadcast->SetupDerivedInstruction(trivial_reshape);
    return trivial_reshape;
  }

  if (ShapeUtil::TrueNumDimensions(final_shape) >= 2 &&
      operand->shape().dimensions().size() == 1 &&
      broadcast->shape().dimensions().size() == 2 &&
      ShapeUtil::ElementsIn(broadcast->shape()) > 1) {
    int64_t remaining = operand->shape().dimensions(0);
    int64_t total_remaining = ShapeUtil::ElementsIn(broadcast->shape());
    DimensionVector partial_sizes;
    DimensionVector broadcast_sizes;
    DimensionVector reshape_after_broadcast_sizes;
    int64_t dim;
    DimensionVector new_dimensions;
    DimensionVector second_dimensions;
    int64_t size;
    if (broadcast->dimensions(0) == 0) {
      for (dim = 0; dim < final_shape.dimensions().size(); ++dim) {
        size = final_shape.dimensions(dim);
        new_dimensions.push_back(dim);
        if (remaining % size != 0) {
          break;
        }
        partial_sizes.push_back(size);
        remaining /= size;
        total_remaining /= size;
      }
      if (size % remaining == 0) {
        reshape_after_broadcast_sizes = partial_sizes;
        reshape_after_broadcast_sizes.push_back(size);
        partial_sizes.push_back(remaining);
        size = size / remaining;
        broadcast_sizes = partial_sizes;
        broadcast_sizes.push_back(size);
        second_dimensions = new_dimensions;
      }
    }
    if (broadcast->dimensions(0) == 1) {
      for (int64_t dim = final_shape.dimensions().size() - 1; dim >= 0; --dim) {
        size = final_shape.dimensions(dim);
        new_dimensions.push_back(dim);
        if (remaining % size != 0) {
          break;
        }
        partial_sizes.push_back(size);
        remaining /= size;
        total_remaining /= size;
      }
      if (size % remaining == 0) {
        reshape_after_broadcast_sizes = partial_sizes;
        reshape_after_broadcast_sizes.push_back(size);
        partial_sizes.push_back(remaining);
        size = size / remaining;
        broadcast_sizes = partial_sizes;
        broadcast_sizes.push_back(size);
      }
      absl::c_reverse(partial_sizes);
      absl::c_reverse(broadcast_sizes);
      absl::c_reverse(reshape_after_broadcast_sizes);
      absl::c_sort(new_dimensions);
      second_dimensions = new_dimensions;
      int64_t dim_offset = new_dimensions[0] - 1;
      for (auto& new_dim : new_dimensions) {
        new_dim -= dim_offset;
      }
    }
    if (!broadcast_sizes.empty()) {
      HloInstruction* partial_operand;
      if (partial_sizes.size() > 1) {
        partial_operand = MakeReshapeHlo(partial_sizes, operand).value();
        broadcast->SetupDerivedInstruction(partial_operand);
      } else {
        if (!transform_trivial &&
            absl::c_equal(broadcast_sizes, broadcast->shape().dimensions())) {
          return nullptr;
        }
        partial_operand = operand;
      }
      HloInstruction* partial_broadcast =
          MakeBroadcastHlo(partial_operand, new_dimensions, broadcast_sizes);
      CHECK_EQ(partial_operand->shape().dimensions().size(),
               new_dimensions.size())
          << shape.ToString() << ": " << broadcast->ToString();
      broadcast->SetupDerivedInstruction(partial_broadcast);
      HloInstruction* reshape =
          MakeReshapeHlo(reshape_after_broadcast_sizes, partial_broadcast)
              .value();
      reshape->SetupDerivedInstruction(partial_broadcast);
      if (reshape_after_broadcast_sizes.size() ==
          final_shape.dimensions().size()) {
        return reshape;
      }
      HloInstruction* final_broadcast =
          MakeBroadcastHlo(reshape, second_dimensions, final_shape);
      CHECK_EQ(reshape->shape().dimensions().size(), second_dimensions.size())
          << shape.ToString() << ": " << broadcast->ToString();
      reshape->SetupDerivedInstruction(final_broadcast);
      return final_broadcast;
    }
  }

  auto invert_dimensions = [](absl::Span<const int64_t> dimensions,
                              int64_t rank) {
    DimensionVector inverted_dimensions;
    for (int64_t i = 0; i < rank; ++i) {
      if (!absl::c_linear_search(dimensions, i)) {
        inverted_dimensions.push_back(i);
      }
    }
    return inverted_dimensions;
  };

  auto unbroadcasted_dimensions = invert_dimensions(
      broadcast->dimensions(), broadcast->shape().dimensions().size());
  auto converted_dimensions = ConvertDimensionNumbers(
      unbroadcasted_dimensions, broadcast->shape().dimensions(),
      final_shape.dimensions());
  if (!converted_dimensions.untransformed_from_dimensions.empty()) {
    Shape unmodified_broadcast_shape = broadcast->shape();
    for (int64_t i = 0; i < converted_dimensions.split_from_dimensions.size();
         ++i) {
      unmodified_broadcast_shape.set_dimensions(
          converted_dimensions.split_from_dimensions[i],
          converted_dimensions.split_from_sizes[i]);
    }
    unmodified_broadcast_shape = ShapeUtil::DeleteDimensions(
        converted_dimensions.transformed_from_dimensions,
        unmodified_broadcast_shape);
    auto broadcast_dims = converted_dimensions.untransformed_from_dimensions;
    for (auto& d : broadcast_dims) {
      for (int64_t j =
               converted_dimensions.transformed_from_dimensions.size() - 1;
           j >= 0; --j) {
        if (d >= converted_dimensions.transformed_from_dimensions[j]) {
          --d;
        }
      }
    }
    broadcast_dims = invert_dimensions(
        broadcast_dims, unmodified_broadcast_shape.dimensions().size());
    if (!transform_trivial &&
        ShapeUtil::SameDimensions(unmodified_broadcast_shape,
                                  broadcast->shape())) {
      return nullptr;
    }
    operand =
        MakeBroadcastHlo(operand, broadcast_dims, unmodified_broadcast_shape);

    broadcast->SetupDerivedInstruction(operand);
    if (ShapeUtil::SameDimensions(operand->shape(), shape)) {
      return operand;
    }
  }

  Shape intermediate_shape = ShapeUtil::DeleteDimensions(
      converted_dimensions.to_dimensions, final_shape);
  if (!ShapeUtil::Equal(operand->shape(), intermediate_shape)) {
    absl::StatusOr<HloInstruction*> reshape =
        MakeReshapeHlo(intermediate_shape, operand);
    CHECK_OK(reshape);
    broadcast->SetupDerivedInstruction(*reshape);
    if (ShapeUtil::SameDimensions((*reshape)->shape(), shape)) {
      return *reshape;
    }
    operand = *reshape;
  }
  auto broadcast_dims = invert_dimensions(converted_dimensions.to_dimensions,
                                          final_shape.dimensions().size());
  CHECK_EQ(broadcast_dims.size(), operand->shape().dimensions().size());
  operand = MakeBroadcastHlo(operand, broadcast_dims, final_shape);
  broadcast->SetupDerivedInstruction(operand);
  return operand;
}

HloInstruction* ReplaceElementwiseGroupSurroundedByReshapesAndBroadcasts(
    const Shape& shape, HloInstruction* root, HloComputation* computation,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>* replacements) {
  if (auto it = replacements->find(root);
      it != replacements->end() && (it->second != nullptr)) {
    return it->second;
  }
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : root->operands()) {
    if (operand->opcode() == HloOpcode::kBroadcast ||
        IsNonTrivialIota(shape, operand)) {
      new_operands.push_back(TransformBroadcast(shape, operand, computation));
      continue;
    }
    if (operand->opcode() == HloOpcode::kReshape) {
      if (ShapeUtil::SameDimensions(operand->operand(0)->shape(), shape)) {
        new_operands.push_back(operand->mutable_operand(0));
        continue;
      }
      auto [it, inserted] = replacements->try_emplace(operand);
      CHECK(inserted || it->second);
      if (inserted || !it->second) {
        it->second = operand->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::ChangeElementType(
                shape, operand->mutable_operand(0)->shape().element_type()),
            operand->mutable_operand(0)));
      }
      new_operands.push_back(it->second);
      continue;
    }
    if (operand->opcode() == HloOpcode::kConcatenate) {
      auto [it, inserted] = replacements->try_emplace(operand);
      CHECK(inserted || it->second);
      if (inserted || !it->second) {
        it->second = TransformConcat(shape, operand);
      }
      new_operands.push_back(it->second);
      continue;
    }
    if (operand->opcode() == HloOpcode::kSlice) {
      auto [it, inserted] = replacements->try_emplace(operand);
      CHECK(inserted || it->second);
      if (inserted || !it->second) {
        it->second = TransformSlice(shape, operand);
      }
      new_operands.push_back(it->second);
      continue;
    }

    if (operand->opcode() == HloOpcode::kReduce ||
        operand->opcode() == HloOpcode::kReduceWindow ||
        operand->opcode() == HloOpcode::kSelectAndScatter ||
        operand->opcode() == HloOpcode::kRng ||
        operand->opcode() == HloOpcode::kGather ||
        operand->opcode() == HloOpcode::kGetTupleElement ||
        operand->opcode() == HloOpcode::kIota) {
      auto [it, inserted] = replacements->try_emplace(operand);
      CHECK(inserted || it->second);
      if (inserted || !it->second) {
        it->second = operand->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::ChangeElementType(shape,
                                         operand->shape().element_type()),
            operand));
      }
      new_operands.push_back(it->second);
      continue;
    }
    if (operand->opcode() == HloOpcode::kConstant) {
      auto [it, inserted] = replacements->try_emplace(operand);
      CHECK(inserted || it->second);
      if (inserted || !it->second) {
        if (ShapeUtil::ElementsIn(operand->shape()) !=
            ShapeUtil::ElementsIn(shape)) {
          it->second = operand;
        } else {
          it->second = operand->AddInstruction(HloInstruction::CreateConstant(
              operand->literal().Reshape(shape.dimensions()).value()));
        }
      }
      new_operands.push_back(it->second);
      continue;
    }
    CHECK(IsTrivialElementwise(*operand)) << operand->ToString();
    new_operands.push_back(
        ReplaceElementwiseGroupSurroundedByReshapesAndBroadcasts(
            shape, operand, computation, replacements));
  }
  HloInstruction* replacement = root->AddInstruction(root->CloneWithNewOperands(
      ShapeUtil::ChangeElementType(shape, root->shape().element_type()),
      new_operands));
  CHECK(replacements->try_emplace(root, replacement).second);
  return replacement;
}

}  // namespace xla
