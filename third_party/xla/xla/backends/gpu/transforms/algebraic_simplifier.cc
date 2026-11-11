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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/conv_rewriter.h"
#include "xla/backends/gpu/transforms/one_hot_gather_utils.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier_options.pb.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace m = ::xla::match;

namespace {

constexpr int64_t kMaxReduceSinkPassthroughDepth = 4;
constexpr int64_t kMaxReduceSinkLeaves = 16;

struct SliceRange {
  std::vector<int64_t> starts;
  std::vector<int64_t> limits;
};

bool IsUnaryElementwise(const HloInstruction* instr) {
  return instr->IsElementwise() && instr->operand_count() == 1 &&
         !instr->HasSideEffect() &&
         HloPredicateIsNotOp<HloOpcode::kFusion, HloOpcode::kCopy,
                             HloOpcode::kBitcastConvert>(instr) &&
         ShapeUtil::SameDimensions(instr->shape(), instr->operand(0)->shape());
}

bool HasNonReplicatedSharding(const HloInstruction* instr) {
  return instr->has_sharding() &&
         !instr->sharding().IsReplicatedOrSingleDevice();
}

bool IsPassthroughUnary(const HloInstruction* instr, bool is_layout_sensitive) {
  return IsUnaryElementwise(instr) && !instr->HasControlDependencies() &&
         !HasNonReplicatedSharding(instr) &&
         (!is_layout_sensitive ||
          LayoutUtil::Equal(instr->shape().layout(),
                            instr->operand(0)->shape().layout()));
}

bool IsPassthroughBitcastOrReshape(const HloInstruction* instr,
                                   bool is_layout_sensitive) {
  if ((instr->opcode() != HloOpcode::kBitcast &&
       instr->opcode() != HloOpcode::kReshape) ||
      instr->HasControlDependencies() || HasNonReplicatedSharding(instr)) {
    return false;
  }
  const Shape& src = instr->operand(0)->shape();
  const Shape& dst = instr->shape();
  if (!src.is_static() || !dst.is_static() || !LayoutUtil::IsDenseArray(src) ||
      !LayoutUtil::IsDenseArray(dst) || !ShapeUtil::SameElementType(src, dst) ||
      ShapeUtil::IsZeroElementArray(src) ||
      ShapeUtil::ElementsIn(src) != ShapeUtil::ElementsIn(dst)) {
    return false;
  }
  if (!is_layout_sensitive) {
    return instr->opcode() == HloOpcode::kReshape;
  }
  return src.has_layout() && dst.has_layout() && src.layout().tiles().empty() &&
         dst.layout().tiles().empty() &&
         src.layout().element_size_in_bits() ==
             dst.layout().element_size_in_bits() &&
         src.layout().memory_space() == dst.layout().memory_space() &&
         (instr->opcode() == HloOpcode::kBitcast ||
          ShapeUtil::ReshapeIsBitcast(src, dst));
}

bool IsPassthroughOp(const HloInstruction* instr, const HloInstruction* root,
                     bool is_layout_sensitive) {
  return instr != root &&
         (IsPassthroughUnary(instr, is_layout_sensitive) ||
          IsPassthroughBitcastOrReshape(instr, is_layout_sensitive));
}

// Returns logical dimension indices ordered from physical major to physical
// minor (or logical 0..rank-1 when layout-insensitive).
std::vector<int64_t> MajorToMinorDims(const Shape& shape,
                                      bool is_layout_sensitive) {
  const int64_t rank = shape.dimensions().size();
  std::vector<int64_t> dims(rank);
  if (is_layout_sensitive) {
    const auto& minor_to_major = shape.layout().minor_to_major();
    for (int64_t i = 0; i < rank; ++i) {
      dims[i] = minor_to_major[rank - 1 - i];
    }
  } else {
    std::iota(dims.begin(), dims.end(), int64_t{0});
  }
  return dims;
}

Shape ShapeWithSliceRange(const Shape& base, const SliceRange& range) {
  Shape out = base;
  for (int64_t d = 0; d < static_cast<int64_t>(range.starts.size()); ++d) {
    out.set_dimensions(d, range.limits[d] - range.starts[d]);
  }
  return out;
}

// Maps an unstrided slice `dst_range` on `dst_shape` backwards across a
// bitcast or reshape `src_shape -> dst_shape` to an equivalent unstrided slice
// on `src_shape`, or returns `std::nullopt` if the slice is not contiguous
// along dimension boundaries of `src_shape`.
std::optional<SliceRange> MapSliceThroughBitcastOrReshape(
    const Shape& src_shape, const Shape& dst_shape, const SliceRange& dst_range,
    bool is_layout_sensitive, HloOpcode opcode) {
  if (ShapeUtil::ElementsIn(src_shape) != ShapeUtil::ElementsIn(dst_shape)) {
    return std::nullopt;
  }
  const std::vector<int64_t> src_phys_dims =
      MajorToMinorDims(src_shape, is_layout_sensitive);
  const std::vector<int64_t> dst_phys_dims =
      MajorToMinorDims(dst_shape, is_layout_sensitive);

  std::vector<int64_t> src_phys_sizes(src_phys_dims.size());
  for (size_t i = 0; i < src_phys_dims.size(); ++i) {
    src_phys_sizes[i] = src_shape.dimensions(src_phys_dims[i]);
  }
  std::vector<int64_t> dst_phys_sizes(dst_phys_dims.size());
  for (size_t i = 0; i < dst_phys_dims.size(); ++i) {
    dst_phys_sizes[i] = dst_shape.dimensions(dst_phys_dims[i]);
  }

  const auto factors = CommonFactors(src_phys_sizes, dst_phys_sizes);
  SliceRange src_range{std::vector<int64_t>(src_shape.dimensions().size(), 0),
                       std::vector<int64_t>(src_shape.dimensions().begin(),
                                            src_shape.dimensions().end())};

  for (size_t k = 0; k + 1 < factors.size(); ++k) {
    const int64_t s_begin = factors[k].first;
    const int64_t s_end = factors[k + 1].first;
    const int64_t d_begin = factors[k].second;
    const int64_t d_end = factors[k + 1].second;

    bool group_sliced = false;
    for (int64_t q = d_begin; q < d_end; ++q) {
      const int64_t d_dim = dst_phys_dims[q];
      if (dst_range.starts[d_dim] != 0 ||
          dst_range.limits[d_dim] != dst_phys_sizes[q]) {
        group_sliced = true;
        break;
      }
    }
    if (!group_sliced) {
      continue;
    }

    int64_t linear_start = 0;
    int64_t linear_len = 1;
    bool seen_extent_gt_1 = false;
    for (int64_t q = d_begin; q < d_end; ++q) {
      const int64_t d_dim = dst_phys_dims[q];
      const int64_t dim_size = dst_phys_sizes[q];
      const int64_t start = dst_range.starts[d_dim];
      const int64_t limit = dst_range.limits[d_dim];
      const int64_t len = limit - start;
      if (len <= 0) {
        return std::nullopt;
      }
      if (seen_extent_gt_1 && (start != 0 || limit != dim_size)) {
        return std::nullopt;
      }
      linear_start = linear_start * dim_size + start;
      linear_len *= len;
      if (len > 1) {
        seen_extent_gt_1 = true;
      }
    }

    int64_t rem_offset = linear_start;
    int64_t rem_len = linear_len;
    for (int64_t p = s_end - 1; p >= s_begin; --p) {
      const int64_t s_dim = src_phys_dims[p];
      const int64_t dim_size = src_phys_sizes[p];
      if (rem_len >= dim_size) {
        if (dim_size <= 0 || rem_len % dim_size != 0 ||
            rem_offset % dim_size != 0) {
          return std::nullopt;
        }
        src_range.starts[s_dim] = 0;
        src_range.limits[s_dim] = dim_size;
        rem_len /= dim_size;
        rem_offset /= dim_size;
      } else {
        const int64_t start_in_dim = rem_offset % dim_size;
        if (start_in_dim + rem_len > dim_size) {
          return std::nullopt;
        }
        src_range.starts[s_dim] = start_in_dim;
        src_range.limits[s_dim] = start_in_dim + rem_len;
        rem_offset /= dim_size;
        rem_len = 1;
      }
    }
    if (rem_offset != 0 || rem_len != 1) {
      return std::nullopt;
    }
  }

  if (is_layout_sensitive && opcode == HloOpcode::kReshape &&
      !ShapeUtil::ReshapeIsBitcast(ShapeWithSliceRange(src_shape, src_range),
                                   ShapeWithSliceRange(dst_shape, dst_range))) {
    return std::nullopt;
  }
  return src_range;
}

// Maps `leaf_slice` backwards across `chain` (`chain[0]` is a user of `reduce`
// and `chain.back()` is `leaf_slice->operand(0)`), returning the `SliceRange`
// at each stage from `reduce` (`ranges[0]`) to `leaf_slice` (`ranges.back()`).
std::optional<std::vector<SliceRange>> MapLeafSliceToReduce(
    const HloInstruction* leaf_slice,
    absl::Span<const HloInstruction* const> chain, bool is_layout_sensitive) {
  std::vector<SliceRange> rev_ranges;
  rev_ranges.reserve(chain.size() + 1);
  rev_ranges.push_back(
      SliceRange{std::vector<int64_t>(leaf_slice->slice_starts().begin(),
                                      leaf_slice->slice_starts().end()),
                 std::vector<int64_t>(leaf_slice->slice_limits().begin(),
                                      leaf_slice->slice_limits().end())});
  for (int64_t i = static_cast<int64_t>(chain.size()) - 1; i >= 0; --i) {
    const HloInstruction* op = chain[i];
    if (IsUnaryElementwise(op)) {
      rev_ranges.push_back(rev_ranges.back());
    } else {
      std::optional<SliceRange> mapped = MapSliceThroughBitcastOrReshape(
          op->operand(0)->shape(), op->shape(), rev_ranges.back(),
          is_layout_sensitive, op->opcode());
      if (!mapped.has_value()) {
        return std::nullopt;
      }
      rev_ranges.push_back(*std::move(mapped));
    }
  }
  absl::c_reverse(rev_ranges);
  return rev_ranges;
}

bool IsScalarConstantZero(const HloInstruction* hlo) {
  if (!hlo->IsConstant() || !ShapeUtil::IsScalar(hlo->shape())) {
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return hlo->literal().GetFirstElement<NativeT>() ==
                 static_cast<NativeT>(0);
        }
        return false;
      },
      hlo->shape().element_type());
}

// Peels at most one widening (or equal-width) byte-aligned non-PRED array
// `kConvert` so that narrow inputs (e.g. bf16/f16 converted to f32) are
// gathered in their narrow type before converting the 1-element result to
// `out_type`. Peeling at most one `kConvert` across the broadcast boundary
// ensures intermediate rounding conversions (e.g. `f32 -> bf16 -> f32`) are
// preserved.
HloInstruction* PeelSingleSafeConvert(HloInstruction* instr,
                                      PrimitiveType out_type) {
  if (instr->opcode() != HloOpcode::kConvert) {
    return instr;
  }
  HloInstruction* src = instr->mutable_operand(0);
  const PrimitiveType src_type = src->shape().element_type();
  if (!primitive_util::IsArrayType(src_type) || src_type == PRED ||
      primitive_util::BitWidth(src_type) < 8 ||
      primitive_util::BitWidth(src_type) > primitive_util::BitWidth(out_type)) {
    return instr;
  }
  return src;
}

// Matched components of a one-hot masked sum reduction:
//   reduce_sum(select(compare(broadcast(indices), iota, EQ),
//                     on_true, broadcast(0)), init_value=0)
struct OneHotReduceMatch {
  HloInstruction* init_value;
  HloInstruction* indices_broadcast;
  HloInstruction* indices;
  HloInstruction* on_true;
  int64_t reduce_dim;
  int64_t reduce_dim_size;
};

// Phase 1 (early rejection): checks whether `reduce` is a compatible one-hot
// masked sum reduction that can be rewritten into a 1-element gather. Returns
// the matched HLO components, or `std::nullopt` if incompatible.
std::optional<OneHotReduceMatch> MatchOneHotReduceToGather(
    HloInstruction* reduce, const AlgebraicSimplifierOptions& options) {
  if (options.is_layout_sensitive() || reduce->shape().IsTuple() ||
      !reduce->shape().is_static() ||
      ShapeUtil::IsZeroElementArray(reduce->shape()) ||
      reduce->dimensions().size() != 1) {
    return std::nullopt;
  }
  HloInstruction* arg = reduce->mutable_operand(0);
  HloInstruction* init_value = reduce->mutable_operand(1);
  if (!arg->shape().is_static()) {
    return std::nullopt;
  }
  const int64_t reduce_dim = reduce->dimensions(0);
  const int64_t reduce_dim_size = arg->shape().dimensions(reduce_dim);
  if (reduce_dim_size <= 1 ||
      reduce_dim_size <= options.very_small_gather_size()) {
    return std::nullopt;
  }
  if (!IsScalarConstantZero(init_value) ||
      !Match(reduce->to_apply()->root_instruction(),
             m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return std::nullopt;
  }

  HloInstruction* pred = nullptr;
  HloInstruction* on_true = nullptr;
  HloInstruction* zero_scalar = nullptr;
  if (!Match(arg, m::Select(m::Op(&pred), m::Op(&on_true),
                            m::Broadcast(m::Op(&zero_scalar)))) ||
      !arg->operand(2)->dimensions().empty() ||
      !IsScalarConstantZero(zero_scalar)) {
    return std::nullopt;
  }

  HloInstruction* compare_lhs = nullptr;
  HloInstruction* compare_rhs = nullptr;
  if (!Match(pred, m::Compare(m::Op(&compare_lhs), m::Op(&compare_rhs))) ||
      pred->comparison_direction() != ComparisonDirection::kEq) {
    return std::nullopt;
  }

  HloInstruction* indices_broadcast = nullptr;
  if (compare_lhs->opcode() == HloOpcode::kIota &&
      Cast<HloIotaInstruction>(compare_lhs)->iota_dimension() == reduce_dim) {
    indices_broadcast = compare_rhs;
  } else if (compare_rhs->opcode() == HloOpcode::kIota &&
             Cast<HloIotaInstruction>(compare_rhs)->iota_dimension() ==
                 reduce_dim) {
    indices_broadcast = compare_lhs;
  } else {
    return std::nullopt;
  }

  if (indices_broadcast->opcode() != HloOpcode::kBroadcast ||
      absl::c_linear_search(indices_broadcast->dimensions(), reduce_dim) ||
      !absl::c_is_sorted(indices_broadcast->dimensions())) {
    return std::nullopt;
  }

  HloInstruction* indices = indices_broadcast->mutable_operand(0);
  const PrimitiveType indices_type = indices->shape().element_type();
  if (!IsValidOneHotIndexType(indices_type, reduce_dim_size)) {
    return std::nullopt;
  }

  return OneHotReduceMatch{
      /*init_value=*/init_value,
      /*indices_broadcast=*/indices_broadcast,
      /*indices=*/indices,
      /*on_true=*/on_true,
      /*reduce_dim=*/reduce_dim,
      /*reduce_dim_size=*/reduce_dim_size,
  };
}

// Phase 2 (lowering): emits the 1-element gather and in-bounds select
// instructions into `reduce`'s parent computation and returns the replacement
// `select` instruction.
absl::StatusOr<HloInstruction*> CreateOneHotReduceGatherReplacement(
    HloInstruction* reduce, const OneHotReduceMatch& match) {
  HloComputation* computation = reduce->parent();
  const OpMetadata* metadata = &reduce->metadata();
  const int64_t out_rank = reduce->shape().dimensions().size();
  const PrimitiveType out_type = reduce->shape().element_type();
  const PrimitiveType indices_type = match.indices->shape().element_type();
  const Shape indices_full_shape =
      ShapeUtil::ChangeElementType(reduce->shape(), indices_type);
  HloInstruction* indices_full = match.indices;
  if (match.indices_broadcast->dimensions().size() != out_rank) {
    std::vector<int64_t> mapped_indices_dims;
    mapped_indices_dims.reserve(match.indices_broadcast->dimensions().size());
    for (int64_t d : match.indices_broadcast->dimensions()) {
      mapped_indices_dims.push_back(d < match.reduce_dim ? d : d - 1);
    }
    indices_full = computation->AddInstruction(
        HloInstruction::CreateBroadcast(indices_full_shape, match.indices,
                                        mapped_indices_dims),
        metadata);
  }

  HloInstruction* base_values = PeelSingleSafeConvert(match.on_true, out_type);
  const bool peeled_outer_convert = (base_values != match.on_true);
  std::vector<int64_t> value_dims;
  if (base_values->opcode() == HloOpcode::kBroadcast &&
      absl::c_is_sorted(base_values->dimensions()) &&
      absl::c_linear_search(base_values->dimensions(), match.reduce_dim)) {
    value_dims.assign(base_values->dimensions().begin(),
                      base_values->dimensions().end());
    base_values = base_values->mutable_operand(0);
    if (!peeled_outer_convert) {
      base_values = PeelSingleSafeConvert(base_values, out_type);
    }
  } else {
    value_dims.resize(base_values->shape().dimensions().size());
    absl::c_iota(value_dims, 0);
  }

  OneHotGatherSpec spec;
  spec.indices = indices_full;
  spec.num_batch_dims = out_rank;
  spec.gather_source = base_values;
  spec.lhs_is_one_hot = true;
  spec.out_type = out_type;
  spec.zero_scalar = match.init_value;
  for (int64_t sub_dim = 0; sub_dim < value_dims.size(); ++sub_dim) {
    const int64_t full_dim = value_dims[sub_dim];
    if (full_dim == match.reduce_dim) {
      spec.source_contract_dim = sub_dim;
    } else {
      const int64_t out_dim =
          full_dim < match.reduce_dim ? full_dim : full_dim - 1;
      spec.source_batch_to_indices_dim.push_back({sub_dim, out_dim});
    }
  }
  return BuildOneHotGatherAndSelect(computation, spec, metadata);
}

}  // namespace

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSinkBroadcastOperandsOfChainedAdds(
    HloInstruction* add) {
  if (!options_.enable_sink_broadcast()) {
    return false;
  }

  HloInstruction *conv, *constant_0, *broadcast_0, *add_0, *constant_1,
      *broadcast_1;
  if (!Match(add, m::AddAnyOrder(
                      m::AddAnyOrder(
                          &add_0, m::Convolution(&conv, m::Op(), m::Op()),
                          m::Broadcast(&broadcast_0, m::Constant(&constant_0))),
                      m::Broadcast(&broadcast_1, m::Constant(&constant_1))))) {
    return false;
  }

  // Skip when the broadcast shapes and dimensions don't match.
  if (!ShapeUtil::Equal(constant_0->shape(), constant_1->shape()) ||
      broadcast_0->dimensions() != broadcast_1->dimensions()) {
    return false;
  }

  HloInstruction* new_constant_add =
      add->AddInstruction(HloInstruction::CreateBinary(
          constant_0->shape(), HloOpcode::kAdd, constant_0, constant_1));
  HloInstruction* new_bcast =
      add->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_0->shape(), new_constant_add, broadcast_0->dimensions()));
  ABSL_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd,
                                        new_bcast, conv)));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  ABSL_ASSIGN_OR_RETURN(bool replaced, TryToSinkBroadcastOperandsOfChainedAdds(add));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleAdd(add);
}

bool GpuAlgebraicSimplifierVisitor::ConvIsLowerable(
    HloInstruction* conv) const {
  return ConvRewriter::ConvIsLowerable(conv);
}

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSimplifyOneHotReduceToGather(
    HloInstruction* reduce) {
  std::optional<OneHotReduceMatch> match =
      MatchOneHotReduceToGather(reduce, options_);
  if (!match.has_value()) {
    return false;
  }
  ABSL_ASSIGN_OR_RETURN(HloInstruction * replacement,
                   CreateOneHotReduceGatherReplacement(reduce, *match));
  ABSL_RETURN_IF_ERROR(ReplaceInstruction(reduce, replacement));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleReduce(
    HloInstruction* reduce) {
  ABSL_ASSIGN_OR_RETURN(bool replaced, TryToSimplifyOneHotReduceToGather(reduce));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleReduce(reduce);
}

bool GpuAlgebraicSimplifierVisitor::IsFusibleUnstridedSlice(
    const HloInstruction* slice, const HloInstruction* operand) const {
  return hlo_instruction_utils::IsUnstridedSlice(slice) && !slice->IsDead() &&
         slice->shape().is_static() &&
         !ShapeUtil::IsZeroElementArray(slice->shape()) &&
         !ShapeUtil::SameDimensions(slice->shape(),
                                    slice->operand(0)->shape()) &&
         !slice->HasControlDependencies() && !HasNonReplicatedSharding(slice) &&
         (!options_.is_layout_sensitive() ||
          LayoutUtil::Equal(slice->shape().layout(),
                            operand->shape().layout()));
}

absl::StatusOr<bool> GpuAlgebraicSimplifierVisitor::TryToSinkReduceThroughSlice(
    HloInstruction* slice) {
  if (!IsFusibleUnstridedSlice(slice, slice->operand(0))) {
    return false;
  }

  const bool is_layout_sensitive = options_.is_layout_sensitive();
  const HloInstruction* root = slice->parent()->root_instruction();

  std::vector<const HloInstruction*> slice_chain;
  HloInstruction* curr = slice->mutable_operand(0);
  while (curr->opcode() != HloOpcode::kReduce) {
    if (static_cast<int64_t>(slice_chain.size()) >=
            kMaxReduceSinkPassthroughDepth ||
        !IsPassthroughOp(curr, root, is_layout_sensitive)) {
      return false;
    }
    slice_chain.push_back(curr);
    curr = curr->mutable_operand(0);
  }
  absl::c_reverse(slice_chain);
  HloInstruction* reduce = curr;

  if (!reduce->shape().IsArray() || ShapeUtil::IsScalar(reduce->shape()) ||
      reduce->HasControlDependencies() || HasNonReplicatedSharding(reduce) ||
      reduce == root) {
    return false;
  }
  HloInstruction* reduce_input = reduce->mutable_operand(0);
  if (!reduce_input->shape().is_static()) {
    return false;
  }

  std::vector<const HloInstruction*> active_chain;
  int64_t remaining_elements = ShapeUtil::ElementsIn(reduce->shape());
  int64_t live_leaf_count = 0;
  auto collect_and_validate_leaves = [&](auto& self,
                                         const HloInstruction* node) -> bool {
    for (const HloInstruction* user : node->users()) {
      if (user->IsDead()) {
        return false;
      }
      if (IsFusibleUnstridedSlice(user, node)) {
        const int64_t leaf_elements = ShapeUtil::ElementsIn(user->shape());
        if (live_leaf_count >= kMaxReduceSinkLeaves ||
            leaf_elements > remaining_elements ||
            !MapLeafSliceToReduce(user, active_chain, is_layout_sensitive)
                 .has_value()) {
          return false;
        }
        remaining_elements -= leaf_elements;
        ++live_leaf_count;
      } else if (static_cast<int64_t>(active_chain.size()) <
                     kMaxReduceSinkPassthroughDepth &&
                 IsPassthroughOp(user, root, is_layout_sensitive)) {
        active_chain.push_back(user);
        const bool ok = self(self, user);
        active_chain.pop_back();
        if (!ok) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  };
  if (!collect_and_validate_leaves(collect_and_validate_leaves, reduce) ||
      live_leaf_count == 0) {
    return false;
  }

  std::optional<std::vector<SliceRange>> slice_ranges =
      MapLeafSliceToReduce(slice, slice_chain, is_layout_sensitive);
  TF_RET_CHECK(slice_ranges.has_value());
  const SliceRange& reduce_slice_range = (*slice_ranges)[0];

  const int64_t input_rank = reduce_input->shape().dimensions().size();
  std::vector<int64_t> new_starts;
  std::vector<int64_t> new_limits;
  std::vector<int64_t> new_strides;
  new_starts.reserve(input_rank);
  new_limits.reserve(input_rank);
  new_strides.reserve(input_rank);

  Shape new_slice_shape = reduce_input->shape();
  int64_t out_dim = 0;
  for (int64_t in_dim = 0; in_dim < input_rank; ++in_dim) {
    if (absl::c_linear_search(reduce->dimensions(), in_dim)) {
      new_starts.push_back(0);
      new_limits.push_back(reduce_input->shape().dimensions(in_dim));
      new_strides.push_back(1);
    } else {
      const int64_t start = reduce_slice_range.starts[out_dim];
      const int64_t limit = reduce_slice_range.limits[out_dim];
      new_starts.push_back(start);
      new_limits.push_back(limit);
      new_strides.push_back(1);
      new_slice_shape.set_dimensions(in_dim, limit - start);
      ++out_dim;
    }
  }
  TF_RET_CHECK(out_dim ==
               static_cast<int64_t>(reduce->shape().dimensions().size()));

  HloComputation* computation = slice->parent();
  HloInstruction* new_slice =
      computation->AddInstruction(HloInstruction::CreateSlice(
          new_slice_shape, reduce_input, new_starts, new_limits, new_strides));
  slice->SetupDerivedInstruction(new_slice);

  Shape new_reduce_shape =
      slice_chain.empty()
          ? slice->shape()
          : ShapeWithSliceRange(reduce->shape(), reduce_slice_range);
  HloInstruction* replacement =
      computation->AddInstruction(HloInstruction::CreateReduce(
          new_reduce_shape, new_slice, reduce->mutable_operand(1),
          reduce->dimensions(), reduce->to_apply()));
  reduce->SetupDerivedInstruction(replacement);

  for (size_t k = 0; k < slice_chain.size(); ++k) {
    const HloInstruction* op = slice_chain[k];
    Shape next_shape =
        (k + 1 == slice_chain.size())
            ? slice->shape()
            : ShapeWithSliceRange(op->shape(), (*slice_ranges)[k + 1]);
    if (IsUnaryElementwise(op)) {
      replacement = computation->AddInstruction(
          op->CloneWithNewOperands(next_shape, {replacement}));
    } else if (op->opcode() == HloOpcode::kBitcast) {
      HloInstruction* new_bitcast = computation->AddInstruction(
          HloInstruction::CreateBitcast(next_shape, replacement));
      op->SetupDerivedInstruction(new_bitcast);
      replacement = new_bitcast;
    } else {
      TF_RET_CHECK(op->opcode() == HloOpcode::kReshape);
      HloInstruction* new_reshape = computation->AddInstruction(
          HloInstruction::CreateReshape(next_shape, replacement));
      op->SetupDerivedInstruction(new_reshape);
      replacement = new_reshape;
    }
  }
  ABSL_RETURN_IF_ERROR(ReplaceInstruction(slice, replacement));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  ABSL_ASSIGN_OR_RETURN(bool replaced, TryToSinkReduceThroughSlice(slice));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleSlice(slice);
}

bool GpuAlgebraicSimplifierVisitor::SupportedDotPrecisionConfig(
    const PrecisionConfig& config, bool has_contracting_dim) {
  if (!has_contracting_dim) {
    return config.algorithm() == PrecisionConfig::ALG_UNSET ||
           config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
  }
  return config.algorithm() == PrecisionConfig::ALG_UNSET ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
}

absl::StatusOr<HloInstruction*>
GpuAlgebraicSimplifierVisitor::MakeMultiplyForPrecisionAlgorithm(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) {
  return MakeMultiplyForDotPrecisionAlgorithm(
      lhs, rhs, dot->precision_config().algorithm());
}

absl::StatusOr<bool> GpuAlgebraicSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  GpuAlgebraicSimplifierVisitor visitor(options_, compute_capability_, this);
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (visitor.Run(comp, options_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), after:\n" + module->ToString());
  return changed;
}

namespace {

class GpuAlgebraicSimplifierFactory : public HloPassFactory {
 public:
  absl::string_view name() const override { return "gpu-algebraic-simplifier"; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext& context,
      const HloPassOptions& options) const override {
    ABSL_ASSIGN_OR_RETURN(const se::DeviceDescription& device_description,
                     context.Get<const se::DeviceDescription>());
    AlgebraicSimplifierOptions simplifier_options;
    options.MergeInto(simplifier_options);
    return std::make_unique<GpuAlgebraicSimplifier>(
        simplifier_options, device_description.gpu_compute_capability());
  }
};

const HloPassFactory* const kGpuAlgebraicSimplifierRegistration =
    RegisterHloPassFactory<GpuAlgebraicSimplifierFactory>();

}  // namespace

}  // namespace xla::gpu
