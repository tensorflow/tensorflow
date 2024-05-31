/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/triton_tiling_propagation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/triton_support.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace {

// The input is a map from dimension index to DimIterationSpec. The function
// removes dimensions that have a trivial DimIterationSpec.
absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>
FilterTrivialDims(
    const absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>&
        dim_iter_specs) {
  absl::flat_hash_map<int, TensorIterationSpec::DimIterationSpec>
      non_trivial_dim_iteration_specs;
  for (const auto& [dim, dim_spec] : dim_iter_specs) {
    if (dim_spec.size() == 1 && dim_spec[0].count == 1) {
      continue;
    }
    non_trivial_dim_iteration_specs[dim] = dim_spec;
  }
  return non_trivial_dim_iteration_specs;
}

}  // namespace

const TensorIterationSpec::DimIterationSpec* TensorIterationSpec::Find(
    const int dimension) const {
  if (auto it = dim_iteration_specs_.find(dimension);
      it != dim_iteration_specs_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<int> TensorIterationSpec::GetDimensions() const {
  std::vector<int> result;
  result.reserve(dim_iteration_specs_.size());
  for (const auto& [dim, _] : dim_iteration_specs_) {
    result.push_back(dim);
  }
  return result;
}

bool TensorIterationSpec::IsPhysicallyEquivalent(
    const TensorIterationSpec& other) const {
  // Filter out trivial dims since they don't affect physical representation.
  const absl::flat_hash_map<int, DimIterationSpec>
      non_trivial_dim_iteration_specs = FilterTrivialDims(dim_iteration_specs_);
  const absl::flat_hash_map<int, DimIterationSpec>
      other_non_trivial_dim_iteration_specs =
          FilterTrivialDims(other.dim_iteration_specs_);

  if (non_trivial_dim_iteration_specs.size() !=
      other_non_trivial_dim_iteration_specs.size()) {
    return false;
  }

  for (const auto& pair : non_trivial_dim_iteration_specs) {
    int dimension = pair.first;
    const DimIterationSpec& dim_iter_spec = pair.second;
    auto other_it = other_non_trivial_dim_iteration_specs.find(dimension);
    if (other_it == other_non_trivial_dim_iteration_specs.end()) {
      return false;
    }
    const DimIterationSpec& other_dim_iter_spec = other_it->second;
    if (dim_iter_spec.size() != other_dim_iter_spec.size()) {
      return false;
    }
    for (size_t i = 0; i < dim_iter_spec.size(); i++) {
      if (!dim_iter_spec[i].IsPhysicallyEquivalent(other_dim_iter_spec[i])) {
        return false;
      }
    }
  }
  return true;
}

std::string TensorIterationSpec::IterationSpecFragment::ToString() const {
  return absl::StrCat("{stride=", stride, ", count=", count,
                      ", slice_start=", slice_start,
                      ", sliced_count=", sliced_count, ", subfragments=[",
                      absl::StrJoin(subfragments, ", "), "]}");
}

std::string TensorIterationSpec::ToString() const {
  return absl::StrCat(
      "{",
      absl::StrJoin(dim_iteration_specs_, ", ",
                    [&](std::string* s, const auto& kv) {
                      absl::StrAppend(
                          s, kv.first, ": ", "[",
                          absl::StrJoin(kv.second, ", ",
                                        [&](std::string* ss, const auto& v) {
                                          absl::StrAppend(ss, v.ToString());
                                        }),
                          "]");
                    }),
      "}");
}

namespace triton_fusion {

using Fragment = DimensionOrder::Fragment;
using Fragments = DimensionOrder::Fragments;
using FragmentOrders = DimensionOrder::FragmentOrders;

/*static*/ DimensionOrder DimensionOrder::FromDotOperandOrOutput(
    const HloInstruction& hlo, const int split_k_dimension_index) {
  DimensionOrder dim_order;
  dim_order.tensor_fragments_order_.reserve(hlo.shape().rank());
  for (const int i : hlo.shape().layout().minor_to_major()) {
    int target_dim_number = i;
    if (i == split_k_dimension_index) {
      CHECK(!dim_order.tensor_fragments_order_.empty())
          << "The split-K batch dimension has be preceded by the contracting "
             "dimension it originates from by construction.";
      target_dim_number =
          dim_order.tensor_fragments_order_.back().dst_dim_number();
    }
    dim_order.dim_fragments_orders_[target_dim_number].push_back(
        dim_order.tensor_fragments_order_.size());
    dim_order.tensor_fragments_order_.push_back(
        Fragment{target_dim_number, hlo.shape().dimensions(i)});
  }
  return dim_order;
}

/*static*/ DimensionOrder DimensionOrder::FromSoftmaxRoot(
    const HloInstruction& hlo) {
  DimensionOrder dim_order;
  dim_order.tensor_fragments_order_.reserve(hlo.shape().rank());
  dim_order.dim_fragments_orders_[kSoftmaxReductionDimension].push_back(
      dim_order.tensor_fragments_order_.size());
  dim_order.tensor_fragments_order_.push_back(
      Fragment{kSoftmaxReductionDimension, hlo.shape().dimensions_minor(0)});
  for (int i = 1; i < hlo.shape().rank(); ++i) {
    dim_order.dim_fragments_orders_[kSoftmaxBatchDimension].push_back(
        dim_order.tensor_fragments_order_.size());
    dim_order.tensor_fragments_order_.push_back(
        Fragment{kSoftmaxBatchDimension, hlo.shape().dimensions_minor(i)});
  }
  return dim_order;
}

std::string DimensionOrder::Fragment::ToString() const {
  return absl::StrCat(dst_dim_number_, ":", count_, ":", slice_start_, "-",
                      sliced_count_);
}

std::string DimensionOrder::ToString() const {
  std::string ret = absl::StrJoin(tensor_fragments_order_, " - ",
                                  [](std::string* out, const Fragment& f) {
                                    absl::StrAppend(out, f.ToString(), " ");
                                  });
  absl::StrAppend(&ret, "|");
  for (const auto& [dim, fragments] : dim_fragments_orders_) {
    absl::StrAppend(&ret, dim, ":", absl::StrJoin(fragments, ","), " ");
  }
  return ret;
}

TensorIterationSpec DimensionOrder::ToTensorIterationSpec() const {
  const Fragments& dim_fragments = TensorFragmentsOrder();
  TensorIterationSpec tensor_spec;
  int64_t accumulated_stride = 1;
  int last_dim = -1;
  for (int dim_order_index = 0; dim_order_index < dim_fragments.size();
       ++dim_order_index) {
    const DimensionOrder::Fragment& fragment = dim_fragments[dim_order_index];
    VLOG(6) << fragment.ToString();

    TensorIterationSpec::DimIterationSpec& dim_spec =
        tensor_spec[fragment.dst_dim_number()];
    if (last_dim == fragment.dst_dim_number()) {
      // Remove previous 1-sized subfragment if present.
      if (!dim_spec.empty() && !dim_spec.back().subfragments.empty() &&
          dim_spec.back().subfragments.back() == 1) {
        dim_spec.back().subfragments.pop_back();
      }
      // Contiguous dimension, split only logically. Merge it back.
      if (fragment.full_count() > 1) {
        CHECK(!dim_spec.empty());
        CHECK(!dim_spec.back().is_sliced())
            << "Only the major-most fragment can have an offset.";
        dim_spec.back().slice_start =
            fragment.slice_start() * dim_spec.back().count;
        dim_spec.back().sliced_count =
            fragment.sliced_count() * dim_spec.back().count;
        dim_spec.back().count *= fragment.full_count();
        dim_spec.back().subfragments.push_back(fragment.sliced_count());
      }
    } else {
      // Add part of the dimension.
      dim_spec.push_back(TensorIterationSpec::IterationSpecFragment{
          accumulated_stride,
          fragment.full_count(),
          fragment.slice_start(),
          fragment.sliced_count(),
          {fragment.sliced_count()}});
    }

    accumulated_stride *= fragment.full_count();
    last_dim = fragment.dst_dim_number();
  }

  // Remove degenerate fragments.
  for (int dim_idx : tensor_spec.GetDimensions()) {
    TensorIterationSpec::DimIterationSpec& dim_spec = tensor_spec[dim_idx];

    // We should not remove the only fragment in a dimension, because if it is
    // removed, the dimension will be removed from the TensorIterationSpec.
    if (dim_spec.size() <= 1) continue;

    TensorIterationSpec::DimIterationSpec filtered_dim_spec;
    absl::c_copy_if(dim_spec, std::back_inserter(filtered_dim_spec),
                    [](const TensorIterationSpec::IterationSpecFragment& f) {
                      return f.count != 1;
                    });
    tensor_spec[dim_idx] = filtered_dim_spec;
  }

  tensor_spec.RemoveEmptyDimensions();
  return tensor_spec;
}

namespace {

// Logical index of a dimension in `shape` labeled with `label` in the
// `dim_order` describing the shape.
std::optional<int> LogicalIndexOfLabeledDimension(
    const Shape& shape, const DimensionOrder& dim_order, const int label) {
  auto fragment_it = dim_order.TensorFragmentsOrder().cbegin();
  for (int dim : shape.layout().minor_to_major()) {
    const int64_t dim_size = shape.dimensions()[dim];
    int64_t fragments_size = 1;
    while (fragments_size < dim_size) {
      fragments_size *= fragment_it->full_count();
      if (fragment_it->dst_dim_number() == label) {
        return dim;
      }
      ++fragment_it;
    }
  }
  return std::nullopt;
}

using Int64OrError = std::variant<int64_t, FusionDecision>;
Int64OrError CombineSplitDimMajorPartSizeReqs(int64_t a, int64_t b) {
  if (a == b || b == kNoSplitRequirement) {
    return a;
  }
  if (a == kNoSplitRequirement) {
    return b;
  }
  return FusionDecision("Conflicting splits of splittable dimension");
}

RequirementsOrError CombineDotRequirements(DotRequirements a,
                                           DotRequirements b) {
  Int64OrError combined_size_req =
      CombineSplitDimMajorPartSizeReqs(a.splittable_dimension_major_part_size,
                                       b.splittable_dimension_major_part_size);
  if (std::holds_alternative<FusionDecision>(combined_size_req)) {
    return std::get<FusionDecision>(combined_size_req);
  }
  return DotRequirements(std::get<int64_t>(combined_size_req));
}

RequirementsOrError CombineSoftmaxRequirements(SoftmaxRequirements a,
                                               SoftmaxRequirements b) {
  // SoftmaxRequirements is an empty class for now.
  return a;
}

}  // namespace

RequirementsOrError CombineRequirements(Requirements a,
                                        RequirementsOrError b_or_error) {
  if (std::holds_alternative<FusionDecision>(b_or_error)) {
    return b_or_error;
  }
  const Requirements& b = std::get<Requirements>(b_or_error);
  if (std::holds_alternative<DotRequirements>(b)) {
    return CombineDotRequirements(std::get<DotRequirements>(a),
                                  std::get<DotRequirements>(b));
  }
  return CombineSoftmaxRequirements(std::get<SoftmaxRequirements>(a),
                                    std::get<SoftmaxRequirements>(b));
}

namespace {

// If the dimension order is supported by the triton emitters, this returns
// which requirements does this order impose on the fusion.
//
// All subdimensions within a dimension have to be ordered.
RequirementsOrError GetRequirementsIfSupportedOrder(
    const DimensionOrder& order, const HeroProperties& properties) {
  VLOG(8) << order.ToString();
  int64_t split_dim_major_part = kNoSplitRequirement;
  const Fragments& tensor_dim_fragments = order.TensorFragmentsOrder();
  for (const auto& [dim_index, dim_fragments] : order.DimFragmentsOrders()) {
    CHECK(!dim_fragments.empty());
    for (int i = 0; i < dim_fragments.size() - 1; ++i) {
      if (tensor_dim_fragments[dim_fragments[i]].is_sliced()) {
        return "Sliced non-major-most fragment.";
      }
    }
    int group_counter = 0;
    int last_seen_group_last_fragment_index = -1;
    auto fragment_it = dim_fragments.cbegin();
    while (true) {
      if (fragment_it == dim_fragments.cend()) {
        break;
      }
      int64_t grouped_size = tensor_dim_fragments[*fragment_it].full_count();
      // Gather contiguous fragments: they have consecutive indices.
      while ((fragment_it + 1) != dim_fragments.cend() &&
             *(fragment_it + 1) == *fragment_it + 1) {
        ++fragment_it;
        grouped_size *= tensor_dim_fragments[*fragment_it].full_count();
      }
      // Ignore 1-sized groups of fragments.
      if (grouped_size == 1) {
        ++fragment_it;
        continue;
      }

      if (last_seen_group_last_fragment_index > *fragment_it) {
        return "Transpose within a dimension.";
      }

      ++group_counter;
      if (group_counter > 1) {
        if (!std::holds_alternative<DotProperties>(properties)) {
          return "Splitting a dimension is not supported for Softmax.";
        }
        // Only the dimension indicated by `splittable_dimension_index` (if any)
        // can be split physically once by other dimensions. Other ones can be
        // only split logically.
        const int splittable_dimension_index =
            std::get<DotProperties>(properties).splittable_dimension_index;
        if (dim_index == splittable_dimension_index) {
          if (group_counter == 2) {
            if (split_dim_major_part != kNoSplitRequirement &&
                split_dim_major_part != grouped_size) {
              return "Conflicting splits of splittable dimension";
            }
            split_dim_major_part = grouped_size;
          } else if (group_counter > 2) {
            return "2nd split of a splittable dimension.";
          }
        } else {
          return "Unsupported split of a dimension.";
        }
      }

      last_seen_group_last_fragment_index = *fragment_it;
      ++fragment_it;
    }
  }

  if (std::holds_alternative<DotProperties>(properties)) {
    return DotRequirements(split_dim_major_part);
  }
  return SoftmaxRequirements{};
}

// Apply GetRequirementsIfSupportedOrder() to all known
// dimension orders around `hlo` and combine the result.
RequirementsOrError GetRequirementsIfSupportedOrders(
    const HloInstruction& hlo, const DimOrderMap& dim_orders,
    const HeroProperties& properties) {
  const Requirements empty_requirements =
      std::holds_alternative<DotProperties>(properties)
          ? Requirements(DotRequirements(kNoSplitRequirement))
          : Requirements(SoftmaxRequirements{});
  auto get_requirements =
      [&](const HloInstruction& instr) -> RequirementsOrError {
    if (auto it = dim_orders.find(&instr); it != dim_orders.end()) {
      return GetRequirementsIfSupportedOrder(it->second, properties);
    }
    return empty_requirements;
  };

  Requirements requirements = empty_requirements;
  for (const HloInstruction* operand : hlo.operands()) {
    RequirementsOrError requirements_or_error =
        CombineRequirements(requirements, get_requirements(*operand));
    if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
      return requirements_or_error;
    }
    requirements = std::get<Requirements>(requirements_or_error);
  }

  return CombineRequirements(requirements, get_requirements(hlo));
}

DimOrderMap GetPropagatedDimOrdersForElementwise(
    const HloInstruction& hlo, TransformDirection direction,
    const DimensionOrder& src_dim_order) {
  if (direction == TransformDirection::kOutputToInput) {
    DimOrderMap map;
    for (const HloInstruction* operand : hlo.operands()) {
      map.insert({operand, src_dim_order});
    }
    return map;
  }

  return {{&hlo, src_dim_order}};
}

const HloInstruction& GetSourceHlo(const HloInstruction& hlo,
                                   TransformDirection direction) {
  CHECK_GE(hlo.operand_count(), 1);

  if (direction == TransformDirection::kOutputToInput) {
    return hlo;
  }
  return *hlo.operand(0);
}

using ConstInstructionVector = absl::InlinedVector<const HloInstruction*, 2>;
ConstInstructionVector GetDestHlos(const HloInstruction& hlo,
                                   TransformDirection direction) {
  if (direction == TransformDirection::kInputToOutput) {
    return {&hlo};
  }

  ConstInstructionVector hlos;
  hlos.reserve(hlo.operands().size());
  for (const HloInstruction* operand : hlo.operands()) {
    hlos.push_back(operand);
  }
  return hlos;
}

const HloInstruction& GetDestHlo(const HloInstruction& hlo,
                                 TransformDirection direction) {
  CHECK_EQ(hlo.operand_count(), 1);

  if (direction == TransformDirection::kInputToOutput) {
    return hlo;
  }

  return *hlo.operand(0);
}

DimOrderMapOrError GetPropagatedDimOrdersForBitcast(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const HeroProperties& properties) {
  const HloInstruction& dst = GetDestHlo(hlo, direction);
  const Shape& dst_shape = dst.shape();
  const Fragments& src_fragments_order = src_dim_order.TensorFragmentsOrder();
  DimOrderMap dst_dim_orders;
  DimensionOrder& dst_dim_order =
      dst_dim_orders.insert({&dst, DimensionOrder()}).first->second;
  Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
  bool dst_remainder_comes_from_reduce_dim = false;
  // Size of not yet assigned part of current target dimension.
  int64_t dst_remaining_size = 1;
  // Track destination fragments created from a source one.
  absl::flat_hash_map<const Fragment*, std::vector<int>> src_to_dst;
  // Iterate in parallel over source dimension order and target dimensions
  // in minor_to_major order. Find groups of dimensions of equal size
  // and project the source dimension order onto the destination.
  auto dst_dim_it = dst_shape.layout().minor_to_major().cbegin();
  const auto dst_dim_end = dst_shape.layout().minor_to_major().cend();
  for (auto src_dim = src_fragments_order.cbegin();
       src_dim != src_fragments_order.cend(); ++src_dim) {
    auto add_new_fragment = [&](const Fragment& fragment) {
      dst_fragments_order.push_back(fragment);
      src_to_dst[&*src_dim].push_back(dst_fragments_order.size() - 1);
    };
    if (std::holds_alternative<SoftmaxProperties>(properties) &&
        src_dim->dst_dim_number() ==
            std::get<SoftmaxProperties>(properties).softmax_batch_dimension) {
      // Special handling for softmax batch dimension: allow arbitrary reshapes
      // on it because it's guaranteed by the construction of the fusion to have
      // no physical alterations like transposes.
      // Find a continuous group of fragments corresponding to this dimension in
      // the source and assign the corresponding size in fragments of the
      // destination ignoring the source ones.

      // If there is dst_remaining_size leftover from our previous src_dim,
      // and it came from a reduce dim, we cannot tile it in a batch dim.
      if (dst_remainder_comes_from_reduce_dim) {
        return R"(Unsupported bitcast splits dimension between batch and
                  reduction dimensions in softmax)";
      }

      dst_remaining_size = src_dim->full_count();
      while (src_dim + 1 != src_fragments_order.cend() &&
             (src_dim + 1)->dst_dim_number() == src_dim->dst_dim_number()) {
        ++src_dim;
        dst_remaining_size *= src_dim->full_count();
      }
      while (dst_remaining_size > 1) {
        CHECK(dst_dim_it != dst_dim_end);
        add_new_fragment(Fragment{src_dim->dst_dim_number(),
                                  dst_shape.dimensions(*dst_dim_it)});
        dst_remaining_size /= dst_shape.dimensions(*dst_dim_it);
        ++dst_dim_it;
      }
      continue;
    }
    if (dst_remaining_size >= src_dim->full_count()) {
      if (dst_remaining_size % src_dim->full_count()) {
        return "Unsupported bitcast";
      }
      // Source dimension fragment completely fits into the destination one:
      // just copy it as is.
      add_new_fragment(*src_dim);
      // Update the size of the remaining part of the destination that is
      // carried over to next source dimensions.
      dst_remaining_size /= src_dim->full_count();
    } else {
      // Source is larger than destination.
      // Assign further destination dimensions.
      // Size of the not yet assigned part of the source dimension.
      int64_t src_remaining_size = src_dim->full_count();
      // Handle dimension splits.
      if (dst_remaining_size > 1) {
        // If there is a remaining fragment of a previous destination dimension
        // assign it first.
        if (src_remaining_size % dst_remaining_size || (src_dim->is_sliced())) {
          return "Unsupported bitcast";
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), dst_remaining_size});
        // Update the size of the fragment remaining to assign.
        src_remaining_size /= dst_remaining_size;
        dst_remaining_size = 1;
      }
      while (src_remaining_size > 1) {
        // Assign destination dimensions until the source remainder is covered.
        CHECK(dst_dim_it != dst_dim_end);
        int64_t dst_dim_size = dst_shape.dimensions(*dst_dim_it);
        int64_t new_fragment_size = dst_dim_size;
        if (dst_dim_size > src_remaining_size) {
          // If adding the next destination dimension exceeds source fragment
          // size assign the remainder of the source and carry over the
          // remainder of the destination.
          if (dst_dim_size % src_remaining_size) {
            return "Unsupported bitcast";
          }
          dst_remaining_size = dst_dim_size / src_remaining_size;
          new_fragment_size = src_remaining_size;
        }
        if (src_dim->is_sliced()) {
          return "Unsupported bitcast";
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), new_fragment_size});
        src_remaining_size /= new_fragment_size;
        ++dst_dim_it;
      }
    }

    // We cannot tile a single dim with fragments across both reduce and batch
    // dimensions. As such, if we have a dst remainder leftover from tiling a
    // src fragment on the reduce dimension in softmax, we must only tile it
    // with other src_dim fragments on the reduce dimension.
    dst_remainder_comes_from_reduce_dim =
        (dst_remaining_size > 1 &&
         std::holds_alternative<SoftmaxProperties>(properties) &&
         src_dim->dst_dim_number() == std::get<SoftmaxProperties>(properties)
                                          .softmax_reduction_dimension);
  }
  CHECK_EQ(dst_remaining_size, 1);

  // Handle remaining major dimensions of the destination. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  while (dst_dim_it != dst_dim_end) {
    if (dst_shape.dimensions(*dst_dim_it) != 1) {
      return "Unsupported bitcast";
    }
    if (!dst_fragments_order.empty()) {
      dst_fragments_order.push_back(
          Fragment{dst_fragments_order.back().dst_dim_number(), 1});
      src_to_dst[&src_fragments_order.back()].push_back(
          dst_fragments_order.size() - 1);
    }
    ++dst_dim_it;
  }

  FragmentOrders& dst_dim_fragment_orders = dst_dim_order.DimFragmentsOrders();
  for (const auto& [dim_index, dim_sequence] :
       src_dim_order.DimFragmentsOrders()) {
    std::vector<int>& dst = dst_dim_fragment_orders[dim_index];
    dst.reserve(dim_sequence.size());
    for (const int src : dim_sequence) {
      std::copy(src_to_dst[&src_fragments_order[src]].cbegin(),
                src_to_dst[&src_fragments_order[src]].cend(),
                std::back_inserter(dst));
    }
  }

  return dst_dim_orders;
}

// Handle copy, transpose, broadcast or reduce.
// Common between them is that they alter the tensor dimensions or their order
// and the way to handle layouts.
DimOrderMapOrError GetPropagatedDimOrdersForDimAlteringOp(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const HeroProperties& properties) {
  // Temporary storage for new fragments local to this function.
  // Please keep this as the first local variable of this function, with type
  // std::list to make sure that all pointers to elements of this remain valid
  // throughout the entire function. std::deque would also work but it is
  // unnecessarily big for a typical size of 1.
  std::list<Fragment> new_fragments;

  const HloInstruction& src = GetSourceHlo(hlo, direction);
  // Note: copying instead of using a const reference because
  // some operations (slice) will modify fragment properties in-place.
  Fragments src_fragments_order = src_dim_order.TensorFragmentsOrder();
  if (hlo.opcode() == HloOpcode::kSlice &&
      ShapeUtil::IsEffectiveScalar(hlo.shape())) {
    return FusionDecision("Slice to scalar is not implemented yet.");
  }
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // full dimensions and matching by total size.
  std::vector<std::vector<Fragment*>> src_physical;
  src_physical.reserve(src.shape().rank());
  if (src_fragments_order.size() < src.shape().rank()) {
    // It's not supported currently to further propagate dimensions after
    // reaching a trivial sized tensor. We could probably support it, but now we
    // just prevent crashing here.
    return FusionDecision("Cannot propagate further from trivial sized tensor");
  }
  auto src_fragment_it = src_fragments_order.begin();
  for (int64_t dim_index : src.shape().layout().minor_to_major()) {
    const int64_t dim_size = src.shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    std::vector<Fragment*> subdim_group;
    do {
      CHECK(src_fragment_it != src_fragments_order.end());
      subdim_size_accumulator *= src_fragment_it->full_count();
      subdim_group.push_back(&*src_fragment_it);
      ++src_fragment_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    src_physical.push_back(subdim_group);
  }

  // Source physical -> source logical.
  std::vector<std::vector<Fragment*>> src_logical;
  src_logical.resize(src_physical.size());
  for (int i = 0; i < src_physical.size(); ++i) {
    src_logical[src.shape().layout().minor_to_major(i)] = src_physical[i];
  }

  DimOrderMap dst_dim_orders;
  int64_t concat_accumulated_size = 0;
  for (const HloInstruction* dst : GetDestHlos(hlo, direction)) {
    DimensionOrder& dst_dim_order =
        dst_dim_orders.insert({dst, DimensionOrder()}).first->second;
    // Source logical -> destination logical.
    std::vector<std::vector<Fragment*>> dst_logical;
    if (hlo.opcode() == HloOpcode::kTranspose) {
      const auto* transpose = Cast<HloTransposeInstruction>(&hlo);
      std::vector<int64_t> permutation(transpose->dimensions().cbegin(),
                                       transpose->dimensions().cend());
      if (direction == TransformDirection::kInputToOutput) {
        permutation = InversePermutation(permutation);
      }
      dst_logical.resize(permutation.size());
      for (int i = 0; i < permutation.size(); ++i) {
        dst_logical[permutation[i]] = src_logical[i];
      }
    } else if (hlo.opcode() == HloOpcode::kBroadcast) {
      const auto* broadcast = Cast<HloBroadcastInstruction>(&hlo);
      dst_logical.resize(broadcast->dimensions().size());
      for (int i = 0; i < broadcast->dimensions().size(); ++i) {
        dst_logical[i] = src_logical[broadcast->dimensions()[i]];
      }
    } else if (hlo.opcode() == HloOpcode::kReduce) {
      // Operand 1 (the neutral value) has to be a scalar.
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* reduce = Cast<HloReduceInstruction>(&hlo);
      dst_logical.resize(src_logical.size() + reduce->dimensions().size());

      if (reduce->dimensions().size() != 1) {
        return FusionDecision("Unsupported reduction.");
      } else if (reduce->dimensions().front() !=
                 reduce->operand(0)->shape().rank() - 1) {
        return FusionDecision("Only row reductions are supported.");
      }
      for (int i = 0; i < dst_logical.size(); ++i) {
        if (i == reduce->dimensions().front()) {
          // This way to assign the reduction dimension will only work for
          // softmax fusions with known patterns for now. Generally a reduction
          // should create a new tiled dimension.
          dst_logical[i] = {&new_fragments.emplace_back(
              std::get<SoftmaxProperties>(properties)
                  .softmax_reduction_dimension,
              reduce->operand(0)->shape().dimensions(i))};
        } else {
          dst_logical[i] = src_logical[i];
        }
      }
    } else if (hlo.opcode() == HloOpcode::kConcatenate) {
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        if (i == hlo.concatenate_dimension()) {
          if (src_logical[i].size() != 1 || src_logical[i][0]->is_sliced()) {
            return FusionDecision("Unsupported concatenation.");
          }
          const Fragment& src_fragment = *src_logical[i][0];
          Fragment& dst_fragment = new_fragments.emplace_back(
              src_fragment.dst_dim_number(), dst->shape().dimensions(i));
          dst_fragment.set_slice(-concat_accumulated_size,
                                 dst->shape().dimensions(i));
          concat_accumulated_size += dst->shape().dimensions(i);
          dst_logical[i].push_back(&dst_fragment);
        } else {
          dst_logical[i] = src_logical[i];
        }
      }
    } else if (hlo.opcode() == HloOpcode::kCopy) {
      // Copy preserves the logical shape, just permutes the layout.
      CHECK(ShapeUtil::SameDimensions(src.shape(), dst->shape()));
      dst_logical = src_logical;
    } else if (hlo.opcode() == HloOpcode::kPad) {
      // Operand 1 (the padding value) has to be a scalar.
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* pad = Cast<HloPadInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        // This only handles the padding added by
        // PadDotOperandsIfNeededForSplitK, which sets only edge_padding_high.
        const int padding =
            pad->padding_config().dimensions(i).edge_padding_high();
        CHECK_EQ(pad->padding_config().dimensions(i).edge_padding_low(), 0);
        CHECK_EQ(pad->padding_config().dimensions(i).interior_padding(), 0);
        if (padding == 0) {
          dst_logical[i] = src_logical[i];
        } else {
          // This case is executed for the contracting dimension when we run the
          // TritonFusionAnalysis after the padding and the split-k transform
          // are applied.
          const std::vector<Fragment*>& fragments = src_logical[i];

          // We must have 2 non-trivial fragments at this point. We may have
          // more than 2 fragments if there are trivial fragments with count 1.
          CHECK_GE(fragments.size(), 2);
          // The dst_dim_numbers must be the same for all fragments of the
          // contracting dimension after applying split-k.
          CHECK(absl::c_all_of(fragments, [&](const Fragment* fragment) {
            return fragment->dst_dim_number() ==
                   fragments.front()->dst_dim_number();
          }));

          std::vector<Fragment*> non_trivial_fragments;
          absl::c_copy_if(fragments, std::back_inserter(non_trivial_fragments),
                          [](const Fragment* fragment) {
                            return fragment->full_count() > 1;
                          });
          CHECK_EQ(non_trivial_fragments.size(), 2);
          new_fragments.emplace_back(
              non_trivial_fragments[0]->dst_dim_number(),
              non_trivial_fragments[0]->full_count() *
                      non_trivial_fragments[1]->full_count() -
                  padding);
          dst_logical[i] = {&new_fragments.back()};
        }
      }
    } else if (hlo.opcode() == HloOpcode::kSlice) {
      const auto slice = Cast<HloSliceInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int dim = 0; dim < src_logical.size(); ++dim) {
        dst_logical[dim] = src_logical[dim];
        if (slice->slice_limits(dim) - slice->slice_starts(dim) !=
            dst->shape().dimensions(dim)) {
          if (dst_logical[dim].size() > 1) {
            return FusionDecision("Slicing of fragmented dimension.");
          }
          auto fragment = dst_logical[dim].front();
          fragment->set_count(dst->shape().dimensions(dim));
          // Slicing of an already sliced dimension means adding offsets.
          fragment->set_slice(
              fragment->slice_start() + slice->slice_starts(dim),
              fragment->sliced_count());
        }
      }
    } else if (hlo.opcode() == HloOpcode::kDynamicSlice) {
      // All operands after idx 0 are scalar indices. As such, we do not want
      // to explicitly define dim orders.
      if (dst != &hlo && hlo.operand_index(dst) >= 1) {
        continue;
      }
      const auto dynamic_slice = Cast<HloDynamicSliceInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int dim = 0; dim < src_logical.size(); ++dim) {
        dst_logical[dim] = src_logical[dim];
        if (dynamic_slice->slice_sizes(dim) != dst->shape().dimensions(dim)) {
          if (dst_logical[dim].size() > 1) {
            return FusionDecision("Slicing of fragmented dimension.");
          }
          auto fragment = dst_logical[dim].front();
          fragment->set_count(dst->shape().dimensions(dim));

          // As we do not know which section of the tensor we keep, we retain
          // the whole part.
          fragment->set_slice(fragment->slice_start(),
                              dst->shape().dimensions(dim));
        }
      }
    } else {
      return FusionDecision("Function called on a wrong instruction.");
    }
    // Destination logical -> destination physical and ungroup subdimensions.
    // Map original fragments to the resulting ones to derive their new
    // logical ordering within each dimension.
    absl::flat_hash_map<const Fragment*, int> src_to_dst;
    Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
    FragmentOrders& dst_dim_fragments_order =
        dst_dim_order.DimFragmentsOrders();
    // Remember which dimensions are present before a broadcast;
    // skip cases when already present dimension is being expanded.
    absl::flat_hash_set<int> dim_numbers_present_in_dst;
    for (const int64_t dim_idx : dst->shape().layout().minor_to_major()) {
      for (const Fragment* subdim : dst_logical[dim_idx]) {
        dst_fragments_order.push_back(*subdim);
        src_to_dst[subdim] = dst_fragments_order.size() - 1;
        dim_numbers_present_in_dst.insert(subdim->dst_dim_number());
      }
    }
    for (const auto& [dim_index, dim_sequence] :
         src_dim_order.DimFragmentsOrders()) {
      for (const int fragment_number : dim_sequence) {
        const auto it = src_to_dst.find(&src_fragments_order[fragment_number]);
        if (it == src_to_dst.cend()) {
          if (hlo.opcode() == HloOpcode::kBroadcast &&
              src_fragments_order[fragment_number].full_count() > 1 &&
              dim_numbers_present_in_dst.contains(dim_index)) {
            return FusionDecision("Unsupported broadcast");
          }
          continue;
        }
        dst_dim_fragments_order[dim_index].push_back(it->second);
      }
    }
  }
  return dst_dim_orders;
}

// If possible, propagates `src_dim_order` (describing one side of `hlo`) to
// the other side and returns those dim orders.
DimOrderMapOrError GetPropagatedDimOrders(const HloInstruction& hlo,
                                          const TransformDirection direction,
                                          const DimensionOrder& src_dim_order,
                                          const HeroProperties& properties) {
  VLOG(7) << "Analyzing " << hlo.ToString();
  if (hlo.opcode() != HloOpcode::kParameter &&
      direction == TransformDirection::kOutputToInput &&
      absl::c_any_of(hlo.users(), [](const HloInstruction* user) {
        return (user->opcode() == HloOpcode::kConcatenate ||
                user->opcode() == HloOpcode::kDynamicSlice);
      })) {
    return "No fusion into concatenations or dynamic slice.";
  }
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo_query::IsScalarConstant(&hlo)) {
    CHECK(direction == TransformDirection::kOutputToInput);
    return DimOrderMap{};
  } else if (hlo.opcode() == HloOpcode::kTranspose ||
             hlo.opcode() == HloOpcode::kCopy) {
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kBroadcast) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported broadcast direction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kReduce) {
    if (!std::holds_alternative<SoftmaxProperties>(properties)) {
      return "Reductions are not supported in GEMM fusions yet.";
    }
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported direction of reduction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kPad) {
    if (std::holds_alternative<SoftmaxProperties>(properties)) {
      return "Pad ops are only supported when they are generated as part of "
             "the split-k transform of dot fusions.";
    }
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported pad direction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.operand_count() > 0 &&
             IsTritonSupportedElementwise(
                 hlo.opcode(), hlo.operand(0)->shape().element_type())) {
    return GetPropagatedDimOrdersForElementwise(hlo, direction, src_dim_order);
  } else if (hlo.opcode() == HloOpcode::kBitcast) {
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kSlice) {
    // TODO(b/316637896) Add support for slices in softmax.
    if (std::holds_alternative<SoftmaxProperties>(properties)) {
      return "Slices are not supported in Softmax fusions yet.";
    }
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported slice direction.";
    }

    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kDynamicSlice &&
             direction == TransformDirection::kOutputToInput) {
    // We handle the dynamic slice within EmitTensorPointer, which is only
    // used for GEMM fusions.
    if (!std::holds_alternative<DotProperties>(properties)) {
      return "Dynamic slices for now are only supported in GEMM fusions.";
    }

    if (CodegenDecision decision = IsTritonSupportedDynamicSlice(
            *Cast<HloDynamicSliceInstruction>(&hlo));
        !decision.CanFuse()) {
      // CodegenDecision is actually the same type as FusionDecision.
      return decision;
    }

    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kReshape) {
    if (!ShapeUtil::ReshapeIsBitcast(hlo.operand(0)->shape(), hlo.shape())) {
      return "Non-bitcast reshape.";
    }
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kConcatenate &&
             direction == TransformDirection::kOutputToInput) {
    if (!std::holds_alternative<DotProperties>(properties)) {
      return "Concatenations for now are only supported in GEMM fusions.";
    }

    int64_t noncontracting_dim_label =
        std::get<DotProperties>(properties).noncontracting_dimension;
    const FragmentOrders& src_dim_fragments_orders =
        src_dim_order.DimFragmentsOrders();

    auto noncontracting_dim_fragment_order_it =
        src_dim_fragments_orders.find(noncontracting_dim_label);
    if (noncontracting_dim_fragment_order_it !=
        src_dim_fragments_orders.end()) {
      if (noncontracting_dim_fragment_order_it->second.size() > 1) {
        return "Concatenations on split non-contracting dimensions are "
               "unsupported.";
      }
    }

    auto dim = LogicalIndexOfLabeledDimension(hlo.shape(), src_dim_order,
                                              noncontracting_dim_label);
    if (!dim.has_value() || dim.value() != hlo.concatenate_dimension()) {
      return "Unsupported concatenation.";
    }
    if (absl::c_any_of(hlo.operands(), [&hlo](const HloInstruction* operand) {
          // In the current simple implementation of concatenation the size of
          // each of its inputs along the concatenated dimension has to be
          // divisible by the tile size used for this dimension. Concatenations
          // with any operand not divisible by kMinConcatFragmentSize will not
          // be fused; tiling configurations with tile size for this dimension
          // larger than kMinConcatFragmentSize will not be emitted.
          constexpr int kMinConcatFragmentSize = 128;
          return operand->shape().dimensions(hlo.concatenate_dimension()) %
                     kMinConcatFragmentSize !=
                 0;
        })) {
      return FusionDecision(
          "At least one operand of concatenation can not be perfectly tiled.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  }
  return "Unimplemented instruction.";
}

// Difference of input and output data volumes of an instruction.
int64_t InputMinusOutputBytes(const HloInstruction& hlo) {
  CHECK(!hlo.shape().IsTuple());
  int64_t input_size = 0;
  for (const HloInstruction* operand : hlo.operands()) {
    CHECK(!operand->shape().IsTuple());
    input_size += ShapeUtil::ByteSizeOf(operand->shape());
  }
  return input_size - ShapeUtil::ByteSizeOf(hlo.shape());
}

// Tells if an instruction has no user into which it could be fused.
// More cases should be added here.
bool CanNotBeFusedIntoAUser(const HloInstruction& hlo) {
  return hlo.IsRoot() || (hlo.user_count() == 1 && hlo.users()[0]->IsRoot() &&
                          hlo.users()[0]->opcode() == HloOpcode::kTuple);
}

// Let input and output data volumes of a fusion grow by small amounts.
constexpr int kIoToleranceBytes = 1024;

// Tells that fusing an instruction as an input is efficient.
bool IsInputWorthFusing(const HloInstruction& hlo) {
  if (InputMinusOutputBytes(hlo) <= kIoToleranceBytes) {
    return true;
  }
  if (hlo.user_count() > 1) {
    return false;
  }
  if (hlo.opcode() == HloOpcode::kSlice &&
      hlo_query::AllOperandsAreParametersOrConstants(hlo)) {
    return true;
  }
  return hlo_query::AllOperandsAreParametersOrConstantsWithSingleUser(hlo);
}

// Tells that fusing an instruction as an output is efficient.
bool IsOutputWorthFusing(const HloInstruction& hlo) {
  return CanNotBeFusedIntoAUser(hlo) ||
         InputMinusOutputBytes(hlo) >= -kIoToleranceBytes;
}

FusionDecision IsConversionWorthFusing(const HloInstruction& input,
                                       se::GpuComputeCapability gpu_version) {
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  if (ShapeUtil::ByteSizeOf(input.operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input.shape())) {
    return "Narrowing conversion.";
  }
  return FusionDecision{};
}

}  // namespace

DimOrdersAndReqsOrError GetPropagatedDimOrdersAndRequirements(
    const HloInstruction& hlo, const DimensionOrder& src_dim_order,
    TransformDirection direction, const HeroProperties& properties) {
  DimOrderMapOrError propagated_dim_orders_or_error =
      GetPropagatedDimOrders(hlo, direction, src_dim_order, properties);
  if (std::holds_alternative<FusionDecision>(propagated_dim_orders_or_error)) {
    return std::get<FusionDecision>(propagated_dim_orders_or_error);
  }
  DimOrderMap propagated_dim_orders =
      std::move(std::get<DimOrderMap>(propagated_dim_orders_or_error));
  RequirementsOrError requirements_or_error =
      GetRequirementsIfSupportedOrders(hlo, propagated_dim_orders, properties);
  if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
    return std::get<FusionDecision>(requirements_or_error);
  }
  return DimOrdersAndReqs{propagated_dim_orders,
                          std::get<Requirements>(requirements_or_error)};
}

DimOrdersAndReqsOrError
GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
    const HloInstruction& hlo, TransformDirection transform_direction,
    const std::optional<int>& src_operand_index,
    const DimensionOrder& src_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties) {
  CHECK_EQ(transform_direction == TransformDirection::kInputToOutput,
           src_operand_index.has_value());

  if (hlo.opcode() == HloOpcode::kTuple ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    return "Unsupported instruction.";
  }
  if (hlo.opcode() == HloOpcode::kReduce) {
    return "Reductions are not fused yet.";
  }
  if (hlo.opcode() == HloOpcode::kPad) {
    return "Pads are not fused yet.";
  }
  for (const HloInstruction* operand : hlo.operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return "Unsupported input data type.";
    }
  }
  if (!IsTritonSupportedDataType(hlo.shape().element_type(), gpu_version)) {
    return "Unsupported output data type.";
  }
  DimOrdersAndReqsOrError result_or_error =
      GetPropagatedDimOrdersAndRequirements(hlo, src_dim_order,
                                            transform_direction, properties);
  if (!std::holds_alternative<DimOrdersAndReqs>(result_or_error)) {
    return result_or_error;
  }
  DimOrdersAndReqs dim_orders_and_requirements =
      std::move(std::get<DimOrdersAndReqs>(result_or_error));
  int fusion_level =
      hlo.GetModule()->config().debug_options().xla_gpu_triton_fusion_level();
  // TODO(ROCm): Check fusion level for ROCm.
  if (transform_direction == TransformDirection::kOutputToInput) {
    if (fusion_level < 2) {
      if (hlo.opcode() == HloOpcode::kConvert) {
        if (FusionDecision decision = IsConversionWorthFusing(hlo, gpu_version);
            !decision) {
          return decision;
        }
      } else if (hlo.IsElementwise() && hlo.opcode() != HloOpcode::kCopy) {
        return "Ignored elementwise operation";
      }
    } else {
      // Exception for binary elementwise operations: in most cases these are
      // not trivial to fuse because they increase DRAM traffic but if one
      // of the inputs is for example a broadcast that can be fused too it
      // becomes worth fusing. Look ahead and analyze operands here.
      bool accepted = false;
      if (hlo.IsElementwise() && hlo.operand_count() == 2) {
        for (const HloInstruction* operand : hlo.operands()) {
          if (operand->opcode() == HloOpcode::kBroadcast &&
              (operand->operand(0)->opcode() == HloOpcode::kParameter ||
               operand->operand(0)->opcode() == HloOpcode::kConstant) &&
              std::holds_alternative<DimOrdersAndReqs>(
                  GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
                      *operand, TransformDirection::kOutputToInput,
                      /*src_operand_index=*/std::nullopt,
                      /*src_dim_order=*/
                      dim_orders_and_requirements.dim_orders.at(operand),
                      gpu_version, properties))) {
            accepted = true;
            break;
          }
        }
      }
      if (!accepted && !IsInputWorthFusing(hlo)) {
        return "Not obviously profitable to fuse as input.";
      }
    }
  } else {
    if (fusion_level < 2) {
      return "Skipping fusing outputs at low fusion levels.";
    }
    for (int i = 0; i < hlo.operand_count(); ++i) {
      const HloInstruction* operand = hlo.operand(i);
      // Skip source operand.
      if (i == *src_operand_index) {
        continue;
      }
      // Currently only broadcasts of scalar constants or parameters
      // are accepted as other inputs of non-unary operations
      // in the output fusion.
      if (hlo_query::IsBroadcastOfScalarConstant(*operand) ||
          operand->opcode() == HloOpcode::kParameter) {
        continue;
      }
      return "Has multiple inputs - not properly analyzed yet.";
    }
    if (!IsOutputWorthFusing(hlo)) {
      return "Not obviously profitable to fuse as output.";
    }
  }
  return dim_orders_and_requirements;
}

}  // namespace triton_fusion
}  // namespace gpu
}  // namespace xla
