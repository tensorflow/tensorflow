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

#include "xla/hlo/analysis/shape_tracker.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// Some invariants to simplify reasoning:
//  - A tracker always has at least one projection.
//  - Tracker doesn't support zero-element shapes (i.e. when any dimension is
//  0).
//  - Projections never have extents of size 1, except for a 1-element view
//  which is represented by a single segment with stride 1 and extent 1.

namespace xla {
namespace {

// Returns the dimensions of `shape` in major-to-minor order, i.e. the
// permutation that transposes `shape` into its physical order.
std::vector<int64_t> MajorToMinor(const Shape& shape) {
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  return std::vector<int64_t>(minor_to_major.rbegin(), minor_to_major.rend());
}

// Returns the indices in [0, size) that are not present in `sorted_indices`.
std::vector<int64_t> ComplementIndices(
    int64_t size, absl::Span<const int64_t> sorted_indices) {
  std::vector<int64_t> result;
  result.reserve(size - sorted_indices.size());
  size_t idx = 0;
  for (int64_t i = 0; i < size; ++i) {
    if (idx < sorted_indices.size() && sorted_indices[idx] == i) {
      ++idx;
      continue;
    }
    result.push_back(i);
  }
  return result;
}

// Returns a copy of `values` with elements at `sorted_indices` removed.
template <typename T>
std::vector<T> RemoveIndices(absl::Span<const T> values,
                             absl::Span<const int64_t> sorted_indices) {
  std::vector<T> result;
  result.reserve(values.size() - sorted_indices.size());
  size_t idx = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (idx < sorted_indices.size() &&
        sorted_indices[idx] == static_cast<int64_t>(i)) {
      ++idx;
      continue;
    }
    result.push_back(values[i]);
  }
  return result;
}

}  // namespace

ShapeTracker::BufferView ShapeTracker::BufferView::FromBroadcastedExtent(
    int64_t extent) {
  BufferView view;
  view.strides_.push_back(0);
  view.extents_.push_back(extent);
  return view;
}

void ShapeTracker::BufferView::SetSubviews(
    absl::Span<const BufferView> sub_views) {
  if (sub_views.empty()) {
    strides_ = {1};
    extents_ = {1};
  } else {
    BufferView combined = FromSubviews(sub_views);
    strides_ = std::move(combined.strides_);
    extents_ = std::move(combined.extents_);
  }
  MergeAdjacentDimensions();
}

void ShapeTracker::BufferView::AppendCollapsedSegments(
    const BufferView& reduced_view) {
  for (size_t k = 0; k < reduced_view.strides_.size(); ++k) {
    if (reduced_view.strides_[k] > 0 && reduced_view.extents_[k] > 1) {
      collapsed_strides_.push_back(reduced_view.strides_[k]);
      collapsed_extents_.push_back(reduced_view.extents_[k]);
    }
  }
}

void ShapeTracker::TryFoldProjections(
    std::vector<ShapeTracker::BufferView>& projections) {
  while (projections.size() > 1) {
    auto transformation = projections.back().AsTransformation();
    auto& prev = projections[projections.size() - 2];

    auto opt_sub_views = prev.TryUnflatten(transformation.input_reshape);
    if (!opt_sub_views.has_value()) {
      break;
    }

    for (int64_t r_dim : transformation.reduced_dims) {
      prev.AppendCollapsedSegments((*opt_sub_views)[r_dim]);
    }
    std::vector<BufferView> kept_views =
        RemoveIndices<BufferView>(*opt_sub_views, transformation.reduced_dims);

    std::vector<BufferView> final_views;
    final_views.reserve(transformation.output_dims.size());
    size_t b_idx = 0;
    for (size_t j = 0; j < transformation.output_dims.size(); ++j) {
      if (b_idx < transformation.broadcast_dims.size() &&
          transformation.broadcast_dims[b_idx] == static_cast<int64_t>(j)) {
        final_views.push_back(
            std::move(kept_views[transformation.transpose[b_idx++]]));
        continue;
      }
      final_views.push_back(
          BufferView::FromBroadcastedExtent(transformation.output_dims[j]));
    }

    prev.SetSubviews(final_views);
    projections.pop_back();
  }
}

void ShapeTracker::BufferView::MergeAdjacentDimensions() {
  if (!strides_.empty()) {
    size_t w = 0;
    for (size_t i = 1; i < strides_.size(); ++i) {
      if (strides_[i] * extents_[i] == strides_[w]) {
        extents_[w] *= extents_[i];
        strides_[w] = strides_[i];
      } else {
        ++w;
        strides_[w] = strides_[i];
        extents_[w] = extents_[i];
      }
    }
    strides_.resize(w + 1);
    extents_.resize(w + 1);
  }

  if (!collapsed_strides_.empty()) {
    llvm::SmallVector<std::pair<int64_t, int64_t>, 6> collapsed;
    collapsed.reserve(collapsed_strides_.size());
    for (size_t i = 0; i < collapsed_strides_.size(); ++i) {
      if (collapsed_strides_[i] > 0 && collapsed_extents_[i] > 1) {
        collapsed.push_back({collapsed_strides_[i], collapsed_extents_[i]});
      }
    }
    absl::c_sort(collapsed, [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
    collapsed_strides_.clear();
    collapsed_extents_.clear();
    for (const auto& [s, e] : collapsed) {
      if (!collapsed_strides_.empty() && s * e == collapsed_strides_.back()) {
        collapsed_extents_.back() *= e;
        collapsed_strides_.back() = s;
      } else {
        collapsed_strides_.push_back(s);
        collapsed_extents_.push_back(e);
      }
    }
  }
}

void ShapeTracker::BufferView::RemoveDegenerateDimensions() {
  llvm::SmallVector<int64_t, 6> new_strides;
  llvm::SmallVector<int64_t, 6> new_extents;
  for (size_t i = 0; i < strides_.size(); ++i) {
    if (extents_[i] != 1) {
      new_strides.push_back(strides_[i]);
      new_extents.push_back(extents_[i]);
    }
  }
  if (new_strides.empty() && !strides_.empty()) {
    new_strides.push_back(1);
    new_extents.push_back(1);
  }
  strides_ = std::move(new_strides);
  extents_ = std::move(new_extents);
}

ShapeTracker::BufferView ShapeTracker::BufferView::FromSubviews(
    absl::Span<const BufferView> sub_views) {
  BufferView result;
  for (const auto& sub_view : sub_views) {
    result.strides_.insert(result.strides_.end(), sub_view.strides_.begin(),
                           sub_view.strides_.end());
    result.extents_.insert(result.extents_.end(), sub_view.extents_.begin(),
                           sub_view.extents_.end());
  }
  return result;
}

absl::StatusOr<ShapeTracker::BufferView>
ShapeTracker::BufferView::FromStridesAndExtents(
    absl::Span<const int64_t> strides, absl::Span<const int64_t> extents) {
  if (strides.size() != extents.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Strides and extents size mismatch: ", strides.size(),
                     " vs ", extents.size()));
  }

  BufferView view;
  view.strides_.assign(strides.begin(), strides.end());
  view.extents_.assign(extents.begin(), extents.end());
  return view;
}

std::optional<std::vector<ShapeTracker::BufferView>>
ShapeTracker::BufferView::TryUnflatten(
    absl::Span<const int64_t> logical_dims) const {
  if (IsEmpty() || Product(logical_dims) != ElementsIn()) {
    return std::nullopt;
  }

  std::vector<BufferView> sub_views;
  sub_views.reserve(logical_dims.size());

  size_t atom_idx = 0;
  int64_t rem_stride = strides_[0];
  int64_t rem_extent = extents_[0];

  for (int64_t d : logical_dims) {
    BufferView dim_view;
    int64_t rem_d = d;

    while (rem_d > 1) {
      int64_t take_extent = std::min(rem_extent, rem_d);
      if (rem_extent % take_extent != 0 || rem_d % take_extent != 0) {
        return std::nullopt;
      }

      int64_t take_stride = rem_stride * (rem_extent / take_extent);
      dim_view.strides_.push_back(take_stride);
      dim_view.extents_.push_back(take_extent);

      rem_d /= take_extent;
      rem_extent /= take_extent;

      if (rem_extent == 1) {
        atom_idx++;
        if (atom_idx < strides_.size()) {
          rem_stride = strides_[atom_idx];
          rem_extent = extents_[atom_idx];
        }
      }
    }
    sub_views.push_back(dim_view);
  }

  return sub_views;
}

ShapeTracker::BufferView ShapeTracker::BufferView::FromShape(
    const xla::Shape& shape) {
  BufferView view;
  const int64_t num_dims = shape.dimensions().size();
  view.strides_.reserve(num_dims);
  view.extents_.reserve(num_dims);
  int64_t stride = 1;
  int64_t i = num_dims;
  while (i--) {
    view.strides_.push_back(stride);
    view.extents_.push_back(shape.dimensions(i));
    stride *= shape.dimensions(i);
  }
  absl::c_reverse(view.strides_);
  absl::c_reverse(view.extents_);
  if (view.strides_.empty()) {
    view.strides_.push_back(1);
    view.extents_.push_back(1);
  }
  return view;
}

ShapeTracker::BufferView ShapeTracker::BufferView::FromShapeAndIndices(
    const xla::Shape& shape, absl::Span<const int64_t> indices) {
  BufferView full_view = FromShape(shape);
  BufferView view;
  view.strides_.reserve(indices.size());
  view.extents_.reserve(indices.size());
  for (int64_t dim : indices) {
    view.strides_.push_back(full_view.strides_[dim]);
    view.extents_.push_back(full_view.extents_[dim]);
  }
  return view;
}

ShapeTracker::BufferView ShapeTracker::BufferView::FromShapeCompacted(
    const xla::Shape& shape) {
  BufferView view;
  int64_t total_elements = ShapeUtil::ElementsIn(shape);
  view.strides_.push_back(1);
  view.extents_.push_back(total_elements);
  return view;
}

int64_t ShapeTracker::BufferView::ElementsIn() const {
  return IsEmpty() ? 0 : xla::Product<int64_t>(extents_);
}

int64_t ShapeTracker::BufferView::InputElementsIn() const {
  if (IsEmpty()) {
    return 0;
  }
  int64_t product = 1;
  for (size_t i = 0; i < strides_.size(); ++i) {
    if (strides_[i] > 0) {
      product *= extents_[i];
    }
  }
  for (size_t i = 0; i < collapsed_strides_.size(); ++i) {
    if (collapsed_strides_[i] > 0) {
      product *= collapsed_extents_[i];
    }
  }
  return product;
}

std::optional<ShapeTracker::BufferView>
ShapeTracker::BufferView::TryIntersectWith(int64_t stride,
                                           int64_t extent) const {
  BufferView result;
  if (stride == 0) {
    return result;
  }
  for (int64_t i = 0; i < strides_.size(); ++i) {
    if (strides_[i] == 0) {
      continue;
    }
    int64_t out_s = std::max(stride, strides_[i]);
    int64_t min_s = std::min(stride, strides_[i]);
    if (out_s % min_s != 0) {
      return std::nullopt;
    }
    int64_t out_upper_slice =
        std::min(stride * extent, strides_[i] * extents_[i]);
    if (out_upper_slice <= out_s) {
      continue;
    }
    if (out_upper_slice % out_s != 0) {
      return std::nullopt;
    }
    result.strides_.push_back(out_s);
    result.extents_.push_back(out_upper_slice / out_s);
  }
  return result;
}

std::optional<ShapeTracker::BufferView>
ShapeTracker::BufferView::TryIntersectWith(
    const ShapeTracker::BufferView& other) const {
  if (other.IsEmpty()) {
    return BufferView();
  }
  BufferView other_normalized = other;
  other_normalized.SortByStrideDescending();
  other_normalized.MergeAdjacentDimensions();

  BufferView result;
  for (int64_t i = 0; i < strides_.size(); ++i) {
    auto intersection =
        other_normalized.TryIntersectWith(strides_[i], extents_[i]);
    if (!intersection.has_value()) {
      return std::nullopt;
    }
    result.strides_.insert(result.strides_.end(),
                           intersection->strides_.begin(),
                           intersection->strides_.end());
    result.extents_.insert(result.extents_.end(),
                           intersection->extents_.begin(),
                           intersection->extents_.end());
  }
  for (int64_t i = 0; i < collapsed_strides_.size(); ++i) {
    auto intersection = other_normalized.TryIntersectWith(
        collapsed_strides_[i], collapsed_extents_[i]);
    if (!intersection.has_value()) {
      return std::nullopt;
    }
    result.collapsed_strides_.insert(result.collapsed_strides_.end(),
                                     intersection->strides_.begin(),
                                     intersection->strides_.end());
    result.collapsed_extents_.insert(result.collapsed_extents_.end(),
                                     intersection->extents_.begin(),
                                     intersection->extents_.end());
  }
  return result;
}

void ShapeTracker::BufferView::Pack() {
  if (strides_.empty() && !collapsed_strides_.empty()) {
    strides_.push_back(1);
    extents_.push_back(1);
  }
  struct SegmentRef {
    int64_t* stride;
    int64_t extent;
  };
  llvm::SmallVector<SegmentRef, 8> segments;
  segments.reserve(strides_.size() + collapsed_strides_.size());
  auto collect_segments = [&](llvm::SmallVectorImpl<int64_t>& strides,
                              absl::Span<const int64_t> extents) {
    for (size_t i = 0; i < strides.size(); ++i) {
      if (strides[i] <= 0) {
        continue;
      }
      if (extents[i] <= 1) {
        strides[i] = 1;
        continue;
      }
      segments.push_back({&strides[i], extents[i]});
    }
  };
  collect_segments(strides_, extents_);
  collect_segments(collapsed_strides_, collapsed_extents_);
  absl::c_stable_sort(segments, [](const SegmentRef& a, const SegmentRef& b) {
    return *a.stride < *b.stride;  // innermost (smallest stride) first
  });

  int64_t running = 1;
  for (SegmentRef& seg : segments) {
    *seg.stride = running;
    running *= seg.extent;
  }
}

void ShapeTracker::BufferView::SortByStrideDescending() {
  llvm::SmallVector<int64_t, 6> order(strides_.size());
  absl::c_iota(order, 0);
  absl::c_sort(order,
               [&](int64_t a, int64_t b) { return strides_[a] > strides_[b]; });

  llvm::SmallVector<int64_t, 6> permuted_strides(strides_.size());
  llvm::SmallVector<int64_t, 6> permuted_extents(extents_.size());
  for (size_t i = 0; i < order.size(); ++i) {
    permuted_strides[i] = strides_[order[i]];
    permuted_extents[i] = extents_[order[i]];
  }
  strides_ = std::move(permuted_strides);
  extents_ = std::move(permuted_extents);
}

ShapeTracker::~ShapeTracker() = default;
ShapeTracker::ShapeTracker(const ShapeTracker&) = default;
ShapeTracker::ShapeTracker(ShapeTracker&&) noexcept = default;
ShapeTracker& ShapeTracker::operator=(const ShapeTracker&) = default;
ShapeTracker& ShapeTracker::operator=(ShapeTracker&&) noexcept = default;

ShapeTracker::ShapeTracker(xla::Shape shape)
    : input_shape_(shape), output_shape_(shape) {
  projections_.push_back(BufferView::FromShapeCompacted(shape));
}

void ShapeTracker::SetElementType(PrimitiveType element_type) {
  input_shape_.set_element_type(element_type);
  output_shape_.set_element_type(element_type);
}

bool ShapeTracker::operator==(const ShapeTracker& other) const {
  return ShapeUtil::Compatible(input_shape_, other.input_shape_) &&
         ShapeUtil::Compatible(output_shape_, other.output_shape_) &&
         projections_ == other.projections_;
}

absl::StatusOr<ShapeTracker> ShapeTracker::FromProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer) {
  std::vector<const HloInstruction*> chain;
  const HloInstruction* current = consumer;

  while (current != producer) {
    if (current->operand_count() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Instruction in the chain does not have exactly one operand: ",
          current->ToString()));
    }
    chain.push_back(current);
    current = current->operand(0);
  }

  ShapeTracker tracker(producer->shape());
  absl::c_reverse(chain);
  for (const auto* inst : chain) {
    ABSL_RETURN_IF_ERROR(tracker.AppendInstruction(inst));
  }
  return tracker;
}

// To compute the tracker between two instructions which have a common ancestor,
// build trackers from the common ancestor, and then concatenate one with
// inverted second one.
absl::StatusOr<ShapeTracker> ShapeTracker::FromSiblings(
    const HloInstruction* source, const HloInstruction* destination) {
  absl::flat_hash_set<const HloInstruction*> ancestors;
  const HloInstruction* current = source;
  // Build set of ancestors.
  while (true) {
    ancestors.insert(current);
    if (current->operand_count() != 1) {
      break;
    }
    current = current->operand(0);
  }

  const HloInstruction* lca = nullptr;
  current = destination;
  // First ancestor which is also an ancestor of source will be the lowest
  // common ancestor.
  while (true) {
    if (ancestors.contains(current)) {
      lca = current;
      break;
    }
    if (current->operand_count() != 1) {
      break;
    }
    current = current->operand(0);
  }

  if (lca == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("No common ancestor found between ", source->name(),
                     " and ", destination->name()));
  }

  ABSL_ASSIGN_OR_RETURN(ShapeTracker tracker1, FromProducerConsumer(lca, source));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker tracker2,
                   FromProducerConsumer(lca, destination));
  ABSL_RETURN_IF_ERROR(tracker1.Invert());
  ABSL_RETURN_IF_ERROR(tracker1.ConcatenateFrom(tracker2));
  return tracker1;
}

absl::StatusOr<std::vector<ShapeTracker::BufferView>>
ShapeTracker::UnflattenOutputDimensions() {
  const auto& old_dimensions = output_shape_.dimensions();
  std::vector<int64_t> non_degenerate_dims;
  absl::c_copy_if(old_dimensions, std::back_inserter(non_degenerate_dims),
                  [](int64_t d) { return d != 1; });

  std::optional<std::vector<BufferView>> opt_sub_views;
  if (!non_degenerate_dims.empty()) {
    opt_sub_views = projections_.back().TryUnflatten(non_degenerate_dims);
    if (!opt_sub_views.has_value()) {
      projections_.push_back(BufferView::FromShapeCompacted(output_shape_));
      opt_sub_views = projections_.back().TryUnflatten(non_degenerate_dims);
      if (!opt_sub_views.has_value()) {
        return absl::InternalError(
            "Failed to unflatten contiguous layout after copy");
      }
    }
  }

  std::vector<BufferView> dim_views(old_dimensions.size(), BufferView());
  size_t physical_idx = 0;
  for (size_t i = 0; i < old_dimensions.size(); ++i) {
    if (old_dimensions[i] != 1) {
      dim_views[i] = std::move((*opt_sub_views)[physical_idx++]);
    }
  }
  return dim_views;
}

// Tries to transpose without introducing a copy (flattening). If transposed
// dimension is not divisible by the neighboring stride, flatten and transpose.
// This is a user facing function which may attempt to transpose degenerate
// dimensions, therefore we need to expand mapping.
absl::Status ShapeTracker::AppendTranspose(
    absl::Span<const int64_t> permutation) {
  if (permutation.size() != output_shape_.dimensions().size()) {
    return absl::InvalidArgumentError("Rank mismatch");
  }
  if (!IsPermutation(permutation)) {
    return absl::InvalidArgumentError("Invalid permutation");
  }
  if (ShapeUtil::ElementsIn(output_shape_) == 1) {
    output_shape_ = ShapeUtil::PermuteDimensions(permutation, output_shape_);
    LayoutUtil::SetToDefaultLayout(&output_shape_);
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<BufferView> dim_views,
                   UnflattenOutputDimensions());
  std::vector<BufferView> permuted_sub_views;
  permuted_sub_views.reserve(permutation.size());
  for (int64_t dim : permutation) {
    if (output_shape_.dimensions(dim) != 1) {
      permuted_sub_views.push_back(std::move(dim_views[dim]));
    }
  }

  projections_.back().SetSubviews(permuted_sub_views);
  output_shape_ = ShapeUtil::PermuteDimensions(permutation, output_shape_);
  LayoutUtil::SetToDefaultLayout(&output_shape_);

  TryFoldProjections(projections_);
  return absl::OkStatus();
}

// Reshapes never introduce a copy (even when new shape's strides are not
// divisible by underlying strides), it defines how a view is cut.
absl::Status ShapeTracker::AppendReshape(absl::Span<const int64_t> dimensions) {
  if (Product(dimensions) != ShapeUtil::ElementsIn(output_shape_)) {
    return absl::InvalidArgumentError("Product of dimensions mismatch");
  }

  output_shape_ =
      ShapeUtil::MakeShape(output_shape_.element_type(), dimensions);
  return absl::OkStatus();
}

// A bitcast reinterprets the physical buffer, so it is a transpose of the
// source into its physical order, a reshape to the destination's physical
// dimensions and a transpose back into the destination's logical order.
absl::Status ShapeTracker::AppendBitcast(const xla::Shape& src_shape,
                                         const xla::Shape& dst_shape) {
  if (ShapeUtil::ElementsIn(src_shape) != ShapeUtil::ElementsIn(dst_shape)) {
    return absl::InvalidArgumentError(
        "Bitcast must preserve the total number of elements");
  }

  if (!ShapeUtil::Compatible(src_shape, output_shape_)) {
    return absl::InvalidArgumentError(
        "Bitcast operand shape does not match current output shape");
  }

  std::vector<int64_t> dst_major_to_minor = MajorToMinor(dst_shape);
  ABSL_RETURN_IF_ERROR(AppendTranspose(MajorToMinor(src_shape)));
  ABSL_RETURN_IF_ERROR(
      AppendReshape(Permute(dst_shape.dimensions(), dst_major_to_minor)));
  return AppendTranspose(InversePermutation(dst_major_to_minor));
}

absl::Status ShapeTracker::AppendBroadcast(
    absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> broadcast_dimensions) {
  if (absl::c_any_of(dimensions, [](int64_t d) { return d <= 0; })) {
    return absl::InvalidArgumentError(
        "Broadcast target dimensions must be positive");
  }
  if (broadcast_dimensions.size() != output_shape_.dimensions().size()) {
    return absl::InvalidArgumentError(
        "broadcast_dimensions size must match current output rank");
  }
  llvm::SmallVector<int64_t, 6> out_to_in(dimensions.size(), -1);
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    int64_t b_dim = broadcast_dimensions[i];
    if (b_dim < 0 || b_dim >= static_cast<int64_t>(dimensions.size())) {
      return absl::InvalidArgumentError(
          "Broadcast dimension index out of bounds");
    }
    if (out_to_in[b_dim] != -1) {
      return absl::InvalidArgumentError("broadcast_dimensions must be unique");
    }
    if (dimensions[b_dim] != output_shape_.dimensions(i)) {
      return absl::InvalidArgumentError(
          "Broadcast dimension size mismatch with input shape");
    }
    out_to_in[b_dim] = static_cast<int64_t>(i);
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<BufferView> dim_views,
                   UnflattenOutputDimensions());
  std::vector<BufferView> new_sub_views;
  new_sub_views.reserve(dimensions.size());
  for (size_t out_dim = 0; out_dim < dimensions.size(); ++out_dim) {
    if (dimensions[out_dim] == 1) {
      continue;
    }
    int64_t in_dim = out_to_in[out_dim];
    if (in_dim != -1) {
      new_sub_views.push_back(std::move(dim_views[in_dim]));
      continue;
    }
    new_sub_views.push_back(
        BufferView::FromBroadcastedExtent(dimensions[out_dim]));
  }

  projections_.back().SetSubviews(new_sub_views);
  output_shape_ =
      ShapeUtil::MakeShape(output_shape_.element_type(), dimensions);
  TryFoldProjections(projections_);
  return absl::OkStatus();
}

absl::Status ShapeTracker::AppendReduce(
    absl::Span<const int64_t> dimensions_to_reduce) {
  llvm::SmallVector<bool, 6> is_reduced(output_shape_.dimensions().size(),
                                        false);
  for (int64_t d : dimensions_to_reduce) {
    if (d < 0 || d >= static_cast<int64_t>(output_shape_.dimensions().size())) {
      return absl::InvalidArgumentError("Dimension to reduce is out of bounds");
    }
    if (is_reduced[d]) {
      return absl::InvalidArgumentError(
          "dimensions_to_reduce must contain unique dimensions");
    }
    is_reduced[d] = true;
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<BufferView> dim_views,
                   UnflattenOutputDimensions());
  std::vector<BufferView> kept_sub_views;
  std::vector<int64_t> new_output_dims;
  for (size_t i = 0; i < output_shape_.dimensions().size(); ++i) {
    int64_t dim_size = output_shape_.dimensions(i);
    if (is_reduced[i]) {
      if (dim_size != 1) {
        projections_.back().AppendCollapsedSegments(dim_views[i]);
      }
      continue;
    }
    new_output_dims.push_back(dim_size);
    if (dim_size != 1) {
      kept_sub_views.push_back(std::move(dim_views[i]));
    }
  }

  projections_.back().SetSubviews(kept_sub_views);
  output_shape_ =
      ShapeUtil::MakeShape(output_shape_.element_type(), new_output_dims);
  TryFoldProjections(projections_);
  return absl::OkStatus();
}

absl::Status ShapeTracker::AppendInstruction(const HloInstruction* inst) {
  if (inst->operand_count() == 0) {
    return absl::InvalidArgumentError(
        "Instruction must have at least one operand");
  }
  if (!ShapeUtil::Compatible(inst->operand(0)->shape(), output_shape_)) {
    return absl::InvalidArgumentError(
        "Instruction operand shape does not match current output shape");
  }

  switch (inst->opcode()) {
    case HloOpcode::kTranspose:
      return AppendTranspose(inst->dimensions());
    case HloOpcode::kReshape:
      return AppendReshape(inst->shape().dimensions());
    case HloOpcode::kBitcast:
      return AppendBitcast(inst->operand(0)->shape(), inst->shape());
    case HloOpcode::kBroadcast:
      return AppendBroadcast(inst->shape().dimensions(), inst->dimensions());
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported opcode: ", HloOpcodeString(inst->opcode())));
  }
}

absl::Status ShapeTracker::PrependInstruction(const HloInstruction* inst) {
  if (!ShapeUtil::Compatible(inst->shape(), input_shape_)) {
    return absl::InvalidArgumentError(
        "Instruction shape does not match current input shape");
  }

  switch (inst->opcode()) {
    case HloOpcode::kTranspose:
      return PrependTranspose(inst->dimensions());
    case HloOpcode::kReshape:
      return PrependReshape(inst->operand(0)->shape().dimensions());
    case HloOpcode::kBitcast:
      return PrependBitcast(inst->operand(0)->shape(), inst->shape());
    case HloOpcode::kBroadcast:
      return PrependBroadcast(inst->dimensions());
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported opcode: ", HloOpcodeString(inst->opcode())));
  }
}

// Concatenates the other tracker (by applying its steps).
absl::Status ShapeTracker::ConcatenateFrom(const ShapeTracker& other) {
  if (output_shape_.dimensions() != other.input_shape().dimensions()) {
    return absl::InvalidArgumentError(
        "Output shape of this tracker must match input shape of the other "
        "tracker");
  }

  for (const auto& step : other.GetSteps()) {
    switch (step.type) {
      case Step::Type::kReshape:
        ABSL_RETURN_IF_ERROR(AppendReshape(step.dimensions));
        break;
      case Step::Type::kTranspose:
        ABSL_RETURN_IF_ERROR(AppendTranspose(step.dimensions));
        break;
      case Step::Type::kBroadcast:
        ABSL_RETURN_IF_ERROR(
            AppendBroadcast(step.dimensions, step.broadcast_dimensions));
        break;
      case Step::Type::kReduce:
        ABSL_RETURN_IF_ERROR(AppendReduce(step.dimensions));
        break;
    }
  }

  return absl::OkStatus();
}

ShapeTracker::BufferView::Transformation
ShapeTracker::BufferView::AsTransformation() const {
  Transformation result;

  struct InputAtom {
    int64_t stride;
    int64_t extent;
    int64_t
        kept_idx;  // 0..K-1 for kept positive-stride segments, -1 for collapsed
  };
  llvm::SmallVector<InputAtom, 6> input_atoms;
  int64_t num_kept = 0;
  for (size_t i = 0; i < strides_.size(); ++i) {
    if (strides_[i] > 0 && extents_[i] > 1) {
      input_atoms.push_back({strides_[i], extents_[i], num_kept++});
    }
  }
  for (size_t i = 0; i < collapsed_strides_.size(); ++i) {
    if (collapsed_strides_[i] > 0 && collapsed_extents_[i] > 1) {
      input_atoms.push_back({collapsed_strides_[i], collapsed_extents_[i], -1});
    }
  }

  absl::c_sort(input_atoms, [](const InputAtom& a, const InputAtom& b) {
    return a.stride > b.stride;
  });

  result.input_reshape.reserve(input_atoms.size());
  result.transpose.resize(num_kept);
  int64_t post_reduce_idx = 0;
  for (size_t j = 0; j < input_atoms.size(); ++j) {
    result.input_reshape.push_back(input_atoms[j].extent);
    if (input_atoms[j].kept_idx == -1) {
      result.reduced_dims.push_back(static_cast<int64_t>(j));
    } else {
      result.transpose[input_atoms[j].kept_idx] = post_reduce_idx++;
    }
  }

  for (size_t i = 0; i < strides_.size(); ++i) {
    if (extents_[i] <= 1) {
      continue;
    }
    result.output_dims.push_back(extents_[i]);
    if (strides_[i] > 0) {
      result.broadcast_dims.push_back(
          static_cast<int64_t>(result.output_dims.size()) - 1);
    }
  }

  return result;
}

bool ShapeTracker::BufferView::Transformation::IsTrivial() const {
  return reduced_dims.empty() && IsIdentityPermutation(transpose) &&
         output_dims.size() == broadcast_dims.size();
}

// To construct the inverse, we materialize our tracker to the list of
// reshape-reduce-transpose-broadcast steps, and construct the tracker
// backwards. Note that it doesn't necessarily contain the same number of
// steps/projections.
absl::StatusOr<ShapeTracker> ShapeTracker::GetInverted() const {
  ShapeTracker inverted(output_shape_);
  for (auto it = projections_.rbegin(); it != projections_.rend(); ++it) {
    auto transformation = it->AsTransformation();
    if (transformation.IsTrivial()) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(inverted.AppendReshape(transformation.output_dims));
    if (transformation.output_dims.size() >
        transformation.broadcast_dims.size()) {
      ABSL_RETURN_IF_ERROR(inverted.AppendReduce(ComplementIndices(
          transformation.output_dims.size(), transformation.broadcast_dims)));
    }
    if (!IsIdentityPermutation(transformation.transpose)) {
      ABSL_RETURN_IF_ERROR(inverted.AppendTranspose(
          InversePermutation(transformation.transpose)));
    }
    if (!transformation.reduced_dims.empty()) {
      ABSL_RETURN_IF_ERROR(inverted.AppendBroadcast(
          transformation.input_reshape,
          ComplementIndices(transformation.input_reshape.size(),
                            transformation.reduced_dims)));
    }
  }

  ABSL_RETURN_IF_ERROR(inverted.AppendReshape(input_shape_.dimensions()));
  // The reshape above resets the layout; restore the original input's one.
  inverted.output_shape_ = input_shape_;

  return inverted;
}

absl::Status ShapeTracker::Invert() {
  ABSL_ASSIGN_OR_RETURN(ShapeTracker inverted, GetInverted());
  *this = std::move(inverted);
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependTranspose(
    absl::Span<const int64_t> permutation) {
  if (IsIdentityPermutation(permutation)) {
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(Invert());
  ABSL_RETURN_IF_ERROR(AppendTranspose(InversePermutation(permutation)));
  ABSL_RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependReshape(
    absl::Span<const int64_t> dimensions) {
  ABSL_RETURN_IF_ERROR(Invert());
  ABSL_RETURN_IF_ERROR(AppendReshape(dimensions));
  ABSL_RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependBitcast(const xla::Shape& src_shape,
                                          const xla::Shape& dst_shape) {
  ABSL_RETURN_IF_ERROR(Invert());
  ABSL_RETURN_IF_ERROR(AppendBitcast(dst_shape, src_shape));
  ABSL_RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependBroadcast(
    absl::Span<const int64_t> broadcast_dimensions) {
  const int64_t input_rank = input_shape_.dimensions().size();
  llvm::SmallVector<bool, 6> is_kept(input_rank, false);
  for (int64_t d : broadcast_dimensions) {
    if (d < 0 || d >= input_rank) {
      return absl::InvalidArgumentError(
          "Broadcast dimension index out of bounds");
    }
    if (is_kept[d]) {
      return absl::InvalidArgumentError("broadcast_dimensions must be unique");
    }
    is_kept[d] = true;
  }

  std::vector<int64_t> dims_to_reduce;
  llvm::SmallVector<int64_t, 6> old_to_reduced_idx(input_rank, -1);
  int64_t kept_count = 0;
  for (int64_t d = 0; d < input_rank; ++d) {
    if (is_kept[d]) {
      old_to_reduced_idx[d] = kept_count++;
    } else {
      dims_to_reduce.push_back(d);
    }
  }

  std::vector<int64_t> perm;
  perm.reserve(broadcast_dimensions.size());
  for (int64_t d : broadcast_dimensions) {
    perm.push_back(old_to_reduced_idx[d]);
  }

  ABSL_RETURN_IF_ERROR(Invert());
  ABSL_RETURN_IF_ERROR(AppendReduce(dims_to_reduce));
  if (!IsIdentityPermutation(perm)) {
    ABSL_RETURN_IF_ERROR(AppendTranspose(perm));
  }
  ABSL_RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependReduce(
    absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> dimensions_to_reduce) {
  if (absl::c_any_of(dimensions, [](int64_t d) { return d <= 0; })) {
    return absl::InvalidArgumentError(
        "Reduce source dimensions must be positive");
  }
  llvm::SmallVector<bool, 6> is_reduced(dimensions.size(), false);
  for (int64_t d : dimensions_to_reduce) {
    if (d < 0 || d >= static_cast<int64_t>(dimensions.size())) {
      return absl::InvalidArgumentError("Dimension to reduce is out of bounds");
    }
    if (is_reduced[d]) {
      return absl::InvalidArgumentError(
          "dimensions_to_reduce must contain unique dimensions");
    }
    is_reduced[d] = true;
  }

  std::vector<int64_t> kept_dimensions;
  std::vector<int64_t> expected_input_dims;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (!is_reduced[i]) {
      kept_dimensions.push_back(static_cast<int64_t>(i));
      expected_input_dims.push_back(dimensions[i]);
    }
  }
  if (!absl::c_equal(expected_input_dims, input_shape_.dimensions())) {
    return absl::InvalidArgumentError(
        "Remaining dimensions after reduction do not match current input "
        "shape");
  }

  ABSL_RETURN_IF_ERROR(Invert());
  ABSL_RETURN_IF_ERROR(AppendBroadcast(dimensions, kept_dimensions));
  ABSL_RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

std::vector<ShapeTracker::Step> ShapeTracker::GetSteps(
    bool avoid_combining_reshapes) const {
  std::vector<Step> steps;
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  for (size_t i = 0; i < projections_.size(); ++i) {
    auto transformation = projections_[i].AsTransformation();

    // If it is the last projection, and it has no reduction, transpose, or
    // broadcast, we can skip it completely because the final reshape will
    // handle it.
    if (i == projections_.size() - 1 && transformation.IsTrivial()) {
      continue;
    }

    if (!absl::c_equal(current_dims, transformation.input_reshape)) {
      current_dims.assign(transformation.input_reshape.begin(),
                          transformation.input_reshape.end());
      steps.push_back({Step::Type::kReshape, current_dims});
    }

    if (!transformation.reduced_dims.empty()) {
      steps.push_back(
          {Step::Type::kReduce,
           std::vector<int64_t>(transformation.reduced_dims.begin(),
                                transformation.reduced_dims.end())});
      current_dims =
          RemoveIndices<int64_t>(current_dims, transformation.reduced_dims);
    }

    if (!IsIdentityPermutation(transformation.transpose)) {
      steps.push_back({Step::Type::kTranspose,
                       std::vector<int64_t>(transformation.transpose.begin(),
                                            transformation.transpose.end())});
      current_dims = Permute(current_dims, transformation.transpose);
    }

    if (transformation.output_dims.size() >
        transformation.broadcast_dims.size()) {
      current_dims.assign(transformation.output_dims.begin(),
                          transformation.output_dims.end());
      steps.push_back(
          {Step::Type::kBroadcast, current_dims,
           std::vector<int64_t>(transformation.broadcast_dims.begin(),
                                transformation.broadcast_dims.end())});
    }
  }

  if (!absl::c_equal(current_dims, output_shape_.dimensions())) {
    steps.push_back({Step::Type::kReshape,
                     std::vector<int64_t>(output_shape_.dimensions().begin(),
                                          output_shape_.dimensions().end())});
  }

  if (avoid_combining_reshapes) {
    return OptimizeSteps(steps, input_shape_, output_shape_);
  }
  return steps;
}

namespace {

// std::inclusive_scan is temporarily not used here because not all supported
// compilers support it yet.
std::vector<int64_t> PartialProducts(absl::Span<const int64_t> input) {
  std::vector<int64_t> result;
  result.reserve(input.size());
  int64_t current = 1;
  for (int64_t val : input) {
    current *= val;
    result.push_back(current);
  }
  return result;
}

struct ExpandedStep {
  ShapeTracker::Step step;
  std::vector<int64_t> shape;
  std::vector<int64_t> mapping;
};

// If a reshape tries to glue dimensions, the function keeps them when possible.
// Do to that, it keeps the strides of the target dimensions (which we are
// required to preserve), and inserts existing dimensions when it doesn't
// violate divisibility.
ExpandedStep ExpandReshapeStep(const ShapeTracker::Step& step,
                               absl::Span<const int64_t> current_shape) {
  if (step.dimensions.empty()) {
    return {ShapeTracker::Step{ShapeTracker::Step::Type::kReshape, {}}, {}, {}};
  }
  std::vector<int64_t> strides_to_preserve = PartialProducts(step.dimensions);
  std::vector<int64_t> strides_to_insert = PartialProducts(current_shape);

  size_t preserve_idx = 0;
  size_t insert_idx = 0;
  std::vector<int64_t> new_mapping;
  std::vector<int64_t> new_shapes;

  // Both strides_to_preserve and strides_to_insert contain distinct elements
  // (in increasing order), and have the same last element (total number of
  // elements). In the loop below, `insert_stride` always follows behind
  // `preserve_stride`, and therefore `insert_idx` always points to a valid
  // element.
  while (preserve_idx < strides_to_preserve.size()) {
    int64_t preserve_stride = strides_to_preserve[preserve_idx];
    int64_t insert_stride = strides_to_insert[insert_idx];
    if (insert_stride < preserve_stride) {
      if (preserve_stride % insert_stride == 0 &&
          (new_shapes.empty() || insert_stride % new_shapes.back() == 0)) {
        new_shapes.push_back(insert_stride);
      }
      ++insert_idx;
      continue;
    }

    new_mapping.push_back(new_shapes.size());
    new_shapes.push_back(preserve_stride);
    ++preserve_idx;

    if (insert_stride == preserve_stride) {
      ++insert_idx;
    }
  }

  absl::c_adjacent_difference(new_shapes, new_shapes.begin(),
                              std::divides<int64_t>());

  return {ShapeTracker::Step{ShapeTracker::Step::Type::kReshape, new_shapes},
          new_shapes, std::move(new_mapping)};
}

ExpandedStep ExpandReduceStep(const ShapeTracker::Step& step,
                              absl::Span<const int64_t> current_shape,
                              absl::Span<const int64_t> expansion) {
  std::vector<int64_t> expanded_reduced_dims;
  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_mapping;
  size_t r_idx = 0;
  for (size_t dim = 0; dim < expansion.size(); ++dim) {
    const int64_t begin = dim == 0 ? 0 : expansion[dim - 1] + 1;
    const int64_t end = expansion[dim] + 1;
    if (r_idx < step.dimensions.size() &&
        step.dimensions[r_idx] == static_cast<int64_t>(dim)) {
      ++r_idx;
      for (int64_t k = begin; k < end; ++k) {
        expanded_reduced_dims.push_back(k);
      }
      continue;
    }
    for (int64_t k = begin; k < end; ++k) {
      new_shape.push_back(current_shape[k]);
    }
    new_mapping.push_back(static_cast<int64_t>(new_shape.size()) - 1);
  }
  return {ShapeTracker::Step{ShapeTracker::Step::Type::kReduce,
                             std::move(expanded_reduced_dims)},
          std::move(new_shape), std::move(new_mapping)};
}

// Performs a transpose over an expanded shape.
ExpandedStep ExpandTransposeStep(const ShapeTracker::Step& step,
                                 absl::Span<const int64_t> current_shape,
                                 absl::Span<const int64_t> expansion) {
  std::vector<int64_t> expanded;
  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_mapping;
  if (!expansion.empty()) {
    expanded.reserve(expansion.back() + 1);
    new_shape.reserve(expansion.back() + 1);
    new_mapping.reserve(step.dimensions.size());
  }
  for (const int64_t dim : step.dimensions) {
    const int64_t begin = dim == 0 ? 0 : expansion[dim - 1] + 1;
    const int64_t end = expansion[dim] + 1;
    for (int64_t k = begin; k < end; ++k) {
      expanded.push_back(k);
      new_shape.push_back(current_shape[k]);
    }
    new_mapping.push_back(static_cast<int64_t>(new_shape.size()) - 1);
  }
  return {ShapeTracker::Step{ShapeTracker::Step::Type::kTranspose,
                             std::move(expanded)},
          std::move(new_shape), std::move(new_mapping)};
}

ExpandedStep ExpandBroadcastStep(const ShapeTracker::Step& step,
                                 absl::Span<const int64_t> current_shape,
                                 absl::Span<const int64_t> expansion) {
  std::vector<int64_t> expanded_output_dims;
  std::vector<int64_t> expanded_broadcast_dims;
  std::vector<int64_t> new_mapping;
  new_mapping.reserve(step.dimensions.size());

  size_t b_idx = 0;
  for (size_t j = 0; j < step.dimensions.size(); ++j) {
    if (b_idx < step.broadcast_dimensions.size() &&
        step.broadcast_dimensions[b_idx] == static_cast<int64_t>(j)) {
      const int64_t begin = b_idx == 0 ? 0 : expansion[b_idx - 1] + 1;
      const int64_t end = expansion[b_idx] + 1;
      for (int64_t k = begin; k < end; ++k) {
        expanded_broadcast_dims.push_back(
            static_cast<int64_t>(expanded_output_dims.size()));
        expanded_output_dims.push_back(current_shape[k]);
      }
      new_mapping.push_back(static_cast<int64_t>(expanded_output_dims.size()) -
                            1);
      ++b_idx;
      continue;
    }
    expanded_output_dims.push_back(step.dimensions[j]);
    new_mapping.push_back(static_cast<int64_t>(expanded_output_dims.size()) -
                          1);
  }
  std::vector<int64_t> new_shape = expanded_output_dims;
  return {ShapeTracker::Step{ShapeTracker::Step::Type::kBroadcast,
                             std::move(expanded_output_dims),
                             std::move(expanded_broadcast_dims)},
          std::move(new_shape), std::move(new_mapping)};
}

}  // namespace

std::vector<ShapeTracker::Step> ShapeTracker::OptimizeSteps(
    const std::vector<Step>& steps, const xla::Shape& input_shape,
    const xla::Shape& output_shape) {
  std::vector<Step> optimized_steps;
  std::vector<int64_t> current_shape(input_shape.dimensions().begin(),
                                     input_shape.dimensions().end());

  std::vector<int64_t> mapping(current_shape.size());
  absl::c_iota(mapping, 0);

  // If the final step is a reshape, skip it for two reasons:
  // - Unlike other steps, it may contain degenerate dimensions which
  // ExpandReshapeStep can't handle.
  // - It is redundant, a reshape will be applied at the end anyway.
  absl::Span<const Step> steps_span = steps;
  if (!steps_span.empty() && steps_span.back().type == Step::Type::kReshape) {
    steps_span.remove_suffix(1);
  }

  for (const auto& step : steps_span) {
    ExpandedStep expanded;
    switch (step.type) {
      case Step::Type::kReshape:
        expanded = ExpandReshapeStep(step, current_shape);
        if (expanded.shape != current_shape) {
          optimized_steps.push_back(std::move(expanded.step));
        }
        break;
      case Step::Type::kReduce:
        expanded = ExpandReduceStep(step, current_shape, mapping);
        optimized_steps.push_back(std::move(expanded.step));
        break;
      case Step::Type::kTranspose:
        expanded = ExpandTransposeStep(step, current_shape, mapping);
        optimized_steps.push_back(std::move(expanded.step));
        break;
      case Step::Type::kBroadcast:
        expanded = ExpandBroadcastStep(step, current_shape, mapping);
        optimized_steps.push_back(std::move(expanded.step));
        break;
    }
    current_shape = std::move(expanded.shape);
    mapping = std::move(expanded.mapping);
  }

  if (current_shape != output_shape.dimensions()) {
    std::vector<int64_t> out_dims(output_shape.dimensions().begin(),
                                  output_shape.dimensions().end());
    optimized_steps.push_back({Step::Type::kReshape, out_dims});
  }

  return optimized_steps;
}

absl::StatusOr<HloInstruction*> ShapeTracker::ToInstructionChain(
    HloInstruction* inst, bool avoid_combining_reshapes) const {
  if (inst->shape().dimensions() != input_shape_.dimensions()) {
    return absl::InvalidArgumentError(
        "Input instruction shape dimensions mismatch");
  }

  std::vector<Step> steps = GetSteps(avoid_combining_reshapes);
  if (absl::c_any_of(steps, [](const Step& step) {
        return step.type == Step::Type::kReduce;
      })) {
    return absl::InvalidArgumentError(
        "Cannot materialize reduction step into HLO instruction chain "
        "without a reduction computation");
  }

  HloComputation* computation = inst->parent();
  HloInstruction* current_inst = inst;

  for (const auto& step : steps) {
    switch (step.type) {
      case Step::Type::kReshape: {
        Shape new_shape = ShapeUtil::MakeShape(
            current_inst->shape().element_type(), step.dimensions);
        current_inst = computation->AddInstruction(
            HloInstruction::CreateReshape(new_shape, current_inst));
        break;
      }
      case Step::Type::kTranspose: {
        current_inst =
            computation->AddInstruction(HloInstruction::CreateTranspose(
                ShapeUtil::PermuteDimensions(step.dimensions,
                                             current_inst->shape()),
                current_inst, step.dimensions));
        break;
      }
      case Step::Type::kBroadcast: {
        Shape new_shape = ShapeUtil::MakeShape(
            current_inst->shape().element_type(), step.dimensions);
        current_inst =
            computation->AddInstruction(HloInstruction::CreateBroadcast(
                new_shape, current_inst, step.broadcast_dimensions));
        break;
      }
      case Step::Type::kReduce:
        break;
    }
  }

  return current_inst;
}

std::string ShapeTracker::DebugString(bool avoid_combining_reshapes) const {
  std::string result =
      absl::StrCat("[", absl::StrJoin(input_shape_.dimensions(), ","), "]");
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  for (const auto& step : GetSteps(avoid_combining_reshapes)) {
    switch (step.type) {
      case Step::Type::kReshape:
        current_dims = step.dimensions;
        absl::StrAppend(&result, " -> R[", absl::StrJoin(current_dims, ","),
                        "]");
        break;
      case Step::Type::kTranspose:
        current_dims = Permute(current_dims, step.dimensions);
        absl::StrAppend(&result, " -> T[", absl::StrJoin(current_dims, ","),
                        "]");
        break;
      case Step::Type::kBroadcast:
        current_dims = step.dimensions;
        absl::StrAppend(&result, " -> B[", absl::StrJoin(current_dims, ","),
                        "]");
        break;
      case Step::Type::kReduce:
        current_dims = RemoveIndices<int64_t>(current_dims, step.dimensions);
        absl::StrAppend(&result, " -> Red[", absl::StrJoin(current_dims, ","),
                        "]");
        break;
    }
  }

  return result;
}

struct ShapeTracker::SlicePropagationResult {
  std::vector<ShapeTracker::BufferView> sliced_projections;
  ShapeTracker::BufferView final_slice;
};

// Keeps the parts of the projections that intersect with the @slice. I.e.
// tracks the life of the slice as it goes through the projections.
absl::StatusOr<ShapeTracker::SlicePropagationResult>
ShapeTracker::SliceProjectionChain(
    absl::Span<const ShapeTracker::BufferView> projections,
    const ShapeTracker::BufferView& slice) {
  CHECK_GT(slice.ElementsIn(), 1)
      << "slice.ElementsIn() == 1 should be handled by the caller";

  std::vector<ShapeTracker::BufferView> sliced_projections;
  sliced_projections.reserve(projections.size());

  ShapeTracker::BufferView current_slice = slice;
  ShapeTracker::BufferView final_slice = slice;

  for (size_t i = 0; i < projections.size(); ++i) {
    if (current_slice.IsEmpty()) {
      break;
    }
    int64_t expected_elements = current_slice.ElementsIn();
    auto intersection_opt = projections[i].TryIntersectWith(current_slice);
    if (!intersection_opt.has_value()) {
      return absl::InvalidArgumentError(
          "Slice is incompatible with projection");
    }
    ShapeTracker::BufferView intersection = *intersection_opt;

    int64_t intersected_elements = Product(intersection.extents()) *
                                   Product(intersection.collapsed_extents());
    if (intersected_elements != expected_elements) {
      return absl::InvalidArgumentError(
          "Lost elements during slice propagation");
    }

    final_slice = intersection;

    llvm::SmallVector<int64_t, 6> output_strides(
        projections[i].extents().size());
    int64_t running_stride = 1;
    for (int64_t k = static_cast<int64_t>(output_strides.size()) - 1; k >= 0;
         --k) {
      output_strides[k] = running_stride;
      running_stride *= projections[i].extents()[k];
    }

    llvm::SmallVector<int64_t, 6> next_slice_strides;
    llvm::SmallVector<int64_t, 6> next_slice_extents;
    for (auto [s, e] :
         llvm::zip(intersection.strides(), intersection.extents())) {
      // Find the dimension in the projection that this slice dimension belongs
      // to.
      auto zip_range = llvm::zip(projections[i].strides(),
                                 projections[i].extents(), output_strides);
      auto it = absl::c_find_if(zip_range, [s = s, e = e](const auto& tuple) {
        auto [proj_stride, proj_extent, output_stride] = tuple;
        return proj_stride > 0 && s % proj_stride == 0 &&
               s * e <= proj_stride * proj_extent;
      });

      if (it == zip_range.end()) {
        return absl::InternalError(
            "Slice dimension does not belong to any projection dimension");
      }
      auto [proj_stride, proj_extent, output_stride] = *it;

      // Every projection "reimagines" the previous projection as sorted by
      // dimension order (major-to-minor).
      // Here we map the slice stride `s` into the output space of this
      // projection. `s / proj_stride` computes the logical stride of the
      // slice within dimension `d`, which we then multiply by the row-major
      // output stride for dimension `d` to accumulate and propagate the
      // correct stride to the next iteration.
      next_slice_strides.push_back(output_stride * (s / proj_stride));
      next_slice_extents.push_back(e);
    }

    intersection.Pack();
    sliced_projections.push_back(intersection);
    TryFoldProjections(sliced_projections);
    sliced_projections.back().Pack();

    ABSL_ASSIGN_OR_RETURN(current_slice,
                     ShapeTracker::BufferView::FromStridesAndExtents(
                         next_slice_strides, next_slice_extents));
  }

  return SlicePropagationResult{std::move(sliced_projections),
                                std::move(final_slice)};
}

absl::StatusOr<ShapeTracker> ShapeTracker::Narrow(
    absl::Span<const int64_t> dims_to_keep) const {
  if (absl::c_any_of(dims_to_keep, [this](int64_t dim) {
        return dim < 0 || dim >= input_shape_.dimensions().size();
      })) {
    return absl::InvalidArgumentError("Invalid dimension index to keep");
  }

  std::vector<int64_t> sorted_dims(dims_to_keep.begin(), dims_to_keep.end());
  absl::c_sort(sorted_dims);
  if (absl::c_adjacent_find(sorted_dims) != sorted_dims.end()) {
    return absl::InvalidArgumentError(
        "dims_to_keep must contain unique dimensions");
  }

  // Build the narrowed input shape.
  std::vector<int64_t> sliced_input_dims;
  sliced_input_dims.reserve(sorted_dims.size());
  absl::c_transform(
      sorted_dims, std::back_inserter(sliced_input_dims),
      [this](int64_t dim) { return input_shape_.dimensions(dim); });

  Shape sliced_input_shape =
      ShapeUtil::MakeShape(input_shape_.element_type(), sliced_input_dims);
  ShapeTracker sliced_tracker(sliced_input_shape);

  // If dimensions that we keep are 1, we can just reshape it to a scalar.
  if (absl::c_all_of(dims_to_keep, [this](int64_t dim) {
        return input_shape_.dimensions(dim) == 1;
      })) {
    ABSL_RETURN_IF_ERROR(sliced_tracker.AppendReshape({}));
    return sliced_tracker;
  }

  // If the dims_to_keep came not sorted, start with a transpose to make them
  // sorted.
  {
    std::vector<int64_t> perm(dims_to_keep.size());
    absl::c_iota(perm, 0);
    absl::c_sort(perm, [&](int64_t a, int64_t b) {
      return dims_to_keep[a] < dims_to_keep[b];
    });
    ABSL_RETURN_IF_ERROR(sliced_tracker.PrependTranspose(perm));
  }

  // Build the current slice to keep.
  BufferView input_view = BufferView::FromShape(input_shape_);
  llvm::SmallVector<int64_t, 6> keep_strides;
  llvm::SmallVector<int64_t, 6> keep_extents;
  for (int64_t dim : sorted_dims) {
    keep_strides.push_back(input_view.strides()[dim]);
    keep_extents.push_back(input_view.extents()[dim]);
  }
  ABSL_ASSIGN_OR_RETURN(BufferView keep_view, BufferView::FromStridesAndExtents(
                                             keep_strides, keep_extents));

  // Slice the projections, and pack them.
  ABSL_ASSIGN_OR_RETURN(SlicePropagationResult propagation_result,
                   SliceProjectionChain(projections_, keep_view));

  // Append rather than assign, for the case the tracker has an initial
  // transpose.
  for (const auto& proj : propagation_result.sliced_projections) {
    sliced_tracker.projections_.push_back(proj);
    sliced_tracker.projections_.back().MergeAdjacentDimensions();
    TryFoldProjections(sliced_tracker.projections_);
  }

  // Build the narrowed output shape.
  absl::Span<const int64_t> output_dims =
      propagation_result.final_slice.IsEmpty()
          ? absl::Span<const int64_t>()
          : propagation_result.sliced_projections.back().extents();
  Shape sliced_output_shape =
      ShapeUtil::MakeShape(output_shape_.element_type(), output_dims);
  sliced_tracker.output_shape_ = std::move(sliced_output_shape);

  return sliced_tracker;
}

std::optional<std::vector<int64_t>>
ShapeTracker::MapInputDimensionsToOutputUnordered(
    absl::Span<const int64_t> input_dims) const {
  // Create a view which covers the input dimensions we want to map.
  BufferView keep_view =
      BufferView::FromShapeAndIndices(input_shape_, input_dims);
  keep_view.RemoveDegenerateDimensions();
  if (keep_view.ElementsIn() <= 1) {
    return std::vector<int64_t>{};
  }
  keep_view.SortByStrideDescending();
  keep_view.MergeAdjacentDimensions();

  // Propagate the view through the projection chain.
  auto propagation_result_or = SliceProjectionChain(projections_, keep_view);
  if (!propagation_result_or.ok()) {
    return std::nullopt;
  }
  BufferView final_slice = propagation_result_or->final_slice;
  if (final_slice.IsEmpty()) {
    return std::vector<int64_t>{};
  }

  // Map the final slice to the output logical dimensions.
  auto output_dim_views_opt =
      projections_.back().TryUnflatten(output_shape_.dimensions());
  if (!output_dim_views_opt.has_value()) {
    return std::nullopt;
  }
  const std::vector<BufferView>& output_dim_views = *output_dim_views_opt;

  std::vector<int64_t> kept_output_dims;
  kept_output_dims.reserve(output_dim_views.size());
  for (size_t i = 0; i < output_dim_views.size(); ++i) {
    if (output_shape_.dimensions(i) == 1) {
      // Skip the degenerate dimensions.
      continue;
    }
    auto intersection_opt = output_dim_views[i].TryIntersectWith(final_slice);
    if (!intersection_opt.has_value()) {
      // Couldn't cleanly intersect, so we can't map this input to output.
      return std::nullopt;
    }
    BufferView intersection = *intersection_opt;
    if (intersection.IsEmpty()) {
      continue;
    }
    if (intersection != output_dim_views[i]) {
      // Only partially intersected, so we can't map this input to output.
      return std::nullopt;
    }
    kept_output_dims.push_back(i);
  }

  return kept_output_dims;
}

bool ShapeTracker::MapsToOneStride(absl::Span<const int64_t> input_dims,
                                   bool allow_swaps) const {
  std::optional<std::vector<int64_t>> mapped =
      MapInputDimensionsToOutputUnordered(input_dims);
  if (!mapped.has_value()) {
    return false;
  }
  if (mapped->empty()) {
    return true;
  }

  // Check if output dimensions are consecutive.
  auto is_invalid_transition = [this](int64_t prev, int64_t next) {
    if (prev + 1 == next) {
      return false;
    }
    // Check if intermediate dimensions are degenerate.
    for (int64_t i = prev + 1; i < next; ++i) {
      if (output_shape_.dimensions(i) != 1) {
        return true;
      }
    }
    return false;
  };
  if (absl::c_adjacent_find(*mapped, is_invalid_transition) != mapped->end()) {
    return false;
  }

  if (allow_swaps) {
    return true;
  }

  // Check for swaps.
  absl::StatusOr<ShapeTracker> narrowed = Narrow(input_dims);
  if (!narrowed.ok()) {
    return false;
  }
  bool has_swaps =
      absl::c_any_of(narrowed->GetSteps(), [](const ShapeTracker::Step& step) {
        return step.type == ShapeTracker::Step::Type::kTranspose;
      });
  return !has_swaps;
}

absl::StatusOr<ShapeTracker> ShapeTracker::Zip(
    absl::Span<const ShapeTracker> trackers) {
  if (trackers.empty()) {
    return absl::InvalidArgumentError("Zip requires at least one ShapeTracker");
  }

  if (absl::c_any_of(trackers, [&](const ShapeTracker& a) {
        return a.input_shape().element_type() !=
               trackers[0].input_shape().element_type();
      })) {
    return absl::InvalidArgumentError("Element types must match for Zip");
  }
  xla::PrimitiveType element_type = trackers[0].input_shape().element_type();

  auto concat_shapes = [&](auto get_shape) {
    std::vector<int64_t> joint_dims;
    for (const auto& tracker : trackers) {
      absl::c_copy(get_shape(tracker).dimensions(),
                   std::back_inserter(joint_dims));
    }
    Shape joint_shape = ShapeUtil::MakeShape(element_type, joint_dims);
    return joint_shape;
  };

  Shape zipped_input_shape =
      concat_shapes([](const ShapeTracker& t) { return t.input_shape(); });
  Shape zipped_output_shape =
      concat_shapes([](const ShapeTracker& t) { return t.output_shape(); });

  ShapeTracker zipped(zipped_input_shape);
  zipped.output_shape_ = std::move(zipped_output_shape);

  auto is_trivial_one_element_tracker = [](const ShapeTracker& t) {
    return ShapeUtil::ElementsIn(t.input_shape()) == 1 &&
           ShapeUtil::ElementsIn(t.output_shape()) == 1 &&
           t.projections_.size() == 1 &&
           t.projections_[0].IsTrivialOneElement();
  };

  if (absl::c_all_of(trackers, is_trivial_one_element_tracker)) {
    return zipped;
  }

  size_t max_projections = 0;
  for (const auto& tracker : trackers) {
    if (!is_trivial_one_element_tracker(tracker)) {
      max_projections = std::max(max_projections, tracker.projections_.size());
    }
  }

  zipped.projections_.clear();
  zipped.projections_.resize(max_projections, ShapeTracker::BufferView());

  auto get_step_input_elements = [](const ShapeTracker& t, size_t s) {
    if (s < t.projections_.size()) {
      return t.projections_[s].InputElementsIn();
    }
    return ShapeUtil::ElementsIn(t.output_shape());
  };

  for (size_t s = 0; s < max_projections; ++s) {
    int64_t total_step_input_elements = 1;
    for (const auto& tracker : trackers) {
      total_step_input_elements *= get_step_input_elements(tracker, s);
    }

    int64_t scale = total_step_input_elements;
    for (const auto& tracker : trackers) {
      int64_t tracker_input_elements = get_step_input_elements(tracker, s);
      scale /= tracker_input_elements;
      if (is_trivial_one_element_tracker(tracker)) {
        continue;
      }

      if (s < tracker.projections_.size()) {
        const auto& view = tracker.projections_[s];
        for (size_t j = 0; j < view.strides_.size(); ++j) {
          if (view.extents_[j] <= 1) {
            continue;
          }
          int64_t scaled_stride =
              view.strides_[j] == 0 ? 0 : view.strides_[j] * scale;
          zipped.projections_[s].strides_.push_back(scaled_stride);
          zipped.projections_[s].extents_.push_back(view.extents_[j]);
        }
        for (size_t k = 0; k < view.collapsed_strides_.size(); ++k) {
          if (view.collapsed_strides_[k] > 0 &&
              view.collapsed_extents_[k] > 1) {
            zipped.projections_[s].collapsed_strides_.push_back(
                view.collapsed_strides_[k] * scale);
            zipped.projections_[s].collapsed_extents_.push_back(
                view.collapsed_extents_[k]);
          }
        }
      } else if (tracker_input_elements > 1) {
        zipped.projections_[s].strides_.push_back(scale);
        zipped.projections_[s].extents_.push_back(tracker_input_elements);
      }
    }

    if (zipped.projections_[s].strides_.empty()) {
      zipped.projections_[s].strides_.push_back(1);
      zipped.projections_[s].extents_.push_back(1);
    }
    zipped.projections_[s].MergeAdjacentDimensions();
  }

  TryFoldProjections(zipped.projections_);
  return zipped;
}

}  // namespace xla
