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
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
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
//  - The support of 1-element shapes happens in the outer layer of the class
//  and doesn't use projections logic.
//  - Projections never have extents of size 1. (except for the case of
//  1-element shapes, which don't use projections anyway).

namespace xla {
namespace {

struct PhysicalDimension {
  int64_t size = 0;
  int64_t src_logical_idx = -1;
  int64_t dst_logical_idx = -1;
  int64_t idx = -1;
  int64_t src_expanded_idx = -1;
};

// Returns a common physical decomposition of dimensions for both shapes to
// facilitate bitcast lowering. The returned dimensions are ordered major to
// minor.
absl::StatusOr<std::vector<PhysicalDimension>> BuildPhysicalDimensions(
    const Shape& src_shape, const Shape& dst_shape) {
  std::vector<PhysicalDimension> result;
  int64_t src_idx = 0;
  int64_t dst_idx = 0;
  int64_t src_rem = 1;
  int64_t dst_rem = 1;

  auto get_dim = [](const Shape& shape, int64_t idx) {
    return shape.dimensions(shape.layout().minor_to_major(idx));
  };
  auto get_logical = [](const Shape& shape, int64_t idx) {
    return shape.layout().minor_to_major(idx);
  };

  while (src_idx < src_shape.dimensions().size() ||
         dst_idx < dst_shape.dimensions().size()) {
    if (src_rem == 1 && src_idx < src_shape.dimensions().size()) {
      src_rem = get_dim(src_shape, src_idx);
    }
    if (dst_rem == 1 && dst_idx < dst_shape.dimensions().size()) {
      dst_rem = get_dim(dst_shape, dst_idx);
    }

    int64_t size = std::min(src_rem, dst_rem);
    PhysicalDimension dim;
    dim.size = size;
    dim.src_logical_idx =
        src_idx < src_shape.dimensions().size()
            ? get_logical(src_shape, src_idx)
            : get_logical(src_shape, src_shape.dimensions().size() - 1);
    dim.dst_logical_idx =
        dst_idx < dst_shape.dimensions().size()
            ? get_logical(dst_shape, dst_idx)
            : get_logical(dst_shape, dst_shape.dimensions().size() - 1);
    result.push_back(dim);

    if (src_rem % size != 0 || dst_rem % size != 0) {
      return absl::InvalidArgumentError("Cannot decompose bitcast");
    }

    src_rem /= size;
    dst_rem /= size;

    if (src_rem == 1) {
      src_idx++;
    }
    if (dst_rem == 1) {
      dst_idx++;
    }
  }

  absl::c_reverse(result);
  size_t idx = 0;
  for (PhysicalDimension& dim : result) {
    dim.idx = idx++;
  }
  return result;
}

void TryFoldProjections(std::vector<ShapeTracker::BufferView>& projections) {
  while (projections.size() > 1) {
    const auto& last = projections.back();
    auto transformation = last.AsTransformation();
    auto& prev = projections[projections.size() - 2];

    auto opt_sub_views = prev.TryUnflatten(transformation.input_reshape);
    if (!opt_sub_views.has_value()) {
      break;
    }

    std::vector<ShapeTracker::BufferView> permuted_views;
    permuted_views.reserve(opt_sub_views->size());
    for (size_t i = 0; i < transformation.transpose.size(); ++i) {
      permuted_views.push_back((*opt_sub_views)[transformation.transpose[i]]);
    }
    prev = ShapeTracker::BufferView::FromSubviews(permuted_views);
    prev.MergeAdjacentDimensions();
    projections.pop_back();
  }
}

}  // namespace

void ShapeTracker::BufferView::MergeAdjacentDimensions() {
  if (strides_.empty()) {
    return;
  }
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

std::optional<ShapeTracker::BufferView>
ShapeTracker::BufferView::TryIntersectWith(int64_t stride,
                                           int64_t extent) const {
  BufferView result;
  for (int64_t i = 0; i < strides_.size(); ++i) {
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
  return result;
}

void ShapeTracker::BufferView::Pack() {
  llvm::SmallVector<int64_t, 6> order(strides_.size());
  absl::c_iota(order, 0);
  absl::c_stable_sort(order, [&](unsigned a, unsigned b) {
    return strides_[a] < strides_[b];  // innermost (smallest stride) first
  });

  int64_t running = 1;
  for (unsigned i : order) {
    strides_[i] = running;
    running *= extents_[i];
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
    RETURN_IF_ERROR(tracker.AppendInstruction(inst));
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

  ASSIGN_OR_RETURN(ShapeTracker tracker1, FromProducerConsumer(lca, source));
  ASSIGN_OR_RETURN(ShapeTracker tracker2,
                   FromProducerConsumer(lca, destination));
  RETURN_IF_ERROR(tracker1.Invert());
  RETURN_IF_ERROR(tracker1.ConcatenateFrom(tracker2));
  return tracker1;
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

  const auto& old_dimensions = output_shape_.dimensions();
  std::vector<int64_t> non_degenerate_dims;
  absl::c_copy_if(old_dimensions, std::back_inserter(non_degenerate_dims),
                  [](int64_t d) { return d != 1; });

  auto* current_view = &projections_.back();
  auto opt_sub_views = current_view->TryUnflatten(non_degenerate_dims);

  if (!opt_sub_views.has_value()) {
    projections_.push_back(BufferView::FromShapeCompacted(output_shape_));
    current_view = &projections_.back();
    opt_sub_views = current_view->TryUnflatten(non_degenerate_dims);
    if (!opt_sub_views.has_value()) {
      return absl::InternalError(
          "Failed to unflatten contiguous layout after copy");
    }
  }

  std::vector<BufferView> permuted_sub_views;
  permuted_sub_views.reserve(non_degenerate_dims.size());
  for (int64_t dim : permutation) {
    if (old_dimensions[dim] == 1) {
      continue;
    }
    // Locate the physical index of this logical dimension by counting the
    // non-degenerate dimensions to its left in the old shape.
    int64_t physical_idx = absl::c_count_if(
        absl::Span<const int64_t>(old_dimensions).subspan(0, dim),
        [](int64_t d) { return d != 1; });
    permuted_sub_views.push_back((*opt_sub_views)[physical_idx]);
  }

  *current_view = BufferView::FromSubviews(permuted_sub_views);
  current_view->MergeAdjacentDimensions();

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

  if (ShapeUtil::ElementsIn(output_shape_) == 1) {
    output_shape_ =
        ShapeUtil::MakeShape(output_shape_.element_type(), dimensions);
    return absl::OkStatus();
  }

  output_shape_ =
      ShapeUtil::MakeShape(output_shape_.element_type(), dimensions);
  return absl::OkStatus();
}

// Decomposes a bitcast into a reshape-transpose-reshape sequence and appends
// it to the tracker.
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

  if (ShapeUtil::ElementsIn(output_shape_) == 1) {
    return AppendReshape(dst_shape.dimensions());
  }

  ASSIGN_OR_RETURN(std::vector<PhysicalDimension> dims,
                   BuildPhysicalDimensions(src_shape, dst_shape));
  absl::c_stable_sort(
      dims, [](const PhysicalDimension& a, const PhysicalDimension& b) {
        return a.src_logical_idx < b.src_logical_idx;
      });
  size_t idx = 0;
  for (PhysicalDimension& dim : dims) {
    dim.src_expanded_idx = idx++;
  }

  std::vector<int64_t> sizes;
  sizes.reserve(dims.size());
  for (const PhysicalDimension& dim : dims) {
    sizes.push_back(dim.size);
  }
  // Decompose bitcast into reshape-transpose-reshape:
  // 1. Reshape to physical dimensions.
  RETURN_IF_ERROR(AppendReshape(sizes));

  absl::c_sort(dims, [](const PhysicalDimension& a,
                        const PhysicalDimension& b) { return a.idx < b.idx; });
  absl::c_stable_sort(
      dims, [](const PhysicalDimension& a, const PhysicalDimension& b) {
        return a.dst_logical_idx < b.dst_logical_idx;
      });
  std::vector<int64_t> permutation;
  permutation.reserve(dims.size());
  for (const PhysicalDimension& dim : dims) {
    permutation.push_back(dim.src_expanded_idx);
  }
  // 2. Transpose to match destination physical layout.
  RETURN_IF_ERROR(AppendTranspose(permutation));
  // 3. Reshape to final destination shape.
  RETURN_IF_ERROR(AppendReshape(dst_shape.dimensions()));

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

  if (ShapeUtil::ElementsIn(output_shape_) == 1) {
    output_shape_ = other.output_shape();
    return absl::OkStatus();
  }

  for (const auto& step : other.GetSteps()) {
    switch (step.type) {
      case Step::Type::kReshape:
        RETURN_IF_ERROR(AppendReshape(step.dimensions));
        break;
      case Step::Type::kTranspose:
        RETURN_IF_ERROR(AppendTranspose(step.dimensions));
        break;
    }
  }

  return absl::OkStatus();
}

ShapeTracker::BufferView::Transformation
ShapeTracker::BufferView::AsTransformation() const {
  Transformation result;

  struct Atom {
    int64_t stride;
    int64_t extent;
    int64_t original_idx;
  };
  llvm::SmallVector<Atom, 6> atoms;
  atoms.reserve(strides_.size());
  for (size_t i = 0; i < strides_.size(); ++i) {
    atoms.push_back({strides_[i], extents_[i], static_cast<int64_t>(i)});
  }

  auto sorted_atoms = atoms;
  absl::c_sort(sorted_atoms, [](const Atom& a, const Atom& b) {
    return a.stride > b.stride;
  });

  result.input_reshape.reserve(sorted_atoms.size());
  for (const auto& atom : sorted_atoms) {
    result.input_reshape.push_back(atom.extent);
  }

  result.transpose.resize(atoms.size());
  for (size_t j = 0; j < sorted_atoms.size(); ++j) {
    result.transpose[sorted_atoms[j].original_idx] = j;
  }

  return result;
}

// To construct the inverse, we materialize our tracker to the list of
// reshape-transpose steps, and construct the tracker backwards. Note that it
// doesn't necessarily contain the same number of steps/projections.
absl::StatusOr<ShapeTracker> ShapeTracker::GetInverted() const {
  if (ShapeUtil::ElementsIn(output_shape_) == 1) {
    ShapeTracker inverted(output_shape_);
    inverted.output_shape_ = input_shape_;
    return inverted;
  }

  ShapeTracker inverted(output_shape_);
  for (auto it = projections_.rbegin(); it != projections_.rend(); ++it) {
    auto transformation = it->AsTransformation();

    std::vector<int64_t> transposed_dims(transformation.transpose.size());
    for (size_t i = 0; i < transformation.transpose.size(); ++i) {
      transposed_dims[i] =
          transformation.input_reshape[transformation.transpose[i]];
    }

    RETURN_IF_ERROR(inverted.AppendReshape(transposed_dims));

    std::vector<int64_t> inv_transpose(transformation.transpose.size());
    for (size_t i = 0; i < transformation.transpose.size(); ++i) {
      inv_transpose[transformation.transpose[i]] = i;
    }

    RETURN_IF_ERROR(inverted.AppendTranspose(inv_transpose));
  }

  RETURN_IF_ERROR(inverted.AppendReshape(input_shape_.dimensions()));

  return inverted;
}

absl::Status ShapeTracker::Invert() {
  ASSIGN_OR_RETURN(ShapeTracker inverted, GetInverted());
  *this = std::move(inverted);
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependTranspose(
    absl::Span<const int64_t> permutation) {
  if (IsIdentityPermutation(permutation)) {
    return absl::OkStatus();
  }
  llvm::SmallVector<int64_t, 6> inv_perm(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    inv_perm[permutation[i]] = i;
  }
  RETURN_IF_ERROR(Invert());
  RETURN_IF_ERROR(AppendTranspose(inv_perm));
  RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependReshape(
    absl::Span<const int64_t> dimensions) {
  RETURN_IF_ERROR(Invert());
  RETURN_IF_ERROR(AppendReshape(dimensions));
  RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

absl::Status ShapeTracker::PrependBitcast(const xla::Shape& src_shape,
                                          const xla::Shape& dst_shape) {
  RETURN_IF_ERROR(Invert());
  RETURN_IF_ERROR(AppendBitcast(dst_shape, src_shape));
  RETURN_IF_ERROR(Invert());
  return absl::OkStatus();
}

std::vector<ShapeTracker::Step> ShapeTracker::GetSteps() const {
  if (ShapeUtil::ElementsIn(input_shape_) == 1) {
    if (input_shape_.dimensions() == output_shape_.dimensions()) {
      return {};
    }
    return {{Step::Type::kReshape,
             std::vector<int64_t>(output_shape_.dimensions().begin(),
                                  output_shape_.dimensions().end())}};
  }

  std::vector<Step> steps;
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  for (size_t i = 0; i < projections_.size(); ++i) {
    const auto& projection = projections_[i];
    auto transformation = projection.AsTransformation();

    // If it is the last projection, and its transpose is identity,
    // we can skip it completely because the final reshape will handle it.
    if (i == projections_.size() - 1 &&
        IsIdentityPermutation(transformation.transpose)) {
      continue;
    }

    if (!absl::c_equal(current_dims, transformation.input_reshape)) {
      std::vector<int64_t> input_reshape_dims(
          transformation.input_reshape.begin(),
          transformation.input_reshape.end());
      steps.push_back({Step::Type::kReshape, input_reshape_dims});
      current_dims = std::move(input_reshape_dims);
    }

    if (IsIdentityPermutation(transformation.transpose)) {
      continue;
    }

    std::vector<int64_t> transpose_perm(transformation.transpose.begin(),
                                        transformation.transpose.end());
    steps.push_back({Step::Type::kTranspose, transpose_perm});

    std::vector<int64_t> new_dims(transpose_perm.size());
    for (size_t i = 0; i < transpose_perm.size(); ++i) {
      new_dims[i] = current_dims[transpose_perm[i]];
    }
    current_dims = std::move(new_dims);
  }

  if (!absl::c_equal(current_dims, output_shape_.dimensions())) {
    steps.push_back({Step::Type::kReshape,
                     std::vector<int64_t>(output_shape_.dimensions().begin(),
                                          output_shape_.dimensions().end())});
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

// If a reshape tries to glue dimensions, the function keeps them when possible.
// Do to that, it keeps the strides of the target dimensions (which we are
// required to preserve), and inserts existing dimensions when it doesn't
// violate divisibility.
std::pair<ShapeTracker::Step, std::vector<int64_t>> ExpandReshapeStep(
    const ShapeTracker::Step& step, const std::vector<int64_t>& current_shape) {
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
          new_mapping};
}

// Performs a transpose over an expanded shape.
ShapeTracker::Step ExpandTransposeStep(const ShapeTracker::Step& step,
                                       const std::vector<int64_t>& expansion) {
  std::vector<int64_t> expanded;
  expanded.reserve(expansion.back() + 1);
  for (const int64_t dim : step.dimensions) {
    const int64_t begin = dim == 0 ? 0 : expansion[dim - 1] + 1;
    const int64_t end = expansion[dim] + 1;
    const auto first = expanded.insert(expanded.end(), end - begin, 0);
    std::iota(first, expanded.end(), begin);
  }
  return ShapeTracker::Step{ShapeTracker::Step::Type::kTranspose, expanded};
}

}  // namespace

std::vector<ShapeTracker::Step> ShapeTracker::OptimizeSteps(
    const std::vector<Step>& steps, const xla::Shape& input_shape,
    const xla::Shape& output_shape) {
  if (ShapeUtil::ElementsIn(input_shape) == 1) {
    return steps;
  }
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
    Step new_step;
    switch (step.type) {
      case Step::Type::kReshape:
        std::tie(new_step, mapping) = ExpandReshapeStep(step, current_shape);
        if (new_step.dimensions != current_shape) {
          current_shape = new_step.dimensions;
          optimized_steps.push_back(std::move(new_step));
        }
        break;
      case Step::Type::kTranspose:
        new_step = ExpandTransposeStep(step, mapping);
        std::vector<int64_t> new_shape(new_step.dimensions.size());
        for (size_t i = 0; i < new_step.dimensions.size(); ++i) {
          new_shape[i] = current_shape[new_step.dimensions[i]];
        }
        current_shape = std::move(new_shape);
        optimized_steps.push_back(std::move(new_step));
        break;
    }
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

  HloComputation* computation = inst->parent();
  HloInstruction* current_inst = inst;

  std::vector<Step> steps = GetSteps();
  if (avoid_combining_reshapes) {
    steps = OptimizeSteps(steps, input_shape_, output_shape_);
  }

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
    }
  }

  return current_inst;
}

std::string ShapeTracker::DebugString(bool avoid_combining_reshapes) const {
  std::string result =
      absl::StrCat("[", absl::StrJoin(input_shape_.dimensions(), ","), "]");
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  std::vector<Step> steps = GetSteps();
  if (avoid_combining_reshapes) {
    steps = OptimizeSteps(steps, input_shape_, output_shape_);
  }

  for (const auto& step : steps) {
    switch (step.type) {
      case Step::Type::kReshape: {
        absl::StrAppend(&result, " -> R[", absl::StrJoin(step.dimensions, ","),
                        "]");
        current_dims = step.dimensions;
        break;
      }
      case Step::Type::kTranspose: {
        std::vector<int64_t> new_dims(step.dimensions.size());
        for (size_t i = 0; i < step.dimensions.size(); ++i) {
          new_dims[i] = current_dims[step.dimensions[i]];
        }
        absl::StrAppend(&result, " -> T[", absl::StrJoin(new_dims, ","), "]");
        current_dims = std::move(new_dims);
        break;
      }
    }
  }

  return result;
}

namespace {

struct SlicePropagationResult {
  std::vector<ShapeTracker::BufferView> sliced_projections;
  ShapeTracker::BufferView final_slice;
};

// Keeps the parts of the projections that intersect with the @slice. I.e.
// tracks the life of the slice as it goes through the projections.
absl::StatusOr<SlicePropagationResult> SliceProjectionChain(
    absl::Span<const ShapeTracker::BufferView> projections,
    const ShapeTracker::BufferView& slice) {
  int64_t expected_elements = slice.ElementsIn();
  CHECK_GT(expected_elements, 1)
      << "slice.ElementsIn() == 1 should be handled by the caller";

  std::vector<ShapeTracker::BufferView> sliced_projections;
  sliced_projections.reserve(projections.size());

  ShapeTracker::BufferView current_slice = slice;

  for (size_t i = 0; i < projections.size(); ++i) {
    auto intersection_opt = projections[i].TryIntersectWith(current_slice);
    if (!intersection_opt.has_value()) {
      return absl::InvalidArgumentError(
          "Slice is incompatible with projection");
    }
    ShapeTracker::BufferView intersection = *intersection_opt;

    ShapeTracker::BufferView packed_proj = projections[i];
    packed_proj.Pack();

    llvm::SmallVector<int64_t, 6> next_slice_strides;
    llvm::SmallVector<int64_t, 6> next_slice_extents;
    for (auto [s, e] :
         llvm::zip(intersection.strides(), intersection.extents())) {
      // Find the dimension in the projection that this slice dimension belongs
      // to.
      auto zip_range =
          llvm::zip(projections[i].strides(), projections[i].extents(),
                    packed_proj.strides());
      auto it = absl::c_find_if(zip_range, [s = s, e = e](const auto& tuple) {
        auto [proj_stride, proj_extent, packed_stride] = tuple;
        return s % proj_stride == 0 && s * e <= proj_stride * proj_extent;
      });

      if (it == zip_range.end()) {
        return absl::InternalError(
            "Slice dimension does not belong to any projection dimension");
      }
      auto [proj_stride, proj_extent, packed_stride] = *it;

      // Every projection "reimagines" the previous projection as sorted by
      // dimension order (major-to-minor).
      // Here we map the slice stride `s` into the output space of this
      // projection. `s / proj_stride` computes the logical stride of the
      // slice within dimension `d`, which we then multiply by the packed
      // output stride for dimension `d` to accumulate and propagate the
      // correct stride to the next iteration.
      next_slice_strides.push_back(packed_stride * (s / proj_stride));
      next_slice_extents.push_back(e);
    }

    if (intersection.ElementsIn() != expected_elements) {
      return absl::InternalError("Lost elements during slice propagation");
    }

    sliced_projections.push_back(intersection);
    TryFoldProjections(sliced_projections);
    sliced_projections.back().Pack();

    ASSIGN_OR_RETURN(current_slice,
                     ShapeTracker::BufferView::FromStridesAndExtents(
                         next_slice_strides, next_slice_extents));
  }

  return SlicePropagationResult{std::move(sliced_projections),
                                std::move(current_slice)};
}

}  // namespace

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
    RETURN_IF_ERROR(sliced_tracker.AppendReshape({}));
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
    RETURN_IF_ERROR(sliced_tracker.PrependTranspose(perm));
  }

  // Build the current slice to keep.
  BufferView input_view = BufferView::FromShape(input_shape_);
  llvm::SmallVector<int64_t, 6> keep_strides;
  llvm::SmallVector<int64_t, 6> keep_extents;
  for (int64_t dim : sorted_dims) {
    keep_strides.push_back(input_view.strides()[dim]);
    keep_extents.push_back(input_view.extents()[dim]);
  }
  ASSIGN_OR_RETURN(BufferView keep_view, BufferView::FromStridesAndExtents(
                                             keep_strides, keep_extents));

  // Slice the projections, and pack them.
  ASSIGN_OR_RETURN(SlicePropagationResult propagation_result,
                   SliceProjectionChain(projections_, keep_view));

  // Append rather than assign, for the case the tracker has an initial
  // transpose.
  for (const auto& proj : propagation_result.sliced_projections) {
    sliced_tracker.projections_.push_back(proj);
    TryFoldProjections(sliced_tracker.projections_);
  }

  // Build the narrowed output shape.
  Shape sliced_output_shape = ShapeUtil::MakeShape(
      output_shape_.element_type(),
      propagation_result.sliced_projections.back().extents());
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

  int64_t total_elements = ShapeUtil::ElementsIn(zipped_input_shape);
  if (total_elements == 1) {
    return zipped;
  }

  size_t max_projections = 0;
  for (const auto& tracker : trackers) {
    if (ShapeUtil::ElementsIn(tracker.input_shape()) > 1) {
      max_projections = std::max(max_projections, tracker.projections_.size());
    }
  }

  zipped.projections_.clear();
  zipped.projections_.resize(max_projections, ShapeTracker::BufferView());

  int64_t scale = total_elements;
  for (const auto& tracker : trackers) {
    int64_t tracker_elements = ShapeUtil::ElementsIn(tracker.input_shape());
    scale /= tracker_elements;
    if (tracker_elements == 1) {
      continue;
    }

    for (size_t s = 0; s < tracker.projections_.size(); ++s) {
      const auto& view = tracker.projections_[s];
      for (size_t j = 0; j < view.strides_.size(); ++j) {
        zipped.projections_[s].strides_.push_back(view.strides_[j] * scale);
        zipped.projections_[s].extents_.push_back(view.extents_[j]);
      }
    }

    // If the current tracker has fewer projections than the max, pad it with a
    // noop.
    for (size_t s = tracker.projections_.size(); s < max_projections; ++s) {
      zipped.projections_[s].strides_.push_back(scale);
      zipped.projections_[s].extents_.push_back(tracker_elements);
    }
  }

  for (auto& projection : zipped.projections_) {
    projection.MergeAdjacentDimensions();
  }

  return zipped;
}

}  // namespace xla
