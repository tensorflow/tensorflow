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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

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

ShapeTracker::BufferView ShapeTracker::BufferView::FromShapeCompacted(
    const xla::Shape& shape) {
  BufferView view;
  int64_t total_elements = ShapeUtil::ElementsIn(shape);
  view.strides_.push_back(1);
  view.extents_.push_back(total_elements);
  return view;
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

  TryFoldProjection();
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

// Attempts to fold the latest projection into the previous one (by trying to
// apply the reshape) to minimize projections.
void ShapeTracker::TryFoldProjection() {
  while (projections_.size() > 1) {
    const auto& last = projections_.back();
    auto transformation = last.AsTransformation();
    auto& prev = projections_[projections_.size() - 2];

    auto opt_sub_views = prev.TryUnflatten(transformation.input_reshape);
    if (!opt_sub_views.has_value()) {
      break;
    }

    std::vector<BufferView> permuted_views;
    permuted_views.reserve(opt_sub_views->size());
    for (size_t i = 0; i < transformation.transpose.size(); ++i) {
      permuted_views.push_back((*opt_sub_views)[transformation.transpose[i]]);
    }
    prev = BufferView::FromSubviews(permuted_views);
    prev.MergeAdjacentDimensions();
    projections_.pop_back();
  }
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
  if (inst->operand_count() == 0) {
    return absl::InvalidArgumentError(
        "Instruction must have at least one operand");
  }
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
// required to preserve), and inserts existing dimensions when it doens't
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
  std::iota(mapping.begin(), mapping.end(), 0);

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
    if (!optimized_steps.empty() &&
        optimized_steps.back().type == Step::Type::kReshape) {
      // Overwrite the last reshape to avoid consecutive reshapes
      optimized_steps.back().dimensions = out_dims;
    } else {
      optimized_steps.push_back({Step::Type::kReshape, out_dims});
    }
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

}  // namespace xla
