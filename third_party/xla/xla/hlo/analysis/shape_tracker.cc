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
#include <iterator>
#include <string>
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

// Represents the view of a single logical dimension, which may consist of
// multiple physical segments (strides and extents) if the dimension is not
// contiguous.
struct DimensionView {
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> extents;

  bool is_contiguous() const { return strides.size() == 1; }
  bool operator==(const DimensionView& other) const {
    return strides == other.strides && extents == other.extents;
  }
};

// Represents a view of a buffer, defined by a sequence of dimension views.
struct BufferView {
  llvm::SmallVector<DimensionView, 6> views;

  DimensionView GetFlattenedAndCompactedView() const {
    DimensionView compacted;
    for (const auto& dim_view : views) {
      for (size_t i = 0; i < dim_view.strides.size(); ++i) {
        int64_t stride = dim_view.strides[i];
        int64_t extent = dim_view.extents[i];

        if (compacted.strides.empty()) {
          compacted.strides.push_back(stride);
          compacted.extents.push_back(extent);
        } else if (compacted.strides.back() == stride * extent) {
          compacted.extents.back() *= extent;
          compacted.strides.back() = stride;
        } else {
          compacted.strides.push_back(stride);
          compacted.extents.push_back(extent);
        }
      }
    }
    return compacted;
  }

  bool operator==(const BufferView& other) const {
    return views == other.views;
  }
};

BufferView ShapeToBufferView(const xla::Shape& shape) {
  BufferView view;

  const auto& dimensions = shape.dimensions();
  int rank = dimensions.size();

  std::vector<int64_t> strides(rank);

  // Default layout: major-to-minor.
  int64_t current_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = current_stride;
    current_stride *= dimensions[i];
  }

  for (int i = 0; i < rank; ++i) {
    if (dimensions[i] == 1) {
      continue;  // Skip degenerate dimensions.
    }
    DimensionView dim_view;
    dim_view.strides.push_back(strides[i]);
    dim_view.extents.push_back(dimensions[i]);
    view.views.push_back(dim_view);
  }

  return view;
}

bool TryReshape(BufferView& view, absl::Span<const int64_t> new_dims) {
  DimensionView compacted = view.GetFlattenedAndCompactedView();

  if (compacted.strides.empty()) {
    view = BufferView{};
    return true;
  }

  BufferView new_view;
  size_t atom_idx = 0;
  int64_t rem_stride = compacted.strides[0];
  int64_t rem_extent = compacted.extents[0];

  for (int64_t d : new_dims) {
    if (d == 1) {
      continue;
    }

    DimensionView dim_view;
    int64_t rem_d = d;

    while (rem_d > 1) {
      int64_t take_extent = std::min(rem_extent, rem_d);
      if (rem_extent % take_extent != 0 || rem_d % take_extent != 0) {
        return false;
      }

      int64_t take_stride = rem_stride * (rem_extent / take_extent);
      if (!dim_view.strides.empty() &&
          take_stride != dim_view.strides.back() * dim_view.extents.back()) {
        return false;
      }

      dim_view.strides.push_back(take_stride);
      dim_view.extents.push_back(take_extent);

      rem_d /= take_extent;
      rem_extent /= take_extent;

      if (rem_extent == 1) {
        atom_idx++;
        if (atom_idx < compacted.strides.size()) {
          rem_stride = compacted.strides[atom_idx];
          rem_extent = compacted.extents[atom_idx];
        }
      }
    }
    new_view.views.push_back(dim_view);
  }

  view = new_view;
  return true;
}

}  // namespace

// Represents a mapping between an input buffer view and an output buffer
// view. This is used to track how data is projected or reshaped.
struct ShapeTracker::ViewMapping {
  BufferView output;

  struct Transformation {
    llvm::SmallVector<int64_t, 6> input_reshape;
    llvm::SmallVector<int64_t, 6> transpose;
  };

  explicit ViewMapping(const xla::Shape& shape);

  Transformation output_transformation() const;

  bool operator==(const ViewMapping& other) const {
    return output == other.output;
  }
};

ShapeTracker::~ShapeTracker() = default;
ShapeTracker::ShapeTracker(const ShapeTracker&) = default;
ShapeTracker::ShapeTracker(ShapeTracker&&) noexcept = default;
ShapeTracker& ShapeTracker::operator=(const ShapeTracker&) = default;
ShapeTracker& ShapeTracker::operator=(ShapeTracker&&) noexcept = default;

ShapeTracker::ViewMapping::ViewMapping(const xla::Shape& shape) {
  output = ShapeToBufferView(shape);
}

ShapeTracker::ShapeTracker(xla::Shape shape)
    : input_shape_(shape), output_shape_(shape) {
  projections_.push_back(ViewMapping(shape));
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

// Transposes are always just shuffling of the existing projection.
absl::Status ShapeTracker::AppendTranspose(
    absl::Span<const int64_t> permutation) {
  if (permutation.size() != output_shape_.dimensions().size()) {
    return absl::InvalidArgumentError("Rank mismatch");
  }

  const auto& old_dimensions = output_shape_.dimensions();
  std::vector<int> old_logical_to_view_idx(old_dimensions.size(), -1);
  int view_idx = 0;
  for (int i = 0; i < old_dimensions.size(); ++i) {
    if (old_dimensions[i] != 1) {
      old_logical_to_view_idx[i] = view_idx++;
    }
  }

  BufferView new_output_view;
  const BufferView& old_output_view = projections_.back().output;

  for (int64_t dim : permutation) {
    int idx = old_logical_to_view_idx[dim];
    if (idx != -1) {
      new_output_view.views.push_back(old_output_view.views[idx]);
    }
  }

  projections_.back().output = new_output_view;
  output_shape_ = ShapeUtil::PermuteDimensions(permutation, output_shape_);
  LayoutUtil::SetToDefaultLayout(&output_shape_);

  return absl::OkStatus();
}

// Tries to reshape without introducing a copy. If not possible, introduces a
// copy and reshapes.
absl::Status ShapeTracker::AppendReshape(absl::Span<const int64_t> dimensions) {
  int64_t new_elements = Product(dimensions);
  if (new_elements != ShapeUtil::ElementsIn(output_shape_)) {
    return absl::InvalidArgumentError("Product of dimensions mismatch");
  }

  if (!TryReshape(projections_.back().output, dimensions)) {
    // Introduce a copy as the reshape is not a bitcast.
    projections_.push_back(ViewMapping(output_shape_));
    if (!TryReshape(projections_.back().output, dimensions)) {
      return absl::InternalError("Failed to reshape after introducing copy");
    }
  }

  output_shape_ =
      ShapeUtil::MakeShape(output_shape_.element_type(), dimensions);

  TryFoldProjection();
  return absl::OkStatus();
}

// Attempts to fold the latest projection into the previous one (by trying to
// apply the reshape) to minimize projections.
void ShapeTracker::TryFoldProjection() {
  while (projections_.size() > 1) {
    const auto& last = projections_.back();
    auto transformation = last.output_transformation();
    auto& prev = projections_[projections_.size() - 2];

    BufferView temp_output = prev.output;
    if (!TryReshape(temp_output, transformation.input_reshape)) {
      break;
    }

    llvm::SmallVector<DimensionView, 6> permuted_views(
        temp_output.views.size());
    for (size_t i = 0; i < transformation.transpose.size(); ++i) {
      permuted_views[i] = temp_output.views[transformation.transpose[i]];
    }
    temp_output.views = std::move(permuted_views);

    prev.output = std::move(temp_output);
    projections_.pop_back();
  }
}

// Decomposes a bitcast into a reshape-transpose-reshape sequence and appends
// it to the tracker.
absl::Status ShapeTracker::AppendBitcast(const xla::Shape& src_shape,
                                         const xla::Shape& dst_shape) {
  if (!ShapeUtil::Compatible(src_shape, output_shape_)) {
    return absl::InvalidArgumentError(
        "Bitcast operand shape does not match current output shape");
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

// Converts the projection into reshape+transpose.
ShapeTracker::ViewMapping::Transformation
ShapeTracker::ViewMapping::output_transformation() const {
  Transformation result;

  struct Atom {
    int64_t stride;
    int64_t extent;
    int64_t original_idx;
  };
  llvm::SmallVector<Atom, 6> atoms;
  int64_t atom_idx = 0;
  for (const auto& dim_view : output.views) {
    for (size_t i = 0; i < dim_view.strides.size(); ++i) {
      atoms.push_back({dim_view.strides[i], dim_view.extents[i], atom_idx++});
    }
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
  for (size_t i = 0; i < atoms.size(); ++i) {
    auto it = absl::c_find_if(
        sorted_atoms, [&](const Atom& a) { return a.original_idx == i; });
    result.transpose[i] = std::distance(sorted_atoms.begin(), it);
  }

  return result;
}

// To construct the inverse, we materialize our tracker to the list of
// reshape-transpose steps, and construct the tracker backwards. Note that it
// doesn't necessarily contain the same number of steps/projections.
absl::StatusOr<ShapeTracker> ShapeTracker::GetInverted() const {
  ShapeTracker inverted(output_shape_);
  for (auto it = projections_.rbegin(); it != projections_.rend(); ++it) {
    auto transformation = it->output_transformation();

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
  std::vector<int64_t> inv_perm(permutation.size());
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
  std::vector<Step> steps;
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  for (size_t i = 0; i < projections_.size(); ++i) {
    const auto& projection = projections_[i];
    auto transformation = projection.output_transformation();

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

absl::StatusOr<HloInstruction*> ShapeTracker::ToInstructionChain(
    HloInstruction* inst) const {
  if (inst->shape().dimensions() != input_shape_.dimensions()) {
    return absl::InvalidArgumentError(
        "Input instruction shape dimensions mismatch");
  }

  HloComputation* computation = inst->parent();
  HloInstruction* current_inst = inst;

  for (const auto& step : GetSteps()) {
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

std::string ShapeTracker::DebugString() const {
  std::string result =
      absl::StrCat("[", absl::StrJoin(input_shape_.dimensions(), ","), "]");
  std::vector<int64_t> current_dims(input_shape_.dimensions().begin(),
                                    input_shape_.dimensions().end());

  for (const auto& step : GetSteps()) {
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
