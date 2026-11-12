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

#ifndef XLA_HLO_ANALYSIS_SHAPE_TRACKER_H_
#define XLA_HLO_ANALYSIS_SHAPE_TRACKER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class HloInstruction;

// Helper class to track and simplify chains of shape transformations
// (reshapes, transposes, bitcasts, broadcasts, and abstract reductions). It
// tries to canonicalize and minimize operations by folding them when possible.
// The tracker ignores the layout of the shape (i.e. intended to run pre-layout
// assignment).
//
// Example usage:
//   ShapeTracker tracker(initial_shape);
//   ABSL_RETURN_IF_ERROR(tracker.AppendReshape(new_dims));
//   ABSL_RETURN_IF_ERROR(tracker.AppendTranspose(permutation));
//   ...
//   ABSL_ASSIGN_OR_RETURN(HloInstruction* chain,
//                       tracker.ToInstructionChain(base_inst));
//
// This class works correctly when there are degenerate dimensions (size 1), but
// the resulting chain may not always be minimal as it may insert reshapes to
// remove or add degenerate dimensions during simplification.
//
// Internally, the tracker represents the view as a sequence of projections,
// each decomposing into Reshape -> Reduce -> Transpose -> Broadcast, with
// reshapes before and after the sequence to remove or introduce degenerate
// dimensions.
class ShapeTracker {
 public:
  // Constructs a ShapeTracker with the given initial shape.
  explicit ShapeTracker(xla::Shape shape);

  ~ShapeTracker();
  ShapeTracker(const ShapeTracker&);
  ShapeTracker(ShapeTracker&&) noexcept;
  ShapeTracker& operator=(const ShapeTracker&);
  ShapeTracker& operator=(ShapeTracker&&) noexcept;

  // Returns true if the trackers represent the same input/output shapes
  // (ignoring layouts) and transformations. Note that (in theory) different
  // transformations chains can result in the same input/output elements
  // mapping. In this case, this operator will return false.
  bool operator==(const ShapeTracker& other) const;
  bool operator!=(const ShapeTracker& other) const { return !(*this == other); }

  // Sets the element type of input/output shapes.
  void SetElementType(PrimitiveType element_type);

  // Constructs a ShapeTracker from a chain of operations from `producer` to
  // `consumer`. The tracker tracks transformations from the output of
  // `producer` to the output of `consumer`. `producer` is excluded from the
  // chain. Assumes a linear chain of shape-modifying operations between them.
  static absl::StatusOr<ShapeTracker> FromProducerConsumer(
      const HloInstruction* producer, const HloInstruction* consumer);

  // Constructs a ShapeTracker representing the transformation path from one
  // "sibling" HLO instruction to another, via their lowest common ancestor.
  static absl::StatusOr<ShapeTracker> FromSiblings(
      const HloInstruction* source, const HloInstruction* destination);

  const xla::Shape& input_shape() const { return input_shape_; }
  const xla::Shape& output_shape() const { return output_shape_; }

  // Appends operations to the tracker.
  absl::Status AppendTranspose(absl::Span<const int64_t> permutation);
  absl::Status AppendReshape(absl::Span<const int64_t> dimensions);
  // Note that strictly speaking bitcast is a layout-aware transformation which
  // may feel like a contradiction to the "layout agnostic" claim above.
  // However, what that function does is reordering the logical dimensions
  // accordingly. It may e.g. be used to convert bitcasts back to
  // reshapes+transposes.
  absl::Status AppendBitcast(const xla::Shape& src_shape,
                             const xla::Shape& dst_shape);
  // Broadcasts the current output shape to `dimensions`, where each existing
  // dimension `i` maps to output dimension `broadcast_dimensions[i]`.
  absl::Status AppendBroadcast(absl::Span<const int64_t> dimensions,
                               absl::Span<const int64_t> broadcast_dimensions);
  // Collapses the specified output dimensions (formal inverse of broadcast).
  absl::Status AppendReduce(absl::Span<const int64_t> dimensions_to_reduce);

  // Appends operation represented by the HLO instruction.
  // Supports transpose, reshape, bitcast, and broadcast.
  absl::Status AppendInstruction(const HloInstruction* inst);

  // Prepends operation represented by the HLO instruction.
  // Supports transpose, reshape, bitcast, and broadcast.
  absl::Status PrependInstruction(const HloInstruction* inst);

  // Prepends operations to the tracker.
  absl::Status PrependTranspose(absl::Span<const int64_t> permutation);
  absl::Status PrependReshape(absl::Span<const int64_t> dimensions);
  absl::Status PrependBitcast(const xla::Shape& src_shape,
                              const xla::Shape& dst_shape);
  // Prepends a broadcast whose output matches the current `input_shape()`,
  // where `broadcast_dimensions` lists the dimensions of `input_shape()` that
  // originate from the pre-broadcast operand.
  absl::Status PrependBroadcast(absl::Span<const int64_t> broadcast_dimensions);
  // Prepends a reduction from `dimensions` (collapsing `dimensions_to_reduce`)
  // whose result matches the current `input_shape()`.
  absl::Status PrependReduce(absl::Span<const int64_t> dimensions,
                             absl::Span<const int64_t> dimensions_to_reduce);

  // Concatenates another ShapeTracker to this one.
  absl::Status ConcatenateFrom(const ShapeTracker& other);

  // Returns a new ShapeTracker with inverted operations.
  absl::StatusOr<ShapeTracker> GetInverted() const;

  // Inverts this ShapeTracker in-place.
  absl::Status Invert();

  // Slices the tracker keeping only the specified input dimensions (and
  // tracking them through the tracker). If dimensions are not sorted, the
  // tracker implicitly starts with a transpose that puts them in order. The
  // output shape of the returned tracker won't have degenerate dimensions as
  // it's not possible to tell which of the original dimensions they correspond
  // to. If you expect the particular output shape (you usually do), append a
  // reshape to the returned tracker.
  absl::StatusOr<ShapeTracker> Narrow(
      absl::Span<const int64_t> dims_to_keep) const;

  // Returns the set of final (output) dimensions that cover the same elements
  // as the specified `input_dims`. Returns std::nullopt if the mapping is not
  // possible (e.g. some output dimensions span elements both within and outside
  // the specified input dimensions). Degenerate dimensions (size 1) are
  // ignored in the input and skipped in the output. The returned dimensions are
  // sorted by index.
  std::optional<std::vector<int64_t>> MapInputDimensionsToOutputUnordered(
      absl::Span<const int64_t> input_dims) const;

  // Returns true if the specified input dimensions map to a single contiguous
  // stride in the output shape. If `allow_swaps` is false (default), the
  // dimensions must not be swapped with each other. In either case, there must
  // be no other non-degenerate dimensions interleaved between them in the
  // output layout.
  bool MapsToOneStride(absl::Span<const int64_t> input_dims,
                       bool allow_swaps = false) const;

  // Zips multiple ShapeTrackers into a single one.
  // For example, suppose we have two trackers:
  // [10,42] -> (reshape)[5,4,7,3] -> (transpose)[3,4,5,7]
  // [6,2] -> (transpose)[2,6] -> (reshape)[12]
  // Then the zipped tracker will be:
  // [10,42,6,2] -> (reshape)[5,4,7,3,6,2] -> (transpose)[3,4,5,7,2,6] ->
  // (reshape)[3,4,5,7,12]
  static absl::StatusOr<ShapeTracker> Zip(
      absl::Span<const ShapeTracker> trackers);

  // Generates HLO instructions corresponding to the tracked operations.
  absl::StatusOr<HloInstruction*> ToInstructionChain(
      HloInstruction* inst, bool avoid_combining_reshapes = true) const;

  struct Step {
    enum class Type { kReshape, kTranspose, kBroadcast, kReduce };
    Type type;
    // For kReshape: target shape dimensions.
    // For kTranspose: permutation of dimensions.
    // For kBroadcast: target shape dimensions.
    // For kReduce: sorted dimension indices to collapse.
    std::vector<int64_t> dimensions;
    // For kBroadcast: maps each operand dimension to its index in `dimensions`
    // (matching HloBroadcastInstruction::dimensions()). Empty for other steps.
    std::vector<int64_t> broadcast_dimensions = {};
  };

  // Returns the sequence of simplified steps represented by this tracker.
  std::vector<Step> GetSteps(bool avoid_combining_reshapes = true) const;

  // Returns a debug string representation of the tracked operations.
  std::string DebugString(bool avoid_combining_reshapes = true) const;

  class BufferView;

 private:
  struct SlicePropagationResult;

  absl::StatusOr<std::vector<BufferView>> UnflattenOutputDimensions();
  static void TryFoldProjections(std::vector<BufferView>& projections);
  static absl::StatusOr<SlicePropagationResult> SliceProjectionChain(
      absl::Span<const BufferView> projections, const BufferView& slice);
  static std::vector<Step> OptimizeSteps(const std::vector<Step>& steps,
                                         const xla::Shape& input_shape,
                                         const xla::Shape& output_shape);

  std::vector<BufferView> projections_;
  xla::Shape input_shape_;
  xla::Shape output_shape_;
};

// Represents a view of a tensor's buffer.
// Always has at least one segment when pointing to some data (e.g. scalar).
// Has no segments when pointing to no data (e.g. empty slice).
class ShapeTracker::BufferView {
 public:
  BufferView(const BufferView&) = default;
  BufferView(BufferView&&) noexcept = default;
  BufferView& operator=(const BufferView&) = default;
  BufferView& operator=(BufferView&&) noexcept = default;

  // Creates a view representing a shape, with each dimension as a separate
  // segment.
  static BufferView FromShape(const xla::Shape& shape);
  // Creates a view representing a subset of dimensions of a shape.
  // Assumes indices are valid and unique. The resulting view is not packed
  // (i.e. retains the original strides and extents).
  static BufferView FromShapeAndIndices(const xla::Shape& shape,
                                        absl::Span<const int64_t> indices);
  // Creates a view representing a shape as a single contiguous segment.
  static BufferView FromShapeCompacted(const xla::Shape& shape);
  // Flattens a set of sub-views.
  static BufferView FromSubviews(absl::Span<const BufferView> sub_views);
  // Creates a view from explicit strides and extents, validating that they do
  // not overlap.
  static absl::StatusOr<BufferView> FromStridesAndExtents(
      absl::Span<const int64_t> strides, absl::Span<const int64_t> extents);

  struct Transformation {
    llvm::SmallVector<int64_t, 6> input_reshape;
    llvm::SmallVector<int64_t, 6> reduced_dims;
    llvm::SmallVector<int64_t, 6> transpose;
    llvm::SmallVector<int64_t, 6> broadcast_dims;
    llvm::SmallVector<int64_t, 6> output_dims;

    bool IsTrivial() const;
  };
  Transformation AsTransformation() const;

  // Combines contiguous adjacent strides/extents in decreasing-stride order.
  void MergeAdjacentDimensions();

  // Removes degenerate dimensions (extent == 1).
  // If the view becomes empty but was not empty before, it keeps a single
  // degenerate dimension (stride 1, extent 1) to represent a 1-element view.
  void RemoveDegenerateDimensions();

  // Removes "gaps" between segments.
  void Pack();

  // Sorts strides and corresponding extents in descending order of strides.
  void SortByStrideDescending();

  // Attempts to partition the flat view into logical dimensions. Returns
  // nullopt if layout is incompatible. A single logical dimension is allowed to
  // span multiple non-contiguous segments.
  std::optional<std::vector<BufferView>> TryUnflatten(
      absl::Span<const int64_t> logical_dims) const;

  // Outputs a new BufferView representing the intersection of this view with
  // the given stride and extent. Returns nullopt if the intersection is not
  // possible (e.g. dimensions are not divisible). Returns empty view if
  // intersection is empty.
  std::optional<BufferView> TryIntersectWith(int64_t stride,
                                             int64_t extent) const;

  // Outputs a new BufferView representing the intersection of this view with
  // the other view. Returns nullopt if the intersection is not possible.
  // Returns empty view if intersection is empty. The order of the dimensions
  // is taken from @this.
  std::optional<BufferView> TryIntersectWith(const BufferView& other) const;

  bool IsEmpty() const { return strides_.empty(); }
  int64_t ElementsIn() const;
  int64_t InputElementsIn() const;
  bool IsTrivialOneElement() const {
    return strides_.size() == 1 && strides_[0] == 1 && extents_[0] == 1 &&
           collapsed_strides_.empty();
  }
  bool operator==(const BufferView& other) const {
    return strides_ == other.strides_ && extents_ == other.extents_ &&
           collapsed_strides_ == other.collapsed_strides_ &&
           collapsed_extents_ == other.collapsed_extents_;
  }
  bool operator!=(const BufferView& other) const { return !(*this == other); }
  absl::Span<const int64_t> strides() const { return strides_; }
  absl::Span<const int64_t> extents() const { return extents_; }
  absl::Span<const int64_t> collapsed_strides() const {
    return collapsed_strides_;
  }
  absl::Span<const int64_t> collapsed_extents() const {
    return collapsed_extents_;
  }

 private:
  friend class ShapeTracker;
  BufferView() = default;

  static BufferView FromBroadcastedExtent(int64_t extent);
  void SetSubviews(absl::Span<const BufferView> sub_views);
  void AppendCollapsedSegments(const BufferView& reduced_view);

  llvm::SmallVector<int64_t, 6> strides_;
  llvm::SmallVector<int64_t, 6> extents_;
  llvm::SmallVector<int64_t, 6> collapsed_strides_;
  llvm::SmallVector<int64_t, 6> collapsed_extents_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_SHAPE_TRACKER_H_
