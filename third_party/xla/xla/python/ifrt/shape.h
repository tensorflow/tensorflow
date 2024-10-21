/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SHAPE_H_
#define XLA_PYTHON_IFRT_SHAPE_H_

#include <stdbool.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/shape.pb.h"

namespace xla {
namespace ifrt {

// Shape of an array. Only supports static shapes (dynamic shapes are supported
// through `ifrt::DynamicShape`). Every dimension size must be equal to or
// greater than 0.
class Shape {
 public:
  // Maximum dimensions to inline.
  static constexpr int kInlineDimensionSize = 6;

  using Dimensions = absl::InlinedVector<int64_t, kInlineDimensionSize>;

  explicit Shape(absl::Span<const int64_t> dims)
      : dims_(Dimensions(dims.begin(), dims.end())) {}
  Shape(const Shape&) = default;
  Shape(Shape&&) = default;
  Shape& operator=(const Shape&) = default;
  Shape& operator=(Shape&&) = default;

  // Constructs `Shape` from `ShapeProto`.
  static absl::StatusOr<Shape> FromProto(const ShapeProto& proto);

  // Returns a `ShapeProto` representation.
  ShapeProto ToProto() const;

  absl::Span<const int64_t> dims() const { return dims_; }

  bool operator==(const Shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& shape);

  // Total number of elements in this shape.
  int64_t num_elements() const;

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Shape& shape) {
    sink.Append(shape.DebugString());
  }

 private:
  Dimensions dims_;
};

template <typename H>
H AbslHashValue(H h, const Shape& shape) {
  return H::combine(std::move(h), shape.dims_);
}

// A tag for `Shape` to indicate bounded dynamism. Should be used together with
// `Shape` to represent a bounded dynamic shape where the number of dimensions
// of the shape is fixed, but certain dimensions in the shape have no fixed
// size and only a size upper bound.
class BoundedDynamicShapeTag {
 public:
  // Maximum dimensions to inline.
  static constexpr int kInlineDimensionSize = 6;

  using DynamicDimensions = absl::InlinedVector<bool, kInlineDimensionSize>;

  explicit BoundedDynamicShapeTag(absl::Span<const bool> dynamic_dims)
      : dynamic_dims_(
            DynamicDimensions(dynamic_dims.begin(), dynamic_dims.end())) {
    CHECK(absl::c_any_of(dynamic_dims_, [](bool b) { return b; }))
        << "At least one dimension needs to be dynamically sized.";
  }

  BoundedDynamicShapeTag(const BoundedDynamicShapeTag&) = default;
  BoundedDynamicShapeTag(BoundedDynamicShapeTag&&) = default;
  BoundedDynamicShapeTag& operator=(const BoundedDynamicShapeTag&) = default;
  BoundedDynamicShapeTag& operator=(BoundedDynamicShapeTag&&) = default;

  absl::Span<const bool> DynamicDims() const { return dynamic_dims_; }

  bool operator==(const BoundedDynamicShapeTag& other) const {
    return dynamic_dims_ == other.dynamic_dims_;
  }

  bool operator!=(const BoundedDynamicShapeTag& other) const {
    return !(*this == other);
  }

  // Constructs `BoundedDynamicShapeTag` from `BoundedDynamicShapeTagProto`.
  static absl::StatusOr<BoundedDynamicShapeTag> FromProto(
      const BoundedDynamicShapeTagProto& proto);

  // Returns a `BoundedDynamicShapeTagProto` representation.
  BoundedDynamicShapeTagProto ToProto() const;

 private:
  // This vector is the same size as `Shape`'s 'dims()' and indicates whether
  // the respective dimension is dynamically sized.
  DynamicDimensions dynamic_dims_;
};

// Use static polymorphism to facilitate type checking. Currently only support
// one type of dynamism.
using DynamicShapeTag = std::variant<BoundedDynamicShapeTag>;

// Shape with dynamism in dimension sizes, etc.
class DynamicShape {
 public:
  // Constructs `DynamicShape` from `Shape` and `DynamicShapeTag`. Fails if
  // the dimensions mismatch.
  //
  // When `tag` is a `BoundedDynamicShapeTag`: for any dimension that is dynamic
  // as indicated by `tag`, the corresponding dimension in `shape` represents
  // the upper bound of the dimension size.
  static absl::StatusOr<DynamicShape> Create(Shape shape, DynamicShapeTag tag);

  DynamicShape(const DynamicShape&) = default;
  DynamicShape(DynamicShape&&) = default;
  DynamicShape& operator=(const DynamicShape&) = default;
  DynamicShape& operator=(DynamicShape&&) = default;

  const DynamicShapeTag& GetTag() const { return tag_; }

  bool operator==(const DynamicShape& other) const {
    return tag_ == other.tag_ && shape_ == other.shape_;
  }
  bool operator!=(const DynamicShape& other) const { return !(*this == other); }

  // Gets the shape after padding. Only works for bounded dynamic shape for now.
  absl::StatusOr<Shape> GetPaddedShape() const;

  // Returns whether a certain dimension in the shape is dynamic.
  bool IsDynamicDim(int dimension) const;

  // Constructs `DynamicShape` from `DynamicShapeProto`.
  static absl::StatusOr<DynamicShape> FromProto(const DynamicShapeProto& proto);

  // Returns a `DynamicShapeProto` representation.
  DynamicShapeProto ToProto() const;

  std::string DebugString() const;

 private:
  DynamicShape(Shape shape, DynamicShapeTag tag)
      : shape_(std::move(shape)), tag_(std::move(tag)) {}

  Shape shape_;
  DynamicShapeTag tag_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);
std::ostream& operator<<(std::ostream& os, const DynamicShape& dynamic_shape);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SHAPE_H_
