/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_SHAPE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_SHAPE_H_

#include <cstddef>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

namespace shlo_ref {

// The SHLO Spec states that dimensions are non-negative. We diverge from the
// spec here to use negative values to represent dynamic dimensions.
using DimensionSize = int64_t;
using Axis = size_t;

inline constexpr DimensionSize kDynamicDimension = -1;
inline constexpr Axis kMaxNumDimensions = 6;

using Strides = absl::InlinedVector<DimensionSize, kMaxNumDimensions>;

class Shape {
 public:
  Shape() = default;
  ~Shape() = default;
  Shape(const Shape&) = default;
  Shape& operator=(const Shape&) = default;
  Shape(Shape&&) = default;
  Shape& operator=(Shape&&) = default;

  explicit Shape(absl::Span<const DimensionSize> dims);

  absl::Span<const DimensionSize> Dimensions() const;
  absl::Span<DimensionSize> MutableDimensions();

  // range(rank(x))
  absl::InlinedVector<Axis, kMaxNumDimensions> Axes() const;

  // shape(x)[axis]
  DimensionSize Dim(Axis axis) const;

  // list(map(lambda axis: dim(x, axis), axes))
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> Dims(
      absl::Span<const Axis> axes) const;

  // size(shape(x))
  size_t Rank() const;

  // reduce(lambda x, y: x * y, shape(x))
  // Note: in the SHLO spec, this is called size. We've diverged for readability
  // and possible confusion with C++ container's usage of size().
  DimensionSize NumElements() const;

  // The following members are provided for compatibility with the standard
  // library.
  using value_type = DimensionSize;

  const value_type& operator[](int dim) const { return dims_[dim]; }
  value_type& operator[](int dim) { return dims_[dim]; }

  auto cbegin() const { return dims_.begin(); }
  auto begin() const { return dims_.begin(); }
  auto begin() { return dims_.begin(); }
  auto cend() const { return dims_.end(); }
  auto end() const { return dims_.end(); }
  auto end() { return dims_.end(); }
  bool empty() const { return dims_.empty(); }
  size_t size() const { return dims_.size(); }
  const value_type* data() const { return dims_.data(); }
  value_type* data() { return dims_.data(); }

 private:
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dims_;
};

bool operator==(const Shape& lhs, const Shape& rhs);
bool operator!=(const Shape& lhs, const Shape& rhs);

Strides ComputeStrides(const Shape& shape);

template <class T>
Strides ComputeStrides(const absl::Span<const T> shape) {
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> strides(shape.size());
  if (!shape.empty()) {
    strides[shape.size() - 1] = 1;
    if (shape.size() > 1) {
      for (size_t i = shape.size() - 1; i != 0; --i) {
        strides[i - 1] = shape[i] * strides[i];
      }
    }
  }
  return strides;
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_SHAPE_H_
