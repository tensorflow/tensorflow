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

#include "tensorflow/lite/experimental/shlo/shape.h"

#include <cstddef>
#include <functional>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

namespace shlo_ref {

Shape::Shape(absl::Span<const DimensionSize> dims)
    : dims_(dims.begin(), dims.end()) {}

absl::Span<const DimensionSize> Shape::Dimensions() const { return dims_; }

absl::Span<DimensionSize> Shape::MutableDimensions() {
  return absl::MakeSpan(dims_);
}

absl::InlinedVector<Axis, kMaxNumDimensions> Shape::Axes() const {
  absl::InlinedVector<Axis, kMaxNumDimensions> axes(dims_.size());
  absl::c_iota(axes, 0);
  return axes;
}

DimensionSize Shape::Dim(Axis axis) const { return dims_[axis]; }

absl::InlinedVector<DimensionSize, kMaxNumDimensions> Shape::Dims(
    absl::Span<const Axis> axes) const {
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dims;
  for (const auto axis : axes) {
    // Ignore invalid axis
    if (axis < dims_.size()) {
      dims.push_back(Dim(axis));
    }
  }
  return dims;
}

size_t Shape::Rank() const { return dims_.size(); }

DimensionSize Shape::NumElements() const {
  if (dims_.empty()) {
    return 0;
  }
  return absl::c_accumulate(dims_, 1, std::multiplies<>());
}

bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.Dimensions() == rhs.Dimensions();
}

bool operator!=(const Shape& lhs, const Shape& rhs) { return !(lhs == rhs); }

Strides ComputeStrides(const Shape& shape) {
  return ComputeStrides(shape.Dimensions());
}

}  // namespace shlo_ref
