/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHAPE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHAPE_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

namespace xla {
namespace ifrt {

// Shape of an array. Only supports static shapes at the moment. Every dimension
// size must be equal to or greater than 0.
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

  absl::Span<const int64_t> dims() const { return dims_; }

  bool operator==(const Shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

  // Total number of elements in this shape.
  int64_t num_elements() const;

  std::string DebugString() const;

 private:
  Dimensions dims_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_SHAPE_H_
