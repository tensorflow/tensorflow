/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_SHAPE_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_SHAPE_H_

#include <initializer_list>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tflite {
namespace shim {

// Shape of a tensor. When unset it means the rank is unknown. Individual dims
// can also be unknown.
class Shape {
 public:
  using ValueType = std::vector<int>;

  Shape() = default;
  Shape(const Shape& o) = default;
  Shape(Shape&& o) = default;
  Shape& operator=(const Shape& o) = default;
  Shape& operator=(Shape&& o) = default;

  // Ctors
  Shape(const std::initializer_list<int>& o) : value_(o), has_value_(true) {}
  template <typename... Args>
  explicit Shape(Args&&... args)  // forward ctor args to that of std::vector
      : value_(std::forward<Args>(args)...), has_value_(true) {}
  explicit Shape(const absl::Span<int> value)
      : value_(value.data(), value.data() + value.size()), has_value_(true) {}

  // Accessors
  inline bool has_value() const { return has_value_; }
  inline ValueType& value() { return value_; }
  inline const ValueType& value() const { return value_; }
  ValueType* operator->() { return &value_; }
  const ValueType* operator->() const { return &value_; }
  ValueType& operator*() { return value_; }
  const ValueType& operator*() const { return value_; }
  // Get the specified dimension if known
  int Dim(const int idx) const;

  // Returns the rank of the shape
  int Rank() const { return has_value_ ? value_.size() : kUnknownRank; }

  // Whether all the dimensions of the shape are known
  bool FullyDefined() const;

  // Pretty printer
  std::string ToString() const;

  // Adds two dimension taking into account unknown dims.
  static int AddDims(const int dim1, const int dim2);

  // Comparison

  // Strict equality of the shapes. Unknown dims or rank on one side will
  // result in false
  bool operator==(const Shape& rhs) const;
  bool operator!=(const Shape& rhs) const;

  // Compatibility of the shapes. If there are two known and incompatible
  // dimensions it returns false
  bool Compatible(const Shape& rhs) const;

  // The value for unknown dimensions and rank. There are static_asserts to
  // ensure this matches the one defined in ::tensorflow namespace
  static constexpr int kUnknownDim = -1;
  static constexpr int kUnknownRank = -1;

 private:
  ValueType value_;
  bool has_value_ = false;
};
using ShapeOr = absl::StatusOr<Shape>;

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_SHAPE_H_
