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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_INCLUDE_SHLO_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_INCLUDE_SHLO_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace stablehlo {

enum class ElementType { kUnknown, kI1, kSI8, kSI16, kSI32, kBF16, kF16, kF32 };

using DimensionSize = size_t;
using Axes = std::vector<size_t>;
using Dims = absl::Span<const DimensionSize>;

/*
  A tensor shape represents non-negative dimension sizes in the ascending order
  of the corresponding dimensions (which are also called axes) numbered from 0
  to R-1. The number of dimensions R is called rank. For example,
  tensor<2x3xf32> is a tensor type with shape 2x3 and element type f32. It has
  two dimensions (or, in other words, two axes) - 0th dimension and 1st
  dimension - whose sizes are 2 and 3. Its rank is 2.
*/
class Shape {
 public:
  // Scalar shape, with rank 0
  Shape() = default;
  // Tensor shape, with rank > 0
  explicit Shape(std::vector<DimensionSize>&& dims) : dims_(std::move(dims)) {}
  Shape(std::initializer_list<DimensionSize>&& dims) : dims_(std::move(dims)) {}
  bool operator==(const Shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return !(*this == other); }
  size_t rank() const { return dims_.size(); }
  DimensionSize dim(size_t idx) const { return dims_[idx]; }
  Axes axes() const;
  Dims dims() const { return dims_; }
  size_t num_elements() const;

 private:
  std::vector<DimensionSize> dims_;
};

class TensorType {
 public:
  TensorType(Shape&& shape, ElementType tensor_element_type)
      : shape_(std::move(shape)), tensor_element_type_(tensor_element_type) {}
  bool operator==(const TensorType& other) const {
    return shape_ == other.shape_ and
           tensor_element_type_ == other.tensor_element_type_;
  }
  bool operator!=(const TensorType& other) const { return !(*this == other); }
  const Shape& shape() const { return shape_; }
  ElementType element_type() const { return tensor_element_type_; }
  size_t num_bytes() const;

 private:
  Shape shape_;
  ElementType tensor_element_type_;
};

// Tensor layout has is not part of the StableHLO standard. At this time we
// support only strides for the minor dimension.
class Layout {
 public:
  explicit Layout(size_t minor_dim_stride = 1)
      : minor_dim_stride_(minor_dim_stride) {}
  bool has_strides() const { return minor_dim_stride_ > 1; }
  size_t minor_dim_stride() const { return minor_dim_stride_; }

 private:
  size_t minor_dim_stride_ = 1;
};

class Tensor {
 public:
  Tensor(TensorType&& type, void* buffer, Layout&& layout = Layout())
      : type_(std::move(type)), buffer_(buffer), layout_(std::move(layout)) {}
  bool operator==(const Tensor& other) const;
  bool operator!=(const Tensor& other) const { return !(*this == other); }
  const TensorType& type() const { return type_; }
  void* buffer() { return buffer_; }
  const void* buffer() const { return buffer_; }
  const Layout& layout() const { return layout_; }
  const Shape& shape() const { return type_.shape(); }
  ElementType element_type() const { return type_.element_type(); }
  TensorType baseline_type() const { return type_; }
  ElementType baseline_element_type() const {
    return baseline_type().element_type();
  }
  size_t rank() const { return shape().rank(); }
  DimensionSize dim(size_t idx) const { return shape().dim(idx); }
  Axes axes() const { return shape().axes(); }
  Dims dims() const { return shape().dims(); }
  size_t num_elements() const { return shape().num_elements(); }
  size_t num_bytes() const { return type_.num_bytes(); }
  bool is_per_tensor_quantized() const { return false; }
  bool is_per_axis_quantized() const { return false; }

 private:
  TensorType type_;
  void* buffer_;
  Layout layout_;
};

struct QuantizedParameter {
  float scale;
  int32_t zero_point;
  bool operator==(const QuantizedParameter& other) const {
    return scale == other.scale and zero_point == other.zero_point;
  }
  bool operator!=(const QuantizedParameter& other) const {
    return !(*this == other);
  }
};

class QuantizedTensorElementType {
 public:
  // Constructor for per-tensor quantization.
  QuantizedTensorElementType(ElementType storage_type,
                             ElementType expressed_type,
                             QuantizedParameter&& quantized_parameter,
                             std::optional<int32_t> storage_min = std::nullopt,
                             std::optional<int32_t> storage_max = std::nullopt)
      : storage_type_(storage_type), expressed_type_(expressed_type) {
    parameters_.emplace_back(std::move(quantized_parameter));
  }

  // Constructor for per-axis quantization.
  QuantizedTensorElementType(ElementType storage_type,
                             ElementType expressed_type,
                             std::vector<QuantizedParameter>&& parameters,
                             std::optional<int32_t> storage_min = std::nullopt,
                             std::optional<int32_t> storage_max = std::nullopt)
      : storage_type_(storage_type),
        expressed_type_(expressed_type),
        parameters_(std::move(parameters)) {}

  ElementType storage_type() const { return storage_type_; }
  ElementType expressed_type() const { return expressed_type_; }
  std::optional<int32_t> storage_min() const { return storage_min_; }
  std::optional<int32_t> storage_max() const { return storage_max_; }
  std::optional<DimensionSize> quantized_dimension() const {
    return quantized_dimension_;
  }
  const QuantizedParameter& parameters(size_t index) const {
    return parameters_[index];
  }
  size_t num_parameters() const { return parameters_.size(); }
  bool operator==(const QuantizedTensorElementType& other) const {
    return storage_type_ == other.storage_type_ and
           expressed_type_ == other.expressed_type_ and
           storage_min_ == other.storage_min_ and
           storage_max_ == other.storage_max_ and
           quantized_dimension_ == other.quantized_dimension_ and
           parameters_ == other.parameters_;
  }
  bool operator!=(const QuantizedTensorElementType& other) const {
    return !(*this == other);
  }
  bool is_per_tensor_quantized() const { return !quantized_dimension_; }
  bool is_per_axis_quantized() const { return !is_per_tensor_quantized(); }

 private:
  ElementType storage_type_;
  ElementType expressed_type_;
  std::optional<int32_t> storage_min_;
  std::optional<int32_t> storage_max_;
  std::optional<DimensionSize> quantized_dimension_;
  std::vector<QuantizedParameter> parameters_;
};

class QuantizedTensorType {
 public:
  QuantizedTensorType(
      Shape&& shape, QuantizedTensorElementType&& quantized_tensor_element_type)
      : shape_(std::move(shape)),
        quantized_tensor_element_type_(
            std::move(quantized_tensor_element_type)) {}
  const Shape& shape() const { return shape_; }
  const QuantizedTensorElementType& element_type() const {
    return quantized_tensor_element_type_;
  }
  bool operator==(const QuantizedTensorType& other) const {
    return shape_ == other.shape_ and quantized_tensor_element_type_ ==
                                          other.quantized_tensor_element_type_;
  }
  bool operator!=(const QuantizedTensorType& other) const {
    return !(*this == other);
  }
  size_t num_bytes() const;

 private:
  Shape shape_;
  QuantizedTensorElementType quantized_tensor_element_type_;
};

class QuantizedTensor {
 public:
  QuantizedTensor(QuantizedTensorType&& type, void* buffer,
                  Layout&& layout = Layout())
      : type_(std::move(type)), buffer_(buffer), layout_(std::move(layout)) {}
  bool operator==(const QuantizedTensor& other) const;
  bool operator!=(const QuantizedTensor& other) const {
    return !(*this == other);
  }
  const QuantizedTensorType& type() const { return type_; }
  void* buffer() { return buffer_; }
  const void* buffer() const { return buffer_; }
  const Layout& layout() const { return layout_; }
  const Shape& shape() const { return type_.shape(); }
  ElementType storage_type() const { return element_type().storage_type(); }
  ElementType expressed_type() const { return element_type().expressed_type(); }
  const QuantizedTensorElementType& element_type() const {
    return type_.element_type();
  }
  QuantizedTensorType baseline_type() const;
  QuantizedTensorElementType baseline_element_type() const {
    return baseline_type().element_type();
  }
  size_t rank() const { return shape().rank(); }
  DimensionSize dim(size_t idx) const { return shape().dim(idx); }
  Axes axes() const { return shape().axes(); }
  Dims dims() const { return shape().dims(); }
  size_t num_elements() const { return shape().num_elements(); }
  size_t num_bytes() const { return type_.num_bytes(); }
  bool is_per_tensor_quantized() const {
    return element_type().is_per_tensor_quantized();
  }
  bool is_per_axis_quantized() const {
    return element_type().is_per_axis_quantized();
  }
  auto quantized_dimension() const {
    return element_type().quantized_dimension();
  }
  auto storage_min() const { return element_type().storage_min(); }
  auto storage_max() const { return element_type().storage_max(); }
  auto scales(size_t idx) const { return element_type().parameters(idx).scale; }
  auto zero_points(size_t idx) const {
    return element_type().parameters(idx).zero_point;
  }

 private:
  QuantizedTensorType type_;
  void* buffer_;
  Layout layout_;
};

inline bool IsSignedInteger(ElementType element_type) {
  return element_type == ElementType::kSI8 ||
         element_type == ElementType::kSI16 ||
         element_type == ElementType::kSI32;
}

inline bool IsUnsignedInteger(ElementType element_type) { return false; }

inline bool IsBoolean(ElementType element_type) {
  return element_type == ElementType::kI1;
}

inline bool IsFloat(ElementType element_type) {
  return element_type == ElementType::kBF16 ||
         element_type == ElementType::kF16 || element_type == ElementType::kF32;
}

inline bool IsSignedInteger(const QuantizedTensorElementType& element_type) {
  return IsSignedInteger(element_type.expressed_type());
}
inline bool IsUnsignedInteger(const QuantizedTensorElementType& element_type) {
  return IsUnsignedInteger(element_type.expressed_type());
}
inline bool IsBoolean(const QuantizedTensorElementType& element_type) {
  return IsBoolean(element_type.expressed_type());
}
inline bool IsFloat(const QuantizedTensorElementType& element_type) {
  return IsFloat(element_type.expressed_type());
}

enum class ComparisonDirection { kEQ, kNE, kGE, kGT, kLE, kLT };
enum class CompareType {
  kFloat,
  kTotalOrder /* Unsupported */,
  kSigned,
  kUnsigned
};

// /////////////////////////////////////////////////////////////////////////////

absl::Status Abs(const Tensor& operand, Tensor& result);
absl::Status Abs(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Add(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Add(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                 QuantizedTensor& result);
absl::Status And(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Atan2(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Atan2(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                   QuantizedTensor& result);
absl::Status BroadcastInDim(
    const Tensor& operand, absl::Span<const DimensionSize> broadcast_dimensions,
    Tensor& result);
absl::Status BroadcastInDim(
    const QuantizedTensor& operand,
    absl::Span<const DimensionSize> broadcast_dimensions,
    QuantizedTensor& result);
absl::Status Clamp(const Tensor& min, const Tensor& operand, const Tensor& max,
                   Tensor& result);
absl::Status Clamp(const QuantizedTensor& min, const QuantizedTensor& operand,
                   const QuantizedTensor& max, QuantizedTensor& result);
absl::Status Cbrt(const Tensor& operand, Tensor& result);
absl::Status Cbrt(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Ceil(const Tensor& operand, Tensor& result);
absl::Status Ceil(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Compare(const Tensor& lhs, const Tensor& rhs,
                     ComparisonDirection comparison_direction,
                     CompareType compare_type, Tensor& result);
absl::Status Compare(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     ComparisonDirection comparison_direction,
                     CompareType compare_type, Tensor& result);
absl::Status Concatenate(absl::Span<const Tensor*> inputs,
                         DimensionSize dimension, Tensor& result);
absl::Status Concatenate(absl::Span<const QuantizedTensor*> inputs,
                         DimensionSize dimension, QuantizedTensor& result);
absl::Status Cosine(const Tensor& operand, Tensor& result);
absl::Status Cosine(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status CountLeadingZeros(const Tensor& operand, Tensor& result);
absl::Status Divide(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Divide(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                    QuantizedTensor& result);
absl::Status Exponential(const Tensor& operand, Tensor& result);
absl::Status Exponential(const QuantizedTensor& operand,
                         QuantizedTensor& result);
absl::Status ExponentialMinusOne(const Tensor& operand, Tensor& result);
absl::Status ExponentialMinusOne(const QuantizedTensor& operand,
                                 QuantizedTensor& result);
absl::Status Floor(const Tensor& operand, Tensor& result);
absl::Status Floor(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Iota(DimensionSize iota_dimension, Tensor& result);
absl::Status Iota(DimensionSize iota_dimension, QuantizedTensor& result);
absl::Status IsFinite(const Tensor& operand, Tensor& result);
absl::Status IsFinite(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Log(const Tensor& operand, Tensor& result);
absl::Status Log(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status LogPlusOne(const Tensor& operand, Tensor& result);
absl::Status LogPlusOne(const QuantizedTensor& operand,
                        QuantizedTensor& result);
absl::Status Logistic(const Tensor& operand, Tensor& result);
absl::Status Logistic(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Maximum(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Maximum(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     QuantizedTensor& result);
absl::Status Minimum(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Minimum(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     QuantizedTensor& result);
absl::Status Multiply(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Multiply(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                      QuantizedTensor& result);
absl::Status Negate(const Tensor& operand, Tensor& result);
absl::Status Negate(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Not(const Tensor& operand, Tensor& result);
absl::Status Or(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Popcnt(const Tensor& operand, Tensor& result);
absl::Status Power(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Power(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                   QuantizedTensor& result);
absl::Status Remainder(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Remainder(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                       QuantizedTensor& result);
absl::Status RoundNearestAfz(const Tensor& operand, Tensor& result);
absl::Status RoundNearestAfz(const QuantizedTensor& operand,
                             QuantizedTensor& result);
absl::Status RoundNearestEven(const Tensor& operand, Tensor& result);
absl::Status RoundNearestEven(const QuantizedTensor& operand,
                              QuantizedTensor& result);
absl::Status Rsqrt(const Tensor& operand, Tensor& result);
absl::Status Rsqrt(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Select(const Tensor& pred, const Tensor& on_true,
                    const Tensor& on_false, Tensor& result);
absl::Status Select(const Tensor& pred, const QuantizedTensor& on_true,
                    const QuantizedTensor& on_false, QuantizedTensor& result);
absl::Status ShiftLeft(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status ShiftRightArithmetic(const Tensor& lhs, const Tensor& rhs,
                                  Tensor& result);
absl::Status ShiftRightLogical(const Tensor& lhs, const Tensor& rhs,
                               Tensor& result);
absl::Status Sign(const Tensor& operand, Tensor& result);
absl::Status Sign(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Sine(const Tensor& operand, Tensor& result);
absl::Status Sine(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Subtract(const Tensor& lhs, const Tensor& rhs, Tensor& result);
absl::Status Subtract(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                      QuantizedTensor& result);
absl::Status Sqrt(const Tensor& operand, Tensor& result);
absl::Status Sqrt(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status Tanh(const Tensor& operand, Tensor& result);
absl::Status Tanh(const QuantizedTensor& operand, QuantizedTensor& result);
absl::Status UniformDequantize(const QuantizedTensor& operand, Tensor& result);
absl::Status UniformQuantize(const Tensor& operand, QuantizedTensor& result);
absl::Status Xor(const Tensor& lhs, const Tensor& rhs, Tensor& result);

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_INCLUDE_SHLO_H_
