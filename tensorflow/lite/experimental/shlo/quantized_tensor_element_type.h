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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZED_TENSOR_ELEMENT_TYPE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZED_TENSOR_ELEMENT_TYPE_H_

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"

namespace shlo_ref {

class QuantizedTensorElementType {
 public:
  template <DataType storage_type, DataType expressed_type>
  static QuantizedTensorElementType PerTensor(
      StorageType<expressed_type> scale, StorageType<storage_type> zero_point) {
    static_assert(IsInteger(storage_type),
                  "Storage type must be an integer type");
    static_assert(IsFloat(expressed_type),
                  "Expressed type must be a floating point type");
    using StorageT = typename Storage<storage_type>::Type;
    using ExpressedT = typename Storage<expressed_type>::Type;

    return QuantizedTensorElementType(
        storage_type, expressed_type, std::nullopt,
        SmallInlinedVector<ExpressedT>({scale}),
        SmallInlinedVector<StorageT>({zero_point}));
  }

  template <DataType storage_type, DataType expressed_type>
  static QuantizedTensorElementType PerAxis(
      absl::Span<const StorageType<expressed_type>> scales,
      absl::Span<const StorageType<storage_type>> zero_points,
      Axis quantized_dimension) {
    static_assert(IsInteger(storage_type),
                  "Storage type must be an integer type");
    static_assert(IsFloat(expressed_type),
                  "Expressed type must be a floating point type");
    using StorageT = typename Storage<storage_type>::Type;
    using ExpressedT = typename Storage<expressed_type>::Type;

    ABSL_CHECK(scales.size() == zero_points.size());
    return QuantizedTensorElementType(
        storage_type, expressed_type, quantized_dimension,
        SmallInlinedVector<ExpressedT>(scales.begin(), scales.end()),
        SmallInlinedVector<StorageT>(zero_points.begin(), zero_points.end()));
  }

  DataType StorageType() const { return storage_type_; }
  DataType ExpressedType() const { return expressed_type_; }

  bool IsPerTensorQuantized() const { return !quantized_dimension_; }
  bool IsPerAxisQuantized() const { return !IsPerTensorQuantized(); }

  Axis QuantizedDimension() const {
    ABSL_CHECK(IsPerAxisQuantized());
    return quantized_dimension_.value();
  }

  template <DataType expressed_type,
            typename T = typename Storage<expressed_type>::Type>
  absl::Span<const T> Scales() const {
    ABSL_CHECK(expressed_type == expressed_type_);
    ABSL_CHECK(std::holds_alternative<SmallInlinedVector<T>>(scales_));
    return std::get<SmallInlinedVector<T>>(scales_);
  }

  template <DataType storage_type,
            typename T = typename Storage<storage_type>::Type>
  absl::Span<const T> ZeroPoints() const {
    ABSL_CHECK(storage_type == storage_type_);
    ABSL_CHECK(std::holds_alternative<SmallInlinedVector<T>>(zero_points_));
    return std::get<SmallInlinedVector<T>>(zero_points_);
  }

  friend bool operator==(const QuantizedTensorElementType& lhs,
                         const QuantizedTensorElementType& rhs) {
    return lhs.storage_type_ == rhs.storage_type_ &&
           lhs.expressed_type_ == rhs.expressed_type_ &&
           lhs.quantized_dimension_ == rhs.quantized_dimension_ &&
           lhs.scales_ == rhs.scales_ && lhs.zero_points_ == rhs.zero_points_;
  }

  friend bool operator!=(const QuantizedTensorElementType& lhs,
                         const QuantizedTensorElementType& rhs) {
    return !(lhs == rhs);
  }

  friend QuantizedTensorElementType BaselineType(
      const QuantizedTensorElementType& type) {
    QuantizedTensorElementType baseline = type;
    std::visit(
        [](auto& scales) -> void {
          using Container = std::remove_reference_t<decltype(scales)>;
          absl::c_fill(scales, static_cast<typename Container::value_type>(1));
        },
        baseline.scales_);
    std::visit(
        [](auto& zero_points) -> void {
          using Container = std::remove_reference_t<decltype(zero_points)>;
          absl::c_fill(zero_points,
                       static_cast<typename Container::value_type>(0));
        },
        baseline.zero_points_);
    return baseline;
  }

 private:
  // Most quantized tensors will likely be per tensor quantized, which will have
  // a single element in the vector. Use an InlinedVector with a single element
  // so we only allocate when using per axis quantization.
  template <typename T>
  using SmallInlinedVector = absl::InlinedVector<T, 1>;

  template <typename StorageT, typename ExpressedT>
  QuantizedTensorElementType(DataType storage_type, DataType expressed_type,
                             std::optional<Axis> quantized_dimension,
                             SmallInlinedVector<ExpressedT> scales,
                             SmallInlinedVector<StorageT> zero_points)
      : storage_type_(storage_type),
        expressed_type_(expressed_type),
        quantized_dimension_(quantized_dimension),
        scales_(std::move(scales)),
        zero_points_(std::move(zero_points)) {}

  DataType storage_type_;
  DataType expressed_type_;

  std::optional<Axis> quantized_dimension_;

  std::variant<SmallInlinedVector<Storage<DataType::kBF16>::Type>,
               SmallInlinedVector<Storage<DataType::kF16>::Type>,
               SmallInlinedVector<Storage<DataType::kF32>::Type>>
      scales_;

  // There is no need for kSI4 because it currently uses the same underlying
  // storage type as kSI8, which complicates accessing the variant. If they ever
  // use different underlying types, please add an alternative for kSI4.
  std::variant<SmallInlinedVector<Storage<DataType::kSI8>::Type>,
               SmallInlinedVector<Storage<DataType::kSI16>::Type>,
               SmallInlinedVector<Storage<DataType::kSI32>::Type>>
      zero_points_;
};

}  // namespace shlo_ref
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZED_TENSOR_ELEMENT_TYPE_H_
