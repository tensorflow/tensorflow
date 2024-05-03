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

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"

namespace shlo_ref {

constexpr bool IsValidQuantizationTypePair(DataType storage_type,
                                           DataType expressed_type) {
  switch (storage_type) {
    case DataType::kSI4:
    case DataType::kSI8:
    case DataType::kSI16:
      break;
    default:
      return false;
  }
  switch (expressed_type) {
    case DataType::kBF16:
    case DataType::kF16:
    case DataType::kF32:
      break;
    default:
      return false;
  }
  return SizeOf(storage_type) < SizeOf(expressed_type);
}

class QuantizedElementTypePerTensor {
 public:
  using ZeroPointVariant =
      std::variant<Storage<DataType::kSI4>::Type, Storage<DataType::kSI8>::Type,
                   Storage<DataType::kSI16>::Type>;
  using ScaleVariant = std::variant<Storage<DataType::kBF16>::Type,
                                    Storage<DataType::kF16>::Type,
                                    Storage<DataType::kF32>::Type>;

  template <class T, class U>
  QuantizedElementTypePerTensor(DataType storage_type, T zero_point,
                                DataType expressed_type, U scale) {
#define SHLO_STORAGE_CASE(TYPE)                                             \
  case DataType ::k##TYPE:                                                  \
    zero_point_ =                                                           \
        static_cast<typename Storage<DataType::k##TYPE>::Type>(zero_point); \
    break;
    switch (storage_type) {
      SHLO_STORAGE_CASE(SI4);
      SHLO_STORAGE_CASE(SI8);
      SHLO_STORAGE_CASE(SI16);
      default:
        ABSL_LOG(FATAL) << "Unsupported quantization storage type ("
                        << ToString(storage_type) << ").";
    }
#undef SHLO_STORAGE_CASE
#define SHLO_EXPRESSED_CASE(TYPE)                                           \
  case DataType ::k##TYPE:                                                  \
    scale_ = static_cast<typename Storage<DataType::k##TYPE>::Type>(scale); \
    break;
    switch (expressed_type) {
      SHLO_EXPRESSED_CASE(BF16);
      SHLO_EXPRESSED_CASE(F16);
      SHLO_EXPRESSED_CASE(F32);
      default:
        ABSL_LOG(FATAL) << "Unsupported quantization expressed type ("
                        << ToString(expressed_type) << ").";
    }
#undef SHLO_EXPRESSED_CASE
    ABSL_CHECK(IsValidQuantizationTypePair(StorageType(), ExpressedType()));
  }

  DataType ExpressedType() const {
    const DataType scale_types[] = {DataType::kBF16, DataType::kF16,
                                    DataType::kF32};
    return scale_types[scale_.index()];
  }

  DataType StorageType() const {
    const DataType zero_point_types[] = {DataType::kSI4, DataType::kSI8,
                                         DataType::kSI16, DataType::kSI32};
    return zero_point_types[zero_point_.index()];
  }

  ScaleVariant& Scale() { return scale_; }

  const ScaleVariant& Scale() const { return scale_; }

  template <DataType expressed_type>
  const typename Storage<expressed_type>::Type& ScaleAs() const {
    return std::get<typename Storage<expressed_type>::Type>(scale_);
  }

  ZeroPointVariant& ZeroPoint() { return zero_point_; }

  const ZeroPointVariant& ZeroPoint() const { return zero_point_; }

  template <DataType storage_type>
  const typename Storage<storage_type>::Type& ZeroPointAs() const {
    return std::get<typename Storage<storage_type>::Type>(zero_point_);
  }

  friend bool operator==(const QuantizedElementTypePerTensor& lhs,
                         const QuantizedElementTypePerTensor& rhs) {
    return lhs.zero_point_ == rhs.zero_point_ && lhs.scale_ == rhs.scale_;
  }

  friend bool operator!=(const QuantizedElementTypePerTensor& lhs,
                         const QuantizedElementTypePerTensor& rhs) {
    return !(lhs == rhs);
  }

 private:
  ZeroPointVariant zero_point_;
  ScaleVariant scale_;
};

class QuantizedElementTypePerAxis {
  template <class To, class FromRange, class... Ts>
  void ConvertAndAssign(std::variant<Ts...>& dest, FromRange&& range) {
    using std::begin;
    using std::end;
    dest = To(begin(range), end(range));
  }

 public:
  template <typename T>
  using SmallInlinedVector = absl::InlinedVector<T, 8>;

  using ScalesVariant =
      std::variant<SmallInlinedVector<Storage<DataType::kBF16>::Type>,
                   SmallInlinedVector<Storage<DataType::kF16>::Type>,
                   SmallInlinedVector<Storage<DataType::kF32>::Type>>;

  // There is no need for kSI4 because it currently uses the same underlying
  // storage type as kSI8, which complicates accessing the variant. If they ever
  // use different underlying types, please add an alternative for kSI4.
  using ZeroPointsVariant =
      std::variant<SmallInlinedVector<Storage<DataType::kSI4>::Type>,
                   SmallInlinedVector<Storage<DataType::kSI8>::Type>,
                   SmallInlinedVector<Storage<DataType::kSI16>::Type>>;

  template <class RangeT = std::initializer_list<int32_t>,
            class RangeU = std::initializer_list<float>>
  QuantizedElementTypePerAxis(DataType storage_type, RangeT&& zero_points,
                              DataType expressed_type, RangeU&& scales,
                              Axis quantized_dimension)
      : quantized_dimension_(quantized_dimension) {
#define SHLO_STORAGE_CASE(TYPE)                                             \
  case DataType ::k##TYPE:                                                  \
    ConvertAndAssign<SmallInlinedVector<Storage<DataType::k##TYPE>::Type>>( \
        zero_points_, static_cast<RangeT&&>(zero_points));                  \
    break;
    switch (storage_type) {
      SHLO_STORAGE_CASE(SI4);
      SHLO_STORAGE_CASE(SI8);
      SHLO_STORAGE_CASE(SI16);
      default:
        ABSL_LOG(FATAL) << "Unsupported quantization storage type ("
                        << ToString(storage_type) << ").";
    }
#undef SHLO_STORAGE_CASE
#define SHLO_EXPRESSED_CASE(TYPE)                                           \
  case DataType ::k##TYPE:                                                  \
    ConvertAndAssign<SmallInlinedVector<Storage<DataType::k##TYPE>::Type>>( \
        scales_, static_cast<RangeU&&>(scales));                            \
    break;
    switch (expressed_type) {
      SHLO_EXPRESSED_CASE(BF16);
      SHLO_EXPRESSED_CASE(F16);
      SHLO_EXPRESSED_CASE(F32);
      default:
        ABSL_LOG(FATAL) << "Unsupported quantization expressed type ("
                        << ToString(expressed_type) << ").";
    }
#undef SHLO_EXPRESSED_CASE
    ABSL_CHECK(IsValidQuantizationTypePair(StorageType(), ExpressedType()));
  }

  DataType ExpressedType() const {
    const DataType scale_types[] = {DataType::kBF16, DataType::kF16,
                                    DataType::kF32};
    return scale_types[scales_.index()];
  }

  DataType StorageType() const {
    const DataType zero_point_types[] = {DataType::kSI4, DataType::kSI8,
                                         DataType::kSI16, DataType::kSI32};
    return zero_point_types[zero_points_.index()];
  }

  Axis& QuantizedDimension() { return quantized_dimension_; }

  const Axis& QuantizedDimension() const { return quantized_dimension_; }

  ScalesVariant& Scales() { return scales_; }

  const ScalesVariant& Scales() const { return scales_; }

  template <DataType expressed_type>
  const SmallInlinedVector<typename Storage<expressed_type>::Type>& ScalesAs()
      const {
    return std::get<SmallInlinedVector<typename Storage<expressed_type>::Type>>(
        scales_);
  }

  ZeroPointsVariant& ZeroPoints() { return zero_points_; }

  const ZeroPointsVariant& ZeroPoints() const { return zero_points_; }

  template <DataType storage_type>
  const SmallInlinedVector<typename Storage<storage_type>::Type>& ZeroPointsAs()
      const {
    return std::get<SmallInlinedVector<typename Storage<storage_type>::Type>>(
        zero_points_);
  }

  friend bool operator==(const QuantizedElementTypePerAxis& lhs,
                         const QuantizedElementTypePerAxis& rhs) {
    return lhs.zero_points_ == rhs.zero_points_ && lhs.scales_ == rhs.scales_;
  }

  friend bool operator!=(const QuantizedElementTypePerAxis& lhs,
                         const QuantizedElementTypePerAxis& rhs) {
    return !(lhs == rhs);
  }

 private:
  Axis quantized_dimension_;
  ScalesVariant scales_;
  ZeroPointsVariant zero_points_;
};

// Gets a string representation of the given element type.
std::string ToString(const QuantizedElementTypePerTensor& t);

// Gets a string representation of the given element type.
std::string ToString(const QuantizedElementTypePerAxis& t);

QuantizedElementTypePerTensor BaselineType(
    const QuantizedElementTypePerTensor& type);

QuantizedElementTypePerAxis BaselineType(
    const QuantizedElementTypePerAxis& type);

}  // namespace shlo_ref
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_QUANTIZED_TENSOR_ELEMENT_TYPE_H_
