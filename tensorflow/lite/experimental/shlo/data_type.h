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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DATA_TYPE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DATA_TYPE_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/i4.h"

namespace shlo_ref {

// For more information on StableHLO types, see the spec., search for "Element
// types". The SHLO Device Profile does not include unsigned or 64 bit types.
enum class DataType {
  kI1,
  kSI4,
  kSI8,
  kSI16,
  kSI32,
  kSI64,
  kBF16,
  kF16,
  kF32,
};

template <class T>
struct DefaultStorageDescription {
  using Type = T;
  static constexpr Type kMinValue = std::numeric_limits<Type>::lowest();
  static constexpr Type kMaxValue = std::numeric_limits<Type>::max();
};

// Storage provides the corresponding C++ type for the given DataType.
template <DataType data_type>
struct Storage {};

template <>
struct Storage<DataType::kI1> : DefaultStorageDescription<bool> {};

template <>
struct Storage<DataType::kSI4> : DefaultStorageDescription<I4> {};

template <>
struct Storage<DataType::kSI8> : DefaultStorageDescription<int8_t> {};

template <>
struct Storage<DataType::kSI16> : DefaultStorageDescription<int16_t> {};

template <>
struct Storage<DataType::kSI32> : DefaultStorageDescription<int32_t> {};

template <>
struct Storage<DataType::kSI64> : DefaultStorageDescription<int64_t> {};

template <>
struct Storage<DataType::kBF16> : DefaultStorageDescription<BF16> {};

template <>
struct Storage<DataType::kF16> : DefaultStorageDescription<F16> {};

template <>
struct Storage<DataType::kF32> : DefaultStorageDescription<float> {};

template <DataType data_type>
using StorageType = typename Storage<data_type>::Type;

constexpr bool IsBool(DataType data_type) { return data_type == DataType::kI1; }

constexpr bool IsSignedInteger(DataType data_type) {
  return data_type == DataType::kSI4 || data_type == DataType::kSI8 ||
         data_type == DataType::kSI16 || data_type == DataType::kSI32 ||
         data_type == DataType::kSI64;
}

constexpr bool IsUnsignedInteger(DataType data_type) { return false; }

constexpr bool IsInteger(DataType data_type) {
  return IsSignedInteger(data_type) || IsUnsignedInteger(data_type);
}

constexpr bool IsFloat(DataType data_type) {
  return data_type == DataType::kBF16 || data_type == DataType::kF16 ||
         data_type == DataType::kF32;
}

template <DataType data_type>
constexpr int64_t SizeOf() {
  return sizeof(StorageType<data_type>);
}

constexpr int64_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::kI1:
      return SizeOf<DataType::kI1>();
    case DataType::kSI4:
      return SizeOf<DataType::kSI4>();
    case DataType::kSI8:
      return SizeOf<DataType::kSI8>();
    case DataType::kSI16:
      return SizeOf<DataType::kSI16>();
    case DataType::kSI32:
      return SizeOf<DataType::kSI32>();
    case DataType::kSI64:
      return SizeOf<DataType::kSI64>();
    case DataType::kBF16:
      return SizeOf<DataType::kBF16>();
    case DataType::kF16:
      return SizeOf<DataType::kF16>();
    case DataType::kF32:
      return SizeOf<DataType::kF32>();
  }
}

// Gets a string representation of the given DataType.
constexpr const char* ToString(DataType t) {
  switch (t) {
    case DataType::kI1:
      return "I1";
      break;
    case DataType::kSI4:
      return "SI4";
      break;
    case DataType::kSI8:
      return "SI8";
      break;
    case DataType::kSI16:
      return "SI16";
      break;
    case DataType::kSI32:
      return "SI32";
      break;
    case DataType::kSI64:
      return "SI64";
      break;
    case DataType::kBF16:
      return "BF16";
      break;
    case DataType::kF16:
      return "F16";
      break;
    case DataType::kF32:
      return "F32";
      break;
  }
  return "Unknown data type";
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DATA_TYPE_H_
