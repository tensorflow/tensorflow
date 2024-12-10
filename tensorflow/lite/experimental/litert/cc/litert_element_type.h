// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ELEMENT_TYPE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ELEMENT_TYPE_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"

namespace litert {

// Data type of tensor elements. C++ equivalent to LiteRtElementType.
enum class ElementType {
  None = kLiteRtElementTypeNone,
  Bool = kLiteRtElementTypeBool,
  Int4 = kLiteRtElementTypeInt4,
  Int8 = kLiteRtElementTypeInt8,
  Int16 = kLiteRtElementTypeInt16,
  Int32 = kLiteRtElementTypeInt32,
  Int64 = kLiteRtElementTypeInt64,
  UInt8 = kLiteRtElementTypeUInt8,
  UInt16 = kLiteRtElementTypeUInt16,
  UInt32 = kLiteRtElementTypeUInt32,
  UInt64 = kLiteRtElementTypeUInt64,
  Float16 = kLiteRtElementTypeFloat16,
  BFloat16 = kLiteRtElementTypeBFloat16,
  Float32 = kLiteRtElementTypeFloat32,
  Float64 = kLiteRtElementTypeFloat64,
  Complex64 = kLiteRtElementTypeComplex64,
  Complex128 = kLiteRtElementTypeComplex128,
  TfResource = kLiteRtElementTypeTfResource,
  TfString = kLiteRtElementTypeTfString,
  TfVariant = kLiteRtElementTypeTfVariant,
};

// Get number of bytes of a single element of given type.
inline constexpr std::optional<size_t> GetByteWidth(ElementType ty) {
  if (ty == ElementType::Bool)
    return 1;
  else if (ty == ElementType::Int8)
    return 1;
  else if (ty == ElementType::Int16)
    return 2;
  else if (ty == ElementType::Int32)
    return 4;
  else if (ty == ElementType::Int64)
    return 8;
  else if (ty == ElementType::UInt8)
    return 1;
  else if (ty == ElementType::UInt16)
    return 2;
  else if (ty == ElementType::UInt32)
    return 4;
  else if (ty == ElementType::UInt64)
    return 8;
  else if (ty == ElementType::Float16)
    return 2;
  else if (ty == ElementType::BFloat16)
    return 2;
  else if (ty == ElementType::Float32)
    return 4;
  else if (ty == ElementType::Float64)
    return 8;
  else
    return std::nullopt;
}

// Get number of bytes of a single element of given type via template.
template <ElementType Ty>
inline constexpr size_t GetByteWidth() {
  constexpr auto byte_width = GetByteWidth(Ty);
  static_assert(byte_width.has_value(), "Type does not have byte width");
  return byte_width.value();
}

// Get the litert::ElementType associated with given C++ type.
template <typename T>
inline constexpr ElementType GetElementType() {
  static_assert(false, "Uknown C++ type");
  return ElementType::None;
}

template <>
inline constexpr ElementType GetElementType<bool>() {
  return ElementType::Bool;
}

template <>
inline constexpr ElementType GetElementType<int8_t>() {
  return ElementType::Int8;
}

template <>
inline constexpr ElementType GetElementType<uint8_t>() {
  return ElementType::UInt8;
}

template <>
inline constexpr ElementType GetElementType<int16_t>() {
  return ElementType::Int16;
}

template <>
inline constexpr ElementType GetElementType<uint16_t>() {
  return ElementType::UInt16;
}

template <>
inline constexpr ElementType GetElementType<int32_t>() {
  return ElementType::Int32;
}

template <>
inline constexpr ElementType GetElementType<uint32_t>() {
  return ElementType::UInt32;
}

template <>
inline constexpr ElementType GetElementType<int64_t>() {
  return ElementType::Int64;
}

template <>
inline constexpr ElementType GetElementType<uint64_t>() {
  return ElementType::UInt64;
}

template <>
inline constexpr ElementType GetElementType<float>() {
  return ElementType::Float32;
}

template <>
inline constexpr ElementType GetElementType<double>() {
  return ElementType::Float64;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ELEMENT_TYPE_H_
