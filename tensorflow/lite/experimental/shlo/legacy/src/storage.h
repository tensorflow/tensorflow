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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_STORAGE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_STORAGE_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/bf16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/f16.h"

namespace stablehlo {

template <ElementType element_type>
struct Storage;

template <>
struct Storage<ElementType::kI1> {
  using Type = uint8_t;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kSI8> {
  using Type = int8_t;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kSI16> {
  using Type = int16_t;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kSI32> {
  using Type = int32_t;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kBF16> {
  using Type = BF16;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kF16> {
  using Type = F16;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

template <>
struct Storage<ElementType::kF32> {
  using Type = float;
  static Type Get(const void* buffer, size_t idx) {
    auto p = static_cast<const Type*>(buffer);
    return p[idx];
  }
  static void Set(void* buffer, size_t idx, Type value) {
    auto p = static_cast<Type*>(buffer);
    p[idx] = value;
  }
};

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_STORAGE_H_
