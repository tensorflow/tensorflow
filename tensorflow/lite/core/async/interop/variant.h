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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_INTEROP_VARIANT_H_
#define TENSORFLOW_LITE_CORE_ASYNC_INTEROP_VARIANT_H_

#include <cstddef>
#include <string>
#include <utility>

namespace tflite {
namespace interop {

// Tagged union implementation for variant type.
// Getters and Setters have compile time check to ensure the type is supported
// in the variant. But this class won't perform runtime type check. Callers
// are required to ensure type used in getters are the same as setters.
// For pointer type values hold in the variant (including C-style string literal
// const char*), Variant does not hold the ownership of the value.
struct Variant {
  Variant();

  template <typename T>
  explicit Variant(T v) {
    Set(v);
  }

  template <typename T>
  Variant& operator=(T v) {
    Set(v);
    return *this;
  }

  // Getter. Disabled if the type is not supported in the variant.
  // Returns nullptr if requested type doesn't match the actual value type.
  template <typename T>
  const T* Get() const = delete;

  // Setter. Disabled if the type is not supported in the variant.
  template <typename T>
  void Set(T v) = delete;

  // Returns the opaque data pointer.
  // Callers are responsible for ensuring to cast to correct type.
  void const* GetPtr() const { return &val; }

  // Comparator.
  // If the underlying data is string type (const char*), performs a string
  // comparison. Otherwise checks equality of the data.
  bool operator==(const Variant& other) const;
  bool operator!=(const Variant& other) const;

  // Data types supported in the variant.
  union {
    int i;
    size_t s;
    const char* c;
  } val;

  // Tracking bit used for equality comparison.
  enum { kInvalid, kInt, kSizeT, kString } type;
};

// Copyable.
template <>
inline Variant::Variant(const Variant& v) : val(v.val), type(v.type) {}

// Copy assign with copy-and-swap.
template <>
inline Variant& Variant::operator=(Variant v) {
  std::swap(val, v.val);
  std::swap(type, v.type);
  return *this;
}

// Accessor specializations.
template <>
inline const int* Variant::Get<int>() const {
  if (type != kInt) return nullptr;
  return &val.i;
}
template <>
inline const size_t* Variant::Get<size_t>() const {
  if (type != kSizeT) return nullptr;
  return &val.s;
}
template <>
inline const char* const* Variant::Get<const char*>() const {
  if (type != kString) return nullptr;
  return &val.c;
}
template <>
inline void Variant::Set(int v) {
  val.i = v;
  type = kInt;
}
template <>
inline void Variant::Set(size_t v) {
  val.s = v;
  type = kSizeT;
}
template <>
inline void Variant::Set(const char* v) {
  val.c = v;
  type = kString;
}

}  // namespace interop
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_INTEROP_VARIANT_H_
