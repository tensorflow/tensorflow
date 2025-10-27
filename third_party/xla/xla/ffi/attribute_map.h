/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_FFI_ATTRIBUTE_MAP_H_
#define XLA_FFI_ATTRIBUTE_MAP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace xla::ffi {
namespace internal {
// A little bit of template metaprogramming to append type to std::variant.
template <typename V, class T>
struct AppendType;

template <typename... Ts, class T>
struct AppendType<std::variant<Ts...>, T> {
  using Type = std::variant<Ts..., T>;
};
}  // namespace internal

// A single scalar value.
using Scalar = std::variant<bool, int8_t, int16_t, int32_t, int64_t, uint8_t,
                            uint16_t, uint32_t, uint64_t, float, double>;

// An array of elements of the same Scalar type.
using Array = std::variant<std::vector<int8_t>, std::vector<int16_t>,
                           std::vector<int32_t>, std::vector<int64_t>,
                           std::vector<uint8_t>, std::vector<uint16_t>,
                           std::vector<uint32_t>, std::vector<uint64_t>,
                           std::vector<float>, std::vector<double>>;

// Attributes that do not support nested dictionaries.
using FlatAttribute = std::variant<Scalar, Array, std::string>;

// A map that maps from an arbitrary name (string key) to a flat attribute.
using FlatAttributesMap = absl::flat_hash_map<std::string, FlatAttribute>;

// Forward declaration of the recursive type.
struct AttributesDictionary;

// Attributes that support arbitrary nesting.
using Attribute =
    internal::AppendType<FlatAttribute, AttributesDictionary>::Type;

// AttributesMap is a map from an arbitrary name (string key) to an attribute.
using AttributesMap = absl::flat_hash_map<std::string, Attribute>;

// Dictionary is just a wrapper around `AttributesMap`. We need an indirection
// through `std::shared_ptr` to be able to define recursive `std::variant`. We
// use shared pointer to keep `AttributesMap` copyable.
struct AttributesDictionary {
  std::shared_ptr<AttributesMap> attrs;
};

// Converts MLIR dictionary attribute attached to a custom call operation to a
// custom call handler attributes that are forwarded to the FFI handler.
absl::StatusOr<AttributesMap> BuildAttributesMap(mlir::DictionaryAttr dict);

}  // namespace xla::ffi

#endif  // XLA_FFI_ATTRIBUTE_MAP_H_
