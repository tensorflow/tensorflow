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
#include "xla/ffi/attribute_map.pb.h"

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

using ScalarBase =
    std::variant<bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                 uint32_t, uint64_t, float, double>;

// A single scalar value.
class Scalar : public ScalarBase {
 public:
  using ScalarBase::ScalarBase;

  ScalarProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const ScalarBase& AsVariant() const { return *this; }
  ScalarBase& AsVariant() { return *this; }

  static absl::StatusOr<Scalar> FromProto(const ScalarProto& proto);
};

using ArrayBase = std::variant<std::vector<int8_t>, std::vector<int16_t>,
                               std::vector<int32_t>, std::vector<int64_t>,
                               std::vector<uint8_t>, std::vector<uint16_t>,
                               std::vector<uint32_t>, std::vector<uint64_t>,
                               std::vector<float>, std::vector<double>>;

// An array of elements of the same Scalar type.
class Array : public ArrayBase {
 public:
  using ArrayBase::ArrayBase;

  ArrayProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const ArrayBase& AsVariant() const { return *this; }
  ArrayBase& AsVariant() { return *this; }

  static absl::StatusOr<Array> FromProto(const ArrayProto& proto);
};

using FlatAttributeBase = std::variant<Scalar, Array, std::string>;

// Attributes that do not support nested dictionaries.
class FlatAttribute : public FlatAttributeBase {
 public:
  using FlatAttributeBase::FlatAttributeBase;

  FlatAttributeProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const FlatAttributeBase& AsVariant() const { return *this; }
  FlatAttributeBase& AsVariant() { return *this; }

  static absl::StatusOr<FlatAttribute> FromProto(
      const FlatAttributeProto& proto);
};

using FlatAttributesMapBase = absl::flat_hash_map<std::string, FlatAttribute>;

// A map that maps from an arbitrary name (string key) to a flat attribute.
class FlatAttributesMap : public FlatAttributesMapBase {
 public:
  using FlatAttributesMapBase::FlatAttributesMapBase;

  FlatAttributesMapProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const FlatAttributesMapBase& AsVariant() const { return *this; }
  FlatAttributesMapBase& AsVariant() { return *this; }

  static absl::StatusOr<FlatAttributesMap> FromProto(
      const FlatAttributesMapProto& proto);
};

// Dictionary is just a wrapper around `AttributesMap`. We need an indirection
// through `std::shared_ptr` to be able to define recursive `std::variant`. We
// use shared pointer to keep `AttributesMap` copyable.
struct AttributesDictionary {
  std::shared_ptr<class AttributesMap> attrs;

  AttributesMapProto ToProto() const;

  static absl::StatusOr<AttributesDictionary> FromProto(
      const AttributesMapProto& proto);

  friend bool operator==(const AttributesDictionary& lhs,
                         const AttributesDictionary& rhs);

  friend bool operator!=(const AttributesDictionary& lhs,
                         const AttributesDictionary& rhs) {
    return !(lhs == rhs);
  }
};

using AttributeBase =
    internal::AppendType<FlatAttributeBase, AttributesDictionary>::Type;

// Attributes that support arbitrary nesting.
class Attribute : public AttributeBase {
 public:
  using AttributeBase::AttributeBase;

  explicit Attribute(const FlatAttribute& flat);

  xla::ffi::AttributeProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const AttributeBase& AsVariant() const { return *this; }
  AttributeBase& AsVariant() { return *this; }

  static absl::StatusOr<Attribute> FromProto(const AttributeProto& proto);
};

using AttributesMapBase = absl::flat_hash_map<std::string, Attribute>;

// AttributesMap is a map from an arbitrary name (string key) to an attribute.
class AttributesMap : public AttributesMapBase {
 public:
  using AttributesMapBase::AttributesMapBase;

  AttributesMapProto ToProto() const;

  // Older versions of libstdc++ don't implement P2162R2, therefore we need
  // these explicit casts to be able to use std::visit.
  const AttributesMapBase& AsVariant() const { return *this; }
  AttributesMapBase& AsVariant() { return *this; }

  static absl::StatusOr<AttributesMap> FromProto(
      const AttributesMapProto& proto);
};

// Converts MLIR dictionary attribute attached to a custom call operation to a
// custom call handler attributes that are forwarded to the FFI handler.
absl::StatusOr<AttributesMap> BuildAttributesMap(mlir::DictionaryAttr dict);

}  // namespace xla::ffi

#endif  // XLA_FFI_ATTRIBUTE_MAP_H_
