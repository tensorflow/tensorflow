/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/ffi/attributes_storage.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "xla/custom_options.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"

namespace xla::ffi {

// ------------------------    !!! !!! !!!     ------------------------------ //

// WARNING: In the structs defined below we use a pattern where we declare
// a storage (e.g. an `std::string` member) and an XLA FFI reference type
// pointing into that storage in the same struct (XLA_FFI_ByteSpan). Extra care
// should be taken of keeping reference type up to date, e.g. if a parent
// struct put into an `std::vector` container, every time vector will reallocate
// storage all reference types will become invalid.

// We intentionally do not use smart pointers that would guarantee pointer
// stability for storage, as we are trying to minimize the number of heap
// allocations required for building the attributes storage.

// This is a low level internal implementation detail: the structs below are
// only forward declared in the public header and can be changed at any time in
// the future.

//----------------------------------------------------------------------------//
// Attributes storage + reference types
//----------------------------------------------------------------------------//

struct AttributesStorage::Dictionary {
  std::unique_ptr<Attributes> attrs;
};

struct AttributesStorage::Array {
  xla::ffi::Array value;  // XLA_FFI_Array::data

  XLA_FFI_Array array = {};
};

struct AttributesStorage::Scalar {
  xla::ffi::Scalar value;  // XLA_FFI_Scalar::value

  XLA_FFI_Scalar scalar = {};
};

struct AttributesStorage::String {
  std::string value;  // XLA_FFI_ByteSpan::ptr

  XLA_FFI_ByteSpan span = {};
};

struct AttributesStorage::NamedAttribute {
  String name;
  Attribute value;
};

struct AttributesStorage::Attributes {
  std::vector<NamedAttribute> attributes;

  std::vector<XLA_FFI_ByteSpan*> names;  // XLA_FFI_Attrs::names
  std::vector<XLA_FFI_AttrType> types;   // XLA_FFI_Attrs::types
  std::vector<void*> attrs;              // XLA_FFI_Attrs::attrs

  XLA_FFI_Attrs ffi_attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr};
};

//===----------------------------------------------------------------------===//
// AttributesStorage
//===----------------------------------------------------------------------===//

AttributesStorage::AttributesStorage(std::unique_ptr<Attributes> attributes)
    : attributes_(std::move(attributes)) {}

AttributesStorage::~AttributesStorage() = default;

std::unique_ptr<const AttributesStorage> AttributesStorage::Create(
    const AttributesMap& attrs) {
  // Not `std::make_unique`: the constructor is private.
  return std::unique_ptr<const AttributesStorage>(
      new AttributesStorage(CreateAttrs(attrs)));
}

const XLA_FFI_Attrs& AttributesStorage::ffi_attrs() const {
  return attributes_->ffi_attrs;
}

// An std::visit overload set for converting an `xla::ffi::Attribute` to an
// `AttributesStorage::Attribute`.
struct AttributesStorage::ConvertAttribute {
  AttributesStorage::Attribute operator()(const xla::ffi::Array& array) {
    return AttributesStorage::Array{array};
  }

  AttributesStorage::Attribute operator()(const xla::ffi::Scalar& scalar) {
    return AttributesStorage::Scalar{scalar};
  }

  AttributesStorage::Attribute operator()(const std::string& str) {
    return AttributesStorage::String{str};
  }

  AttributesStorage::Attribute operator()(
      const xla::ffi::AttributesDictionary& dict) {
    // `AttributesDictionary` is an aggregate whose `attrs` defaults to null,
    // and both `AttributesDictionary::ToProto` and `operator==` treat a null
    // `attrs` as an empty dictionary. Do the same here instead of
    // dereferencing null.
    if (dict.attrs == nullptr) {
      return Dictionary{CreateAttrs(xla::ffi::AttributesMap())};
    }
    return Dictionary{CreateAttrs(*dict.attrs)};
  }
};

// An std::visit overload set for converting an `xla::CustomOptions::Value` to
// an `AttributesStorage::Attribute`.
struct AttributesStorage::ConvertOption {
  AttributesStorage::Attribute operator()(bool value) {
    return AttributesStorage::Scalar{xla::ffi::Scalar{value}};
  }

  AttributesStorage::Attribute operator()(int64_t value) {
    return AttributesStorage::Scalar{xla::ffi::Scalar{value}};
  }

  AttributesStorage::Attribute operator()(float value) {
    return AttributesStorage::Scalar{xla::ffi::Scalar{value}};
  }

  AttributesStorage::Attribute operator()(const std::string& str) {
    return AttributesStorage::String{str};
  }

  AttributesStorage::Attribute operator()(const std::vector<int64_t>& array) {
    return AttributesStorage::Array{xla::ffi::Array{array}};
  }
};

// An std::visit overload set to fix up AttributesStorage::Attribute storage and
// initialize XLA FFI structs with valid pointers into storage objects.
struct AttributesStorage::FixUpAttribute {
  void operator()(AttributesStorage::Array& array) {
    auto visitor = [&](auto& value) {
      using T = typename std::remove_reference_t<decltype(value)>::value_type;
      array.array.dtype = internal::NativeTypeToCApiDataType<T>();
      array.array.size = value.size();
      array.array.data = value.data();
    };
    std::visit(visitor, array.value.AsVariant());
  }

  void operator()(AttributesStorage::Scalar& scalar) {
    auto visitor = [&](auto& value) {
      using T = std::remove_reference_t<decltype(value)>;
      scalar.scalar.dtype = internal::NativeTypeToCApiDataType<T>();
      scalar.scalar.value = &value;
    };
    std::visit(visitor, scalar.value.AsVariant());
  }

  void operator()(AttributesStorage::String& str) {
    str.span.ptr = str.value.data();
    str.span.len = str.value.size();
  }

  void operator()(AttributesStorage::Dictionary&) {}
};

// An std::visit overload set to get AttributesStorage::Attribute XLA FFI type.
struct AttributesStorage::AttributeType {
  XLA_FFI_AttrType operator()(AttributesStorage::Array&) {
    return XLA_FFI_AttrType_ARRAY;
  }

  XLA_FFI_AttrType operator()(AttributesStorage::Scalar&) {
    return XLA_FFI_AttrType_SCALAR;
  }

  XLA_FFI_AttrType operator()(AttributesStorage::String&) {
    return XLA_FFI_AttrType_STRING;
  }

  XLA_FFI_AttrType operator()(AttributesStorage::Dictionary&) {
    return XLA_FFI_AttrType_DICTIONARY;
  }
};

// An std::visit overload set to get AttributesStorage::Attribute storage
// pointer.
struct AttributesStorage::AttributeStorage {
  template <typename T>
  void* operator()(T& value) {
    return &value;
  }

  void* operator()(AttributesStorage::Array& array) { return &array.array; }

  void* operator()(AttributesStorage::Scalar& scalar) { return &scalar.scalar; }

  void* operator()(AttributesStorage::String& str) { return &str.span; }

  void* operator()(AttributesStorage::Dictionary& dict) {
    return &dict.attrs->ffi_attrs;
  }
};

std::unique_ptr<AttributesStorage::Attributes> AttributesStorage::CreateAttrs(
    const xla::ffi::AttributesMap& battrs) {
  auto attrs = std::make_unique<Attributes>();

  // Convert the attributes map to a collection of named attributes.
  attrs->attributes.reserve(battrs.size());
  for (auto& [name, battr] : battrs) {
    NamedAttribute attr = {String{name},
                           std::visit(ConvertAttribute(), battr.AsVariant())};
    attrs->attributes.push_back(std::move(attr));
  }

  return FixUpAttrs(std::move(attrs));
}

std::unique_ptr<const AttributesStorage> AttributesStorage::Create(
    const xla::CustomOptions& options) {
  auto attrs = std::make_unique<Attributes>();
  attrs->attributes.reserve(options.size());
  for (const auto& [name, value] : options.map()) {
    NamedAttribute attr = {String{name}, std::visit(ConvertOption(), value)};
    attrs->attributes.push_back(std::move(attr));
  }
  return std::unique_ptr<const AttributesStorage>(
      new AttributesStorage(FixUpAttrs(std::move(attrs))));
}

std::unique_ptr<AttributesStorage::Attributes> AttributesStorage::FixUpAttrs(
    std::unique_ptr<Attributes> attrs) {
  // Sort attributes by name to enable binary search at run time.
  absl::c_sort(attrs->attributes,
               [](const NamedAttribute& a, const NamedAttribute& b) {
                 return a.name.value < b.name.value;
               });

  size_t num_attrs = attrs->attributes.size();
  DCHECK(attrs->names.empty() && attrs->types.empty() && attrs->attrs.empty());

  attrs->names.reserve(num_attrs);
  attrs->types.reserve(num_attrs);
  attrs->attrs.reserve(num_attrs);

  // Fix up XLA FFI structs to point to correct storage.
  for (NamedAttribute& attr : attrs->attributes) {
    std::invoke(FixUpAttribute{}, attr.name);
    std::visit(FixUpAttribute{}, attr.value);
  }

  // Initialize vectors required for building XLA_FFI_Attributes.
  for (NamedAttribute& attr : attrs->attributes) {
    attrs->names.push_back(&attr.name.span);
    attrs->types.push_back(std::visit(AttributeType(), attr.value));
    attrs->attrs.push_back(std::visit(AttributeStorage(), attr.value));
  }

  // Finally initialize XLA FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  attrs->ffi_attrs.size = attrs->attributes.size();
  attrs->ffi_attrs.names = attrs->names.data();
  attrs->ffi_attrs.types = attrs->types.data();
  attrs->ffi_attrs.attrs = attrs->attrs.data();

  return attrs;
}

}  // namespace xla::ffi
