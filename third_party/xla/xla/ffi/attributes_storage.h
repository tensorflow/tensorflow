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

#ifndef XLA_FFI_ATTRIBUTES_STORAGE_H_
#define XLA_FFI_ATTRIBUTES_STORAGE_H_

#include <memory>
#include <variant>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// AttributesStorage
//===----------------------------------------------------------------------===//

// Owns the XLA FFI C API storage backing a set of custom call attributes.
//
// An `AttributesMap` is a map of nested `std::variant`s, whereas the XLA FFI C
// API passes attributes as an `XLA_FFI_Attrs` struct: parallel arrays of names,
// type tags and type-erased pointers into storage owned by the caller.
// `AttributesStorage` owns that storage and keeps the `XLA_FFI_Attrs` struct
// pointing into it. This is what makes `AttrDecoding` (and therefore all user
// defined decodings registered with `XLA_FFI_REGISTER_ENUM_ATTR_DECODING` and
// `XLA_FFI_REGISTER_STRUCT_ATTR_DECODING`) work.
//
// For any particular instance of a custom call in the XLA program, attributes
// are compile time constants. An `AttributesStorage` is therefore immutable and
// shared: `Create` hands out a `std::shared_ptr` that can be held by any number
// of call frames (see `CallFrame::Copy`) and `ffi::Attributes` views.
class AttributesStorage {
 public:
  // Builds the C API storage for `attrs`.
  static std::shared_ptr<const AttributesStorage> Create(
      const AttributesMap& attrs);

  ~AttributesStorage();

  AttributesStorage(const AttributesStorage&) = delete;
  AttributesStorage& operator=(const AttributesStorage&) = delete;

  // Returns the XLA FFI attributes struct. It points into storage owned by
  // `*this` and is only valid for as long as `*this` is alive.
  const XLA_FFI_Attrs& ffi_attrs() const;

 private:
  // Declare implementation detail structs to grant access to private members.
  struct AttributeStorage;
  struct AttributeType;
  struct ConvertAttribute;
  struct FixUpAttribute;

  // Declare implementation detail structs for attributes storage.
  struct Array;
  struct Attributes;
  struct Dictionary;
  struct NamedAttribute;
  struct Scalar;
  struct String;

  using Attribute = std::variant<Scalar, Array, String, Dictionary>;

  explicit AttributesStorage(std::unique_ptr<Attributes> attributes);

  // Creates attributes storage from an attributes map.
  static std::unique_ptr<Attributes> CreateAttrs(const AttributesMap& attrs);

  // Fixes up attributes storage by initializing XLA FFI structs with valid
  // pointers into storage objects.
  static std::unique_ptr<Attributes> FixUpAttrs(
      std::unique_ptr<Attributes> attrs);

  std::unique_ptr<Attributes> attributes_;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_ATTRIBUTES_STORAGE_H_
