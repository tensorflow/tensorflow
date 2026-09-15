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

#ifndef XLA_FFI_ATTRIBUTES_H_
#define XLA_FFI_ATTRIBUTES_H_

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/attributes_storage.h"
#include "xla/ffi/ffi.h"

namespace xla::ffi {

// An owning, type-safe view over a set of FFI attributes.
//
// Compile-time consumers of custom-call attributes (for example XLA:GPU native
// custom-call handlers, see
// `backends/gpu/libraries/native_custom_call_thunks/`) get attributes as an
// `AttributesMap`, which is a map of nested `std::variant`s. Unwrapping those
// variants by hand is verbose and easy to get wrong.
//
// `Attributes` bridges that gap: it owns the C API storage for an
// `AttributesMap` (an `AttributesStorage`) and exposes the very same type-safe
// accessors that runtime FFI handlers use via `ffi::Dictionary`, so both sides
// agree on what a "float attribute named epsilon" means.
//
//   ffi::Attributes attrs = ffi::Attributes::Create(attrs_map);
//   ABSL_ASSIGN_OR_RETURN(float epsilon, attrs.Get<float>("epsilon"));
//   ABSL_ASSIGN_OR_RETURN(auto dims, attrs.Get<absl::Span<const int64_t>>("dims"));
//   ABSL_ASSIGN_OR_RETURN(auto mode, attrs.Get<absl::string_view>("mode"));
//
// Types that work out of the box are exactly the ones supported by
// `AttrDecoding`: scalars, `absl::Span<const T>` arrays, `absl::string_view`,
// nested `Dictionary`, `std::variant`, and any user type registered with
// `XLA_FFI_REGISTER_ENUM_ATTR_DECODING` or
// `XLA_FFI_REGISTER_STRUCT_ATTR_DECODING`.
//
// Decoding is strict about widths: an attribute created from an `i32` MLIR
// attribute must be read as `int32_t`, not `int64_t`. This mirrors the runtime
// FFI behavior.
//
// Lifetime: values that alias the underlying storage (`absl::string_view`,
// `absl::Span`, nested `Dictionary`) stay valid only as long as the
// `Attributes` object that produced them.
class Attributes {
 public:
  // Builds the C API storage for `attrs`.
  static Attributes Create(const AttributesMap& attrs);

  Attributes(Attributes&&) = default;
  Attributes& operator=(Attributes&&) = default;

  Attributes(const Attributes&) = delete;
  Attributes& operator=(const Attributes&) = delete;

  // Returns a non-owning type-safe view. The returned `Dictionary` is only
  // valid for the lifetime of `*this`.
  Dictionary dict() const { return Dictionary(&storage_->ffi_attrs()); }

  // Decodes the attribute named `name` as a `T`. Returns an error if the
  // attribute is missing or has a different type.
  template <typename T>
  absl::StatusOr<T> Get(absl::string_view name) const {
    return dict().get<T>(name);
  }

  // Returns true if an attribute named `name` exists, regardless of its type.
  bool Contains(absl::string_view name) const { return dict().contains(name); }

  // Returns true if an attribute named `name` exists and can be decoded as a
  // `T`. Only available for types whose `AttrDecoding` specialization provides
  // `Isa` (scalars, variants and registered enums/structs), matching
  // `ffi::Dictionary`.
  template <typename T>
  bool Contains(absl::string_view name) const {
    return dict().contains<T>(name);
  }

  size_t size() const { return dict().size(); }
  bool empty() const { return size() == 0; }

 private:
  explicit Attributes(std::shared_ptr<const AttributesStorage> storage);

  std::shared_ptr<const AttributesStorage> storage_;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_ATTRIBUTES_H_
