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

#ifndef XLA_PYTHON_IFRT_CUSTOM_OPTIONS_UTIL_H_
#define XLA_PYTHON_IFRT_CUSTOM_OPTIONS_UTIL_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/python/ifrt/attribute_map.h"

namespace xla {
namespace ifrt {

// Registers `key` as an IFRT-internal `ExecuteOptions::custom_options` key
// consumed by IFRT layers rather than forwarded to the underlying XLA runtime /
// FFI `ffi::Ctx<ffi::CustomOptions>()`.
//
// Thread-safe. Copies `key` into the registry, so `key` does not need to
// outlive this call. Typically called from a module initializer inside a
// dedicated `alwayslink = True` registration target (cf. `RegisterSerDes`).
// Registering the same key more than once crashes the process.
void RegisterIfrtInternalCustomOption(absl::string_view key);

// Returns true if `key` has been registered as an IFRT-internal
// `ExecuteOptions::custom_options` key that should not be forwarded to the
// underlying XLA runtime / FFI `ffi::Ctx<ffi::CustomOptions>()`.
bool IsIfrtInternalCustomOption(absl::string_view key);

// Returns a copy of `custom_options` with all registered IFRT-internal keys
// removed, or `std::nullopt` if `custom_options` is empty or contains only
// internal keys.
std::optional<AttributeMap> FilterOutIfrtInternalCustomOptions(
    const AttributeMap& custom_options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_CUSTOM_OPTIONS_UTIL_H_
