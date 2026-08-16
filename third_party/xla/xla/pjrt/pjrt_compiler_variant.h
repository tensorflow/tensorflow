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

#ifndef XLA_PJRT_PJRT_COMPILER_VARIANT_H_
#define XLA_PJRT_PJRT_COMPILER_VARIANT_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

// Represents the PjRt compiler variant to use.
using PjRtCompilerVariant = uint64_t;

// XLA linked in as a library.
inline constexpr absl::string_view kLinkedVariant = "linked";
inline constexpr absl::string_view kUnknownVariant = "unknown";

inline PjRtCompilerVariant LinkedCompilerVariantId() {
  static const PjRtCompilerVariant kLinkedVariantId =
      tsl::Fingerprint64(kLinkedVariant);
  return kLinkedVariantId;
}

// This function has a weak default implementation in the
// pjrt_compiler_variant.cc file.
// Users can override it by providing a strong definition elsewhere.
// It is deliberately not marked as weak in this header so that any
// overriding definitions are correctly treated as strong symbols by the linker.
std::string CompilerVariantToString(xla::PjRtCompilerVariant variant);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPILER_VARIANT_H_
