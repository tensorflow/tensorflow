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

#ifndef XLA_BACKENDS_CPU_CODEGEN_OBJECT_BUFFER_IDENTIFIER_H_
#define XLA_BACKENDS_CPU_CODEGEN_OBJECT_BUFFER_IDENTIFIER_H_

#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla::cpu {

// Delimiter that separates the memory region name (used for profiling) from the
// unique module identifier (used for sanitizers and symbol identification).
inline constexpr absl::string_view kMemoryRegionSeparator =
    "::__xla_mem_region_separator__::";

// Encodes the memory region name and module identifier into a single object
// buffer identifier.
inline std::string EncodeBufferIdentifier(absl::string_view memory_region_name,
                                          absl::string_view module_identifier) {
  return absl::StrCat(memory_region_name, kMemoryRegionSeparator,
                      module_identifier);
}

// Extracts the memory region name from an object buffer identifier. If the
// separator is found, returns the region name prefix. Otherwise, returns the
// identifier itself for backward compatibility with legacy or raw buffer
// identifiers.
inline absl::string_view ExtractMemoryRegionName(
    absl::string_view buffer_identifier) {
  size_t pos = buffer_identifier.find(kMemoryRegionSeparator);
  if (pos != absl::string_view::npos) {
    return buffer_identifier.substr(0, pos);
  }
  return buffer_identifier;
}

// Extracts the unique module identifier from an object buffer identifier.
inline absl::string_view ExtractModuleIdentifier(
    absl::string_view buffer_identifier) {
  size_t pos = buffer_identifier.find(kMemoryRegionSeparator);
  if (pos != absl::string_view::npos) {
    return buffer_identifier.substr(pos + kMemoryRegionSeparator.size());
  }
  return buffer_identifier;
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_OBJECT_BUFFER_IDENTIFIER_H_
