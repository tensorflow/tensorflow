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

#ifndef XLA_BACKENDS_PROFILER_UTIL_METADATA_REGISTRY_H_
#define XLA_BACKENDS_PROFILER_UTIL_METADATA_REGISTRY_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {

// Sets a key-value metadata entry for the current profiling session.
// This function is thread-safe.
void SetProfilerMetadata(absl::string_view key, absl::string_view value);

// Gets a metadata value by key. Returns empty string if not found.
// This function is thread-safe.
std::string GetProfilerMetadata(absl::string_view key);

// Gets all metadata.
absl::flat_hash_map<std::string, std::string> GetAllProfilerMetadata();

// Clears all metadata.
void ClearProfilerMetadata();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_UTIL_METADATA_REGISTRY_H_
