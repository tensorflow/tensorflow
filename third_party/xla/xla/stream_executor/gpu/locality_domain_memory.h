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

#ifndef XLA_STREAM_EXECUTOR_GPU_LOCALITY_DOMAIN_MEMORY_H_
#define XLA_STREAM_EXECUTOR_GPU_LOCALITY_DOMAIN_MEMORY_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

inline constexpr absl::string_view kLocalityDomainMemoryKindPrefix =
    "locality_domain:";

// Memory-space colors [16, 48) are reserved for locality domains, one color per
// domain. Lower colors are used by other GPU memory spaces, and colors from 48
// up are left for future use.
inline constexpr int64_t kLocalityDomainMemorySpaceBase = 16;
inline constexpr int64_t kMaxLocalityDomains = 32;

// Encodes a locality domain as a GPU memory-space color.
absl::StatusOr<int64_t> EncodeLocalityDomainMemorySpace(
    int64_t locality_domain_id);

// Decodes a GPU memory-space color in the reserved locality-domain range.
// Callers with topology information must separately validate the decoded id
// against the actual domain count.
absl::StatusOr<int64_t> DecodeLocalityDomainMemorySpace(int64_t memory_space);

// Returns true if `memory_space` is an encoded locality-domain color.
bool IsLocalityDomainMemorySpace(int64_t memory_space);

// Returns a process-lifetime-stable view of the canonical memory kind
// "locality_domain:<id>".
absl::StatusOr<absl::string_view> LocalityDomainMemoryKind(
    int64_t locality_domain_id);

// Returns the stable PjRT kind id derived from the canonical memory kind.
absl::StatusOr<int> LocalityDomainMemoryKindId(int64_t locality_domain_id);

// Parses a canonical locality-domain memory kind. Non-canonical spellings
// (including whitespace, signs, and leading zeroes) are rejected.
absl::StatusOr<int64_t> ParseLocalityDomainMemoryKind(
    absl::string_view memory_kind);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_LOCALITY_DOMAIN_MEMORY_H_
