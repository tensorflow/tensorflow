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

#include "xla/stream_executor/gpu/locality_domain_memory.h"

#include <cstdint>
#include <map>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor::gpu {

namespace {

absl::Status ValidateLocalityDomainMemoryId(int64_t locality_domain_id) {
  if (locality_domain_id < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Locality-domain memory id must be nonnegative, got ",
                     locality_domain_id));
  }
  if (locality_domain_id >= kMaxLocalityDomains) {
    return absl::InvalidArgumentError(
        absl::StrCat("Locality-domain memory id must be less than ",
                     kMaxLocalityDomains, ", got ", locality_domain_id));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<int64_t> EncodeLocalityDomainMemorySpace(
    int64_t locality_domain_id) {
  ABSL_RETURN_IF_ERROR(ValidateLocalityDomainMemoryId(locality_domain_id));
  return kLocalityDomainMemorySpaceBase + locality_domain_id;
}

absl::StatusOr<int64_t> DecodeLocalityDomainMemorySpace(int64_t memory_space) {
  if (!IsLocalityDomainMemorySpace(memory_space)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Memory space ", memory_space,
        " is not an encoded locality-domain memory space; expected a value "
        "in [",
        kLocalityDomainMemorySpaceBase, ", ",
        kLocalityDomainMemorySpaceBase + kMaxLocalityDomains, ")"));
  }
  return memory_space - kLocalityDomainMemorySpaceBase;
}

bool IsLocalityDomainMemorySpace(int64_t memory_space) {
  return memory_space >= kLocalityDomainMemorySpaceBase &&
         memory_space < kLocalityDomainMemorySpaceBase + kMaxLocalityDomains;
}

absl::StatusOr<absl::string_view> LocalityDomainMemoryKind(
    int64_t locality_domain_id) {
  ABSL_RETURN_IF_ERROR(ValidateLocalityDomainMemoryId(locality_domain_id));
  // the map needs to be static to outlive any caller since we are returning a
  // string_view with no ownership semantics.
  static auto* const kinds = new std::map<int64_t, std::string>();
  static auto* const mu = new absl::Mutex();
  absl::MutexLock lock(mu);
  auto it = kinds
                ->try_emplace(locality_domain_id,
                              absl::StrCat(kLocalityDomainMemoryKindPrefix,
                                           locality_domain_id))
                .first;
  return it->second;
}

absl::StatusOr<int> LocalityDomainMemoryKindId(int64_t locality_domain_id) {
  ABSL_ASSIGN_OR_RETURN(absl::string_view kind,
                   LocalityDomainMemoryKind(locality_domain_id));
  return static_cast<int>(tsl::Fingerprint32(kind));
}

absl::StatusOr<int64_t> ParseLocalityDomainMemoryKind(
    absl::string_view memory_kind) {
  if (!absl::StartsWith(memory_kind, kLocalityDomainMemoryKindPrefix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid locality-domain memory kind: ", memory_kind));
  }

  absl::string_view id_text =
      memory_kind.substr(kLocalityDomainMemoryKindPrefix.size());
  int64_t locality_domain_id;
  if (id_text.empty() || !absl::SimpleAtoi(id_text, &locality_domain_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid locality domain id in memory kind: ", memory_kind));
  }
  ABSL_RETURN_IF_ERROR(ValidateLocalityDomainMemoryId(locality_domain_id));

  ABSL_ASSIGN_OR_RETURN(absl::string_view canonical,
                   LocalityDomainMemoryKind(locality_domain_id));
  if (memory_kind != canonical) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-canonical locality-domain memory kind: ", memory_kind,
                     "; expected ", canonical));
  }
  return locality_domain_id;
}

}  // namespace stream_executor::gpu
