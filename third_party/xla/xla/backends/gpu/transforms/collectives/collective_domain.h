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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_H_

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {

using CollectiveCommunicationDomain =  // NOLINT
    CollectiveBackendConfig::CollectiveCommunicationDomain;

// Shorthands to avoid typing extremely long protobuf types.
inline constexpr CollectiveCommunicationDomain kUnspecifiedCollectiveDomain =
    CollectiveBackendConfig::COLLECTIVE_COMMUNICATION_DOMAIN_UNSPECIFIED;
inline constexpr CollectiveCommunicationDomain kScaleUpFabricCollectiveDomain =
    CollectiveBackendConfig::COLLECTIVE_COMMUNICATION_DOMAIN_SCALE_UP_FABRIC;

// Human-readable string representations of enums.
inline constexpr absl::string_view kUnspecifiedCollectiveDomainName =
    "unspecified";
inline constexpr absl::string_view kScaleUpFabricCollectiveDomainName =
    "scale_up_fabric";

template <typename Sink>
void AbslStringify(Sink& sink, CollectiveCommunicationDomain domain) {
  switch (domain) {
    case kUnspecifiedCollectiveDomain:
      sink.Append(kUnspecifiedCollectiveDomainName);
      return;
    case kScaleUpFabricCollectiveDomain:
      sink.Append(kScaleUpFabricCollectiveDomainName);
      return;
    default:
      absl::Format(&sink, "unknown(%d)", static_cast<int>(domain));
      return;
  }
}

// Parses the frontend spelling of a collective communication domain.
absl::StatusOr<CollectiveCommunicationDomain>
ParseCollectiveCommunicationDomain(absl::string_view value);

// Joins two compatible domain facts. UNSPECIFIED is an unknown fact and
// therefore yields to an explicitly known domain.
absl::StatusOr<CollectiveCommunicationDomain>
JoinCollectiveCommunicationDomains(CollectiveCommunicationDomain lhs,
                                   CollectiveCommunicationDomain rhs);

// Returns true if `instruction` can carry a collective communication domain.
bool SupportsCollectiveCommunicationDomain(const HloInstruction& instruction);

// Returns the typed communication domain from the canonical async start (or
// from `instruction` itself for a synchronous collective).
absl::StatusOr<CollectiveCommunicationDomain> GetCollectiveCommunicationDomain(
    const HloInstruction& instruction);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_H_
