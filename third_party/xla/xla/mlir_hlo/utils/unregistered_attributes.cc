/* Copyright 2025 The OpenXLA Authors.

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

#include "utils/unregistered_attributes.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace xla {

bool IsKnownUnregisteredAttribute(std::string_view attr_name) {
  static constexpr std::array<std::string_view, 17>
      kKnownUnregisteredAttributes{
          // Function argument attributes
          xla::kJaxBufferDonor,
          xla::kExecutionThread,

          // Module level attributes.
          xla::kMhloCrossProgramPrefetches,
          xla::kMhloInputOutputAlias,
          xla::kMhloIsDynamic,
          xla::kMhloLiteral,
          xla::kMhloReplication,
          xla::kMhloSpmdOutputSharding,
          xla::kMhloSpmdParametersShardings,
          xla::kMhloUseAutoSpmdPartitioning,
          xla::kMhloXlaEntryComputationParameterLayouts,
          xla::kMhloXlaEntryComputationParameterTiles,
          xla::kMhloXlaEntryComputationResultLayout,
          xla::kMhloXlaEntryComputationResultTiles,
          xla::kMhloNumPartitions,
          xla::kMhloNumReplicas,

          // Op level attributes.
          xla::kMhloFrontendAttributes,
      };

  return std::find(kKnownUnregisteredAttributes.begin(),
                   kKnownUnregisteredAttributes.end(),
                   attr_name) != kKnownUnregisteredAttributes.end();
}

}  // namespace xla
