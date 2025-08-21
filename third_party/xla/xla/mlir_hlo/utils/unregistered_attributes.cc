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

bool IsKnownDiscardableModuleAttribute(std::string_view attr_name) {
  {
    static constexpr std::array<std::string_view, 17>
        kKnownDiscardableAttributes{
            xla::kMhloNumPartitions,
            xla::kMhloNumReplicas,
            xla::kMhloCrossProgramPrefetches,
            xla::kMhloInputOutputAlias,
            xla::kMhloIsDynamic,
            xla::kMhloReplication,
            xla::kMhloSpmdOutputSharding,
            xla::kMhloSpmdParametersShardings,
            xla::kMhloUseAutoSpmdPartitioning,
            xla::kMhloXlaEntryComputationParameterLayouts,
            xla::kMhloXlaEntryComputationParameterTiles,
            xla::kMhloXlaEntryComputationResultLayout,
            xla::kMhloXlaEntryComputationResultTiles,
        };

    return std::find(kKnownDiscardableAttributes.begin(),
                     kKnownDiscardableAttributes.end(),
                     attr_name) != kKnownDiscardableAttributes.end();
  }
}
bool IsKnownDiscardableFuncAttribute(std::string_view attr_name) {
  static constexpr std::array<std::string_view, 3> kKnownDiscardableAttributes{
      xla::kJaxBufferDonor,
      xla::kExecutionThread,
      xla::kMhloFrontendAttributes,
  };

  return std::find(kKnownDiscardableAttributes.begin(),
                   kKnownDiscardableAttributes.end(),
                   attr_name) != kKnownDiscardableAttributes.end();
}

bool IsKnownDiscardableOpAttribute(std::string_view attr_name) {
  static constexpr std::array<std::string_view, 3> kKnownDiscardableAttributes{
      xla::kMhloFrontendAttributes,
      xla::kMhloLiteral,
      xla::kMhloSharding,
  };

  return std::find(kKnownDiscardableAttributes.begin(),
                   kKnownDiscardableAttributes.end(),
                   attr_name) != kKnownDiscardableAttributes.end();
}

}  // namespace xla
