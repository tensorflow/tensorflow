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
    static constexpr std::array<std::string_view, 14>
        kKnownDiscardableAttributes{
            xla::kMhloCrossProgramPrefetches,
            xla::kMhloFrontendAttributes,
            xla::kMhloInputOutputAlias,
            xla::kMhloIsDynamic,
            xla::kMhloNumPartitions,
            xla::kMhloNumReplicas,
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
  static constexpr std::array<std::string_view, 8> kKnownDiscardableAttributes{
      xla::kExecutionThread,        xla::kJaxBufferDonor,
      xla::kMhloFrontendAttributes, xla::kMhloLayoutMode,
      xla::kMhloMemoryKind,         xla::kMhloParameterReplication,
      xla::kMhloSharding,           xla::kTfAliasingOutput,
  };

  return std::find(kKnownDiscardableAttributes.begin(),
                   kKnownDiscardableAttributes.end(),
                   attr_name) != kKnownDiscardableAttributes.end();
}

bool IsKnownDiscardableOpAttribute(std::string_view attr_name) {
  static constexpr std::array<std::string_view, 8> kKnownDiscardableAttributes{
      xla::kBitcastResultLayout, xla::kBitcastSourceLayout,
      xla::kInfeedLayout,        xla::kMhloFrontendAttributes,
      xla::kMhloLiteral,         xla::kMhloOriginalValueAttr,
      xla::kMhloSharding,        xla::kXlaShape,
  };

  return std::find(kKnownDiscardableAttributes.begin(),
                   kKnownDiscardableAttributes.end(),
                   attr_name) != kKnownDiscardableAttributes.end();
}

}  // namespace xla
