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

#include "xla/backends/autotuner/tiered_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

TieredCache::TieredCache(
    std::unique_ptr<AutotunerCacheInterface> primary_cache,
    std::unique_ptr<AutotunerCacheInterface> secondary_cache)
    : primary_cache_(std::move(primary_cache)),
      secondary_cache_(std::move(secondary_cache)) {}

std::optional<AutotunerCacheInterface::Config> TieredCache::Lookup(
    const HloInstruction* instr) {
  // Query the primary cache first.
  std::optional<Config> config = primary_cache_->Lookup(instr);
  if (config.has_value()) {
    return config;
  }

  // On a primary cache miss, query the secondary cache.
  config = secondary_cache_->Lookup(instr);
  if (config.has_value()) {
    // Populate the primary cache for subsequent queries.
    primary_cache_->Insert(instr, *config).IgnoreError();
    return config;
  }

  return std::nullopt;
}

absl::Status TieredCache::Insert(const HloInstruction* instr,
                                 const Config& config) {
  // Write to both the primary and secondary caches.
  absl::Status status1 = primary_cache_->Insert(instr, config);
  absl::Status status2 = secondary_cache_->Insert(instr, config);

  if (!status1.ok()) {
    return status1;
  }
  return status2;
}

AutotunerCacheInterface::CacheStats TieredCache::GetCacheStats() const {
  CacheStats stats1 = primary_cache_->GetCacheStats();
  CacheStats stats2 = secondary_cache_->GetCacheStats();
  CacheStats stats;
  stats.hits = stats1.hits + stats2.hits;
  stats.misses = stats2.misses;
  return stats;
}

absl::StatusOr<std::string> TieredCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  return secondary_cache_->Serialize(instructions_to_serialize);
}

absl::Status TieredCache::Deserialize(absl::string_view serialized_cache) {
  // Update the secondary cache.
  RETURN_IF_ERROR(secondary_cache_->Deserialize(serialized_cache));
  // Invalidate or update primary cache by deserializing into it as well.
  return primary_cache_->Deserialize(serialized_cache);
}

}  // namespace xla
