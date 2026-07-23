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

#ifndef XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// TieredCache is an AutotunerCacheInterface implementation using AutotuneEntry
// protos. It owns a primary and an optional secondary AutotuneCacheStore and
// centralizes all of the logic that used to be duplicated across cache
// implementations:
//   - building the AutotuneTargetKey / AutotuneKey from an instruction.
//   - strict/loose environment matching.
//   - AutotuneCache proto (de)serialization.
//
// Lookups query tiers in primary and then in secondary cache, promoting any
// secondary cache hits to the primary cache. Inserts write to every
// writable tier.
class TieredCache : public AutotunerCacheInterface {
 public:
  TieredCache(AutotuneCacheContext context, KeyMatchingMode matching_mode,
              std::unique_ptr<AutotuneCacheStore> primary,
              std::unique_ptr<AutotuneCacheStore> secondary = nullptr);

  ~TieredCache() override = default;

  std::optional<Config> Lookup(const HloInstruction* instr) override;

  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override;

  CacheStats GetCacheStats() const override;

  // Serialize from the primary cache.
  absl::StatusOr<std::string> Serialize(absl::Span<const HloInstruction* const>
                                            instructions_to_serialize) override;

  // Deserialize and insert into the primary cache.
  absl::Status Deserialize(absl::string_view serialized_cache) override;

  CacheMode GetMode() const override;
  KeyMatchingMode GetKeyMatchingMode() const override { return matching_mode_; }

 private:
  // Builds the target key (device, explicit_version, hlo_fingerprint) for the
  // instruction. This is the sole caller of GetHloFingerprint on the hot path.
  autotuner::AutotuneTargetKey BuildTargetKey(
      const HloInstruction& instr) const;

  // Builds a full entry (key + value) for insertion.
  autotuner::AutotuneEntry BuildEntry(const HloInstruction& instr,
                                      const Config& config) const;

  // Returns the first entry from `entries` that matches the current context
  // (and, in strict mode, `codegen_options_fp`), or nullopt.
  std::optional<autotuner::AutotuneEntry> MatchEntry(
      const std::vector<autotuner::AutotuneEntry>& entries,
      const std::optional<std::string>& codegen_options_fp) const;

  // Writes `entry` to `store` honoring the store's CacheMode.
  absl::Status WriteToStore(AutotuneCacheStore& store,
                            const autotuner::AutotuneEntry& entry) const;

  AutotuneCacheContext context_;
  KeyMatchingMode matching_mode_;
  std::unique_ptr<AutotuneCacheStore> primary_;
  std::unique_ptr<AutotuneCacheStore> secondary_;

  mutable absl::Mutex stats_mutex_;
  CacheStats stats_ ABSL_GUARDED_BY(stats_mutex_);
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_
