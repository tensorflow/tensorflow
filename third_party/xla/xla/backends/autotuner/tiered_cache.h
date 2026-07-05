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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// TieredCache combines a primary cache (e.g. local in-memory L1 cache)
// and a secondary cache (e.g. persistent L2 cache) into a single interface.
//
// Read requests check the primary cache first. On a miss, they query the
// secondary cache and populate the primary cache with the result on a hit.
// Write requests update both the primary and secondary caches.
class TieredCache : public AutotunerCacheInterface {
 public:
  TieredCache(std::unique_ptr<AutotunerCacheInterface> primary_cache,
              std::unique_ptr<AutotunerCacheInterface> secondary_cache);

  ~TieredCache() override = default;

  std::optional<Config> Lookup(const HloInstruction* instr) override;

  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override;

  CacheStats GetCacheStats() const override;

  // Serialize from the secondary cache.
  absl::StatusOr<std::string> Serialize(absl::Span<const HloInstruction* const>
                                            instructions_to_serialize) override;

  // Deserialize updates both the primary and secondary caches.
  absl::Status Deserialize(absl::string_view serialized_cache) override;

  CacheMode GetMode() const override { return secondary_cache_->GetMode(); }
  KeyMatchingMode GetKeyMatchingMode() const override {
    return secondary_cache_->GetKeyMatchingMode();
  }

 private:
  std::unique_ptr<AutotunerCacheInterface> primary_cache_;
  std::unique_ptr<AutotunerCacheInterface> secondary_cache_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_
