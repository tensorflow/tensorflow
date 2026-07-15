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

#ifndef XLA_BACKENDS_AUTOTUNER_LOCAL_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_LOCAL_CACHE_H_

#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

class LocalCache;

// LocalCacheStorage owns the underlying cache map and corresponding
// statistics.
class LocalCacheStorage {
 public:
  LocalCacheStorage() = default;
  ~LocalCacheStorage() = default;

  LocalCacheStorage(const LocalCacheStorage&) = delete;
  LocalCacheStorage& operator=(const LocalCacheStorage&) = delete;

  // Returns the process-wide context-specific instance of LocalCacheStorage.
  static LocalCacheStorage& GetInstance(const AutotuneCacheContext& ctx);

  friend class LocalCache;

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, AutotunerCacheInterface::Config> cache_
      ABSL_GUARDED_BY(mutex_);
  AutotunerCacheInterface::CacheStats stats_ ABSL_GUARDED_BY(mutex_);
};

// LocalCache is an implementation of AutotunerCacheInterface that queries
// and updates configurations stored in a LocalCacheStorage.
class LocalCache : public AutotunerCacheInterface {
 public:
  // Constructs a LocalCache with the given matching mode, and backing storage.
  LocalCache(KeyMatchingMode matching_mode,
             LocalCacheStorage* absl_nonnull storage);

  std::optional<Config> Lookup(const HloInstruction* instr) override;
  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override;

  absl::StatusOr<std::string> Serialize(absl::Span<const HloInstruction* const>
                                            instructions_to_serialize) override;
  absl::Status Deserialize(absl::string_view serialized_cache) override;

  CacheStats GetCacheStats() const override;
  CacheMode GetMode() const override { return CacheMode::kReadWrite; }
  KeyMatchingMode GetKeyMatchingMode() const override { return matching_mode_; }

 private:
  std::string GetCacheKey(const HloInstruction* instr) const;

  KeyMatchingMode matching_mode_;
  LocalCacheStorage* absl_nonnull storage_;  // Not owned.
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_LOCAL_CACHE_H_
