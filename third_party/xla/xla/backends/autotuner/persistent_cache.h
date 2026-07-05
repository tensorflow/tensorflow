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

#ifndef XLA_BACKENDS_AUTOTUNER_PERSISTENT_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_PERSISTENT_CACHE_H_

#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// PersistentCache is an abstract base class for caches that persist entries
// to a backing store (e.g., filesystem, database).
class PersistentCache : public AutotunerCacheInterface {
 public:
  PersistentCache(AutotuneCacheContext context, CacheMode mode,
                  KeyMatchingMode matching_mode);

  std::optional<Config> Lookup(const HloInstruction* instr) override;

  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override;

  CacheStats GetCacheStats() const override;
  CacheMode GetMode() const override { return mode_; }
  KeyMatchingMode GetKeyMatchingMode() const override { return matching_mode_; }

 protected:
  // TargetKey-based read, used for loose key matching.
  // Returns a vector of entries that match the target key, or an empty vector
  // if no match is found.
  virtual absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) = 0;
  // Full-key based read, for strict key matching.
  virtual absl::StatusOr<std::optional<autotuner::AutotuneValue>> Read(
      const autotuner::AutotuneKey& key);
  // Write an entry to the cache.
  virtual absl::Status Write(const autotuner::AutotuneEntry& entry) = 0;

 private:
  AutotuneCacheContext context_;
  CacheMode mode_;
  KeyMatchingMode matching_mode_;

  mutable absl::Mutex stats_mutex_;
  CacheStats stats_ ABSL_GUARDED_BY(stats_mutex_);
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_PERSISTENT_CACHE_H_
