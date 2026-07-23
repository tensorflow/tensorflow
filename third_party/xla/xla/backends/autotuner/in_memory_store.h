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

#ifndef XLA_BACKENDS_AUTOTUNER_IN_MEMORY_STORE_H_
#define XLA_BACKENDS_AUTOTUNER_IN_MEMORY_STORE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotuning.pb.h"

namespace xla {

// InMemoryStore is a process-wide, in-memory AutotuneCacheStore. It replaces
// the former LocalCache/LocalCacheStorage pair as the hot in-memory tier.
// Entries are held in a global static flat_hash_map keyed by TargetKey, which
// is shared across all instances of InMemoryStore.
class InMemoryStore : public AutotuneCacheStore {
 public:
  InMemoryStore() = default;
  ~InMemoryStore() override = default;

  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) override;
  absl::Status Write(const autotuner::AutotuneEntry& entry) override;
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> ReadAll() override;

  CacheMode GetMode() const override { return CacheMode::kReadWrite; }

  // Clears the process-wide in-memory cache. Used to isolate unit tests.
  static void Clear();
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_IN_MEMORY_STORE_H_
