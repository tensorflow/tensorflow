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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"

namespace xla {

// AutotuneCacheStore is a minimal interface for reading and writing autotune
// entries. It abstracts away where the entries are stored (e.g. process memory,
// a local directory, CNS) and how keys are matched, leaving that matching/keys
// logic to TieredCache.
class AutotuneCacheStore {
 public:
  virtual ~AutotuneCacheStore() = default;

  // Reads all entries matching the given target key.
  virtual absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) = 0;

  // Writes an entry to the store.
  virtual absl::Status Write(const autotuner::AutotuneEntry& entry) = 0;

  // Reads all entries in the store. Mainly used for serialization.
  virtual absl::StatusOr<std::vector<autotuner::AutotuneEntry>> ReadAll() = 0;

  virtual CacheMode GetMode() const = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_
