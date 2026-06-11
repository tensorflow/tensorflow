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

#ifndef XLA_HLO_IR_BACKEND_CONFIG_POOL_H_
#define XLA_HLO_IR_BACKEND_CONFIG_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla {

// A process-wide pool for deduplicating backend config strings.
// It allows multiple instructions to share the same large JSON strings in
// memory.
class BackendConfigPool {
 public:
  // Returns the singleton instance of the pool.
  static BackendConfigPool* Get();

  // Returns a shared pointer to an interned string corresponding to the
  // provided JSON. If the string is already in the pool, returns the existing
  // instance. Otherwise, creates a new one.
  std::shared_ptr<const std::string> Intern(absl::string_view json);

  // Runs garbage collection to remove expired weak pointers from the pool.
  // Returns the number of entries removed.
  size_t GarbageCollect();

  // Resets the pool, clearing all entries. For testing only.
  void ResetForTesting();

 private:
  friend class absl::NoDestructor<BackendConfigPool>;
  BackendConfigPool() = default;

  absl::Mutex mutex_;
  size_t GarbageCollectLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // We use a hash of the string as the key to avoid storing the large string
  // twice (once as key and once in the value). Collisions are handled by
  // a linear search in the InlinedVector, which should almost always have
  // size 1.
  //
  // TODO(b/510802287): Consider sharding this pool if lock contention becomes
  // a bottleneck during concurrent module loading.
  absl::flat_hash_map<uint64_t,
                      absl::InlinedVector<std::weak_ptr<const std::string>, 1>>
      registry_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_HLO_IR_BACKEND_CONFIG_POOL_H_
