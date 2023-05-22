/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// Persistent cache repository.
//
// Defines the interface for a persistent cache repository. Caching solutions
// will use this interface to persist entries, e.g., XLA compilation output.

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_PERSISTENT_CACHE_REPOSITORY_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_PERSISTENT_CACHE_REPOSITORY_H_

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tsl {

// Persistent cache repository interface.
//
// Usage:
//   Assume an implementation:
//     class FooPersistentCacheRepository : public PersistentCacheRepository {
//       ...
//     };
//
//   StatusOr<std::unique_ptr> pcr = FooPersistentCacheRepository::Create(
//       PersistentCacheRepository::Options{
//           .reset_ttl_on_access = true,
//       }, ...);
//   CHECK_OK(pcr.status());
//   auto repo = *std::move(pcr);
//
//   std::string key = ...;
//   std::string entry = ...;
//   Status put_status = repo->Put(key, entry);
//   CHECK_OK(put_status);
//
//   StatusOr<std::string> entry = repo->Get(key);
//   CHECK_OK(entry.status());
//   ... use *entry ...
//
// Implementation notes:
// . An implementation can declare a factory or constructor that accepts
//   additional options unique to that implementation.
//
// . Put must fail with ALREADY_EXISTS if it is called with a key that already
//   exists in the repository and the entry specified is different from the
//   entry in the repository. Put requests for the same key for a given user
//   must be idempotent.
//
// . Get must fail with NOT_FOUND if it is called with a key that does not
//   exist in the repository.
//
// . Put and Get can return other errors that are implementation dependent.
//
// . An implementation does not have to guard against:
//   - two Puts for the same key racing with each other, or
//   - a Put and a Get for the same key racing with each other.
//
// . If Put(k, e) succeeds, a subsequent Get(k) must return e if no errors or
//   races occur.
//
// . If the implementation uses RPC services, it must ensure that appropriate
//   deadlines are set to prevent hangs.
class PersistentCacheRepository {
 public:
  struct Options {
    // If true, reset the TTL of the entry on each Get() or Put(). If false,
    // retain the TTL from the time of creation of the entry.
    bool reset_ttl_on_access = false;
  };
  explicit PersistentCacheRepository(Options options)
      : options_(std::move(options)) {}

  PersistentCacheRepository(const PersistentCacheRepository&) = delete;
  PersistentCacheRepository& operator=(const PersistentCacheRepository&) =
      delete;

  virtual ~PersistentCacheRepository() = default;

  virtual absl::Status Put(const std::string& key,
                           const std::string& serialized_entry) = 0;
  virtual absl::StatusOr<std::string> Get(const std::string& key) = 0;

 protected:
  const Options options_;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_PERSISTENT_CACHE_REPOSITORY_H_
