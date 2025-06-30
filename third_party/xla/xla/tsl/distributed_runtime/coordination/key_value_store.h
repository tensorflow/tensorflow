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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_KEY_VALUE_STORE_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_KEY_VALUE_STORE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"

namespace tsl {

// A thread-safe in-memory key-value store.
class KeyValueStore {
 public:
  using Callback =
      absl::AnyInvocable<void(const absl::StatusOr<absl::string_view>&)>;

  KeyValueStore() = default;
  ~KeyValueStore();

  // KeyValueStore is not copyable or movable.
  KeyValueStore(const KeyValueStore&) = delete;
  KeyValueStore(KeyValueStore&&) = delete;
  KeyValueStore& operator=(const KeyValueStore&) = delete;
  KeyValueStore& operator=(KeyValueStore&&) = delete;

  // Inserts a key-value pair. If allow_overwrite is false, then Put returns an
  // error if the provided key is already in the store.
  absl::Status Put(absl::string_view key, absl::string_view value,
                   bool allow_overwrite);

  // Returns the value associated with the provided key, if one exists.
  std::optional<std::string> Get(absl::string_view key);

  // Returns all key-value pairs where the key has the provided prefix.
  //
  // The empty string "" is a prefix of every key, so GetPrefix("") can be used
  // to retrieve every element in the store.
  std::vector<tensorflow::KeyValueEntry> GetPrefix(absl::string_view prefix);

  // Adds a callback that is called when the provided key exists in the map.
  void AddCallbackForKey(absl::string_view key, Callback callback);

  // Deletes the provided key.
  void Delete(absl::string_view key);

  // Deletes all key-value pairs where the key has the provided prefix.
  void DeletePrefix(absl::string_view prefix);

 private:
  // Notifies all callbacks registered for the provided key.
  void NotifyCallbacksForKey(absl::string_view key,
                             const absl::StatusOr<absl::string_view>& value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  absl::btree_map<std::string, std::string> data_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::vector<Callback>> callbacks_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_KEY_VALUE_STORE_H_
