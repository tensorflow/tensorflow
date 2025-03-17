/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DISTRIBUTED_IN_MEMORY_KEY_VALUE_STORE_H_
#define XLA_PJRT_DISTRIBUTED_IN_MEMORY_KEY_VALUE_STORE_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"

namespace xla {

class InMemoryKeyValueStore : public KeyValueStoreInterface {
 public:
  // This is the default behavior in in-memory key-value store to
  // allow overwriting.
  InMemoryKeyValueStore() : allow_overwrite_(true) {}
  explicit InMemoryKeyValueStore(bool allow_overwrite)
      : allow_overwrite_(allow_overwrite) {};
  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration timeout) override;

  absl::StatusOr<std::string> TryGet(absl::string_view key) override;

  absl::Status Set(absl::string_view key, absl::string_view value) override;

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::string> kv_store_ ABSL_GUARDED_BY(mu_);
  bool allow_overwrite_;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_IN_MEMORY_KEY_VALUE_STORE_H_
