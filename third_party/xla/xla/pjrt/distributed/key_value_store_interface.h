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

#ifndef XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_INTERFACE_H_
#define XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_INTERFACE_H_

#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace xla {

// In the multi-node case, the caller of PjRtClient can provide a key-value
// store accessible across nodes. The caller can provide the two callbacks
// below to access the key-value store. There are a few requirements:
// (1) Get and Set must be thread-safe.
// (2) The caller that provides the two callbacks is responsible for avoiding
// key collisions between different users of key-value store (i.e. between
// different plugins, but not between different GPU plugin nodes).
class KeyValueStoreInterface {
 public:
  virtual ~KeyValueStoreInterface() = default;

  // Blocking Get().
  // There are no concurrency guarantees. To avoid a race / impose an ordering
  // on potentially concurrent ops (e.g. set, delete), use WaitAtBarrier().
  virtual absl::StatusOr<std::string> Get(std::string_view key,
                                          absl::Duration timeout) = 0;

  virtual absl::Status Set(std::string_view key, std::string_view value) = 0;
};

struct MultiProcessKeyValueStore {
  std::shared_ptr<KeyValueStoreInterface> key_value_store;
  int process_index = 0;
  int process_count = 1;
};

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_INTERFACE_H_
