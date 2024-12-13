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

#include "xla/pjrt/distributed/in_memory_key_value_store.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace xla {

absl::StatusOr<std::string> InMemoryKeyValueStore::Get(absl::string_view key,
                                                       absl::Duration timeout) {
  absl::MutexLock lock(&mu_);
  auto cond = [&]() {
    mu_.AssertHeld();
    return kv_store_.find(key) != kv_store_.end();
  };
  bool exists = mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  if (!exists) {
    return absl::NotFoundError(
        absl::StrCat(key, " is not found in the kv store."));
  }
  return kv_store_.find(key)->second;
}

absl::Status InMemoryKeyValueStore::Set(absl::string_view key,
                                        absl::string_view value) {
  absl::MutexLock lock(&mu_);
  kv_store_[key] = value;
  return absl::OkStatus();
}

}  // namespace xla
