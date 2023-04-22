/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/distributed/key_value_store.h"

namespace xla {

KeyValueStore::KeyValueStore() = default;

::grpc::Status KeyValueStore::Get(const std::string& key,
                                  absl::Duration timeout, std::string* value) {
  auto key_is_present = [&]() {
    mu_.AssertHeld();
    return entries_.find(key) != entries_.end();
  };
  absl::MutexLock lock(&mu_);
  // TODO(phawkins): the synchronization here is very coarse, but probably
  // sufficient for its current application.
  if (!mu_.AwaitWithTimeout(absl::Condition(&key_is_present), timeout)) {
    return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, key);
  }
  *value = entries_.find(key)->second;
  return ::grpc::Status::OK;
}

::grpc::Status KeyValueStore::Set(const std::string& key, std::string value) {
  absl::MutexLock lock(&mu_);
  entries_[key] = std::move(value);
  return ::grpc::Status::OK;
}

}  // namespace xla
