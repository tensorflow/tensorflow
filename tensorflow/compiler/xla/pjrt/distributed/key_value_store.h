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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_H_

#include "grpcpp/grpcpp.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace xla {

// A simple blocking key-value store class.
class KeyValueStore {
 public:
  KeyValueStore();

  KeyValueStore(const KeyValueStore&) = delete;
  KeyValueStore(KeyValueStore&&) = delete;
  KeyValueStore& operator=(const KeyValueStore&) = delete;
  KeyValueStore&& operator=(KeyValueStore&&) = delete;

  // Looks up `key`. If present, returns its value. If the key is not present,
  // waits until `timeout` expires for the key to arrive. If the key does not
  // arrive by the expiry of `timeout`, returns NOT_FOUND.
  ::grpc::Status Get(const std::string& key, absl::Duration timeout,
                     std::string* value);

  // Replaces the value of `key` with `value`.
  ::grpc::Status Set(const std::string& key, std::string value);

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::string> entries_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_KEY_VALUE_STORE_H_
