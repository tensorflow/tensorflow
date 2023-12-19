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

#include "xla/pjrt/cpu/gloo_kv_store.h"

#include <chrono>  // NOLINT
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gloo/gloo/rendezvous/store.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"

namespace xla::cpu {

GlooKeyValueStore::GlooKeyValueStore(PjRtClient::KeyValueGetCallback kv_get,
                                     PjRtClient::KeyValuePutCallback kv_put)
    : kv_get_(kv_get), kv_put_(kv_put) {}

GlooKeyValueStore::~GlooKeyValueStore() = default;

void GlooKeyValueStore::set(const std::string& key,
                            const std::vector<char>& data) {
  ThrowIfError(kv_put_(key, std::string_view(data.data(), data.size())));
}

std::vector<char> GlooKeyValueStore::get(const std::string& key) {
  std::string result = ValueOrThrow(kv_get_(key, kv_get_timeout_));
  std::vector<char> data(result.begin(), result.end());
  return data;
}

void GlooKeyValueStore::wait(const std::vector<std::string>& keys) {
  wait(keys, Store::kDefaultTimeout);
}

void GlooKeyValueStore::wait(const std::vector<std::string>& keys,
                             const std::chrono::milliseconds& timeout) {
  // TODO(phawkins): add a wait-many feature to the distributed service.
  absl::Time deadline = absl::Now() + absl::FromChrono(timeout);
  for (const std::string& key : keys) {
    absl::Time now = absl::Now();
    if (now >= deadline) {
      throw std::runtime_error("Deadline exceeded in wait()");
    }
    ThrowIfError(kv_get_(key, deadline - now).status());
  }
}

}  // namespace xla::cpu
