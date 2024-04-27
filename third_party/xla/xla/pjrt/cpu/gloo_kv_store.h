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

#ifndef XLA_PJRT_CPU_GLOO_KV_STORE_H_
#define XLA_PJRT_CPU_GLOO_KV_STORE_H_

#include <chrono>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "third_party/gloo/gloo/rendezvous/store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"

namespace xla::cpu {

class GlooKeyValueStore : public ::gloo::rendezvous::Store {
 public:
  explicit GlooKeyValueStore(std::shared_ptr<KeyValueStoreInterface> kv_store);
  ~GlooKeyValueStore() override;

  void set(const std::string& key, const std::vector<char>& data) override;

  std::vector<char> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

 private:
  std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Duration kv_get_timeout_ = absl::Minutes(1);
};

}  // namespace xla::cpu

#endif  // XLA_PJRT_CPU_GLOO_KV_STORE_H_
