/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/test_util.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace ifrt {
namespace test_util {

namespace {

class ClientFactory {
 public:
  void Register(std::function<StatusOr<std::unique_ptr<Client>>()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_) << "Client factory has been already registered.";
    factory_ = std::move(factory);
  }

  std::function<StatusOr<std::unique_ptr<Client>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<StatusOr<std::unique_ptr<Client>>()> factory_
      ABSL_GUARDED_BY(mu_);
};

ClientFactory& GetGlobalClientFactory() {
  static auto* const factory = new ClientFactory;
  return *factory;
}

}  // namespace

void RegisterClientFactory(
    std::function<StatusOr<std::unique_ptr<Client>>()> factory) {
  GetGlobalClientFactory().Register(std::move(factory));
}

StatusOr<std::unique_ptr<Client>> GetClient() {
  auto factory = GetGlobalClientFactory().Get();
  CHECK(factory) << "Client factory has not been registered.";
  return factory();
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
