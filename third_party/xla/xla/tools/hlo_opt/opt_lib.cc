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

#include "xla/tools/hlo_opt/opt_lib.h"

#include "absl/container/flat_hash_map.h"
#include "xla/types.h"

namespace xla {

using ProviderMap =
    absl::flat_hash_map<se::Platform::Id, std::unique_ptr<OptProvider>>;
static absl::Mutex provider_mu(absl::kConstInit);

static ProviderMap& GetProviderMap() {
  static auto& provider_map = *new ProviderMap();
  return provider_map;
}

/*static*/ void OptProvider::RegisterForPlatform(
    se::Platform::Id platform,
    std::unique_ptr<OptProvider> translate_provider) {
  absl::MutexLock l(&provider_mu);
  CHECK(!GetProviderMap().contains(platform));
  GetProviderMap()[platform] = std::move(translate_provider);
}

/*static*/ OptProvider* OptProvider::ProviderForPlatform(
    se::Platform::Id platform) {
  absl::MutexLock l(&provider_mu);
  auto it = GetProviderMap().find(platform);
  if (it == GetProviderMap().end()) {
    return nullptr;
  }

  return it->second.get();
}

}  // namespace xla
