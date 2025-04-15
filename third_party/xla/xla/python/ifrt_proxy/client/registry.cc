// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/registry.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

using FactoryFn =
    std::function<absl::StatusOr<std::unique_ptr<xla::ifrt::Client>>(
        absl::string_view, const ClientConnectionOptions&)>;

struct Registry {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, FactoryFn> factories ABSL_GUARDED_BY(mu);
};

Registry* registry() {
  static auto* r = new Registry();
  return r;
}

}  // namespace

void RegisterClientFactory(absl::string_view transport_name,
                           FactoryFn factory) {
  absl::MutexLock l(&registry()->mu);
  const bool inserted =
      registry()
          ->factories.insert({std::string(transport_name), factory})
          .second;
  CHECK(inserted) << "IFRT proxy transport '" << transport_name
                  << "' already registered";
}

absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateClient(
    absl::string_view proxy_server_address,
    const ClientConnectionOptions& options) {
  const size_t pos = proxy_server_address.find("://");
  if (pos == std::string::npos) {
    return absl::InvalidArgumentError(
        absl::StrCat("IFRT proxy server address must be "
                     "'<transport-type>://<backend-address>' (e.g., "
                     "'grpc://localhost'), but got ",
                     proxy_server_address));
  }

  const absl::string_view transport_name = proxy_server_address.substr(0, pos);
  const absl::string_view address = proxy_server_address.substr(pos + 3);
  LOG(INFO) << "Attempting to create IFRT proxy client with transport name "
            << transport_name << " to address '" << address
            << "' and with global client flags " << *GetGlobalClientFlags();

  FactoryFn factory;
  {
    absl::MutexLock l(&registry()->mu);
    const auto it = registry()->factories.find(transport_name);
    if (it == registry()->factories.end()) {
      return absl::NotFoundError(
          absl::StrCat("IFRT proxy transport '", transport_name,
                       "' not found; available transports are: ",
                       absl::StrJoin(registry()->factories, ", ",
                                     [](std::string* out, const auto& it) {
                                       out->append(it.first);
                                     })));
    }
    factory = it->second;
  }

  return factory(address, options);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
