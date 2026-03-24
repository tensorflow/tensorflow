/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt_proxy/integration_tests/scoped_pjrt_cpu_via_proxy.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace test_util {

absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateIfrtBackendClient(
    AttributeMap initialization_data) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 4;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> pjrt_cpu_client,
                      xla::GetXlaPjrtCpuClient(options));
  return xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client));
}

ScopedPjRtCpuViaProxy::ScopedPjRtCpuViaProxy() {
  std::string address =
      absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
  server_ = *GrpcServer::CreateFromIfrtClientFactory(address,
                                                     CreateIfrtBackendClient);
  client_ = *CreateClient(absl::StrCat("grpc://", address));

  auto client_weak = std::weak_ptr<xla::ifrt::Client>(client_);
  std::function<absl::StatusOr<std::shared_ptr<xla::ifrt::Client>>()>
      client_factory = [client_weak = std::move(client_weak)]()
      -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
    auto result = client_weak.lock();
    if (result == nullptr) {
      return absl::InternalError("Client has already been destroyed!");
    }
    return result;
  };
  xla::ifrt::test_util::RegisterClientFactory(std::move(client_factory));
}

ScopedPjRtCpuViaProxy::~ScopedPjRtCpuViaProxy() {
  LOG(INFO)
      << "ScopedPjRtCpuViaProxy::~ScopedPjRtCpuViaProxy: Destroying client.";
  auto client_weak = std::weak_ptr<xla::ifrt::Client>(client_);
  client_ = nullptr;
  CHECK(client_weak.expired());
  LOG(INFO) << "ScopedPjRtCpuViaProxy::~ScopedPjRtCpuViaProxy: Client "
            << "destroyed; destroying server.";
  server_ = nullptr;
  LOG(INFO)
      << "ScopedPjRtCpuViaProxy::~ScopedPjRtCpuViaProxy: Server destroyed.";
}

}  // namespace test_util
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
