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

// This file registers a factory with `xla::ifrt::test_util` that will spawn a
// IFRT proxy client connected to an instance of a proxy server that is backed
// by the IFRT-PjRt-CPU backend.
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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
namespace {

absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateIfrtBackendClient(
    AttributeMap initialization_data) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 4;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> pjrt_cpu_client,
                      xla::GetXlaPjrtCpuClient(options));
  return xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client));
}

class ClientAndServer {
 public:
  explicit ClientAndServer() {
    std::string address =
        absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
    server_ = GrpcServer::CreateFromIfrtClientFactory(address,
                                                      CreateIfrtBackendClient);
    if (!server_.ok()) {
      client_ = server_.status();
      return;
    }
    client_ = CreateClient(absl::StrCat("grpc://", address));
  }

  absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> client() {
    return client_;
  }

 private:
  absl::StatusOr<std::unique_ptr<GrpcServer>> server_;
  absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> client_;
};

// NOTE: A prior version of this file destructed the client and the server
// at the end of each test case, but that triggered a non-rootcaused bug
// (b/450109296). We work around the bug by keeping a singleton that is reused
// across test cases and is not destroyed.
absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetSingletonClient() {
  static absl::NoDestructor<ClientAndServer> client_and_server;
  return client_and_server->client();
}

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(GetSingletonClient), true);

}  // namespace
}  // namespace test_util
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
