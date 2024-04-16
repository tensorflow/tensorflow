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

#include "xla/python/ifrt_proxy/server/grpc_server.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::Not;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

// A fake IFRT service that fails all the Session creation attempts.
class FakeIfrtService : public grpc::GrpcIfrtService::Service {};

TEST(GrpcServerTest, CreationTest) {
  auto addr = absl::StrCat("[::1]:", tsl::testing::PickUnusedPortOrDie());
  auto grpc_service_impl = std::make_unique<FakeIfrtService>();
  ASSERT_THAT(GrpcServer::Create(addr, std::move(grpc_service_impl)), IsOk());
  // Also implicitly tests that the destruction of the GrpcServer object.
}

TEST(GrpcServerTest, CreationFailsIfImplIsNullptr) {
  auto addr = absl::StrCat("[::1]:", tsl::testing::PickUnusedPortOrDie());
  EXPECT_THAT(GrpcServer::Create(addr, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GrpcServerTest, CreationFailsWithInvalidAddress) {
  auto grpc_service_impl = std::make_unique<FakeIfrtService>();
  EXPECT_THAT(GrpcServer::Create(/*address=*/"invalid-address",
                                 std::move(grpc_service_impl)),
              Not(IsOk()));
}

TEST(GrpcServerTest, RetrievingServerAddressWorks) {
  auto addr = absl::StrCat("[::1]:", tsl::testing::PickUnusedPortOrDie());
  auto grpc_service_impl = std::make_unique<FakeIfrtService>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto grpc_server, GrpcServer::Create(addr, std::move(grpc_service_impl)));
  EXPECT_EQ(grpc_server->address(), addr);
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
