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

#include "xla/python/ifrt_proxy/server/grpc_service_impl.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "xla/python/ifrt_proxy/client/grpc_host_buffer.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/version.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(kServerMaxVersion);
  return version;
}

// Sets up fresh GrpcServer for testing.
absl::StatusOr<std::unique_ptr<GrpcServer>> MakeGrpcServer() {
  // TODO(b/282993619): For external/GKE uses, we may need to find (or build)
  // a utility function that works similar to PickUnusedPortorDie().
  auto addr = absl::StrCat("[::1]:", tsl::testing::PickUnusedPortOrDie());
  return GrpcServer::CreateFromIfrtClientFactory(addr, []() {
    return absl::UnimplementedError(
        "IFRT client creation fails. This test is not expected to "
        "instantiate any IFRT client");
  });
}

TEST(GrpcServiceImplTest, CanBeUsedToSetupAnGrpcServer) {
  ASSERT_THAT(MakeGrpcServer(), IsOk());
  // Also implicitly tests that destruction of both the server and the
  // implementation objects.
}

class GrpcIfrtServiceImplHostBufferTest
    : public testing::TestWithParam</*size=*/int64_t> {
 protected:
  GrpcIfrtServiceImplHostBufferTest()
      : impl_([](IfrtProxyVersion version, uint64_t session_id,
                 std::shared_ptr<HostBufferStore> host_buffer_store) {
          return absl::UnimplementedError(
              "IFRT backend creation is not implemented");
        }) {
    ::grpc::ServerBuilder builder;
    builder.RegisterService(&impl_);
    server_ = builder.BuildAndStart();

    stub_ = grpc::GrpcIfrtService::NewStub(
        server_->InProcessChannel(::grpc::ChannelArguments()));
  }

  // Returns a string to be stored as a host buffer. The length is parameterized
  // so that we can test chunking.
  std::string GetTestData() const {
    std::string data;
    for (int i = 0; i < GetParam(); ++i) {
      data.push_back(i % 7);
    }
    return data;
  }

  GrpcServiceImpl impl_;
  std::unique_ptr<::grpc::Server> server_;
  std::shared_ptr<grpc::GrpcIfrtService::Stub> stub_;
};

TEST_P(GrpcIfrtServiceImplHostBufferTest, StoreAndLookupStringView) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  absl::string_view source(data);

  ASSERT_THAT(client.Store(kHandle, source).Await(), IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(), IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, StoreAndLookupCord) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();

  absl::Cord source(data);
  ASSERT_THAT(client.Store(kHandle, source).Await(), IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(), IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, Lookup) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  ASSERT_THAT(store->Store(kHandle, data), IsOk());

  EXPECT_THAT(client.Lookup(kHandle).Await(), IsOkAndHolds(data));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

TEST_P(GrpcIfrtServiceImplHostBufferTest, Delete) {
  static constexpr uint64_t kSessionId = 1;

  auto store = std::make_shared<HostBufferStore>();
  ASSERT_TRUE(impl_.Test_InsertHostBufferStore(kSessionId, store));
  GrpcClientHostBufferStore client(stub_, Version(), kSessionId);

  constexpr uint64_t kHandle = 2;
  const std::string data = GetTestData();
  ASSERT_THAT(store->Store(kHandle, data), IsOk());

  ASSERT_THAT(client.Delete(kHandle).Await(), IsOk());
  EXPECT_THAT(client.Lookup(kHandle).Await(),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_TRUE(impl_.Test_DeleteHostBufferStore(kSessionId));
}

INSTANTIATE_TEST_SUITE_P(
    DataSize, GrpcIfrtServiceImplHostBufferTest,
    testing::Values(0,                  // Empty host buffer.
                    16,                 // Small enough to fit in one chunk.
                    3 * 1024 * 1024));  // Requires multiple chunks

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
