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

#include "xla/python/ifrt_proxy/client/array.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

using ::testing::_;
using ::testing::Pointee;
using ::testing::Return;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;

#if defined(PLATFORM_GOOGLE)
using ::testing::EquivToProto;
using ::testing::proto::Partially;
#endif

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(kClientMaxVersion);
  return version;
}

class ArrayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    // Default handler that ignores all uninteresting requests, but still
    // invokes the callback in order to avoid hanging the caller forever.
    EXPECT_CALL(*session_, Enqueue(_))
        .WillRepeatedly(Return(Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
};

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(ArrayTest, Destruction) {
  // Destruction may not happen immediately because of batching at the
  // client-side. This test waits until destruction happens.
  absl::Notification destructed;
  EXPECT_CALL(
      *session_,
      Enqueue(Pointee(Partially(EquivToProto(R"pb(destruct_array_request {
                                                    array_handle: 1234
                                                  })pb")))))
      .WillOnce([&](std::unique_ptr<IfrtRequest> request)
                    -> Future<ClientSession::Response> {
        destructed.Notify();
        auto result = std::make_shared<IfrtResponse>();
        return Future<ClientSession::Response>(result);
      });

  MockClient client;
  tsl::MakeRef<Array>(&client, rpc_helper_, DType(DType::Kind::kBF16),
                      Shape({}), /*sharding=*/nullptr, ArrayHandle{1234});

  ASSERT_TRUE(destructed.WaitForNotificationWithTimeout(absl::Seconds(10)));
}
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(ArrayTest, FullyReplicatedShard) {
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {}
           fully_replicated_shard_response { array_handle: 1 })pb",
      &response));

  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(fully_replicated_shard_request {
                                    array_handle: 1234
                                    result_handle: 1
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  MockClient client;
  MockDevice mock_device;

  auto sharding = xla::ifrt::SingleDeviceSharding::Create(
      &mock_device, xla::ifrt::MemoryKind());

  auto array =
      tsl::MakeRef<Array>(&client, rpc_helper_, DType(DType::Kind::kBF16),
                          Shape({}), std::move(sharding), ArrayHandle{1234});

  ASSERT_THAT(array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy),
              IsOk());
}
#endif

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
