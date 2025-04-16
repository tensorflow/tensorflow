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
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
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
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;

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

TEST_F(ArrayTest, Destruction) {
  // Destruction may not happen immediately because of batching at the
  // client-side. This test waits until destruction happens.
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));

  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kDestructArrayRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, IfrtResponse()));

  MockClient client;
  tsl::MakeRef<Array>(&client, rpc_helper_, DType(DType::Kind::kBF16),
                      Shape({}), /*sharding=*/nullptr, ArrayHandle{1234});

  auto destruct_array_request = requests_queue.Pop().destruct_array_request();
  EXPECT_THAT(destruct_array_request.array_handle(), ElementsAre(1234));
}

TEST_F(ArrayTest, FullyReplicatedShard) {
  IfrtResponse response;
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));

  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {}
           fully_replicated_shard_response { array_handle: 1 })pb",
      &response));

  EXPECT_CALL(
      *session_,
      Enqueue(IfrtRequestOfType(IfrtRequest::kFullyReplicatedShardRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  MockClient client;
  ON_CALL(client, MakeDeviceList(_))
      .WillByDefault([](absl::Span<xla::ifrt::Device* const> devices) {
        return xla::ifrt::BasicDeviceList::Create(devices);
      });

  MockDevice mock_device;
  ON_CALL(mock_device, client()).WillByDefault(Return(&client));

  auto sharding = xla::ifrt::SingleDeviceSharding::Create(
      &mock_device, xla::ifrt::MemoryKind());

  auto array =
      tsl::MakeRef<Array>(&client, rpc_helper_, DType(DType::Kind::kBF16),
                          Shape({}), std::move(sharding), ArrayHandle{1234});

  EXPECT_THAT(array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy),
              IsOk());
  auto req = requests_queue.Pop().fully_replicated_shard_request();
  EXPECT_EQ(req.array_handle(), 1234);
  EXPECT_EQ(req.result_handle(), 1);
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
