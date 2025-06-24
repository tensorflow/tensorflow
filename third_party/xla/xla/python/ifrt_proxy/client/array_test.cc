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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

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
  ArrayTest()
      : session_(std::make_shared<MockClientSession>()),
        host_buffer_store_(std::make_shared<MockClientHostBufferStore>()),
        rpc_helper_(std::make_shared<RpcHelper>(Version(), session_)),
        mock_client_(std::make_shared<xla::ifrt::MockClient>()),
        mock_device_(std::make_shared<xla::ifrt::MockDevice>()),
        kLayout1(std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeDescendingLayout(1))),
        kLayout2(std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeDescendingLayout(5))) {
    rpc_helper_->set_host_buffer_store(host_buffer_store_);
  }

  void SetUp() override {
    // Default handler that ignores all uninteresting requests, but still
    // invokes the callback in order to avoid hanging the caller forever.
    EXPECT_CALL(*session_, Enqueue(_))
        .WillRepeatedly(Return(Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
    EXPECT_CALL(*host_buffer_store_, Store(_, testing::An<absl::string_view>()))
        .WillRepeatedly(Return(Future<>(absl::OkStatus())));

    ON_CALL(*mock_client_, MakeDeviceList(_))
        .WillByDefault([](absl::Span<xla::ifrt::Device* const> devices) {
          return xla::ifrt::BasicDeviceList::Create(devices);
        });
    ON_CALL(*mock_device_, client()).WillByDefault(Return(mock_client_.get()));
    sharding_ = xla::ifrt::SingleDeviceSharding::Create(
        mock_device_.get(), xla::ifrt::MemoryKind());
  }

  std::shared_ptr<xla::ifrt::SingleDeviceSharding> sharding_;
  const std::shared_ptr<MockClientSession> session_;
  const std::shared_ptr<MockClientHostBufferStore> host_buffer_store_;
  const std::shared_ptr<RpcHelper> rpc_helper_;
  const std::shared_ptr<xla::ifrt::MockClient> mock_client_;
  const std::shared_ptr<xla::ifrt::MockDevice> mock_device_;
  const std::shared_ptr<xla::PjRtLayout> kLayout1;
  const std::shared_ptr<xla::PjRtLayout> kLayout2;
};

TEST_F(ArrayTest, Destruction) {
  // Destruction may not happen immediately because of batching at the
  // client-side. This test waits until destruction happens.
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));

  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kDestructArrayRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, IfrtResponse()));
  ON_CALL(*mock_client_, GetDefaultLayout).WillByDefault(Return(kLayout1));

  tsl::MakeRef<Array>(mock_client_.get(), rpc_helper_,
                      DType(DType::Kind::kBF16), Shape({}),
                      /*sharding=*/nullptr, ArrayHandle{1234},
                      /*layout=*/nullptr);

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
  ON_CALL(*mock_client_, GetDefaultLayout).WillByDefault(Return(kLayout1));

  auto array = tsl::MakeRef<Array>(
      mock_client_.get(), rpc_helper_, DType(DType::Kind::kBF16), Shape({}),
      sharding_, ArrayHandle{1234}, /*layout=*/nullptr);

  EXPECT_THAT(array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy),
              IsOk());
  auto req = requests_queue.Pop().fully_replicated_shard_request();
  EXPECT_EQ(req.array_handle(), 1234);
  EXPECT_EQ(req.result_handle(), 1);
}

TEST_F(ArrayTest, GetDefaultLayoutSuccess) {
  ON_CALL(*mock_client_, GetDefaultLayout).WillByDefault(Return(kLayout1));

  auto array = tsl::MakeRef<Array>(
      mock_client_.get(), rpc_helper_, DType(DType::Kind::kBF16), Shape({}),
      sharding_, ArrayHandle{1234}, /*layout=*/nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto layout_1, array->layout());
  EXPECT_EQ(*layout_1, *kLayout1);
}

TEST_F(ArrayTest, GetCustomLayoutSuccess) {
  auto array = tsl::MakeRef<Array>(mock_client_.get(), rpc_helper_,
                                   DType(DType::Kind::kBF16), Shape({}),
                                   sharding_, ArrayHandle{1234}, kLayout1);
  TF_ASSERT_OK_AND_ASSIGN(auto layout_1, array->layout());
  EXPECT_EQ(*layout_1, *kLayout1);
}

TEST_F(ArrayTest, MakeArraysFromHostBufferShardsSuccess) {
  IfrtResponse response;
  const absl::string_view data = "test";
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {}
           make_arrays_from_host_buffer_shards_response {
             array_handles: 1
             array_handles: 2
           })pb",
      &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(
                  IfrtRequest::kMakeArraysFromHostBufferShardsRequest)))
      .WillRepeatedly(MockClientSessionReturnResponse(response));
  std::vector<xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs;
  xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec spec_1{
      /*buffers=*/{{/*shard_indices=*/{0}, xla::ifrt::Client::HostBuffer{
                                               /*data=*/(void*)data.data(),
                                               DType(DType::Kind::kBF16),
                                               Shape({}),
                                           }}},
      /*array_spec=*/xla::ifrt::ArraySpec{DType(DType::Kind::kBF16), Shape({}),
                                          sharding_, kLayout1}};
  xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec spec_2{
      /*buffers=*/{{/*shard_indices=*/{0},
                    /*host_buffer=*/xla::ifrt::Client::HostBuffer{
                        /*data=*/(void*)data.data(),
                        /*dtype=*/DType(DType::Kind::kBF16),
                        /*shape=*/Shape({}),
                    }}},
      /*array_spec=*/xla::ifrt::ArraySpec{DType(DType::Kind::kBF16), Shape({}),
                                          sharding_, kLayout2}};
  specs.push_back(spec_1);
  specs.push_back(spec_2);

  auto result = Array::MakeArraysFromHostBufferShards(
      mock_client_.get(), rpc_helper_, absl::MakeSpan(specs),
      xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*user_context=*/tsl::RCReference<xla::ifrt::UserContext>());
  TF_ASSERT_OK(result.status());
  TF_ASSERT_OK_AND_ASSIGN(auto layout_1, result.value().at(0)->layout());
  EXPECT_EQ(*layout_1, *kLayout1);
  TF_ASSERT_OK_AND_ASSIGN(auto layout_2, result.value().at(1)->layout());
  EXPECT_EQ(*layout_2, *kLayout2);
}

TEST_F(ArrayTest, MakeErrorArraysSuccess) {
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {}
           make_error_arrays_response { array_handles: 1 })pb",
      &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kMakeErrorArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));
  std::vector<xla::ifrt::ArraySpec> specs;
  specs.push_back(xla::ifrt::ArraySpec{DType(DType::Kind::kBF16), Shape({}),
                                       sharding_, kLayout1});

  auto result = Array::MakeErrorArrays(
      mock_client_.get(), rpc_helper_, absl::InternalError("test error"),
      absl::MakeSpan(specs),
      /*user_context=*/tsl::RCReference<xla::ifrt::UserContext>());
  TF_ASSERT_OK(result.status());
  TF_ASSERT_OK_AND_ASSIGN(auto layout, result.value().at(0)->layout());
  EXPECT_EQ(*layout, *kLayout1);
}

TEST_F(ArrayTest, AssembleArrayFromSingleDeviceArraysSuccess) {
  IfrtResponse response;
  ASSERT_TRUE(
      TextFormat::ParseFromString(
          R"pb(response_metadata {}
               assemble_array_from_single_device_arrays_response {
                 array_handle: 1
               })pb",
          &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(
                  IfrtRequest::kAssembleArrayFromSingleDeviceArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));

  auto array = tsl::MakeRef<Array>(mock_client_.get(), rpc_helper_,
                                   DType(DType::Kind::kBF16), Shape({}),
                                   sharding_, ArrayHandle{1234}, kLayout1);
  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  arrays.push_back(array);

  auto result = Array::AssembleArrayFromSingleDeviceArrays(
      mock_client_.get(), rpc_helper_, DType(DType::Kind::kBF16), Shape({}),
      sharding_, absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy,
      SingleDeviceShardSemantics::kAllShards);
  TF_ASSERT_OK(result.status());
  TF_ASSERT_OK_AND_ASSIGN(auto layout, result.value()->layout());
  EXPECT_EQ(*layout, *kLayout1);
}

TEST_F(ArrayTest, AssembleArrayFromSingleDeviceArraysDefaultLayoutSuccess) {
  IfrtResponse response;
  ASSERT_TRUE(
      TextFormat::ParseFromString(
          R"pb(response_metadata {}
               assemble_array_from_single_device_arrays_response {
                 array_handle: 1
               })pb",
          &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(
                  IfrtRequest::kAssembleArrayFromSingleDeviceArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));
  ON_CALL(*mock_client_, GetDefaultLayout).WillByDefault(Return(kLayout1));

  auto array = tsl::MakeRef<Array>(
      mock_client_.get(), rpc_helper_, DType(DType::Kind::kBF16), Shape({}),
      sharding_, ArrayHandle{1234}, /*layout=*/nullptr);
  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  arrays.push_back(array);

  auto result = Array::AssembleArrayFromSingleDeviceArrays(
      mock_client_.get(), rpc_helper_, DType(DType::Kind::kBF16), Shape({}),
      sharding_, absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy,
      SingleDeviceShardSemantics::kAllShards);
  TF_ASSERT_OK(result.status());
  TF_ASSERT_OK_AND_ASSIGN(auto layout, result.value()->layout());
  EXPECT_EQ(*layout, *kLayout1);
}

TEST_F(ArrayTest, RemapArraysSuccess) {
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {}
           remap_arrays_response { array_handles: 1 array_handles: 2 })pb",
      &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kRemapArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));

  auto array_1 = tsl::MakeRef<Array>(mock_client_.get(), rpc_helper_,
                                     DType(DType::Kind::kBF16), Shape({}),
                                     sharding_, ArrayHandle{1234}, kLayout1);
  auto array_2 = tsl::MakeRef<Array>(mock_client_.get(), rpc_helper_,
                                     DType(DType::Kind::kBF16), Shape({}),
                                     sharding_, ArrayHandle{1234}, kLayout2);
  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  arrays.push_back(array_1);
  arrays.push_back(array_2);
  std::vector<RemapPlan::Mapping> mappings;
  mappings.push_back({/*in_array=*/0, /*out_array=*/1});
  mappings.push_back({/*in_array=*/1, /*out_array=*/0});
  std::vector<xla::ifrt::ArraySpec> input_specs;
  input_specs.push_back(xla::ifrt::ArraySpec{DType(DType::Kind::kBF16),
                                             Shape({}), sharding_, kLayout1});
  input_specs.push_back(xla::ifrt::ArraySpec{DType(DType::Kind::kBF16),
                                             Shape({}), sharding_, kLayout2});
  std::vector<xla::ifrt::ArraySpec> output_specs;
  output_specs.push_back(
      xla::ifrt::ArraySpec{DType(DType::Kind::kBF16), Shape({}), sharding_});
  output_specs.push_back(
      xla::ifrt::ArraySpec{DType(DType::Kind::kBF16), Shape({}), sharding_});
  RemapPlan plan{input_specs, output_specs,
                 std::make_shared<std::vector<RemapPlan::Mapping>>(mappings)};

  absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>> result =
      Array::RemapArrays(mock_client_.get(), rpc_helper_, plan,
                         absl::MakeSpan(arrays),
                         ArrayCopySemantics::kAlwaysCopy);

  TF_ASSERT_OK(result.status());
  TF_ASSERT_OK_AND_ASSIGN(auto layout_1, result.value().at(0)->layout());
  EXPECT_EQ(*layout_1, *kLayout2);
  TF_ASSERT_OK_AND_ASSIGN(auto layout_2, result.value().at(1)->layout());
  EXPECT_EQ(*layout_2, *kLayout1);
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
