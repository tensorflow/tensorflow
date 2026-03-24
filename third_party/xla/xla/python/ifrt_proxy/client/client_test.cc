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

#include "xla/python/ifrt_proxy/client/client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

using ::tsl::proto_testing::EquivToProto;
using ::tsl::proto_testing::Partially;

class ClientTest : public ::testing::TestWithParam</*protocol_version=*/int> {
 protected:
  ClientTest()
      : layout_1_(std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeDescendingLayout(3))),
        layout_2_(std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeDescendingLayout(5))) {}

  IfrtProxyVersion Version() {
    IfrtProxyVersion version;
    version.set_protocol_version(GetParam());
    // TODO(hyeontaek): For a more realistic test setup, the IFRT SerDes version
    // should vary by the IFRT Proxy protocol version.
    version.set_ifrt_serdes_version_number(
        SerDesVersion::current().version_number().value());
    return version;
  }

  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    InitResponse response;
    ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
        R"pb(
          platform_name: "ifrt-service"
          platform_version: "n/a"
          platform_id: 42
          process_index: 1
          runtime_type: "ifrt-service"
          all_devices {
            id: 0
            local_hardware_id: 1234
            device_kind: "mock"
            default_memory_id: 0
            memory_ids: [ 0 ]
            attributes {
              attributes {
                key: "name"
                value { string_value: "device0" }
              }
            }
          }
          all_devices {
            id: 1
            local_hardware_id: 1234
            device_kind: "mock"
            default_memory_id: 1
            memory_ids: [ 1 ]
            attributes {
              attributes {
                key: "name"
                value { string_value: "device1" }
              }
            }
          }
          primary_device_ids: [ 0, 1 ]
          addressable_device_ids: 1
          memories {
            id: 0
            memory_space_kind: "mock"
            kind_id: 0
            device_ids: [ 0 ]
          }
          memories {
            id: 1
            memory_space_kind: "mock"
            kind_id: 1
            device_ids: [ 1 ]
          }
        )pb",
        &response));

    AttributeMap::Map client_attributes(
        {{"test_key", AttributeMap::StringValue("test_value")}});
    AttributeMap(client_attributes)
        .ToProto(*response.mutable_client_attributes(),
                 rpc_helper_->ifrt_serdes_version());

    TF_ASSERT_OK_AND_ASSIGN(client_, Client::Create(rpc_helper_, response));
    TF_ASSERT_OK_AND_ASSIGN(device_, client_->LookupDevice(DeviceId(0)));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
  std::unique_ptr<Client> client_;
  std::shared_ptr<xla::PjRtLayout> layout_1_;
  std::shared_ptr<xla::PjRtLayout> layout_2_;
  xla::ifrt::Device* device_;
};

TEST_P(ClientTest, Init) {
  EXPECT_EQ(client_->platform_name(), "ifrt-service");
  EXPECT_EQ(client_->platform_version(), "n/a");
  EXPECT_EQ(client_->platform_id(), 42);
  EXPECT_EQ(client_->process_index(), 1);
  EXPECT_EQ(client_->runtime_type(), "proxy/ifrt-service");
  EXPECT_THAT(client_->Attributes().Get<std::string>("test_key"),
              absl_testing::IsOkAndHolds("test_value"));

  ASSERT_EQ(client_->device_count(), 2);
  ASSERT_EQ(client_->addressable_device_count(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto* const device0,
                          client_->LookupDevice(DeviceId(0)));
  EXPECT_EQ(device0->Id(), DeviceId(0));
  EXPECT_EQ(device0->Kind(), "mock");
  EXPECT_THAT(device0->Attributes().Get<std::string>("name"),
              absl_testing::IsOkAndHolds("device0"));

  ASSERT_THAT(device0->Memories(), SizeIs(1));
  auto* const memory0 = device0->Memories()[0];
  EXPECT_EQ(memory0->Id(), 0);
  EXPECT_EQ(memory0->Kind().memory_kind(), "mock");
  EXPECT_THAT(memory0->Devices(), UnorderedElementsAre(device0));
  EXPECT_THAT(device0->DefaultMemory(), absl_testing::IsOkAndHolds(memory0));

  TF_ASSERT_OK_AND_ASSIGN(auto* const device1,
                          client_->LookupDevice(DeviceId(1)));
  EXPECT_EQ(device1->Id(), 1);
  EXPECT_EQ(device1->Kind(), "mock");
  EXPECT_THAT(device1->Attributes().Get<std::string>("name"),
              absl_testing::IsOkAndHolds("device1"));

  ASSERT_THAT(device1->Memories(), SizeIs(1));
  auto* const memory1 = device1->Memories()[0];
  EXPECT_EQ(memory1->Id(), 1);
  EXPECT_EQ(memory1->Kind().memory_kind(), "mock");
  EXPECT_THAT(memory1->Devices(), UnorderedElementsAre(device1));
  EXPECT_THAT(device1->DefaultMemory(), absl_testing::IsOkAndHolds(memory1));

  EXPECT_THAT(client_->addressable_devices(), ElementsAre(device1));
}

TEST_P(ClientTest, GetDefaultLayoutSuccess) {
  xla::PjRtLayout layout(xla::LayoutUtil::MakeDescendingLayout(3));
  IfrtResponse response;
  response.mutable_get_default_layout_response()->set_serialized_pjrt_layout(
      layout.Serialize());
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kGetDefaultLayoutRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));

  TF_ASSERT_OK_AND_ASSIGN(
      auto resolved_layout,
      client_->GetDefaultPjRtLayout(DType(DType::kF64), {1, 2, 3}, device_,
                                    MemoryKind("mock")));
  EXPECT_EQ(resolved_layout->ToString(), layout.ToString());
}

TEST_P(ClientTest, GetCachedDefaultLayoutSuccess) {
  IfrtResponse response;
  response.mutable_get_default_layout_response()->set_serialized_pjrt_layout(
      layout_1_->Serialize());
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kGetDefaultLayoutRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));

  TF_ASSERT_OK_AND_ASSIGN(
      auto resolved_layout,
      client_->GetDefaultPjRtLayout(DType(DType::kF64), {1, 2, 3}, device_,
                                    MemoryKind("mock")));
  EXPECT_EQ(resolved_layout->ToString(), layout_1_->ToString());

  TF_ASSERT_OK_AND_ASSIGN(resolved_layout, client_->GetDefaultPjRtLayout(
                                               DType(DType::kF64), {1, 2, 3},
                                               device_, MemoryKind("mock")));
  EXPECT_EQ(resolved_layout->ToString(), layout_1_->ToString());
}

TEST_P(ClientTest, GetDefaultLayoutFailure) {
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kGetDefaultLayoutRequest)))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          absl::InternalError("injected from test"))));

  EXPECT_THAT(client_->GetDefaultPjRtLayout(DType(DType::kF64), {1, 2, 3},
                                            device_, MemoryKind("mock")),
              Not(absl_testing::IsOk()));
}

TEST_P(ClientTest, CopyArraysDefaultLayoutSuccess) {
  std::shared_ptr<xla::ifrt::SingleDeviceSharding> sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_, xla::ifrt::MemoryKind());
  auto array0 = tsl::MakeRef<Array>(
      client_.get(), rpc_helper_, DType(DType::kF64), Shape({1, 2, 3}),
      sharding, ArrayHandle{1234}, /*layout=*/nullptr);
  auto sharding1 = SingleDeviceSharding::Create(device_, MemoryKind("mock"));
  auto array1 = tsl::MakeRef<Array>(
      client_.get(), rpc_helper_, DType(DType::kF64), Shape({1, 2, 3}),
      sharding, ArrayHandle{5678}, /*layout=*/nullptr);

  IfrtResponse response;
  response.mutable_copy_arrays_response()->add_array_handles(1);
  response.mutable_copy_arrays_response()->add_array_handles(2);

  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kCopyArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kDestructArrayRequest)))
      .WillRepeatedly(MockClientSessionReturnResponse(IfrtResponse()));

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays = {array0, array1};
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client_->MakeDeviceList({device_}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto copied_arrays,
      client_->CopyArrays(absl::MakeSpan(arrays), std::move(device_list),
                          MemoryKind("mock"), ArrayCopySemantics::kAlwaysCopy));
  ASSERT_THAT(copied_arrays, SizeIs(2));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout_1,
                          copied_arrays[0].get()->pjrt_layout());
  EXPECT_EQ(layout_1, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout_2,
                          copied_arrays[1].get()->pjrt_layout());
  EXPECT_EQ(layout_2, nullptr);
}

TEST_P(ClientTest, CopyArraysCustomLayoutSuccess) {
  std::shared_ptr<xla::ifrt::SingleDeviceSharding> sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_, xla::ifrt::MemoryKind());
  auto array0 = tsl::MakeRef<Array>(client_.get(), rpc_helper_,
                                    DType(DType::kF64), Shape({1, 2, 3}),
                                    sharding, ArrayHandle{1234}, layout_1_);
  auto sharding1 = SingleDeviceSharding::Create(device_, MemoryKind("mock"));
  auto array1 = tsl::MakeRef<Array>(client_.get(), rpc_helper_,
                                    DType(DType::kF64), Shape({1, 2, 3}),
                                    sharding, ArrayHandle{5678}, layout_2_);

  IfrtResponse response;
  response.mutable_copy_arrays_response()->add_array_handles(1);
  response.mutable_copy_arrays_response()->add_array_handles(2);

  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kCopyArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kDestructArrayRequest)))
      .WillRepeatedly(MockClientSessionReturnResponse(IfrtResponse()));

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays = {array0, array1};
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client_->MakeDeviceList({device_}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto copied_arrays,
      client_->CopyArrays(absl::MakeSpan(arrays), std::move(device_list),
                          MemoryKind("mock"), ArrayCopySemantics::kAlwaysCopy));
  ASSERT_THAT(copied_arrays, SizeIs(2));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout_1,
                          copied_arrays[0].get()->pjrt_layout());
  EXPECT_EQ(layout_1->ToString(), layout_1_->ToString());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout_2,
                          copied_arrays[1].get()->pjrt_layout());
  EXPECT_EQ(layout_2->ToString(), layout_2_->ToString());
}

TEST_P(ClientTest, GetDefaultDeviceAssignmentSuccess) {
  IfrtResponse response;
  xla::DeviceAssignment assignment(1, 3);
  assignment.Serialize(
      response.mutable_get_default_device_assignment_response()
          ->mutable_device_assignment());

  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(
                               get_default_device_assignment_request {
                                 num_replicas: 1
                                 num_partitions: 3
                               }
                             )pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  TF_ASSERT_OK_AND_ASSIGN(auto assignment_got,
                          client_->GetDefaultDeviceAssignment(1, 3));
  EXPECT_EQ(assignment_got.replica_count(), 1);
  EXPECT_EQ(assignment_got.computation_count(), 3);
}

TEST_P(ClientTest, GetDefaultDeviceAssignmentFailure) {
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(
                               get_default_device_assignment_request {
                                 num_replicas: 1
                                 num_partitions: 3
                               }
                             )pb")))))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          absl::InternalError("injected from test"))));

  EXPECT_THAT(client_->GetDefaultDeviceAssignment(1, 3),
              Not(absl_testing::IsOk()));
}

TEST_P(ClientTest, ReshardArraysSuccess) {
  std::shared_ptr<xla::ifrt::SingleDeviceSharding> sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_, xla::ifrt::MemoryKind());
  auto array = tsl::MakeRef<Array>(client_.get(), rpc_helper_,
                                   DType(DType::kF64), Shape({1, 2, 3}),
                                   sharding, ArrayHandle{1234}, layout_1_);

  IfrtResponse response;
  response.mutable_reshard_arrays_response()->add_array_handles(1);

  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kReshardArraysRequest)))
      .WillOnce(MockClientSessionReturnResponse(response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kDestructArrayRequest)))
      .WillRepeatedly(MockClientSessionReturnResponse(IfrtResponse()));

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays = {array};
  std::vector<xla::ifrt::ArraySpec> specs;
  specs.push_back({array->dtype(), array->shape(), sharding, layout_2_});

  TF_ASSERT_OK_AND_ASSIGN(
      auto reshared_arrays,
      client_->ReshardArrays(absl::MakeSpan(arrays), absl::MakeSpan(specs),
                             ArrayCopySemantics::kAlwaysCopy));
  ASSERT_THAT(reshared_arrays, SizeIs(1));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout,
                          reshared_arrays[0]->pjrt_layout());
  EXPECT_EQ(layout->ToString(), layout_2_->ToString());
}

INSTANTIATE_TEST_SUITE_P(
    ClientTestWithAllVersions, ClientTest,
    testing::Range(protocol_version::kClientMin,
                   protocol_version::kClientMax + 1),
    [](const testing::TestParamInfo<ClientTest::ParamType>& info) {
      return absl::StrCat(info.param);
    });

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
