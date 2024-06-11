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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

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
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;

#if defined(PLATFORM_GOOGLE)
using ::testing::EquivToProto;
using ::testing::proto::Partially;
#endif

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(kClientMinVersion);
  return version;
}

class ClientTest : public ::testing::Test {
 protected:
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
          devices {
            id: 0
            local_hardware_id: 1234
            device_kind: "mock"
            default_memory_id: 0
            memory_ids: [ 0 ]
            attributes {
              key: "name"
              value { string_value: "device0" }
            }
          }
          devices {
            id: 1
            local_hardware_id: 1234
            device_kind: "mock"
            default_memory_id: 1
            memory_ids: [ 1 ]
            attributes {
              key: "name"
              value { string_value: "device1" }
            }
          }
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
    TF_ASSERT_OK_AND_ASSIGN(client_, Client::Create(rpc_helper_, response));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
  std::unique_ptr<Client> client_;
};

TEST_F(ClientTest, Init) {
  EXPECT_EQ(client_->platform_name(), "ifrt-service");
  EXPECT_EQ(client_->platform_version(), "n/a");
  EXPECT_EQ(client_->platform_id(), 42);
  EXPECT_EQ(client_->process_index(), 1);
  EXPECT_EQ(client_->runtime_type(), "proxy/ifrt-service");

  ASSERT_EQ(client_->device_count(), 2);
  ASSERT_EQ(client_->addressable_device_count(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto* const device0,
                          client_->LookupDevice(DeviceId(0)));
  EXPECT_EQ(device0->Id(), DeviceId(0));
  EXPECT_EQ(device0->Kind(), "mock");
  EXPECT_THAT(device0->Attributes(),
              ElementsAre(Pair(
                  "name", xla::PjRtDeviceAttribute(std::string("device0")))));

  ASSERT_THAT(device0->Memories(), SizeIs(1));
  auto* const memory0 = device0->Memories()[0];
  EXPECT_EQ(memory0->Id(), 0);
  EXPECT_EQ(memory0->Kind().memory_kind(), "mock");
  EXPECT_THAT(memory0->Devices(), UnorderedElementsAre(device0));
  EXPECT_THAT(device0->DefaultMemory(), IsOkAndHolds(memory0));

  TF_ASSERT_OK_AND_ASSIGN(auto* const device1,
                          client_->LookupDevice(DeviceId(1)));
  EXPECT_EQ(device1->Id(), 1);
  EXPECT_EQ(device1->Kind(), "mock");
  EXPECT_THAT(device1->Attributes(),
              ElementsAre(Pair(
                  "name", xla::PjRtDeviceAttribute(std::string("device1")))));

  ASSERT_THAT(device1->Memories(), SizeIs(1));
  auto* const memory1 = device1->Memories()[0];
  EXPECT_EQ(memory1->Id(), 1);
  EXPECT_EQ(memory1->Kind().memory_kind(), "mock");
  EXPECT_THAT(memory1->Devices(), UnorderedElementsAre(device1));
  EXPECT_THAT(device1->DefaultMemory(), IsOkAndHolds(memory1));

  EXPECT_THAT(client_->addressable_devices(), ElementsAre(device1));
}

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(ClientTest, GetDefaultDeviceAssignmentSuccess) {
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
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(ClientTest, GetDefaultDeviceAssignmentFailure) {
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(
                               get_default_device_assignment_request {
                                 num_replicas: 1
                                 num_partitions: 3
                               }
                             )pb")))))
      .WillOnce(Return(Future<ClientSession::Response>(
          absl::InternalError("injected from test"))));

  EXPECT_THAT(client_->GetDefaultDeviceAssignment(1, 3), Not(IsOk()));
}
#endif

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
