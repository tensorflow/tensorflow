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

#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(HostSendRecvAsyncEventsTest, EmplaceAndExtractAcrossDevices) {
  se::MockStreamExecutor executor0;
  se::MockStreamExecutor executor1;
  EXPECT_CALL(executor0, device_ordinal()).WillRepeatedly(Return(0));
  EXPECT_CALL(executor1, device_ordinal()).WillRepeatedly(Return(1));

  HostSendRecvAsyncEvents events(/*devices_per_host=*/2);

  auto event0 =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(nullptr);
  auto event1 =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(nullptr);

  ASSERT_OK(events.Emplace(&executor0, /*channel_id=*/42, event0));
  ASSERT_OK(events.Emplace(&executor1, /*channel_id=*/42, event1));
  EXPECT_THAT(
      events.Emplace(&executor0, /*channel_id=*/42, event0),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("already exists")));

  ASSERT_OK_AND_ASSIGN(auto extracted0,
                       events.Extract(&executor0, /*channel_id=*/42));
  EXPECT_EQ(extracted0.GetAsyncValue(), event0.GetAsyncValue());
  EXPECT_THAT(
      events.Extract(&executor0, /*channel_id=*/42),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("was not found")));

  ASSERT_OK_AND_ASSIGN(auto extracted1,
                       events.Extract(&executor1, /*channel_id=*/42));
  EXPECT_EQ(extracted1.GetAsyncValue(), event1.GetAsyncValue());
}

TEST(HostSendRecvAsyncEventsTest, EmplaceAndExtractAcrossChannels) {
  se::MockStreamExecutor executor;
  EXPECT_CALL(executor, device_ordinal()).WillRepeatedly(Return(0));

  HostSendRecvAsyncEvents events(/*devices_per_host=*/1);
  auto send_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(nullptr);
  auto recv_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(nullptr);

  ASSERT_OK(events.Emplace(&executor, /*channel_id=*/10, send_event));
  ASSERT_OK(events.Emplace(&executor, /*channel_id=*/20, recv_event));

  ASSERT_OK_AND_ASSIGN(auto extracted_recv,
                       events.Extract(&executor, /*channel_id=*/20));
  EXPECT_EQ(extracted_recv.GetAsyncValue(), recv_event.GetAsyncValue());
  ASSERT_OK_AND_ASSIGN(auto extracted_send,
                       events.Extract(&executor, /*channel_id=*/10));
  EXPECT_EQ(extracted_send.GetAsyncValue(), send_event.GetAsyncValue());
}

TEST(HostSendRecvThunkTest, LinkSendAndSendDoneAndIgnoreLegacyId) {
  Thunk::ThunkInfo thunk_info;
  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);

  HostSendThunkProto send_proto = ParseTextProtoOrDie<HostSendThunkProto>(R"pb(
    shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
    buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
    channel_id: 42
    async_events_unique_id: 100
  )pb");

  HostSendDoneThunkProto done_proto =
      ParseTextProtoOrDie<HostSendDoneThunkProto>(R"pb(
        channel_id: 42
        async_events_unique_id: 200
      )pb");

  std::shared_ptr<HostSendRecvAsyncEvents> async_events;

  // It can happen that the HostSendDoneThunk is deserialized before the
  // HostSendThunk.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostSendDoneThunk> done_thunk,
      HostSendDoneThunk::FromProto(thunk_info, done_proto, buffer_allocations,
                                   async_events));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostSendThunk> send_thunk,
      HostSendThunk::FromProto(thunk_info, send_proto, buffer_allocations,
                               async_events));

  ASSERT_NE(async_events, nullptr);
  EXPECT_EQ(send_thunk->async_events(), async_events.get());
  EXPECT_EQ(done_thunk->async_events(), async_events.get());

  EXPECT_THAT(
      send_thunk->ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        thunk_info {}
        host_send_thunk {
          shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
          buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
          channel_id: 42
        }
      )pb")));
  EXPECT_THAT(done_thunk->ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                thunk_info {}
                host_send_done_thunk { channel_id: 42 }
              )pb")));
}

TEST(HostSendRecvThunkTest, LinkRecvAndRecvDoneAndIgnoreLegacyId) {
  Thunk::ThunkInfo thunk_info;
  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);

  HostRecvThunkProto recv_proto = ParseTextProtoOrDie<HostRecvThunkProto>(R"pb(
    shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
    buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
    channel_id: 42
    async_events_unique_id: 100
  )pb");

  HostRecvDoneThunkProto done_proto =
      ParseTextProtoOrDie<HostRecvDoneThunkProto>(R"pb(
        channel_id: 42
        async_events_unique_id: 200
      )pb");

  std::shared_ptr<HostSendRecvAsyncEvents> async_events;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostRecvThunk> recv_thunk,
      HostRecvThunk::FromProto(thunk_info, recv_proto, buffer_allocations,
                               async_events));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostRecvDoneThunk> done_thunk,
      HostRecvDoneThunk::FromProto(thunk_info, done_proto, buffer_allocations,
                                   async_events));

  ASSERT_NE(async_events, nullptr);
  EXPECT_EQ(recv_thunk->async_events(), async_events.get());
  EXPECT_EQ(done_thunk->async_events(), async_events.get());

  EXPECT_THAT(
      recv_thunk->ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        thunk_info {}
        host_recv_thunk {
          shape { element_type: F32 dimensions: 1 is_dynamic_dimension: false }
          buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
          channel_id: 42
        }
      )pb")));
  EXPECT_THAT(done_thunk->ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                thunk_info {}
                host_recv_done_thunk { channel_id: 42 }
              )pb")));
}

}  // namespace
}  // namespace xla::gpu
