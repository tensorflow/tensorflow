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
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(HostSendRecvThunkTest, LinkSendAndSendDone) {
  Thunk::ThunkInfo thunk_info;
  Shape shape = ShapeUtil::MakeShape(F32, {1});
  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);

  HostSendThunkProto send_proto = ParseTextProtoOrDie<HostSendThunkProto>(R"pb(
    shape { element_type: F32 dimensions: 1 }
    buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
    channel_id: 42
    async_events_unique_id: 100
  )pb");

  HostSendDoneThunkProto done_proto =
      ParseTextProtoOrDie<HostSendDoneThunkProto>(R"pb(
        channel_id: 42
        async_events_unique_id: 100
      )pb");

  HostSendRecvAsyncEventsMap async_events_map;

  // It can happen that the HostSendDoneThunk is deserialized before the
  // HostSendThunk.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostSendDoneThunk> done_thunk,
      HostSendDoneThunk::FromProto(thunk_info, done_proto, buffer_allocations,
                                   async_events_map));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostSendThunk> send_thunk,
      HostSendThunk::FromProto(thunk_info, send_proto, buffer_allocations,
                               async_events_map));

  // Verify they share the same events.
  EXPECT_EQ(send_thunk->GetAsyncEventsUniqueId(),
            done_thunk->GetAsyncEventsUniqueId());

  // Verify ToProto generates the same ID for both
  ASSERT_OK_AND_ASSIGN(ThunkProto send_thunk_proto, send_thunk->ToProto());
  ASSERT_OK_AND_ASSIGN(ThunkProto done_thunk_proto, done_thunk->ToProto());

  EXPECT_EQ(send_thunk_proto.host_send_thunk().async_events_unique_id(),
            done_thunk_proto.host_send_done_thunk().async_events_unique_id());
}

TEST(HostSendRecvThunkTest, LinkRecvAndRecvDone) {
  Thunk::ThunkInfo thunk_info;
  Shape shape = ShapeUtil::MakeShape(F32, {1});
  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);

  HostRecvThunkProto recv_proto = ParseTextProtoOrDie<HostRecvThunkProto>(R"pb(
    shape { element_type: F32 dimensions: 1 }
    buffer { buffer_allocation_index: 0 offset: 0 size: 4 }
    channel_id: 42
    async_events_unique_id: 100
  )pb");

  HostRecvDoneThunkProto done_proto =
      ParseTextProtoOrDie<HostRecvDoneThunkProto>(R"pb(
        channel_id: 42
        async_events_unique_id: 100
      )pb");

  HostSendRecvAsyncEventsMap async_events_map;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostRecvThunk> recv_thunk,
      HostRecvThunk::FromProto(thunk_info, recv_proto, buffer_allocations,
                               async_events_map));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostRecvDoneThunk> done_thunk,
      HostRecvDoneThunk::FromProto(thunk_info, done_proto, buffer_allocations,
                                   async_events_map));

  // Verify they share the same events.
  EXPECT_EQ(recv_thunk->GetAsyncEventsUniqueId(),
            done_thunk->GetAsyncEventsUniqueId());

  // Verify ToProto generates the same ID for both
  ASSERT_OK_AND_ASSIGN(ThunkProto recv_thunk_proto, recv_thunk->ToProto());
  ASSERT_OK_AND_ASSIGN(ThunkProto done_thunk_proto, done_thunk->ToProto());

  EXPECT_EQ(recv_thunk_proto.host_recv_thunk().async_events_unique_id(),
            done_thunk_proto.host_recv_done_thunk().async_events_unique_id());
}

}  // namespace
}  // namespace xla::gpu
