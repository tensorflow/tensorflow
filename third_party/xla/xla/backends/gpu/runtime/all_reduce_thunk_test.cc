/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        all_reduce_start_thunk {
          async_events_unique_id: 3
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};

  CollectiveThunk::AsyncEventsMap async_events_map;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AllReduceStartThunk> thunk,
      AllReduceStartThunk::FromProto(thunk_info, proto.all_reduce_start_thunk(),
                                     buffer_allocations, async_events_map));
  ASSERT_NE(thunk->async_events(), nullptr);

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  // Ids are unique and expected to differ.
  proto.mutable_all_reduce_start_thunk()->set_async_events_unique_id(
      round_trip_proto.all_reduce_start_thunk().async_events_unique_id());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(CollectiveThunkTest, SyncCollective) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        all_reduce_start_thunk {
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};

  CollectiveThunk::AsyncEventsMap async_events_map;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AllReduceStartThunk> thunk,
      AllReduceStartThunk::FromProto(thunk_info, proto.all_reduce_start_thunk(),
                                     buffer_allocations, async_events_map));
  ASSERT_EQ(thunk->async_events(), nullptr);
}

TEST(ReduceScatterStartThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        reduce_scatter_start_thunk {
          async_events_unique_id: 3
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};

  CollectiveThunk::AsyncEventsMap async_events_map;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ReduceScatterStartThunk> thunk,
                       ReduceScatterStartThunk::FromProto(
                           thunk_info, proto.reduce_scatter_start_thunk(),
                           buffer_allocations, async_events_map));
  ASSERT_NE(thunk->async_events(), nullptr);

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  // Ids are unique and expected to differ.
  proto.mutable_reduce_scatter_start_thunk()->set_async_events_unique_id(
      round_trip_proto.reduce_scatter_start_thunk().async_events_unique_id());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
