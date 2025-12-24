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

#include "xla/backends/gpu/runtime/memset_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(MemzeroThunkTest, ProtoRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        memzero_thunk {
          dest_buffer {
            slice { offset: 0 size: 4 buffer_allocation_index: 0 }
            shape {
              dimensions: 1
              element_type: F32
              is_dynamic_dimension: false
            }
          }
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemzeroThunk> thunk,
      MemzeroThunk::FromProto(thunk_info, proto.memzero_thunk(),
                              buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(Memset32BitValueThunkTest, ProtoRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        memset32bit_value_thunk {
          dest_buffer { offset: 0 size: 4 buffer_allocation_index: 0 }
          value: 123
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Memset32BitValueThunk> thunk,
      Memset32BitValueThunk::FromProto(
          thunk_info, proto.memset32bit_value_thunk(), buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
