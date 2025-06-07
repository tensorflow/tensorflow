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

#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(TriangularSolveThunkTest, ProtoRoundTrip) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 8
        }
        triangular_solve_thunk {
          options {
            lower: true
            left_side: true
            unit_diagonal: false
            transpose_a: TRANSPOSE
          }
          a_buffer { offset: 0 size: 256 buffer_allocation_index: 0 }
          b_buffer { offset: 0 size: 256 buffer_allocation_index: 1 }
          temp_buffer { offset: 0 size: 128 buffer_allocation_index: 2 }
          type: F32
          batch_size: 1
          m: 32
          n: 32
          a_batch_stride: 0
          b_batch_stride: 1
        }
      )pb",
      &proto));
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/1),
      BufferAllocation(/*index=*/2, /*size=*/1024, /*color=*/2)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TriangularSolveThunk> thunk,
      TriangularSolveThunk::FromProto(
          thunk_info, proto.triangular_solve_thunk(), buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
