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

#include "xla/backends/gpu/runtime/nvshmem_recv_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(NvshmemRecvThunkTest, ProtoRoundTrip) {
  ThunkProto reference_proto =
      tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
          R"pb(
            thunk_info {
              profile_annotation: "profile_annotation"
              execution_stream_id: 1
            }
            nvshmem_recv_thunk {
              config {
                config {
                  operand_element_type: F32
                  replica_groups { replica_ids: 0 replica_ids: 1 }
                  group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
                  use_symmetric_buffer: false
                }
                id_to_source_target {
                  key: 0
                  value { source: 1 target: 1 }
                }
              }
              buffer {
                element_count: 10
                source_buffer {
                  shape {}
                  slice { buffer_allocation_index: 0 offset: 0 size: 40 }
                }
                destination_buffer {
                  shape {}
                  slice { buffer_allocation_index: 0 offset: 40 size: 40 }
                }
                source_memory_space: 0
                destination_memory_space: 0
              }
              hlo_name: "hlo_name"
            }
          )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/80, /*color=*/0)};
  std::shared_ptr<NvshmemBufferAddresses> nvshmem_buffer_addresses =
      std::make_shared<NvshmemBufferAddresses>();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<NvshmemRecvThunk> thunk,
                       NvshmemRecvThunk::FromProto(
                           thunk_info, reference_proto.nvshmem_recv_thunk(),
                           buffer_allocations, nvshmem_buffer_addresses));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

}  // namespace
}  // namespace xla::gpu
