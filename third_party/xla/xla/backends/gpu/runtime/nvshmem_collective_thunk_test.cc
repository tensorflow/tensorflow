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

#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/nvshmem_all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_send_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(CollectiveThunkTest, NvshmemAllReduceStartThunkProtoRoundTrip) {
  ThunkProto reference_proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 2
        }
        nvshmem_all_reduce_start_thunk {
          collective_config {
            operand_element_type: 11
            group_mode: 1
            use_symmetric_buffer: false
          }
          buffers {
            element_count: 5
            source_buffer {
              slice { buffer_allocation_index: 0 offset: 10 size: 20 }
              shape {}
            }
            destination_buffer {
              slice { buffer_allocation_index: 1 offset: 30 size: 40 }
              shape {}
            }
          }
          reduction_kind: 1
        }
      )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/100, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/100, /*color=*/0)};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NvshmemAllReduceThunk> thunk,
      NvshmemAllReduceThunk::FromProto(
          thunk_info, reference_proto.nvshmem_all_reduce_start_thunk(),
          buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

TEST(CollectiveThunkTest, NvshmemCollectivePermuteStartThunkProtoRoundTrip) {
  ThunkProto reference_proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 2
        }
        nvshmem_collective_permute_start_thunk {
          p2p_config {
            config {
              operand_element_type: F32
              replica_groups { replica_ids: 0 replica_ids: 1 }
              group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
            }
            id_to_source_target {
              key: 0
              value { target: 1 }
            }
            id_to_source_target {
              key: 1
              value { source: 0 }
            }
          }
          buffers {
            element_count: 5
            source_buffer {
              slice { buffer_allocation_index: 0 offset: 10 size: 20 }
              shape {}
            }
            destination_buffer {
              slice { buffer_allocation_index: 1 offset: 30 size: 40 }
              shape {}
            }
          }
          p2p_memcpy_enabled: false
        }
      )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/100, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/100, /*color=*/0)};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NvshmemCollectivePermuteThunk> thunk,
      NvshmemCollectivePermuteThunk::FromProto(
          thunk_info, reference_proto.nvshmem_collective_permute_start_thunk(),
          buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

TEST(CollectiveThunkTest, NvshmemSendThunkProtoRoundTrip) {
  ThunkProto reference_proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 2
        }
        nvshmem_send_thunk {
          p2p_config {
            config {
              operand_element_type: F32
              replica_groups { replica_ids: 0 replica_ids: 1 }
              group_mode: COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
            }
            id_to_source_target {
              key: 0
              value { target: 1 }
            }
          }
          buffer {
            element_count: 5
            source_buffer {
              slice { buffer_allocation_index: 0 offset: 10 size: 20 }
              shape {}
            }
            destination_buffer {
              slice { buffer_allocation_index: 1 offset: 30 size: 40 }
              shape {}
            }
          }
          hlo_name: "custom_send"
        }
      )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/100, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/100, /*color=*/0)};
  std::shared_ptr<NvshmemBufferAddresses> nvshmem_buffer_addresses =
      std::make_shared<NvshmemBufferAddresses>();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<NvshmemSendThunk> thunk,
                       NvshmemSendThunk::FromProto(
                           thunk_info, reference_proto.nvshmem_send_thunk(),
                           buffer_allocations, nvshmem_buffer_addresses));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

}  // namespace
}  // namespace xla::gpu
